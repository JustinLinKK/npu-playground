from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .providers import ProviderError, ProviderResponse, _strict_json_schema, normalize_usage
from .validation import json_digest


MODEL_LICENSES = {
    "deepseek/deepseek-v3.2": "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/LICENSE",
    "qwen/qwen3-32b": "https://huggingface.co/Qwen/Qwen3-32B/blob/main/LICENSE",
    "qwen/qwen3-coder-30b-a3b-instruct": "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/LICENSE",
}


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["deepseek/deepseek-v3.2", "qwen/qwen3-32b", "qwen/qwen3-coder-30b-a3b-instruct"] = "deepseek/deepseek-v3.2"
    endpoint: str = "alibaba/fp8"
    upstream_provider: str = "Alibaba"
    reasoning: bool = True
    temperature: float = Field(default=0.6, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_tokens: int = Field(default=16384, ge=256, le=65536)
    timeout_seconds: int = Field(default=900, ge=1)

    @model_validator(mode="after")
    def validate_reasoning(self):
        if "coder-30b" in self.model and self.reasoning:
            raise ValueError("the Qwen3-Coder diagnostic cohort does not support reasoning")
        return self


class StudyProvider:
    """One stateless request per recorded call; no routing fallback or hidden retries."""

    name = "openrouter"

    def __init__(self, config: ProviderConfig, repository: Path):
        self.config = config
        self.repository = repository

    def endpoint_identity(self) -> dict:
        url = f"https://openrouter.ai/api/v1/models/{self.config.model}/endpoints"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)["data"]
        endpoint = next((item for item in data["endpoints"] if item["tag"] == self.config.endpoint), None)
        if endpoint is None or endpoint.get("status") != 0:
            raise ValueError("the pinned OpenRouter endpoint is unavailable; choose a new campaign explicitly")
        if endpoint["provider_name"] != self.config.upstream_provider:
            raise ValueError("endpoint provider name conflicts with upstream_provider")
        required = {"response_format", "structured_outputs", "max_tokens", "temperature", "top_p"}
        if self.config.reasoning or "coder-30b" not in self.config.model:
            required.add("reasoning")
        if not required.issubset(endpoint.get("supported_parameters", [])):
            raise ValueError("the pinned endpoint does not advertise all required request parameters")
        if self.config.max_tokens > endpoint["max_completion_tokens"]:
            raise ValueError("max_tokens exceeds the pinned endpoint completion limit")
        return {"model": self.config.model, "license": MODEL_LICENSES[self.config.model],
                "tag": endpoint["tag"], "provider_name": endpoint["provider_name"],
                "quantization": endpoint.get("quantization"), "context_length": endpoint["context_length"],
                "max_completion_tokens": endpoint["max_completion_tokens"],
                "supported_parameters": sorted(endpoint["supported_parameters"])}

    def _key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            import yaml

            try:
                document = yaml.safe_load((self.repository / "config.yaml").read_text())
                key = document["openrouter"]["api_key"].strip()
            except (OSError, KeyError, TypeError, AttributeError, yaml.YAMLError):
                raise ProviderError("set OPENROUTER_API_KEY or openrouter.api_key in config.yaml") from None
        if not key:
            raise ProviderError("the OpenRouter API key is empty")
        return key

    def generate(self, prompt: str, response_model: type[BaseModel], model: str | None = None) -> ProviderResponse:
        if model != self.config.model:
            raise ValueError("model conflicts with the frozen v4 configuration")
        schema = _strict_json_schema(response_model)
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature, "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens, "stream": False,
                "provider": {"only": [self.config.endpoint], "allow_fallbacks": False, "require_parameters": True},
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": response_model.__name__, "strict": True, "schema": schema}}}
        if "coder-30b" not in model:
            body["reasoning"] = {"enabled": self.config.reasoning, "exclude": True}
        metadata = {"provider": self.name, "model": model, "request_settings": self.config.model_dump(),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "schema_sha256": json_digest(schema),
                    "schema_valid": False, "exit_code": 1, "usage": {}, "usage_incomplete": True,
                    "failure_origin": "infrastructure"}
        request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}"}, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                envelope = json.load(response)
        except urllib.error.HTTPError as exc:
            metadata["duration_seconds"] = time.perf_counter() - started
            metadata["http_status"] = exc.code
            raise ProviderError(f"OpenRouter HTTP {exc.code}; response and exact usage unavailable", metadata) from None
        except (urllib.error.URLError, OSError, ValueError):
            metadata["duration_seconds"] = time.perf_counter() - started
            raise ProviderError("OpenRouter transport failed; response and exact usage unavailable", metadata) from None
        metadata["duration_seconds"] = time.perf_counter() - started
        if not isinstance(envelope, dict):
            raise ProviderError("invalid OpenRouter envelope", metadata)
        usage = envelope.get("usage") or {}
        counts = normalize_usage(usage)
        complete = counts["input_tokens"] is not None and counts["output_tokens"] is not None
        metadata.update(usage=usage, usage_incomplete=not complete, generation_id=envelope.get("id"),
                        returned_model=envelope.get("model"), upstream_provider=envelope.get("provider"))
        if envelope.get("error") or envelope.get("model") != model or envelope.get("provider") != self.config.upstream_provider:
            raise ProviderError("provider error or returned model/route differs from the frozen configuration", metadata)
        if not complete or not envelope.get("id"):
            raise ProviderError("OpenRouter omitted complete usage or generation identity", metadata)
        reason_tokens = counts["reasoning_output_tokens"]
        declared_total = usage.get("total_tokens")
        if ((declared_total is not None and (isinstance(declared_total, bool) or declared_total != counts["total_tokens"])) or
                (reason_tokens is not None and reason_tokens > counts["output_tokens"]) or
                (counts["cached_input_tokens"] is not None and counts["cached_input_tokens"] > counts["input_tokens"])):
            metadata["usage_incomplete"] = True
            raise ProviderError("OpenRouter returned inconsistent token accounting", metadata)
        if "coder-30b" not in model and (reason_tokens is None or (not self.config.reasoning and reason_tokens != 0)):
            raise ProviderError("reasoning telemetry is absent or conflicts with the requested mode", metadata)
        content = ""
        try:
            choice = envelope["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise TypeError("expected text content")
            finish = choice.get("finish_reason")
            metadata["finish_reason"] = finish
            if message.get("tool_calls") or message.get("function_call") or finish not in ("stop", "length"):
                raise ProviderError("provider error, tool request, or unsupported finish reason", metadata)
            metadata.update(exit_code=0, failure_origin="model_output")
            if finish == "length":
                raise ValueError("output token limit reached")
            value = response_model.model_validate_json(content)
        except (KeyError, IndexError, TypeError):
            metadata.update(exit_code=1, failure_origin="infrastructure")
            raise ProviderError("malformed OpenRouter response envelope", metadata) from None
        except ValueError:
            # Never embed Pydantic input_value (possibly megabytes) in repair feedback.
            raise ProviderError("output token limit reached" if metadata.get("finish_reason") == "length" else
                                "response does not satisfy the requested JSON schema", metadata, {"content": content}) from None
        metadata.update(schema_valid=True, failure_origin=None)
        return ProviderResponse(value=value, metadata=metadata, raw=value.model_dump(mode="json"))
