from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr


ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ProviderError(RuntimeError):
    def __init__(
        self, message: str, metadata: dict[str, Any] | None = None, raw: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.metadata = metadata
        self.raw = raw or {"error": message}


@dataclass(frozen=True)
class ProviderResponse:
    value: BaseModel
    metadata: dict[str, Any]
    raw: dict[str, Any]


class StructuredProvider(Protocol):
    name: str

    def generate(self, prompt: str, response_model: type[ResponseT], model: str | None = None) -> ProviderResponse: ...


def normalize_usage(usage: dict[str, Any]) -> dict[str, int | None]:
    if not isinstance(usage, dict):
        usage = {}
    def count(*values: Any) -> int | None:
        return next((value for value in values if isinstance(value, int) and not isinstance(value, bool) and value >= 0), None)

    inputs = count(usage.get("input_tokens"), usage.get("prompt_tokens"))
    outputs = count(usage.get("output_tokens"), usage.get("completion_tokens"))
    input_details = usage.get("input_token_details") or usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_token_details") or usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "cached_input_tokens": count(usage.get("cached_input_tokens"), input_details.get("cache_read"), input_details.get("cached_tokens")),
        "reasoning_output_tokens": count(usage.get("reasoning_output_tokens"), output_details.get("reasoning"), output_details.get("reasoning_tokens")),
        "total_tokens": count(inputs + outputs if inputs is not None and outputs is not None else None, usage.get("total_tokens")),
    }


def telemetry_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [event.get("usage") or {} for event in events if event.get("type") == "turn.completed"]
    if len(usages) == 1:
        return usages[0] if isinstance(usages[0], dict) else {}
    if not usages:
        return {}
    normalized = [normalize_usage(usage) for usage in usages]
    totals = {key: sum(value[key] for value in normalized) if all(value[key] is not None for value in normalized) else None
              for key in normalized[0]}
    totals["known_total_tokens"] = sum(value["total_tokens"] or 0 for value in normalized)
    totals["usage_incomplete"] = totals["total_tokens"] is None
    return totals


def _hash_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _strict_json_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    schema = response_model.model_json_schema()

    def close_objects(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                value["additionalProperties"] = False
                properties = value.get("properties")
                if isinstance(properties, dict):
                    value["required"] = list(properties)
            if value.get("default") is None:
                value.pop("default", None)
            for child in value.values():
                close_objects(child)
        elif isinstance(value, list):
            for child in value:
                close_objects(child)

    close_objects(schema)
    return schema


class OpenAIProvider:
    name = "openai"

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model or os.getenv("NPU_AGENT_OPENAI_MODEL")

    def generate(self, prompt: str, response_model: type[ResponseT], model: str | None = None) -> ProviderResponse:
        selected_model = model or self.default_model
        if not selected_model:
            raise ProviderError("OpenAI provider requires --model or NPU_AGENT_OPENAI_MODEL")
        schema = _strict_json_schema(response_model)
        started = time.perf_counter()
        try:
            from langchain_openai import ChatOpenAI

            client = ChatOpenAI(model=selected_model, temperature=0)
            envelope = client.with_structured_output(
                response_model, method="json_schema", include_raw=True
            ).invoke(prompt)
            value = envelope["parsed"]
            if value is None:
                raise ValueError(envelope.get("parsing_error") or "structured response was empty")
            raw_message = envelope.get("raw")
        except Exception as exc:  # provider exceptions are normalized at this boundary
            message = f"OpenAI invocation failed: {exc}"
            raise ProviderError(
                message,
                {
                    "provider": self.name,
                    "model": selected_model,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "schema_sha256": _hash_json(schema),
                    "exit_code": 1,
                    "schema_valid": False,
                    "duration_seconds": time.perf_counter() - started,
                },
            ) from exc
        if not isinstance(value, response_model):
            value = response_model.model_validate(value)
        raw = value.model_dump(mode="json")
        usage = getattr(raw_message, "usage_metadata", None) or {}
        if not usage and raw_message is not None:
            token_usage = getattr(raw_message, "response_metadata", {}).get("token_usage", {})
            usage = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            }
        return ProviderResponse(
            value=value,
            raw=raw,
            metadata={
                "provider": self.name,
                "model": selected_model,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "schema_sha256": _hash_json(schema),
                "exit_code": 0,
                "schema_valid": True,
                "usage": usage,
                "telemetry": [],
                "duration_seconds": time.perf_counter() - started,
            },
        )


class OpenRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    api_key: SecretStr
    model: Literal["qwen/qwen3-coder-30b-a3b-instruct"] = "qwen/qwen3-coder-30b-a3b-instruct"
    endpoint: Literal["siliconflow/fp8"] = "siliconflow/fp8"
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=1)
    max_tokens: int = Field(default=32768, ge=1, le=65536)


class ExperimentOpenRouterProvider:
    """Stateless, generation-only OpenRouter calls with fixed model and routing."""

    name = "openrouter"
    version = "openrouter-chat-completions-v1"

    def __init__(self, repository_path: Path, timeout_seconds: int = 900) -> None:
        import yaml

        path = repository_path / "config.yaml"
        try:
            document = yaml.safe_load(path.read_text())
            self.config = OpenRouterConfig.model_validate(document["openrouter"])
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
            raise ProviderError("OpenRouter requires a valid openrouter section in the repository root config.yaml") from None
        if not self.config.api_key.get_secret_value().strip():
            raise ProviderError("Set openrouter.api_key in the repository root config.yaml")
        self.timeout_seconds = timeout_seconds
        self.public_config = self.config.model_dump(exclude={"api_key"})

    def generate(self, prompt: str, response_model: type[ResponseT], model: str | None = None) -> ProviderResponse:
        if model != self.config.model:
            raise ValueError("OpenRouter model must match the frozen experiment configuration")
        schema = _strict_json_schema(response_model)
        request_body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature, "top_p": self.config.top_p,
            "top_k": self.config.top_k, "max_tokens": self.config.max_tokens, "stream": False,
            "provider": {"only": [self.config.endpoint], "allow_fallbacks": False, "require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": response_model.__name__, "strict": True, "schema": schema}}}
        started = time.perf_counter()
        metadata = {"provider": self.name, "model": model, "reasoning_effort": "none",
            "provider_version": self.version, "request_settings": self.public_config,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "schema_sha256": _hash_json(schema),
            "exit_code": 1, "schema_valid": False, "usage": {}, "usage_incomplete": True}
        request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(request_body).encode(), headers={"Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key.get_secret_value()}"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                envelope = json.load(response)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            metadata["duration_seconds"] = time.perf_counter() - started
            status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            metadata["http_status"] = status
            # Do not persist request headers, credentials or raw server error bodies.
            raise ProviderError(f"OpenRouter request failed (HTTP {status})" if status else
                                "OpenRouter request failed before a complete response; usage unknown", metadata) from None
        metadata["duration_seconds"] = time.perf_counter() - started
        if not isinstance(envelope, dict):
            raise ProviderError("OpenRouter returned an invalid response envelope", metadata)
        usage = envelope.get("usage") or {}
        metadata.update(usage=usage, usage_incomplete=normalize_usage(usage)["total_tokens"] is None,
                        generation_id=envelope.get("id"), returned_model=envelope.get("model"),
                        upstream_provider=envelope.get("provider"))
        if envelope.get("error"):
            raise ProviderError("OpenRouter returned a provider error", metadata)
        if envelope.get("model") != model or envelope.get("provider") != "SiliconFlow":
            raise ProviderError("OpenRouter returned a different model or upstream provider", metadata)
        if metadata["usage_incomplete"]:
            raise ProviderError("OpenRouter response omitted complete token usage", metadata)
        metadata["exit_code"] = 0
        content = ""
        try:
            choice = envelope["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            metadata["finish_reason"] = choice.get("finish_reason")
            if message.get("tool_calls") or message.get("function_call"):
                metadata["exit_code"] = 1
                raise ProviderError("OpenRouter generation attempted an external tool", metadata)
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"generation ended with {choice.get('finish_reason')}")
            value = response_model.model_validate_json(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"OpenRouter structured response rejected: {exc}", metadata,
                                {"content": content}) from None
        metadata["schema_valid"] = True
        return ProviderResponse(value=value, raw=value.model_dump(mode="json"), metadata=metadata)


class CLIProvider:
    executable_name: str
    name: str

    def __init__(self, repository_path: Path, timeout_seconds: int = 900) -> None:
        executable = shutil.which(self.executable_name)
        if not executable:
            raise ProviderError(f"provider executable not found: {self.executable_name}")
        self.executable = executable
        self.repository_path = repository_path.resolve()
        self.timeout_seconds = timeout_seconds
        version = subprocess.run(
            [self.executable, "--version"], capture_output=True, text=True, timeout=15, check=False
        )
        self.version = (version.stdout or version.stderr).strip()[:500]

    def _command(self, schema_path: Path, response_path: Path, model: str | None) -> list[str]:
        raise NotImplementedError

    def _extract(
        self, stdout: str, response_path: Path
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        raise NotImplementedError

    def _temp_parent(self) -> Path | None:
        return self.repository_path / ".npu-agent" / "provider-tmp"

    def _working_directory(self, temp_path: Path) -> Path:
        del temp_path
        return self.repository_path

    def generate(self, prompt: str, response_model: type[ResponseT], model: str | None = None) -> ProviderResponse:
        schema = _strict_json_schema(response_model)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        schema_hash = _hash_json(schema)
        temp_root = self._temp_parent()
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{self.name}-", dir=temp_root) as temp:
            temp_path = Path(temp)
            schema_path = temp_path / "schema.json"
            response_path = temp_path / "response.json"
            schema_path.write_text(json.dumps(schema, sort_keys=True), encoding="utf-8")
            command = self._command(schema_path, response_path, model)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    cwd=self._working_directory(temp_path),
                    timeout=self.timeout_seconds,
                    check=False,
                    env=os.environ.copy(),
                    start_new_session=isinstance(self, ExperimentCodexProvider),
                )
            except subprocess.TimeoutExpired as exc:
                message = f"{self.name} timed out after {self.timeout_seconds}s"
                partial = exc.stdout or ""
                if isinstance(partial, bytes):
                    partial = partial.decode("utf-8", errors="replace")
                telemetry = self._telemetry(partial)
                raise ProviderError(
                    message,
                    {
                        "provider": self.name,
                        "model": model or "cli-default",
                        "executable": self.executable,
                        "provider_version": self.version,
                        "prompt_sha256": prompt_hash,
                        "schema_sha256": schema_hash,
                        "exit_code": 124,
                        "schema_valid": False,
                        "usage": telemetry_usage(telemetry),
                        "telemetry": telemetry,
                        "usage_incomplete": True,
                        "duration_seconds": time.perf_counter() - started,
                    },
                ) from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout)[-4000:]
                message = f"{self.name} exited {completed.returncode}: {detail}"
                raise ProviderError(
                    message,
                    {
                        "provider": self.name,
                        "model": model or "cli-default",
                        "executable": self.executable,
                        "provider_version": self.version,
                        "prompt_sha256": prompt_hash,
                        "schema_sha256": schema_hash,
                        "exit_code": completed.returncode,
                        "schema_valid": False,
                        "usage_incomplete": True,
                        "telemetry": self._telemetry(completed.stdout),
                        "usage": telemetry_usage(self._telemetry(completed.stdout)),
                        "stderr": completed.stderr[-4000:],
                        "duration_seconds": time.perf_counter() - started,
                    },
                )
            try:
                raw, usage, telemetry = self._extract(completed.stdout, response_path)
                value = response_model.model_validate(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                message = f"{self.name} returned invalid structured output: {exc}"
                raise ProviderError(
                    message,
                    {
                        "provider": self.name,
                        "model": model or "cli-default",
                        "executable": self.executable,
                        "provider_version": self.version,
                        "prompt_sha256": prompt_hash,
                        "schema_sha256": schema_hash,
                        "exit_code": completed.returncode,
                        "schema_valid": False,
                        "telemetry": self._telemetry(completed.stdout),
                        "usage": telemetry_usage(self._telemetry(completed.stdout)),
                        "stderr": completed.stderr[-4000:],
                        "duration_seconds": time.perf_counter() - started,
                    },
                    raw if "raw" in locals() and isinstance(raw, dict) else None,
                ) from exc
        return ProviderResponse(
            value=value,
            raw=raw,
            metadata={
                "provider": self.name,
                "model": model or "cli-default",
                "executable": self.executable,
                "provider_version": self.version,
                "prompt_sha256": prompt_hash,
                "schema_sha256": schema_hash,
                "exit_code": 0,
                "schema_valid": True,
                "usage": usage,
                "telemetry": telemetry,
                "stderr": completed.stderr[-4000:],
                "duration_seconds": time.perf_counter() - started,
            },
        )

    @staticmethod
    def _telemetry(stdout: str) -> list[dict[str, Any]]:
        events = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events


class CodexCLIProvider(CLIProvider):
    executable_name = "codex"
    name = "codex-cli"

    def _command(self, schema_path: Path, response_path: Path, model: str | None) -> list[str]:
        command = [
            self.executable,
            "exec",
            "-",
            "--json",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "--cd",
            str(self.repository_path),
        ]
        if model:
            command.extend(["--model", model])
        return command

    def _extract(
        self, stdout: str, response_path: Path
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        text = response_path.read_text(encoding="utf-8") if response_path.exists() else stdout
        telemetry = self._telemetry(stdout)
        usage = telemetry_usage(telemetry)
        return json.loads(text), usage, telemetry


class ExperimentCodexProvider(CodexCLIProvider):
    """A pinned, isolated generation-only provider for the three-scenario experiment."""

    def _temp_parent(self) -> Path | None:
        return None

    def _working_directory(self, temp_path: Path) -> Path:
        return temp_path

    def _command(self, schema_path: Path, response_path: Path, model: str | None) -> list[str]:
        if model != "gpt-5.6-terra":
            raise ValueError("the experiment requires gpt-5.6-terra")
        command = super()._command(schema_path, response_path, model)
        command[command.index("--cd") + 1] = str(schema_path.parent)
        command.extend(["--ignore-user-config", "--skip-git-repo-check", "--strict-config",
                        "-c", 'model_reasoning_effort="xhigh"', "-c", "project_doc_max_bytes=0",
                        "-c", 'web_search="disabled"'])
        for feature in ("shell_tool", "unified_exec", "memories", "multi_agent", "multi_agent_v2", "apps",
                        "plugins", "remote_plugin", "workspace_dependencies", "hooks", "image_generation",
                        "browser_use", "browser_use_external", "browser_use_full_cdp_access", "computer_use",
                        "in_app_browser", "code_mode", "code_mode_host", "code_mode_only", "goals"):
            command.extend(["--disable", feature])
        return command

    def generate(self, prompt: str, response_model: type[ResponseT], model: str | None = None) -> ProviderResponse:
        response = super().generate(prompt, response_model, model)
        response.metadata["reasoning_effort"] = "xhigh"
        for event in response.metadata.get("telemetry", []):
            if event.get("type") in {"item.started", "item.completed"}:
                item = event.get("item", {})
                if item.get("type") == "error" and item.get("message") == (
                        "Code Mode is unavailable because code-mode host is disabled. "
                        "Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."):
                    continue
                if item.get("type") not in {"agent_message", "reasoning"}:
                    raise ProviderError("experimental provider attempted an external tool", response.metadata, response.raw)
        return response


class ClaudeCLIProvider(CLIProvider):
    executable_name = "claude"
    name = "claude-cli"

    def _temp_parent(self) -> Path | None:
        return None

    def _working_directory(self, temp_path: Path) -> Path:
        return temp_path

    def _command(self, schema_path: Path, response_path: Path, model: str | None) -> list[str]:
        del response_path
        command = [
            self.executable,
            "--print",
            "--no-session-persistence",
            "--permission-mode",
            "plan",
            "--output-format",
            "json",
            "--json-schema",
            schema_path.read_text(encoding="utf-8"),
        ]
        if model:
            command.extend(["--model", model])
        return command

    def _extract(
        self, stdout: str, response_path: Path
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        del response_path
        envelope = json.loads(stdout)
        raw = envelope.get("structured_output") or envelope.get("result")
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("Claude response did not contain structured_output")
        usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
        return raw, usage, [envelope]


def create_provider(name: str, repository_path: Path, model: str | None, timeout_seconds: int) -> StructuredProvider:
    if name == "openai":
        return OpenAIProvider(model)
    if name == "codex-cli":
        return CodexCLIProvider(repository_path, timeout_seconds)
    if name == "claude-cli":
        return ClaudeCLIProvider(repository_path, timeout_seconds)
    raise ValueError(f"unknown provider: {name}")
