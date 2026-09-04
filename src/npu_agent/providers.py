from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel


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
                )
            except subprocess.TimeoutExpired as exc:
                message = f"{self.name} timed out after {self.timeout_seconds}s"
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
                        "telemetry": self._telemetry(completed.stdout),
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
        telemetry: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                telemetry.append(event)
                if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                    usage = event["usage"]
        return json.loads(text), usage, telemetry


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
