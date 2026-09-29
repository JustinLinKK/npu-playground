import json
from pathlib import Path

import pytest

from npu_agent.models import KernelIR
from npu_agent.providers import ClaudeCLIProvider, CodexCLIProvider, _strict_json_schema
from npu_agent.providers import ExperimentCodexProvider, ProviderError, telemetry_usage


def test_codex_schema_closes_every_object() -> None:
    schema = _strict_json_schema(KernelIR)

    def check(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value.get("properties", {}))
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)


def test_codex_command_is_ephemeral_schema_constrained_and_read_only(tmp_path: Path) -> None:
    provider = object.__new__(CodexCLIProvider)
    provider.executable = "/usr/bin/codex"
    provider.repository_path = tmp_path
    command = provider._command(tmp_path / "schema.json", tmp_path / "response.json", "test-model")
    assert command[:3] == ["/usr/bin/codex", "exec", "-"]
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command
    assert command[-2:] == ["--model", "test-model"]


def test_claude_envelope_parsing(tmp_path: Path) -> None:
    provider = object.__new__(ClaudeCLIProvider)
    raw, usage, telemetry = provider._extract(
        json.dumps({"structured_output": {"value": 3}, "usage": {"input_tokens": 7}}),
        tmp_path / "unused.json",
    )
    assert raw == {"value": 3}
    assert usage == {"input_tokens": 7}
    assert telemetry[0]["structured_output"] == {"value": 3}


def test_codex_jsonl_usage_and_final_schema_response_are_separate(tmp_path: Path) -> None:
    provider = object.__new__(CodexCLIProvider)
    response_path = tmp_path / "response.json"
    response_path.write_text('{"value": 3}', encoding="utf-8")
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "test"}),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 11, "cached_input_tokens": 2, "output_tokens": 5},
                }
            ),
        ]
    )
    raw, usage, telemetry = provider._extract(stdout, response_path)
    assert raw == {"value": 3}
    assert usage == {"input_tokens": 11, "cached_input_tokens": 2, "output_tokens": 5}
    assert [event["type"] for event in telemetry] == ["thread.started", "turn.completed"]


def test_experiment_provider_pins_model_and_disables_ambient_context(tmp_path):
    provider = object.__new__(ExperimentCodexProvider)
    provider.executable = "/usr/bin/codex"
    provider.repository_path = Path("/repository-with-ir-and-knowledge")
    command = provider._command(tmp_path / "schema.json", tmp_path / "response.json", "gpt-5.6-terra")
    assert command[command.index("--cd") + 1] == str(tmp_path)
    assert command[command.index("--model") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="xhigh"' in command
    assert "project_doc_max_bytes=0" in command
    assert 'web_search="disabled"' in command
    assert "--ignore-user-config" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    disabled = {command[i + 1] for i, arg in enumerate(command) if arg == "--disable"}
    assert {"shell_tool", "unified_exec", "memories", "multi_agent", "plugins", "hooks", "apps", "workspace_dependencies"} <= disabled
    assert provider._temp_parent() is None
    assert provider._working_directory(tmp_path) == tmp_path
    with pytest.raises(ValueError, match="requires"):
        provider._command(tmp_path / "schema.json", tmp_path / "response.json", "another-model")


def test_multiple_completed_turns_are_all_counted():
    usage = telemetry_usage([
        {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 4}},
        {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11}},
    ])
    assert usage["input_tokens"] == 8
    assert usage["output_tokens"] == 10
    assert usage["total_tokens"] == 18


@pytest.mark.parametrize("item_type,message,allowed", [
    ("error", "Code Mode is unavailable because code-mode host is disabled. "
     "Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.", True),
    ("error", "unrecognized runtime failure", False),
    ("command_execution", "", False),
    ("mcp_tool_call", "", False),
    ("web_search", "", False),
])
def test_experiment_allows_only_known_disabled_host_diagnostic(monkeypatch, item_type, message, allowed):
    from npu_agent.providers import ProviderResponse
    from npu_agent.scenarios import IntermediateRepresentation

    value = IntermediateRepresentation(intermediate="saved valid response")
    metadata = {"usage": {"input_tokens": 13, "output_tokens": 8}, "telemetry": [
        {"type": "item.completed", "item": {"type": item_type, "message": message}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": value.model_dump_json()}},
        {"type": "turn.completed", "usage": {"input_tokens": 13, "output_tokens": 8}}]}
    response = ProviderResponse(value=value, raw=value.model_dump(), metadata=metadata)
    monkeypatch.setattr(CodexCLIProvider, "generate", lambda *args: response)
    provider = object.__new__(ExperimentCodexProvider)
    if allowed:
        actual = provider.generate("test", IntermediateRepresentation, "gpt-5.6-terra")
        assert actual is response
        assert actual.metadata["usage"] == {"input_tokens": 13, "output_tokens": 8}
    else:
        with pytest.raises(ProviderError, match="external tool"):
            provider.generate("test", IntermediateRepresentation, "gpt-5.6-terra")


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("provider_type", [CodexCLIProvider, ExperimentCodexProvider])
def test_invalid_cli_output_preserves_usage(tmp_path, monkeypatch, returncode, provider_type):
    import subprocess

    provider = object.__new__(provider_type)
    provider.repository_path, provider.executable, provider.version, provider.timeout_seconds = tmp_path, "/codex", "test", 1
    stdout = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 13, "output_tokens": 8}})
    def invoke(command, **kwargs):
        assert kwargs["start_new_session"] is (provider_type is ExperimentCodexProvider)
        Path(command[command.index("--output-last-message") + 1]).write_text("not json")
        return subprocess.CompletedProcess(command, returncode, stdout, "bad output")
    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(ProviderError) as caught:
        provider.generate("test", KernelIR, "gpt-5.6-terra")
    assert caught.value.metadata["usage"] == {"input_tokens": 13, "output_tokens": 8}


def test_timeout_keeps_partial_usage_and_marks_it_incomplete(tmp_path, monkeypatch):
    import subprocess

    provider = object.__new__(CodexCLIProvider)
    provider.repository_path, provider.executable, provider.version, provider.timeout_seconds = tmp_path, "/codex", "test", 1
    partial = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}}).encode()
    def invoke(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 1, output=partial)
    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(ProviderError) as caught:
        provider.generate("test", KernelIR, "gpt-5.6-terra")
    assert caught.value.metadata["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert caught.value.metadata["usage_incomplete"] is True
