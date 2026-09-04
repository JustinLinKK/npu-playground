import json
from pathlib import Path

from npu_agent.models import KernelIR
from npu_agent.providers import ClaudeCLIProvider, CodexCLIProvider, _strict_json_schema


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
