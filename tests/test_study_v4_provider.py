import io
import json
import urllib.error

import pytest

from npu_agent.providers import ProviderError, normalize_usage
from npu_agent.study_v4 import Hint
from npu_agent.study_v4_provider import ProviderConfig, StudyProvider


def envelope(**overrides):
    return {"id": "generation-1", "model": "deepseek/deepseek-v3.2", "provider": "Alibaba",
            "usage": {"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60,
                      "prompt_tokens_details": {"cached_tokens": 5}, "completion_tokens_details": {"reasoning_tokens": 30}},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"intermediate":"identity"}'}}], **overrides}


def test_request_is_pinned_stateless_and_reasoning_not_double_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    requests = []

    def post(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(envelope()).encode())

    monkeypatch.setattr("urllib.request.urlopen", post)
    provider = StudyProvider(ProviderConfig(), tmp_path)
    response = provider.generate("translate", Hint, provider.config.model)
    body = requests[0]
    assert body["provider"] == {"only": ["alibaba/fp8"], "allow_fallbacks": False, "require_parameters": True}
    assert body["reasoning"] == {"enabled": True, "exclude": True}
    assert body["messages"] == [{"role": "user", "content": "translate"}]
    assert "tools" not in body and body["response_format"]["type"] == "json_schema"
    assert normalize_usage(response.metadata["usage"])["total_tokens"] == 60
    assert "test-secret" not in json.dumps(response.metadata)


@pytest.mark.parametrize("kind,origin", [("length", "model_output"), ("bad_json", "model_output"),
                                         ("error", "infrastructure"), ("wrong_model", "infrastructure"),
                                         ("wrong_provider", "infrastructure"), ("missing_usage", "infrastructure"),
                                         ("missing_reasoning", "infrastructure"), ("tool_calls", "infrastructure")])
def test_rejected_output_and_infrastructure_are_distinct(tmp_path, monkeypatch, kind, origin):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    data = envelope()
    if kind in ("length", "error"):
        data["choices"][0]["finish_reason"] = kind
    elif kind == "bad_json":
        data["choices"][0]["message"]["content"] = "BIG_RAW_OUTPUT" * 1000
    elif kind == "tool_calls":
        data["choices"][0]["message"]["tool_calls"] = [{"id": "tool"}]
    elif kind == "wrong_model":
        data["model"] = "some-other-model"
    elif kind == "wrong_provider":
        data["provider"] = "other"
    elif kind == "missing_usage":
        data.pop("usage")
    else:
        data["usage"].pop("completion_tokens_details")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(data).encode()))
    provider = StudyProvider(ProviderConfig(), tmp_path)
    with pytest.raises(ProviderError) as exc:
        provider.generate("translate", Hint, provider.config.model)
    assert exc.value.metadata["failure_origin"] == origin
    assert "BIG_RAW_OUTPUT" not in str(exc.value)


def test_transport_failure_has_unknown_usage_and_no_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("possibly sensitive upstream body")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    provider = StudyProvider(ProviderConfig(), tmp_path)
    with pytest.raises(ProviderError) as exc:
        provider.generate("translate", Hint, provider.config.model)
    assert len(calls) == 1 and exc.value.metadata["usage_incomplete"]
    assert "sensitive" not in str(exc.value)


def test_reasoning_off_is_explicit_and_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    def post(request, timeout):
        assert json.loads(request.data)["reasoning"]["enabled"] is False
        return io.BytesIO(json.dumps(envelope()).encode())

    monkeypatch.setattr("urllib.request.urlopen", post)
    provider = StudyProvider(ProviderConfig(reasoning=False), tmp_path)
    with pytest.raises(ProviderError, match="reasoning telemetry"):
        provider.generate("translate", Hint, provider.config.model)


def test_http_failure_preserves_status_without_sensitive_body_or_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://openrouter.ai", 502, "sensitive upstream reason", {},
                                     io.BytesIO(b"sensitive upstream body"))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    provider = StudyProvider(ProviderConfig(), tmp_path)
    with pytest.raises(ProviderError, match="HTTP 502") as exc:
        provider.generate("translate", Hint, provider.config.model)
    assert len(calls) == 1 and exc.value.metadata["http_status"] == 502
    assert exc.value.metadata["usage_incomplete"] and exc.value.metadata["failure_origin"] == "infrastructure"
    assert "sensitive" not in str(exc.value) + json.dumps(exc.value.metadata)


def test_endpoint_must_support_actual_schema_routing(tmp_path, monkeypatch):
    data = {"data": {"endpoints": [{"tag": "alibaba/fp8", "provider_name": "Alibaba", "status": 0, "supported_parameters": ["response_format"]}]}}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(data).encode()))
    with pytest.raises(ValueError, match="required request parameters"):
        StudyProvider(ProviderConfig(), tmp_path).endpoint_identity()


def test_unknown_fields_and_nonreasoning_coder_are_rejected():
    with pytest.raises(ValueError):
        ProviderConfig(tools=True)
    with pytest.raises(ValueError, match="does not support reasoning"):
        ProviderConfig(model="qwen/qwen3-coder-30b-a3b-instruct")


def test_inconsistent_token_accounting_blocks_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    data = envelope()
    data["usage"]["total_tokens"] = 999
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(data).encode()))
    provider = StudyProvider(ProviderConfig(), tmp_path)
    with pytest.raises(ProviderError, match="inconsistent token accounting") as exc:
        provider.generate("translate", Hint, provider.config.model)
    assert exc.value.metadata["usage_incomplete"]
