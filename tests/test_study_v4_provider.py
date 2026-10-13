import io
import json
import urllib.error

import pytest

from npu_agent.providers import ProviderError, normalize_usage
from npu_agent.study_v4 import Hint
from npu_agent.study_v4_provider import ProviderConfig, StudyOpenAIProvider, StudyProvider


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


def openai_config(**overrides):
    return ProviderConfig(**({"model": "gpt-5.6-terra", "endpoint": "https://api.openai.com/v1/responses",
                             "upstream_provider": "OpenAI"} | overrides))


def openai_envelope(**overrides):
    return {"id": "resp-1", "model": "gpt-5.6-terra", "status": "completed", "error": None,
            "usage": {"input_tokens": 20, "output_tokens": 40, "total_tokens": 60,
                      "input_tokens_details": {"cached_tokens": 5}, "output_tokens_details": {"reasoning_tokens": 30}},
            "output": [{"type": "reasoning", "summary": []}, {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": '{"intermediate":"identity"}'}]}], **overrides}


@pytest.mark.parametrize("environment_key", [False, True])
@pytest.mark.parametrize("model", ["gpt-6-terra", "gpt-5.6-terra", "gpt-6-sol", "gpt-6-astra"])
def test_openai_request_uses_own_key_responses_schema_and_complete_usage(tmp_path, monkeypatch, environment_key, model):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong-provider-key")
    (tmp_path / "config.yaml").write_text('openai:\n  api_key: yaml-test-secret\n')
    if environment_key:
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-secret")
    requests = []

    def post(request, timeout):
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert request.get_header("Authorization") == "Bearer " + ("env-test-secret" if environment_key else "yaml-test-secret")
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(openai_envelope(model=model)).encode())

    monkeypatch.setattr("urllib.request.urlopen", post)
    provider = StudyOpenAIProvider(openai_config(model=model), tmp_path)
    response = provider.generate("translate", Hint, provider.config.model)
    body = requests[0]
    assert body["model"] == model and body["input"] == [{"role": "user", "content": "translate"}]
    assert body["reasoning"] == {"effort": "medium"} and not body["store"] and not body["stream"]
    assert body["max_output_tokens"] == provider.config.max_tokens
    assert body["text"]["format"]["strict"] and body["text"]["format"]["schema"]["additionalProperties"] is False
    assert not {"tools", "temperature", "top_p", "provider", "previous_response_id"} & body.keys()
    assert response.value.intermediate == "identity" and response.metadata["schema_valid"]
    assert normalize_usage(response.metadata["usage"])["total_tokens"] == 60
    assert normalize_usage(response.metadata["usage"])["cached_input_tokens"] == 5
    assert response.metadata["provider"] == "openai" and response.metadata["generation_id"] == "resp-1"
    assert "cost" not in response.metadata["usage"] and not response.metadata["usage_incomplete"]
    assert "test-secret" not in json.dumps(response.metadata)


@pytest.mark.parametrize("kind,origin", [("length", "model_output"), ("bad_json", "model_output"),
                                         ("refusal", "model_output"), ("wrong_model", "infrastructure"),
                                         ("missing_usage", "infrastructure"), ("missing_reasoning", "infrastructure"),
                                         ("bad_total", "infrastructure"), ("tool_call", "infrastructure"),
                                         ("failed", "infrastructure"), ("bad_output", "infrastructure")])
def test_openai_rejected_responses_preserve_usage_and_failure_origin(tmp_path, monkeypatch, kind, origin):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    data = openai_envelope()
    if kind == "length":
        data.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[])
    elif kind == "bad_json":
        data["output"][1]["content"][0]["text"] = "BIG_RAW_OUTPUT" * 100
    elif kind == "refusal":
        data["output"][1]["content"] = [{"type": "refusal", "refusal": "sensitive model text"}]
    elif kind == "wrong_model":
        data["model"] = "gpt-6-sol"
    elif kind == "missing_usage":
        data.pop("usage")
    elif kind == "missing_reasoning":
        data["usage"].pop("output_tokens_details")
    elif kind == "bad_total":
        data["usage"]["total_tokens"] = 999
    elif kind == "tool_call":
        data["output"].append({"type": "function_call", "name": "tool"})
    elif kind == "failed":
        data["status"] = "failed"
    else:
        data["output"] = {}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(data).encode()))
    provider = StudyOpenAIProvider(openai_config(), tmp_path)
    with pytest.raises(ProviderError) as exc:
        provider.generate("translate", Hint, provider.config.model)
    assert exc.value.metadata["failure_origin"] == origin
    assert exc.value.metadata["generation_id"] == "resp-1"
    assert "BIG_RAW_OUTPUT" not in str(exc.value) and "sensitive" not in str(exc.value)


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_openai_http_errors_do_not_retry_or_expose_response(tmp_path, monkeypatch, status):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://api.openai.com", status, "sensitive reason", {}, io.BytesIO(b"sensitive body"))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    provider = StudyOpenAIProvider(openai_config(), tmp_path)
    with pytest.raises(ProviderError) as exc:
        provider.generate("translate", Hint, provider.config.model)
    assert calls == [1] and exc.value.metadata["http_status"] == status
    assert exc.value.metadata["usage_incomplete"] and "sensitive" not in str(exc.value)


def test_openai_deadline_preserves_request_id_and_unknown_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    clock = [0.]

    class Heartbeats(io.BytesIO):
        headers = {"x-request-id": "request-stalled"}

        def read1(self, size):
            clock[0] += .6
            return b' '

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Heartbeats())
    monkeypatch.setattr("npu_agent.study_v4_provider.time.perf_counter", lambda: clock[0])
    provider = StudyOpenAIProvider(openai_config(timeout_seconds=1), tmp_path)
    with pytest.raises(ProviderError, match="deadline") as exc:
        provider.generate("translate", Hint, provider.config.model)
    assert exc.value.metadata["timeout_kind"] == "total_response_deadline"
    assert exc.value.metadata["request_id"] == "request-stalled" and exc.value.metadata["usage_incomplete"]


@pytest.mark.parametrize("unavailable", [False, True])
def test_openai_access_check_uses_exact_model_and_sends_no_generation(tmp_path, monkeypatch, unavailable):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    calls = []

    def get(request, timeout):
        calls.append(request)
        assert request.full_url == "https://api.openai.com/v1/models/gpt-5.6-terra"
        assert request.data is None and request.get_header("Authorization") == "Bearer test-secret"
        if unavailable:
            raise urllib.error.HTTPError(request.full_url, 404, "sensitive reason", {}, io.BytesIO(b"sensitive body"))
        return io.BytesIO(json.dumps({"id": "gpt-5.6-terra", "created": 1, "owned_by": "openai"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", get)
    provider = StudyOpenAIProvider(openai_config(), tmp_path)
    if unavailable:
        with pytest.raises(ProviderError, match="no generation request sent") as exc:
            provider.endpoint_identity()
        assert exc.value.metadata["http_status"] == 404 and "sensitive" not in str(exc.value)
    else:
        identity = provider.endpoint_identity()
        assert identity["model"] == "gpt-5.6-terra" and identity["provider_name"] == "OpenAI"
    assert len(calls) == 1


def test_openai_empty_key_fails_without_request(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text('openai:\n  api_key: ""\n')
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("must not send"))
    with pytest.raises(ProviderError, match="key is empty"):
        StudyOpenAIProvider(openai_config(), tmp_path).endpoint_identity()
