import io
import json
import urllib.error

import pytest

from npu_agent.providers import ExperimentOpenRouterProvider, ProviderError
from npu_agent.scenarios import GUIDANCE_MODEL, IntermediateRepresentation


@pytest.fixture
def openrouter(tmp_path, monkeypatch):
    secret = "test-secret-never-log"
    (tmp_path / "config.yaml").write_text(f'openrouter:\n  api_key: "{secret}"\n')
    provider = ExperimentOpenRouterProvider(tmp_path)
    envelope = {"id": "generation-1", "model": GUIDANCE_MODEL, "provider": "SiliconFlow",
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.001},
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                    "content": '{"intermediate":"ready"}'}}]}
    requests = []
    def respond(request, timeout):
        requests.append(request)
        assert timeout == 900
        return io.BytesIO(json.dumps(envelope).encode())
    monkeypatch.setattr("urllib.request.urlopen", respond)
    return provider, envelope, requests, secret


def test_openrouter_pins_route_schema_and_excludes_credentials_and_history(openrouter):
    provider, _, requests, secret = openrouter
    response = provider.generate("first", IntermediateRepresentation, GUIDANCE_MODEL)
    provider.generate("second", IntermediateRepresentation, GUIDANCE_MODEL)
    assert response.value.intermediate == "ready"
    assert response.metadata["usage"]["total_tokens"] == 30
    assert response.metadata["usage"]["cost"] == 0.001
    assert response.metadata["upstream_provider"] == "SiliconFlow"
    assert response.metadata["reasoning_effort"] == "none"
    assert response.metadata["generation_id"] == "generation-1"
    assert secret not in json.dumps(response.metadata)
    assert secret not in json.dumps(provider.public_config)
    for request, prompt in zip(requests, ("first", "second")):
        body = json.loads(request.data)
        assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.get_header("Authorization") == f"Bearer {secret}"
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert body["model"] == GUIDANCE_MODEL
        assert body["provider"] == {"only": ["siliconflow/fp8"], "allow_fallbacks": False, "require_parameters": True}
        assert body["temperature"] == 0.7 and body["max_tokens"] == 32768
        assert body["response_format"]["json_schema"]["strict"] is True
        assert not {"tools", "reasoning", "reasoning_effort", "models"} & body.keys()
    with pytest.raises(ValueError, match="model must match"):
        provider.generate("test", IntermediateRepresentation, "other-model")
    assert len(requests) == 2


@pytest.mark.parametrize("failure", ["malformed", "schema", "truncated"])
def test_openrouter_generated_output_rejection_preserves_paid_usage(openrouter, failure):
    provider, envelope, _, _ = openrouter
    if failure == "truncated":
        envelope["choices"][0]["finish_reason"] = "length"
    else:
        envelope["choices"][0]["message"]["content"] = "{" if failure == "malformed" else '{"wrong":1}'
    with pytest.raises(ProviderError) as caught:
        provider.generate("test", IntermediateRepresentation, GUIDANCE_MODEL)
    metadata = caught.value.metadata
    assert metadata["exit_code"] == 0 and metadata["schema_valid"] is False
    assert metadata["usage"]["total_tokens"] == 30
    assert metadata["usage_incomplete"] is False


@pytest.mark.parametrize("failure", ["model", "provider", "tools", "usage", "error"])
def test_openrouter_blocks_uncontrolled_or_incomplete_responses(openrouter, failure):
    provider, envelope, _, _ = openrouter
    if failure in ("model", "provider"):
        envelope[failure] = "unexpected"
    elif failure == "tools":
        envelope["choices"][0]["message"]["tool_calls"] = [{"type": "function", "name": "shell"}]
    elif failure == "usage":
        del envelope["usage"]
    else:
        envelope["error"] = {"code": 503}
    with pytest.raises(ProviderError) as caught:
        provider.generate("test", IntermediateRepresentation, GUIDANCE_MODEL)
    assert caught.value.metadata["exit_code"] == 1


def test_openrouter_http_failure_does_not_retry_or_log_credentials(openrouter, monkeypatch):
    provider, _, _, secret = openrouter
    calls = []
    def fail(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, secret, {"Authorization": secret},
                                     io.BytesIO(secret.encode()))
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(ProviderError) as caught:
        provider.generate("test", IntermediateRepresentation, GUIDANCE_MODEL)
    assert len(calls) == 1
    assert caught.value.metadata["http_status"] == 401
    assert caught.value.metadata["usage_incomplete"] is True
    assert secret not in str(caught.value)
    assert secret not in json.dumps(caught.value.metadata)
    assert secret not in json.dumps(caught.value.raw)


@pytest.mark.parametrize("contents", ['', 'openrouter:\n  api_key: ""\n',
    'openrouter:\n  api_key: "test-secret"\n  model: "other-model"\n',
    'openrouter: [test-secret', 'openrouter:\n  api_key: "test-secret"\n  max_tokens: true\n'])
def test_openrouter_missing_or_invalid_config_fails_without_secret_disclosure(tmp_path, contents):
    (tmp_path / "config.yaml").write_text(contents)
    with pytest.raises(ProviderError) as caught:
        ExperimentOpenRouterProvider(tmp_path)
    assert "config.yaml" in str(caught.value)
    assert "test-secret" not in str(caught.value)
