import json

import httpx
import pytest

from review_triage.providers import ProviderError
from review_triage.providers.base import parse_rate_limit
from review_triage.providers.remote import AnthropicProvider, GeminiProvider, OpenAIProvider


def client_returning(status: int, body: dict, seen: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_openai_request_and_parsing():
    seen: list[httpx.Request] = []
    body = {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://api.test/v1/",
        model="m",
        client=client_returning(200, body, seen),
    )
    response = await provider.complete_json("sys", "user")
    assert response.text == '{"ok": true}'
    assert (response.input_tokens, response.output_tokens) == (12, 3)
    request = seen[0]
    assert str(request.url) == "https://api.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    payload = json.loads(request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0] == {"role": "system", "content": "sys"}


async def test_anthropic_request_and_parsing():
    seen: list[httpx.Request] = []
    body = {
        "content": [{"type": "text", "text": '{"ok": true}'}],
        "usage": {"input_tokens": 7, "output_tokens": 2},
    }
    provider = AnthropicProvider(api_key="k", model="m", client=client_returning(200, body, seen))
    response = await provider.complete_json("sys", "user")
    assert response.text == '{"ok": true}' and response.input_tokens == 7
    assert seen[0].headers["x-api-key"] == "k"
    assert json.loads(seen[0].content)["system"] == "sys"


async def test_gemini_request_and_parsing():
    seen: list[httpx.Request] = []
    body = {
        "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
    }
    provider = GeminiProvider(api_key="k", model="gm", client=client_returning(200, body, seen))
    response = await provider.complete_json("sys", "user")
    assert response.text == '{"ok": true}' and response.output_tokens == 1
    assert seen[0].url.path.endswith("/models/gm:generateContent")
    assert json.loads(seen[0].content)["generationConfig"]["responseMimeType"] == (
        "application/json"
    )


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (401, False)])
async def test_http_errors_are_classified(status, retryable):
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://api.test/v1",
        model="m",
        client=client_returning(status, {"error": "x"}, []),
    )
    with pytest.raises(ProviderError) as exc:
        await provider.complete_json("s", "u")
    assert exc.value.retryable is retryable


def _google_429(quota_id: str, delay: str = "17s") -> httpx.Response:
    body = {
        "error": {
            "code": 429,
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": quota_id}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
            ],
        }
    }
    return httpx.Response(429, json=body)


def test_rate_limit_per_minute_uses_retry_delay():
    retryable, retry_after, _ = parse_rate_limit(
        _google_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    )
    assert retryable is True and retry_after == 17.0


def test_daily_quota_is_not_retried():
    retryable, _, reason = parse_rate_limit(
        _google_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    )
    assert retryable is False and "diária" in reason


def test_retry_after_header():
    response = httpx.Response(429, headers={"retry-after": "5"}, json={})
    assert parse_rate_limit(response)[:2] == (True, 5.0)
