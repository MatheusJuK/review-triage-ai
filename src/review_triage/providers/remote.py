"""Clientes HTTP para OpenAI (e APIs compatíveis), Anthropic e Google Gemini."""

from .base import HTTPProvider, LLMResponse, ProviderError


class OpenAIProvider(HTTPProvider):
    """Chat Completions. Funciona com APIs compatíveis via OPENAI_BASE_URL (ex.: Ollama, Groq)."""

    name = "openai"

    def __init__(self, api_key: str, base_url: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def complete_json(self, system: str, user: str) -> LLMResponse:
        data = await self._post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload={
                "model": self.model,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"openai: resposta inesperada: {data}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


class AnthropicProvider(HTTPProvider):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    async def complete_json(self, system: str, user: str) -> LLMResponse:
        data = await self._post(
            self.url,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            payload={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": self.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        try:
            text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"anthropic: resposta inesperada: {data}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(text, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


class GeminiProvider(HTTPProvider):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    async def complete_json(self, system: str, user: str) -> LLMResponse:
        data = await self._post(
            f"{self.base_url}/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            payload={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "responseMimeType": "application/json",
                },
            },
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"gemini: resposta inesperada: {data}") from exc
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
        )
