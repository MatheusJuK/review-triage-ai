from ..config import Settings
from .base import LLMProvider, LLMResponse, ProviderError
from .mock import MockProvider
from .remote import AnthropicProvider, GeminiProvider, OpenAIProvider

__all__ = ["LLMProvider", "LLMResponse", "ProviderError", "build_provider"]


def build_provider(settings: Settings) -> LLMProvider:
    """Cria o provedor configurado em LLM_PROVIDER."""
    common = {"model": settings.model, "temperature": settings.llm_temperature}
    http = {**common, "timeout": settings.llm_timeout_seconds}
    match settings.llm_provider:
        case "mock":
            return MockProvider(**common)
        case "openai":
            return OpenAIProvider(
                api_key=_require(settings.openai_api_key, "OPENAI_API_KEY"),
                base_url=settings.openai_base_url,
                **http,
            )
        case "anthropic":
            return AnthropicProvider(
                api_key=_require(settings.anthropic_api_key, "ANTHROPIC_API_KEY"), **http
            )
        case "gemini":
            return GeminiProvider(
                api_key=_require(settings.gemini_api_key, "GEMINI_API_KEY"), **http
            )
    raise ValueError(f"Provedor desconhecido: {settings.llm_provider}")


def _require(value: str | None, env_name: str) -> str:
    if not value:
        raise ValueError(f"Defina {env_name} no ambiente ou no arquivo .env")
    return value
