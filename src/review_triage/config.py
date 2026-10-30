"""Configuração via variáveis de ambiente (ou arquivo .env)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "anthropic", "gemini", "mock"]

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-3.8-flash",
    "mock": "keyword-baseline",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: ProviderName = "mock"
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    max_concurrency: int = 5
    max_retries: int = 3
    max_reviews_per_request: int = 200
    redact_pii: bool = True

    @property
    def model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
