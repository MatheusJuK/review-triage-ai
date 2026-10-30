import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(RuntimeError):
    """Erro do provedor. `retryable` indica falhas transitórias (429, 5xx, timeout)."""

    def __init__(self, message: str, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


def parse_rate_limit(response: httpx.Response) -> tuple[bool, float | None, str]:
    """Interpreta um HTTP 429: (vale tentar de novo?, segundos de espera sugeridos, motivo).

    Usa o cabeçalho Retry-After (OpenAI, Anthropic) e os detalhes RetryInfo/QuotaFailure
    da API do Google. Cota diária esgotada não adianta repetir.
    """
    retry_after: float | None = None
    header = response.headers.get("retry-after")
    if header:
        try:
            retry_after = float(header)
        except ValueError:
            retry_after = None
    reason = "limite de requisições"
    try:
        details = response.json().get("error", {}).get("details", []) or []
    except (ValueError, AttributeError):
        details = []
    for detail in details:
        kind = detail.get("@type", "")
        if kind.endswith("RetryInfo") and (
            m := re.match(r"([\d.]+)s", detail.get("retryDelay", ""))
        ):
            retry_after = float(m.group(1))
        if kind.endswith("QuotaFailure"):
            ids = " ".join(v.get("quotaId", "") for v in detail.get("violations", []))
            if "PerDay" in ids:
                return False, None, "cota diária esgotada (renova em até 24h)"
            if ids:
                reason = f"cota excedida ({ids})"
    return True, retry_after, reason


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    async def complete_json(self, system: str, user: str) -> LLMResponse:
        """Envia o prompt e retorna o texto (esperado em JSON) gerado pelo modelo."""

    async def aclose(self) -> None:  # noqa: B027 - opcional para provedores sem recursos
        pass


class HTTPProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model, temperature)
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def _post(self, url: str, headers: dict[str, str], payload: dict) -> dict:
        try:
            response = await self._client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name}: timeout", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: erro de rede: {exc}", retryable=True) from exc
        if response.status_code == 429:
            retryable, retry_after, reason = parse_rate_limit(response)
            raise ProviderError(
                f"{self.name}: HTTP 429: {reason}", retryable=retryable, retry_after=retry_after
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: HTTP {response.status_code}: {response.text[:300]}",
                retryable=response.status_code >= 500,
            )
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
