"""Pipeline de triagem: mascaramento de PII -> LLM -> validação -> reparo/retry."""

import asyncio
import json
import re
import time
from collections.abc import Callable, Sequence

from pydantic import ValidationError

from .pii import redact
from .prompts import TRIAGE_SYSTEM_PROMPT, build_repair_prompt, build_triage_user_prompt
from .providers import LLMProvider, ProviderError
from .schemas import ReviewIn, ReviewTriage, TriageResult

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict:
    """Extrai o primeiro objeto JSON da resposta (tolera cercas ``` e texto extra)."""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("nenhum objeto JSON encontrado na resposta")
    return json.loads(match.group(0))


async def triage_review(
    provider: LLMProvider,
    review: ReviewIn,
    review_id: str,
    max_retries: int = 3,
    redact_pii: bool = True,
    backoff_seconds: float = 1.0,
) -> TriageResult:
    text = redact(review.text) if redact_pii else review.text
    result = TriageResult(review_id=review_id, text=review.text)
    user_prompt = build_triage_user_prompt(text)
    started = time.perf_counter()

    for attempt in range(1, max_retries + 1):
        result.attempts = attempt
        try:
            response = await provider.complete_json(TRIAGE_SYSTEM_PROMPT, user_prompt)
        except ProviderError as exc:
            result.error = str(exc)
            if not exc.retryable or attempt == max_retries:
                break
            await asyncio.sleep(backoff_seconds * 2 ** (attempt - 1))
            continue

        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens
        try:
            result.triage = ReviewTriage.model_validate(extract_json(response.text))
            result.error = None
            break
        except (ValueError, ValidationError) as exc:
            # Reparo: devolve o erro de validação ao modelo para ele corrigir a saída.
            result.error = f"saída inválida: {exc}"
            user_prompt = build_repair_prompt(text, response.text, str(exc))

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


async def triage_batch(
    provider: LLMProvider,
    reviews: Sequence[ReviewIn],
    max_concurrency: int = 5,
    max_retries: int = 3,
    redact_pii: bool = True,
    on_result: Callable[[TriageResult], None] | None = None,
) -> list[TriageResult]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(index: int, review: ReviewIn) -> TriageResult:
        async with semaphore:
            result = await triage_review(
                provider,
                review,
                review_id=review.id or str(index + 1),
                max_retries=max_retries,
                redact_pii=redact_pii,
            )
        if on_result:
            on_result(result)
        return result

    return await asyncio.gather(*(run(i, r) for i, r in enumerate(reviews)))
