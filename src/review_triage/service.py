from collections.abc import Sequence

from .config import Settings
from .insights import build_report, summarize
from .providers import LLMProvider
from .schemas import ReviewIn, TriageResponse
from .triage import triage_batch


async def run_triage(
    settings: Settings,
    provider: LLMProvider,
    reviews: Sequence[ReviewIn],
    include_summary: bool = True,
) -> TriageResponse:
    results = await triage_batch(
        provider,
        reviews,
        max_concurrency=settings.max_concurrency,
        max_retries=settings.max_retries,
        redact_pii=settings.redact_pii,
    )
    report = build_report(results)
    if include_summary and report.processed:
        report.summary = await summarize(provider, report, results)
    return TriageResponse(
        provider=provider.name, model=provider.model, results=results, report=report
    )
