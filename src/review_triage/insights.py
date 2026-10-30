"""Agregação dos resultados e resumo executivo gerado por LLM."""

import json
from collections import Counter

from pydantic import ValidationError

from .prompts import SUMMARY_SYSTEM_PROMPT
from .providers import LLMProvider, ProviderError
from .schemas import ExecutiveSummary, Report, TriageResult
from .triage import extract_json

URGENCY_ORDER = {"alta": 0, "media": 1, "baixa": 2}


def build_report(results: list[TriageResult]) -> Report:
    ok = [r for r in results if r.triage]
    sentiment = Counter(r.triage.sentiment for r in ok)
    urgency = Counter(r.triage.urgency for r in ok)
    topics = Counter(t for r in ok for t in r.triage.topics)
    negative_topics = Counter(
        t for r in ok if r.triage.sentiment == "negativo" for t in r.triage.topics
    )
    return Report(
        total=len(results),
        processed=len(ok),
        failed=len(results) - len(ok),
        sentiment={k: sentiment.get(k, 0) for k in ("positivo", "neutro", "negativo")},
        urgency={k: urgency.get(k, 0) for k in ("alta", "media", "baixa")},
        topics=dict(topics.most_common()),
        negative_topics=dict(negative_topics.most_common()),
        needs_human=sum(1 for r in ok if r.triage.needs_human),
        urgent_review_ids=[r.review_id for r in ok if r.triage.urgency == "alta"],
        avg_latency_ms=int(sum(r.latency_ms for r in results) / len(results)) if results else 0,
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
    )


async def summarize(
    provider: LLMProvider, report: Report, results: list[TriageResult], max_items: int = 40
) -> ExecutiveSummary | None:
    """Gera um resumo executivo a partir das estatísticas e dos casos mais urgentes."""
    ok = sorted(
        (r for r in results if r.triage),
        key=lambda r: URGENCY_ORDER[r.triage.urgency],
    )[:max_items]
    payload = {
        **report.model_dump(exclude={"summary", "urgent_review_ids"}),
        "amostra": [
            {"urgencia": r.triage.urgency, "topicos": r.triage.topics, "resumo": r.triage.summary}
            for r in ok
        ],
    }
    user = "Dados agregados:\n" + json.dumps(payload, ensure_ascii=False)
    try:
        response = await provider.complete_json(SUMMARY_SYSTEM_PROMPT, user)
        return ExecutiveSummary.model_validate(extract_json(response.text))
    except (ProviderError, ValueError, ValidationError):
        return None
