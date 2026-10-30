"""Avaliação contra o conjunto rotulado em data/reviews_sample.csv.

Métricas:
- acurácia e F1 macro de sentimento e urgência;
- recall de urgência "alta" (o erro mais caro é deixar um caso grave na fila comum);
- precisão, recall e F1 micro dos tópicos (multirrótulo);
- taxa de falha, latência média e tokens.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import ReviewIn, TriageResult


@dataclass
class LabeledReview:
    review: ReviewIn
    sentiment: str
    urgency: str
    topics: set[str] = field(default_factory=set)


def load_labeled(path: str | Path) -> list[LabeledReview]:
    with Path(path).open(encoding="utf-8") as f:
        return [
            LabeledReview(
                review=ReviewIn(id=row["id"], text=row["text"]),
                sentiment=row["sentiment"],
                urgency=row["urgency"],
                topics={t for t in row["topics"].split("|") if t},
            )
            for row in csv.DictReader(f)
        ]


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold) if gold else 0.0


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = set(gold)
    scores = []
    for label in labels:
        tp = sum(g == p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(p == label and g != label for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(_f1(precision, recall))
    return sum(scores) / len(scores) if scores else 0.0


def compute_metrics(labeled: list[LabeledReview], results: list[TriageResult]) -> dict:
    by_id = {r.review_id: r for r in results}
    pairs = [(item, by_id.get(item.review.id)) for item in labeled]
    ok = [(item, res.triage) for item, res in pairs if res and res.triage]
    n = len(ok)

    gold_sent = [i.sentiment for i, _ in ok]
    pred_sent = [t.sentiment for _, t in ok]
    gold_urg = [i.urgency for i, _ in ok]
    pred_urg = [t.urgency for _, t in ok]

    high_total = sum(g == "alta" for g in gold_urg)
    high_hit = sum(g == p == "alta" for g, p in zip(gold_urg, pred_urg, strict=True))

    tp = sum(len(i.topics & set(t.topics)) for i, t in ok)
    fp = sum(len(set(t.topics) - i.topics) for i, t in ok)
    fn = sum(len(i.topics - set(t.topics)) for i, t in ok)
    topic_p = tp / (tp + fp) if tp + fp else 0.0
    topic_r = tp / (tp + fn) if tp + fn else 0.0

    valid = [r for _, r in pairs if r]
    return {
        "amostras": len(labeled),
        "processadas": n,
        "taxa_de_falha": round(1 - n / len(labeled), 4) if labeled else 0.0,
        "sentimento_acuracia": round(accuracy(gold_sent, pred_sent), 4),
        "sentimento_f1_macro": round(macro_f1(gold_sent, pred_sent), 4),
        "urgencia_acuracia": round(accuracy(gold_urg, pred_urg), 4),
        "urgencia_f1_macro": round(macro_f1(gold_urg, pred_urg), 4),
        "urgencia_alta_recall": round(high_hit / high_total, 4) if high_total else 0.0,
        "topicos_precisao": round(topic_p, 4),
        "topicos_recall": round(topic_r, 4),
        "topicos_f1": round(_f1(topic_p, topic_r), 4),
        "latencia_media_ms": int(sum(r.latency_ms for r in valid) / len(valid)) if valid else 0,
        "tokens_entrada": sum(r.input_tokens for r in valid),
        "tokens_saida": sum(r.output_tokens for r in valid),
    }


def to_markdown(metrics: dict, provider: str, model: str) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    return f"### {provider} / {model}\n\n| Métrica | Valor |\n|---|---|\n{rows}\n"
