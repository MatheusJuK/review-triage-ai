"""Linha de comando: `review-triage run <csv>` e `review-triage eval <csv>`."""

import argparse
import asyncio
import json
from pathlib import Path

from .config import get_settings
from .evaluate import compute_metrics, load_labeled, to_markdown
from .io import load_csv
from .providers import build_provider
from .service import run_triage
from .triage import triage_batch


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    provider = build_provider(settings)
    try:
        response = await run_triage(settings, provider, load_csv(args.csv), not args.no_summary)
    finally:
        await provider.aclose()
    output = response.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Resultados salvos em {args.out}")
    report = response.report
    print(f"Processadas: {report.processed}/{report.total} | urgência: {report.urgency}")
    print(f"Sentimento: {report.sentiment}")
    if report.summary:
        print(f"Resumo: {report.summary.headline}")


async def _eval(args: argparse.Namespace) -> None:
    settings = get_settings()
    provider = build_provider(settings)
    labeled = load_labeled(args.csv)
    done = 0
    print(f"Avaliando {len(labeled)} avaliações com {provider.name} / {provider.model}...")

    def progress(result) -> None:
        nonlocal done
        done += 1
        status = "ok" if result.triage else f"falha: {(result.error or '')[:200]}"
        print(
            f"[{done}/{len(labeled)}] {result.review_id} {status} "
            f"({result.latency_ms / 1000:.1f}s, {result.attempts} tentativa(s))",
            flush=True,
        )

    try:
        results = await triage_batch(
            provider,
            [item.review for item in labeled],
            max_concurrency=settings.max_concurrency,
            max_retries=settings.max_retries,
            redact_pii=settings.redact_pii,
            on_result=progress,
        )
    finally:
        await provider.aclose()
    metrics = compute_metrics(labeled, results)
    markdown = to_markdown(metrics, provider.name, provider.model)
    print(markdown)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{provider.name}_{provider.model}".replace("/", "-")
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "provider": provider.name,
                "model": provider.model,
                "metrics": metrics,
                "results": [r.model_dump() for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Relatório salvo em {out_dir / stem}.md")


def main() -> None:
    parser = argparse.ArgumentParser(prog="review-triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Faz a triagem de um CSV de avaliações")
    run.add_argument("csv")
    run.add_argument("--out", help="Arquivo JSON de saída")
    run.add_argument("--no-summary", action="store_true", help="Não gerar resumo executivo")

    ev = sub.add_parser("eval", help="Avalia o provedor contra um CSV rotulado")
    ev.add_argument("csv", nargs="?", default="data/reviews_sample.csv")
    ev.add_argument("--out-dir", default="reports")

    args = parser.parse_args()
    try:
        asyncio.run(_run(args) if args.command == "run" else _eval(args))
    except KeyboardInterrupt:
        raise SystemExit("\nInterrompido pelo usuário.") from None


if __name__ == "__main__":
    main()
