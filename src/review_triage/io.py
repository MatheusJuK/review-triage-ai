"""Leitura de avaliações a partir de CSV (coluna `text` obrigatória, `id` opcional)."""

import csv
import io
from pathlib import Path

from .schemas import ReviewIn

TEXT_COLUMNS = ("text", "texto", "review", "avaliacao", "comentario")


def parse_csv(content: str) -> list[ReviewIn]:
    reader = csv.DictReader(io.StringIO(content.lstrip("﻿")))
    if not reader.fieldnames:
        raise ValueError("CSV vazio")
    fields = {name.strip().lower(): name for name in reader.fieldnames}
    text_col = next((fields[c] for c in TEXT_COLUMNS if c in fields), None)
    if not text_col:
        raise ValueError(f"CSV precisa de uma coluna de texto: {', '.join(TEXT_COLUMNS)}")
    id_col = fields.get("id")
    reviews = []
    for row in reader:
        text = (row.get(text_col) or "").strip()
        if text:
            reviews.append(ReviewIn(id=(row.get(id_col) or None) if id_col else None, text=text))
    return reviews


def load_csv(path: str | Path) -> list[ReviewIn]:
    return parse_csv(Path(path).read_text(encoding="utf-8"))
