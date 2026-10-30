"""Mascaramento de dados pessoais antes de enviar texto a um provedor externo (LGPD)."""

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARTAO]"),
]


def redact(text: str) -> str:
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text
