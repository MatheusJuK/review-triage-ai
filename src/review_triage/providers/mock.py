"""Baseline por palavras-chave, sem LLM.

Serve para três coisas: rodar testes e CI sem chave de API, permitir a demo offline
e servir de linha de base na avaliação (quanto o LLM ganha sobre regras simples?).
"""

import json
import re
import unicodedata

from ..prompts import SUMMARY_SYSTEM_PROMPT
from .base import LLMProvider, LLMResponse

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "seguranca_alimentar": [
        "intoxica",
        "passei mal",
        "passamos mal",
        "cabelo",
        "barata",
        "inseto",
        "alergi",
        "estragad",
        "azed",
        "vencid",
        "plastico dentro",
        "mofo",
    ],
    "atraso_entrega": ["atras", "demor", "nao chegou", "nunca chegou", "esperei", "horas"],
    "pedido_errado_incompleto": ["errad", "falt", "trocad", "veio outro", "incomplet", "esquece"],
    "temperatura": ["frio", "fria", "gelad", "derret", "quente demais"],
    "embalagem": ["embalagem", "vaz", "derram", "amassad", "lacre"],
    "atendimento": ["atendimento", "suporte", "atendente", "chat", "restaurante nao respond"],
    "preco": ["caro", "preco", "taxa", "cupom", "valor", "custo"],
    "app_pagamento": ["cobra", "cobran", "estorno", "pagamento", "app ", "aplicativo", "cartao"],
    "entregador": ["entregador", "motoboy", "grosseir", "mal educad"],
    "qualidade_comida": [
        "sabor",
        "gostos",
        "sem gosto",
        "cru",
        "queimad",
        "salgad",
        "porcao",
        "pequen",
        "massa",
        "carne",
        "comida",
    ],
}
POSITIVE = [
    "otim",
    "excelente",
    "perfeit",
    "maravilh",
    "delicios",
    "gostos",
    "amei",
    "adorei",
    "recomendo",
    "rapid",
    "bom",
    "boa",
    "parabens",
    "nota 10",
    "melhor",
]
NEGATIVE = [
    "pessim",
    "horrivel",
    "ruim",
    "nunca mais",
    "decepcion",
    "absurdo",
    "lixo",
    "nojo",
    "reembolso",
    "errad",
    "falt",
    "atras",
    "demor",
    "frio",
    "fria",
    "cobra",
    "passei mal",
    "nao chegou",
    "grosseir",
    "vaz",
    "derram",
    "cru",
    "queimad",
    "caro",
    "insatisfeit",
]
HIGH_URGENCY = [
    "intoxica",
    "passei mal",
    "passamos mal",
    "cabelo",
    "barata",
    "inseto",
    "alergi",
    "duas vezes",
    "cobrado em dobro",
    "cobranca indevida",
    "nao chegou",
    "nunca chegou",
    "estragad",
    "mofo",
    "ameac",
    "agress",
]
MEDIUM_URGENCY = [
    "errad",
    "falt",
    "trocad",
    "reembolso",
    "atras",
    "demor",
    "frio",
    "fria",
    "vaz",
    "derram",
    "cru",
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _count(text: str, words: list[str]) -> int:
    return sum(text.count(w) for w in words)


def classify(review: str) -> dict:
    t = _normalize(review)
    topics = [topic for topic, words in TOPIC_KEYWORDS.items() if _count(t, words)]
    pos, neg = _count(t, POSITIVE), _count(t, NEGATIVE)
    if neg > pos:
        sentiment = "negativo"
    elif pos > neg:
        sentiment = "positivo"
    else:
        sentiment = "neutro"
    if sentiment == "positivo" and "elogio" not in topics:
        topics.insert(0, "elogio")
    if not topics:
        topics = ["qualidade_comida"]
    if _count(t, HIGH_URGENCY):
        urgency = "alta"
    elif sentiment == "negativo" and _count(t, MEDIUM_URGENCY):
        urgency = "media"
    else:
        urgency = "baixa"
    words = review.split()
    summary = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
    reply = (
        "Obrigado pelo feedback! Ficamos muito felizes com a sua experiência."
        if sentiment == "positivo"
        else "Sentimos muito pelo ocorrido. Nossa equipe vai analisar o seu caso."
    )
    return {
        "sentiment": sentiment,
        "topics": topics[:4],
        "urgency": urgency,
        "needs_human": urgency != "baixa",
        "summary": summary,
        "suggested_reply": reply,
    }


class MockProvider(LLMProvider):
    name = "mock"

    async def complete_json(self, system: str, user: str) -> LLMResponse:
        if system == SUMMARY_SYSTEM_PROMPT:
            stats = json.loads(user.split("\n", 1)[1]) if "\n" in user else {}
            top = list((stats.get("negative_topics") or {}).items())[:3]
            issues = [f"{name}: {count} avaliações negativas" for name, count in top]
            payload = {
                "headline": f"{stats.get('urgency', {}).get('alta', 0)} casos de alta urgência "
                f"em {stats.get('processed', 0)} avaliações",
                "key_issues": issues or ["Sem problemas recorrentes"],
                "recommended_actions": ["Priorizar os casos de alta urgência na fila de suporte"],
            }
            return LLMResponse(json.dumps(payload, ensure_ascii=False))
        match = re.search(r"<avaliacao>\n(.*)\n</avaliacao>", user, re.S)
        review = match.group(1) if match else user
        return LLMResponse(json.dumps(classify(review), ensure_ascii=False))
