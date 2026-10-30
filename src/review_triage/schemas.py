"""Contratos de dados. A saída do LLM é validada contra `ReviewTriage`."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positivo", "neutro", "negativo"]
Urgency = Literal["baixa", "media", "alta"]
Topic = Literal[
    "atraso_entrega",
    "pedido_errado_incompleto",
    "qualidade_comida",
    "temperatura",
    "embalagem",
    "atendimento",
    "preco",
    "app_pagamento",
    "entregador",
    "seguranca_alimentar",
    "elogio",
]

TOPIC_DESCRIPTIONS: dict[str, str] = {
    "atraso_entrega": "pedido chegou atrasado, demorou demais ou não chegou",
    "pedido_errado_incompleto": "item trocado, faltando ou diferente do pedido",
    "qualidade_comida": "sabor, ponto, frescor, porção ou aparência da comida",
    "temperatura": "comida fria, bebida quente ou derretida",
    "embalagem": "embalagem vazando, amassada, aberta ou inadequada",
    "atendimento": "contato com o restaurante ou com o suporte",
    "preco": "preço, taxa de entrega, custo-benefício ou cupom",
    "app_pagamento": "problemas no app, cobrança, pagamento ou estorno",
    "entregador": "conduta ou comportamento do entregador",
    "seguranca_alimentar": "intoxicação, corpo estranho, alergia ou alimento estragado",
    "elogio": "elogio ou experiência positiva",
}


class ReviewIn(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1, max_length=4000)


class ReviewTriage(BaseModel):
    """Saída estruturada esperada do modelo para cada avaliação."""

    sentiment: Sentiment
    topics: list[Topic] = Field(min_length=1, max_length=4)
    urgency: Urgency
    needs_human: bool = Field(description="Se um atendente humano deve agir no caso")
    summary: str = Field(min_length=1, max_length=240)
    suggested_reply: str = Field(min_length=1, max_length=600)

    @field_validator("topics")
    @classmethod
    def dedupe_topics(cls, topics: list[str]) -> list[str]:
        return list(dict.fromkeys(topics))


class TriageResult(BaseModel):
    review_id: str
    text: str
    triage: ReviewTriage | None = None
    error: str | None = None
    attempts: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ExecutiveSummary(BaseModel):
    headline: str
    key_issues: list[str] = Field(max_length=5)
    recommended_actions: list[str] = Field(max_length=5)


class Report(BaseModel):
    total: int
    processed: int
    failed: int
    sentiment: dict[str, int]
    urgency: dict[str, int]
    topics: dict[str, int]
    negative_topics: dict[str, int]
    needs_human: int
    urgent_review_ids: list[str]
    avg_latency_ms: int
    total_input_tokens: int
    total_output_tokens: int
    summary: ExecutiveSummary | None = None


class TriageRequest(BaseModel):
    reviews: list[ReviewIn] = Field(min_length=1)
    include_summary: bool = True


class TriageResponse(BaseModel):
    provider: str
    model: str
    results: list[TriageResult]
    report: Report
