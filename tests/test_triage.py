import json

import pytest

from review_triage.insights import build_report
from review_triage.pii import redact
from review_triage.providers import LLMProvider, LLMResponse, ProviderError
from review_triage.providers.mock import MockProvider
from review_triage.schemas import ReviewIn
from review_triage.triage import extract_json, triage_batch, triage_review

VALID = {
    "sentiment": "negativo",
    "topics": ["temperatura", "temperatura"],
    "urgency": "media",
    "needs_human": True,
    "summary": "Comida chegou fria.",
    "suggested_reply": "Sentimos muito, vamos verificar.",
}


class ScriptedProvider(LLMProvider):
    """Devolve respostas (ou exceções) pré-definidas, na ordem."""

    name = "scripted"

    def __init__(self, outputs):
        super().__init__(model="test")
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    async def complete_json(self, system: str, user: str) -> LLMResponse:
        self.prompts.append(user)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMResponse(out, input_tokens=10, output_tokens=5)


def test_extract_json_tolerates_code_fences():
    text = 'Claro! ```json\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_extract_json_without_object_raises():
    with pytest.raises(ValueError):
        extract_json("sem json aqui")


def test_redact_masks_personal_data():
    text = "Sou ana@mail.com, CPF 123.456.789-09, tel (86) 99999-1234"
    masked = redact(text)
    assert "ana@mail.com" not in masked and "123.456.789-09" not in masked
    assert "99999-1234" not in masked
    assert {"[EMAIL]", "[CPF]", "[TELEFONE]"} <= set(masked.replace(",", " ").split())


async def test_valid_output_is_parsed_and_topics_deduplicated():
    provider = ScriptedProvider([json.dumps(VALID)])
    result = await triage_review(provider, ReviewIn(text="Chegou frio"), "1")
    assert result.error is None
    assert result.triage.topics == ["temperatura"]
    assert result.attempts == 1
    assert (result.input_tokens, result.output_tokens) == (10, 5)


async def test_invalid_output_triggers_repair_prompt():
    bad = json.dumps({**VALID, "urgency": "urgentissima"})
    provider = ScriptedProvider([bad, json.dumps(VALID)])
    result = await triage_review(provider, ReviewIn(text="Chegou frio"), "1")
    assert result.triage is not None
    assert result.attempts == 2
    assert "inválida" in provider.prompts[1]


async def test_retryable_error_is_retried():
    provider = ScriptedProvider([ProviderError("429", retryable=True), json.dumps(VALID)])
    result = await triage_review(provider, ReviewIn(text="x"), "1", backoff_seconds=0)
    assert result.triage is not None and result.attempts == 2


async def test_non_retryable_error_fails_fast():
    provider = ScriptedProvider([ProviderError("401", retryable=False)])
    result = await triage_review(provider, ReviewIn(text="x"), "1")
    assert result.triage is None and result.attempts == 1 and "401" in result.error


async def test_pii_is_redacted_before_reaching_the_model():
    provider = ScriptedProvider([json.dumps(VALID)])
    await triage_review(provider, ReviewIn(text="meu email é joao@x.com"), "1")
    assert "joao@x.com" not in provider.prompts[0]


async def test_batch_with_mock_provider_and_report():
    reviews = [
        ReviewIn(id="a", text="Passei mal depois de comer, comida estragada."),
        ReviewIn(id="b", text="Excelente, chegou rápido e delicioso!"),
        ReviewIn(id="c", text="Veio o pedido errado e faltou a bebida."),
    ]
    results = await triage_batch(MockProvider(model="baseline"), reviews)
    assert [r.review_id for r in results] == ["a", "b", "c"]
    report = build_report(results)
    assert report.processed == 3
    assert report.urgent_review_ids == ["a"]
    assert report.sentiment["positivo"] == 1
