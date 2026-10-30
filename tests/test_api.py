import pytest
from fastapi.testclient import TestClient

from review_triage.api import create_app
from review_triage.config import Settings
from review_triage.evaluate import compute_metrics, load_labeled
from review_triage.providers.mock import MockProvider
from review_triage.schemas import ReviewTriage, TriageResult


@pytest.fixture
def client():
    settings = Settings(llm_provider="mock", max_reviews_per_request=5)
    with TestClient(create_app(settings, MockProvider(model="baseline"))) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["provider"] == "mock"


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "Review Triage AI" in response.text


def test_sample_reviews(client):
    assert len(client.get("/sample").json()) == 40


def test_triage_endpoint(client):
    payload = {"reviews": [{"text": "Comida estragada, passei mal."}, {"text": "Amei, delicioso!"}]}
    body = client.post("/triage", json=payload).json()
    assert body["report"]["processed"] == 2
    assert body["report"]["urgency"]["alta"] == 1
    assert body["report"]["summary"]["headline"]


def test_triage_limit(client):
    payload = {"reviews": [{"text": "ok"}] * 6}
    assert client.post("/triage", json=payload).status_code == 413


def test_triage_csv(client):
    csv = "id,text\n1,Pedido veio errado\n2,Muito bom\n"
    response = client.post("/triage/csv", files={"file": ("r.csv", csv, "text/csv")})
    assert [r["review_id"] for r in response.json()["results"]] == ["1", "2"]


def test_triage_csv_without_text_column(client):
    response = client.post("/triage/csv", files={"file": ("r.csv", "a,b\n1,2\n", "text/csv")})
    assert response.status_code == 422


def test_metrics_are_perfect_for_gold_predictions():
    labeled = load_labeled("data/reviews_sample.csv")
    results = [
        TriageResult(
            review_id=item.review.id,
            text=item.review.text,
            triage=ReviewTriage(
                sentiment=item.sentiment,
                urgency=item.urgency,
                topics=sorted(item.topics),
                needs_human=False,
                summary="s",
                suggested_reply="r",
            ),
        )
        for item in labeled
    ]
    metrics = compute_metrics(labeled, results)
    assert metrics["sentimento_acuracia"] == 1.0
    assert metrics["urgencia_alta_recall"] == 1.0
    assert metrics["topicos_f1"] == 1.0


def test_demo_endpoint_serves_precomputed_results(client, tmp_path, monkeypatch):
    from review_triage import api

    missing = tmp_path / "missing.json"
    monkeypatch.setattr(api, "DEMO_PATH", missing)
    assert client.get("/demo").status_code == 404

    payload = client.post("/triage", json={"reviews": [{"text": "Amei!"}]}).json()
    demo = tmp_path / "demo.json"
    import json

    demo.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(api, "DEMO_PATH", demo)
    assert client.get("/demo").json()["report"]["processed"] == 1
