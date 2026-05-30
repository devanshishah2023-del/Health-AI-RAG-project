"""End-to-end API tests using FastAPI's TestClient (no network required)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan startup (builds the index)
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0


def test_general_education_question(client):
    resp = client.post("/ask", json={"question": "What is HFpEF?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["guardrail_triggered"] is False
    assert body["evidence_sufficient"] is True
    assert len(body["evidence_used"]) >= 1
    assert "document_id" in body["evidence_used"][0]


def test_high_risk_question_escalates(client):
    resp = client.post(
        "/ask", json={"question": "I'm having severe chest pain and can't breathe"}
    )
    body = resp.json()
    assert body["guardrail_triggered"] is True
    assert body["evidence_used"] == []
    assert "emergency" in body["answer"].lower()


def test_insufficient_evidence_refuses(client):
    resp = client.post(
        "/ask", json={"question": "What is the best programming language to learn?"}
    )
    body = resp.json()
    assert body["guardrail_triggered"] is False
    assert body["evidence_sufficient"] is False


def test_empty_question_is_rejected(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # pydantic validation
