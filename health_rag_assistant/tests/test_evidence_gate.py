"""Tests for retrieval and the evidence-sufficiency gate."""

import pytest

from app.rag_engine import HealthRAGEngine


@pytest.fixture(scope="module")
def engine():
    return HealthRAGEngine()


def test_relevant_question_is_sufficient(engine):
    retrieved = engine.retrieve("What is HFpEF and what are its symptoms?")
    decision = engine.evaluate_evidence(retrieved)
    assert decision.sufficient is True
    assert retrieved[0].similarity_score > 0


def test_treatment_question_retrieves_treatment_doc(engine):
    retrieved = engine.retrieve("Which medications reduce HFpEF hospitalizations?")
    top_doc = retrieved[0].chunk.document_id
    assert "treatment" in top_doc


def test_off_topic_question_is_insufficient(engine):
    retrieved = engine.retrieve(
        "How do I replace the timing belt on a 2008 Honda Civic?"
    )
    decision = engine.evaluate_evidence(retrieved)
    assert decision.sufficient is False


def test_empty_retrieval_is_insufficient(engine):
    decision = engine.evaluate_evidence([])
    assert decision.sufficient is False
    assert decision.reason == "no_chunks_retrieved"
