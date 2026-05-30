"""
FastAPI application exposing POST /ask plus helper endpoints.

Pipeline order for every request:
1. Validate input.
2. SAFETY FIRST: run the emergency guardrail. If it fires, return the escalation
   message immediately, with no retrieval and no generation.
3. Retrieve top-k chunks.
4. Apply the evidence-sufficiency gate. If insufficient, refuse safely.
5. Generate a grounded, patient-friendly answer.
6. Log everything for research, including latency.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import config, guardrails, logging_store
from app.rag_engine import HealthRAGEngine
from app.schemas import AskRequest, AskResponse, EvidenceItem

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough reliable information in my reference documents to answer "
    "that safely. I can only answer questions covered by my HFpEF and "
    "cardio-kidney-metabolic education materials. Please ask your care team, or "
    "try rephrasing your question to focus on those topics."
)

# Built once at startup and reused for every request.
engine: HealthRAGEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = HealthRAGEngine()
    yield
    engine = None


app = FastAPI(
    title="Health AI RAG Assistant",
    description=(
        "Document-grounded patient-education backend for HFpEF and "
        "cardio-kidney-metabolic conditions, with an evidence-sufficiency gate, "
        "emergency safety guardrails, and research logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe with a little diagnostic info."""
    return {
        "status": "ok",
        "chunks_indexed": len(engine.chunks) if engine else 0,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "vector_backend": engine.store.backend if engine else None,
        "llm_model": config.LLM_MODEL_NAME if config.OPENROUTER_API_KEY else "extractive-fallback",
        "evidence_threshold": config.EVIDENCE_THRESHOLD,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    assert engine is not None  # guaranteed by lifespan
    started = time.perf_counter()
    question = request.question.strip()

    log_entry = {
        "timestamp": logging_store.now_iso(),
        "user_question": question,
        "retrieved_ids": [],
        "similarity_scores": [],
        "evidence_sufficient": False,
        "guardrail_triggered": False,
        "guardrail_type": None,
        "final_answer": "",
        "model_name": None,
        "prompt_summary": None,
        "retrieval_backend": engine.store.backend,
        "latency_ms": None,
    }

    # ----- 1) Safety guardrail (runs before any retrieval/generation) ----------
    guard = guardrails.evaluate(question)
    if guard.triggered:
        log_entry.update(
            guardrail_triggered=True,
            guardrail_type=guard.guardrail_type,
            final_answer=guard.message,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        logging_store.write_log(log_entry)
        return AskResponse(
            answer=guard.message,
            evidence_used=[],
            evidence_sufficient=False,
            guardrail_triggered=True,
            guardrail_type=guard.guardrail_type,
            model_name=None,
        )

    # ----- 2) Retrieval --------------------------------------------------------
    retrieved = engine.retrieve(question)
    evidence_items = [
        EvidenceItem(
            document_id=r.chunk.document_id,
            chunk_id=r.chunk.chunk_id,
            similarity_score=round(r.similarity_score, 4),
            text_preview=(r.chunk.text[:160] + ("…" if len(r.chunk.text) > 160 else "")),
        )
        for r in retrieved
    ]
    log_entry["retrieved_ids"] = [
        f"{r.chunk.document_id}:{r.chunk.chunk_id}" for r in retrieved
    ]
    log_entry["similarity_scores"] = [round(r.similarity_score, 4) for r in retrieved]

    # ----- 3) Evidence-sufficiency gate ---------------------------------------
    decision = engine.evaluate_evidence(retrieved)
    log_entry["evidence_sufficient"] = decision.sufficient

    if not decision.sufficient:
        log_entry.update(
            final_answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        logging_store.write_log(log_entry)
        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            evidence_used=evidence_items,
            evidence_sufficient=False,
            guardrail_triggered=False,
            model_name=None,
        )

    # ----- 4) Grounded generation ---------------------------------------------
    answer, model_used = engine.generate(question, retrieved)
    from app import llm  # local import to avoid an unused import when not generating

    log_entry.update(
        final_answer=answer,
        model_name=model_used,
        prompt_summary=llm.prompt_summary(),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    logging_store.write_log(log_entry)

    return AskResponse(
        answer=answer,
        evidence_used=evidence_items,
        evidence_sufficient=True,
        guardrail_triggered=False,
        model_name=model_used,
    )
