"""Pydantic models that define the public API contract for /ask."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The patient's question in natural language.",
        json_schema_extra={
            "example": "What should I ask my doctor about HFpEF treatment options?"
        },
    )


class EvidenceItem(BaseModel):
    """A single retrieved chunk that supported (or was considered for) the answer."""

    document_id: str
    chunk_id: str
    similarity_score: float = Field(..., description="Cosine similarity in [0, 1].")
    # Extra, additive field (not required by the spec) that makes responses more
    # useful and auditable without breaking the required contract.
    text_preview: Optional[str] = Field(
        default=None, description="Short preview of the chunk text."
    )


class AskResponse(BaseModel):
    """Response body for POST /ask. The first four fields match the assignment spec."""

    answer: str
    evidence_used: List[EvidenceItem]
    evidence_sufficient: bool
    guardrail_triggered: bool
    # Additive fields below — they extend the spec without altering it.
    guardrail_type: Optional[str] = Field(
        default=None,
        description="Which guardrail fired, if any (e.g. 'emergency_escalation').",
    )
    model_name: Optional[str] = Field(
        default=None, description="LLM used, or 'extractive-fallback' / null."
    )
    disclaimer: str = Field(
        default=(
            "This is an educational tool, not medical advice, and cannot diagnose "
            "or treat any condition. Always consult a qualified clinician."
        )
    )
