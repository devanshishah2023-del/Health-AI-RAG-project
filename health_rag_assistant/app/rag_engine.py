"""
RAG engine.

Orchestrates the full pipeline: build the index at startup, retrieve for a
question, apply the evidence-sufficiency gate, and (when evidence is sufficient)
generate a grounded answer. The guardrail check runs *before* this in the API
layer so that emergencies short-circuit everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app import config, llm
from app.chunking import Chunk, load_and_chunk_corpus
from app.embeddings import build_embedder
from app.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity_score: float


@dataclass
class EvidenceDecision:
    sufficient: bool
    reason: str
    top_score: float


class HealthRAGEngine:
    """Document-grounded retrieval and answer generation."""

    def __init__(self) -> None:
        self.embedder = build_embedder(
            config.EMBEDDING_BACKEND, config.EMBEDDING_MODEL_NAME
        )
        self.chunks: List[Chunk] = load_and_chunk_corpus(
            config.DATA_DIR,
            config.CHUNK_SIZE_WORDS,
            config.CHUNK_OVERLAP_WORDS,
        )
        if not self.chunks:
            raise RuntimeError(
                f"No documents found in {config.DATA_DIR}. Add .txt/.md files."
            )

        corpus_texts = [c.text for c in self.chunks]
        # TF-IDF needs to learn its vocabulary; dense models ignore this call.
        self.embedder.fit(corpus_texts)
        matrix = self.embedder.embed(corpus_texts)

        self.store = VectorStore(dimension=matrix.shape[1])
        self.store.add(matrix)

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, question: str, top_k: int | None = None) -> List[RetrievedChunk]:
        top_k = top_k or config.TOP_K
        query_vec = self.embedder.embed([question])[0]
        scores, indices = self.store.search(query_vec, top_k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores, indices):
            if 0 <= idx < len(self.chunks):
                results.append(
                    RetrievedChunk(chunk=self.chunks[idx], similarity_score=float(score))
                )
        return results

    # ----------------------------------------------------------- evidence gate
    def evaluate_evidence(self, retrieved: List[RetrievedChunk]) -> EvidenceDecision:
        """
        Decide whether retrieved evidence is strong enough to answer.

        Rule (documented in the README):
        * The single best chunk must clear EVIDENCE_THRESHOLD, AND
        * at least MIN_SUPPORTING_CHUNKS chunks must clear the softer
          SUPPORTING_FLOOR.
        Otherwise we refuse and return a safe "not enough information" message.
        """
        if not retrieved:
            return EvidenceDecision(False, "no_chunks_retrieved", 0.0)

        top_score = retrieved[0].similarity_score
        supporting = sum(
            1 for r in retrieved if r.similarity_score >= config.SUPPORTING_FLOOR
        )

        if top_score < config.EVIDENCE_THRESHOLD:
            return EvidenceDecision(
                False,
                f"top_score {top_score:.3f} < threshold {config.EVIDENCE_THRESHOLD:.3f}",
                top_score,
            )
        if supporting < config.MIN_SUPPORTING_CHUNKS:
            return EvidenceDecision(
                False,
                f"only {supporting} chunk(s) above floor {config.SUPPORTING_FLOOR:.3f}",
                top_score,
            )
        return EvidenceDecision(
            True,
            f"top_score {top_score:.3f} >= threshold {config.EVIDENCE_THRESHOLD:.3f}",
            top_score,
        )

    # -------------------------------------------------------------- generation
    def generate(self, question: str, retrieved: List[RetrievedChunk]):
        chunks = [r.chunk for r in retrieved]
        return llm.generate_answer(question, chunks)
