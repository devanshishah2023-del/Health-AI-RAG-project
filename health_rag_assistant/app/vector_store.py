"""
Vector store.

A thin wrapper around a FAISS inner-product index over L2-normalised vectors,
which makes inner product equal to cosine similarity. If FAISS is not installed,
the store transparently falls back to a NumPy brute-force search so the system
still runs (the corpus here is tiny, so this is perfectly adequate).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:  # FAISS is the primary, preferred backend.
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - exercised only when FAISS is absent
    _HAS_FAISS = False


class VectorStore:
    """Cosine-similarity search over a fixed set of vectors."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._matrix: np.ndarray | None = None
        self._index = None
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(dimension)

    @property
    def backend(self) -> str:
        return "faiss" if _HAS_FAISS else "numpy"

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        if _HAS_FAISS:
            self._index.add(vectors)
        else:
            self._matrix = (
                vectors
                if self._matrix is None
                else np.vstack([self._matrix, vectors])
            )

    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[float], List[int]]:
        query = np.ascontiguousarray(query.astype("float32")).reshape(1, -1)
        if _HAS_FAISS:
            scores, indices = self._index.search(query, top_k)
            return scores[0].tolist(), indices[0].tolist()
        # NumPy brute force.
        if self._matrix is None or len(self._matrix) == 0:
            return [], []
        sims = (self._matrix @ query[0]).astype("float32")
        top_k = min(top_k, len(sims))
        order = np.argsort(-sims)[:top_k]
        return sims[order].tolist(), order.tolist()
