"""
Pluggable embedding backends.

Two interchangeable implementations are provided behind a common interface:

* ``SentenceTransformerEmbedder`` — dense semantic embeddings using a
  Sentence-Transformers model (default; recommended for quality).
* ``TfidfEmbedder`` — a lexical TF-IDF baseline using scikit-learn that needs no
  model download. It is used for offline demos and the automated test suite, and
  it doubles as a useful baseline to compare dense retrieval against.

Both return L2-normalised float32 vectors so that an inner-product search is
equivalent to cosine similarity.
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embedder(Protocol):
    """Common interface every embedding backend implements."""

    dimension: int

    def fit(self, corpus: List[str]) -> None: ...

    def embed(self, texts: List[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Dense embeddings via sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so the lightweight TF-IDF path has no heavy dependency.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def fit(self, corpus: List[str]) -> None:
        # Dense models are pre-trained; nothing to fit on our corpus.
        return None

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return _l2_normalize(vectors)


class TfidfEmbedder:
    """Lexical TF-IDF baseline (no downloads required)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.dimension = 0
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self.dimension = len(self._vectorizer.get_feature_names_out())
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.embed called before fit().")
        dense = self._vectorizer.transform(texts).toarray()
        return _l2_normalize(dense)


def build_embedder(backend: str, model_name: str) -> Embedder:
    """Factory that returns the configured embedding backend."""
    backend = backend.lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")
