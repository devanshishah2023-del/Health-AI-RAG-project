"""
Central configuration for the Health AI RAG Assistant.

All tunable behaviour lives here so the rest of the codebase reads cleanly and so
that a reviewer can see, in one place, every knob that affects retrieval, the
evidence gate, the safety guardrails, and the LLM call.

Values can be overridden with environment variables (see .env.example), which makes
the service easy to configure in different environments without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if one is present. This is a no-op in
# environments where the variables are already set (e.g. Docker, CI).
load_dotenv()


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", BASE_DIR / "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSONL_LOG_PATH = LOGS_DIR / "research_log.jsonl"
SQLITE_LOG_PATH = LOGS_DIR / "research_log.db"

# --------------------------------------------------------------------------- #
# Embedding / retrieval configuration
# --------------------------------------------------------------------------- #
# Backend options:
#   "sentence-transformers" -> dense semantic embeddings (recommended, default).
#   "tfidf"                 -> lexical baseline using scikit-learn. Requires no
#                              model download, so it is ideal for offline demos
#                              and automated tests.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking (word-based sliding window with overlap).
CHUNK_SIZE_WORDS = _get_int("CHUNK_SIZE_WORDS", 120)
CHUNK_OVERLAP_WORDS = _get_int("CHUNK_OVERLAP_WORDS", 30)

# How many chunks to retrieve per question.
TOP_K = _get_int("TOP_K", 3)

# --------------------------------------------------------------------------- #
# Evidence sufficiency gate
# --------------------------------------------------------------------------- #
# Cosine similarity sits in [-1, 1]; for normalised text embeddings it is
# effectively in [0, 1]. The right threshold depends on the embedding backend,
# because lexical (TF-IDF) and dense models produce different score
# distributions. We therefore keep a separate default per backend.
_DEFAULT_THRESHOLDS = {
    "sentence-transformers": 0.40,
    # TF-IDF produces lower absolute scores on a small corpus, so it uses a
    # lower, separately-calibrated threshold. Dense embeddings are recommended.
    "tfidf": 0.13,
}
EVIDENCE_THRESHOLD = _get_float(
    "EVIDENCE_THRESHOLD",
    _DEFAULT_THRESHOLDS.get(EMBEDDING_BACKEND, 0.40),
)
# Require at least this many retrieved chunks to clear the soft floor before we
# are willing to answer. Guards against answering off a single weak match.
MIN_SUPPORTING_CHUNKS = _get_int("MIN_SUPPORTING_CHUNKS", 1)
# A softer floor used together with MIN_SUPPORTING_CHUNKS.
SUPPORTING_FLOOR = _get_float("SUPPORTING_FLOOR", EVIDENCE_THRESHOLD * 0.8)

# --------------------------------------------------------------------------- #
# LLM (OpenRouter) configuration
# --------------------------------------------------------------------------- #
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# "openrouter/free" is OpenRouter's auto-router that picks an available free
# model at request time. It is the most resilient default because individual
# free model IDs are rotated and retired without notice. See README for
# specific alternatives such as "meta-llama/llama-3.3-70b-instruct:free".
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openrouter/free")
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 500)
LLM_TIMEOUT_SECONDS = _get_float("LLM_TIMEOUT_SECONDS", 40.0)
# Optional attribution headers recommended by OpenRouter.
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost:8000")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Health AI RAG Assistant")

# If no API key is configured, the system falls back to a deterministic
# extractive answer assembled from the retrieved evidence. This lets the whole
# pipeline run end-to-end with zero secrets, which is convenient for grading.
USE_EXTRACTIVE_FALLBACK = os.getenv("USE_EXTRACTIVE_FALLBACK", "true").lower() == "true"
