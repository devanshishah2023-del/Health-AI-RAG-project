"""Test package. Forces the offline-friendly TF-IDF backend and no LLM key."""

import os

# Make tests hermetic: no model downloads, no network, deterministic behaviour.
os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("USE_EXTRACTIVE_FALLBACK", "true")
