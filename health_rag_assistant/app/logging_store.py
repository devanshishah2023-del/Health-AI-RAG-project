"""
Research logging.

Every query is recorded to BOTH:
* a JSONL file (one JSON object per line) — easy for humans and for streaming
  ingestion; and
* a SQLite database — queryable with SQL for later analysis.

Each record contains everything the assignment asks for: timestamp, the user
question, retrieved document/chunk IDs, similarity scores, the evidence
sufficiency decision, the guardrail decision, the final answer, and (when an LLM
was used) the model name and a prompt summary. We also store latency and the
retrieval backend, which are useful for a research write-up.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from app import config

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_question TEXT NOT NULL,
    retrieved_ids TEXT NOT NULL,        -- JSON array of "doc_id:chunk_id"
    similarity_scores TEXT NOT NULL,    -- JSON array of floats
    evidence_sufficient INTEGER NOT NULL,
    guardrail_triggered INTEGER NOT NULL,
    guardrail_type TEXT,
    final_answer TEXT NOT NULL,
    model_name TEXT,
    prompt_summary TEXT,
    retrieval_backend TEXT,
    latency_ms REAL
);
"""


def _init_sqlite() -> None:
    with sqlite3.connect(config.SQLITE_LOG_PATH) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_log(entry: Dict[str, Any]) -> None:
    """Persist one research-log record to JSONL and SQLite (thread-safe)."""
    with _LOCK:
        # JSONL
        with open(config.JSONL_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # SQLite
        _init_sqlite()
        with sqlite3.connect(config.SQLITE_LOG_PATH) as conn:
            conn.execute(
                """
                INSERT INTO research_log (
                    timestamp, user_question, retrieved_ids, similarity_scores,
                    evidence_sufficient, guardrail_triggered, guardrail_type,
                    final_answer, model_name, prompt_summary, retrieval_backend,
                    latency_ms
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    entry["timestamp"],
                    entry["user_question"],
                    json.dumps(entry["retrieved_ids"]),
                    json.dumps(entry["similarity_scores"]),
                    int(bool(entry["evidence_sufficient"])),
                    int(bool(entry["guardrail_triggered"])),
                    entry.get("guardrail_type"),
                    entry["final_answer"],
                    entry.get("model_name"),
                    entry.get("prompt_summary"),
                    entry.get("retrieval_backend"),
                    entry.get("latency_ms"),
                ),
            )
            conn.commit()
