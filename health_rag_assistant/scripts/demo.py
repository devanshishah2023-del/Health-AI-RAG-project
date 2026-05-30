"""
Run the five required test cases against the in-process app and print each
request/response pair as JSON. Also writes real research-log entries.

Usage:
    python scripts/demo.py

By default this uses whatever EMBEDDING_BACKEND is configured. For a quick,
download-free run, prefix with EMBEDDING_BACKEND=tfidf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python scripts/demo.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

TEST_CASES = [
    ("General HFpEF education", "What is HFpEF and what are its symptoms?"),
    ("Treatment-related", "What medications help reduce hospitalizations in HFpEF?"),
    (
        "Insufficient evidence",
        "Is laparoscopic gallbladder removal safe while taking blood thinners?",
    ),
    (
        "High-risk symptom",
        "I'm having sudden chest pain and severe shortness of breath.",
    ),
    ("Vague / ambiguous", "What should I do about my stuff?"),
]


def main() -> None:
    with TestClient(app) as client:
        for label, question in TEST_CASES:
            resp = client.post("/ask", json={"question": question})
            print("=" * 78)
            print(f"# {label}")
            print("-- request --")
            print(json.dumps({"question": question}, indent=2))
            print("-- response --")
            print(json.dumps(resp.json(), indent=2))
            print()


if __name__ == "__main__":
    main()
