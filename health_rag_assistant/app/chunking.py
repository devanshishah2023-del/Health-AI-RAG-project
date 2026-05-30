"""
Document loading and chunking.

Each document is split into overlapping, word-bounded chunks. Every chunk is
tagged with a stable ``document_id`` (derived from the filename) and a
sequential ``chunk_id``. The overlap preserves context that would otherwise be
cut across a chunk boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob
from typing import List


@dataclass
class Chunk:
    """A single retrievable unit of text with its provenance."""

    document_id: str
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def chunk_text(
    text: str,
    document_id: str,
    chunk_size_words: int,
    overlap_words: int,
) -> List[Chunk]:
    """Split a single document's text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []

    if overlap_words >= chunk_size_words:
        # Defensive: an overlap >= window would never advance the cursor.
        overlap_words = max(0, chunk_size_words // 2)

    step = chunk_size_words - overlap_words
    chunks: List[Chunk] = []
    index = 0
    cursor = 0
    while cursor < len(words):
        window = words[cursor : cursor + chunk_size_words]
        chunk_text_value = " ".join(window).strip()
        if chunk_text_value:
            index += 1
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_id=f"chunk_{index}",
                    text=chunk_text_value,
                    metadata={
                        "start_word": cursor,
                        "end_word": cursor + len(window),
                        "word_count": len(window),
                    },
                )
            )
        cursor += step
    return chunks


def _document_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".txt", ".md"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def load_and_chunk_corpus(
    data_dir,
    chunk_size_words: int,
    overlap_words: int,
) -> List[Chunk]:
    """Load every .txt/.md file in ``data_dir`` and return all chunks, ordered."""
    paths = sorted(
        glob(os.path.join(str(data_dir), "*.txt"))
        + glob(os.path.join(str(data_dir), "*.md"))
    )
    all_chunks: List[Chunk] = []
    for path in paths:
        document_id = _document_id_from_path(path)
        text = _read_text(path)
        all_chunks.extend(
            chunk_text(text, document_id, chunk_size_words, overlap_words)
        )
    return all_chunks
