"""Tests for document chunking."""

from app.chunking import chunk_text, load_and_chunk_corpus
from app import config


def test_chunk_ids_are_sequential_and_tagged():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, "doc_x", chunk_size_words=100, overlap_words=20)
    assert len(chunks) >= 3
    assert all(c.document_id == "doc_x" for c in chunks)
    assert [c.chunk_id for c in chunks[:3]] == ["chunk_1", "chunk_2", "chunk_3"]


def test_overlap_creates_shared_words():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(text, "d", chunk_size_words=20, overlap_words=5)
    # The last 5 words of chunk 1 should reappear at the start of chunk 2.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-5:] == second_words[:5]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", "d", 100, 20) == []


def test_corpus_loads_all_documents():
    chunks = load_and_chunk_corpus(
        config.DATA_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS
    )
    doc_ids = {c.document_id for c in chunks}
    # We ship five sample documents.
    assert len(doc_ids) == 5
    assert len(chunks) > 5
