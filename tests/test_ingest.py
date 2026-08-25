from __future__ import annotations

from pathlib import Path

import numpy as np

from database import get_index_metadata
from ingest import create_chunks, ensure_index, read_source
from settings import MAX_CHUNK_CHARS, MAX_CHUNK_WORDS


class FakeEmbedder:
    model_name = "fake-multilingual"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.asarray(
            [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


def test_documents_txt_is_read_as_utf8(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text("İstanbul’da yapay zekâ çalışmaları.", encoding="utf-8")
    assert read_source(source) == "İstanbul’da yapay zekâ çalışmaları."


def test_chunks_keep_words_intact_and_respect_hard_limits() -> None:
    words = [f"kelime{i}" for i in range(180)]
    chunks = create_chunks(" ".join(words) + ".")

    chunk_words = [word.rstrip(".") for chunk in chunks for word in chunk.split()]
    assert set(chunk_words) == set(words)
    assert all(len(chunk.split()) <= MAX_CHUNK_WORDS for chunk in chunks)
    assert all(len(chunk) <= MAX_CHUNK_CHARS for chunk in chunks)


def test_embedding_cache_skips_unchanged_source(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    database = tmp_path / "rag.sqlite3"
    source.write_text("Birinci bilgi.\n\nİkinci bilgi.", encoding="utf-8")
    embedder = FakeEmbedder()

    first = ensure_index(source_path=source, db_path=database, embedder=embedder)
    second = ensure_index(source_path=source, db_path=database, embedder=embedder)

    assert first.rebuilt is True
    assert second.rebuilt is False
    assert embedder.calls == 1

    source.write_text("Birinci bilgi değişti.\n\nİkinci bilgi.", encoding="utf-8")
    third = ensure_index(source_path=source, db_path=database, embedder=embedder)
    assert third.rebuilt is True
    assert embedder.calls == 2


def test_only_documents_txt_is_indexed_not_a_pdf_directory(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text("Tek geçerli kaynak budur.", encoding="utf-8")
    legacy = tmp_path / "documents"
    legacy.mkdir()
    (legacy / "eski.pdf").write_bytes(b"PDF-icerigi-indekslenmemeli")
    database = tmp_path / "rag.sqlite3"

    ensure_index(source_path=source, db_path=database, embedder=FakeEmbedder())
    metadata = get_index_metadata(database)

    assert metadata["source_name"] == "documents.txt"
    assert "PDF" not in database.read_bytes().decode("utf-8", errors="ignore")
