"""SQLite tabanlı küçük ve yerel vektör deposu."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from settings import DATABASE_PATH


@dataclass(frozen=True)
class ChunkRecord:
    position: int
    content: str
    embedding: Sequence[float]
    source_name: str = "documents.txt"


@dataclass(frozen=True)
class RetrievedChunk:
    position: int
    content: str
    score: float
    source_name: str


def _connect(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position INTEGER NOT NULL UNIQUE,
                source_name TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )


def get_index_metadata(db_path: Path = DATABASE_PATH) -> dict[str, str]:
    initialize_database(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute("SELECT key, value FROM index_metadata").fetchall()
    return {row["key"]: row["value"] for row in rows}


def index_is_current(
    source_hash: str,
    embedding_model: str,
    db_path: Path = DATABASE_PATH,
) -> bool:
    metadata = get_index_metadata(db_path)
    return (
        metadata.get("source_name") == "documents.txt"
        and metadata.get("source_hash") == source_hash
        and metadata.get("embedding_model") == embedding_model
        and int(metadata.get("chunk_count", "0")) > 0
    )


def replace_index(
    records: Iterable[ChunkRecord],
    *,
    source_hash: str,
    embedding_model: str,
    db_path: Path = DATABASE_PATH,
) -> int:
    """Eski indeksi tek transaction içinde tamamen yenisiyle değiştirir."""

    prepared: list[tuple[int, str, str, bytes, int]] = []
    for record in records:
        vector = np.asarray(record.embedding, dtype=np.float32).reshape(-1)
        if not record.content.strip() or vector.size == 0:
            raise ValueError("Boş metin veya embedding indekse eklenemez.")
        prepared.append(
            (
                record.position,
                record.source_name,
                record.content,
                vector.tobytes(),
                int(vector.size),
            )
        )

    if not prepared:
        raise ValueError("İndekslenecek metin parçası bulunamadı.")

    initialize_database(db_path)
    metadata = {
        "source_name": "documents.txt",
        "source_hash": source_hash,
        "embedding_model": embedding_model,
        "chunk_count": str(len(prepared)),
        "indexed_at": datetime.now(UTC).isoformat(),
    }

    with _connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM index_metadata")
        connection.executemany(
            """
            INSERT INTO chunks(position, source_name, content, embedding, dimensions)
            VALUES (?, ?, ?, ?, ?)
            """,
            prepared,
        )
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )

    return len(prepared)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def retrieve_chunks(
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
    threshold: float = 0.35,
    db_path: Path = DATABASE_PATH,
) -> list[RetrievedChunk]:
    """Önce Top-K sıralar, ardından eşik altındaki sonuçları eler."""

    if top_k < 1:
        raise ValueError("top_k en az 1 olmalıdır.")

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    initialize_database(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT position, source_name, content, embedding, dimensions
            FROM chunks
            ORDER BY position
            """
        ).fetchall()

    ranked: list[RetrievedChunk] = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32, count=row["dimensions"])
        ranked.append(
            RetrievedChunk(
                position=row["position"],
                content=row["content"],
                score=_cosine_similarity(query, vector),
                source_name=row["source_name"],
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.position))
    return [item for item in ranked[:top_k] if item.score >= threshold]


def count_chunks(db_path: Path = DATABASE_PATH) -> int:
    initialize_database(db_path)
    with _connect(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
    return int(row["total"])
