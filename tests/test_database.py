from __future__ import annotations

from pathlib import Path

from database import ChunkRecord, replace_index, retrieve_chunks


def seed(database: Path) -> None:
    replace_index(
        [
            ChunkRecord(0, "düşük", [0.0, 1.0]),
            ChunkRecord(1, "en yüksek", [1.0, 0.0]),
            ChunkRecord(2, "ikinci", [0.8, 0.2]),
        ],
        source_hash="abc",
        embedding_model="fake",
        db_path=database,
    )


def test_top_k_is_sorted_by_cosine_similarity(tmp_path: Path) -> None:
    database = tmp_path / "rag.sqlite3"
    seed(database)

    results = retrieve_chunks([1.0, 0.0], top_k=2, threshold=-1.0, db_path=database)

    assert [item.content for item in results] == ["en yüksek", "ikinci"]
    assert results[0].score > results[1].score


def test_threshold_filters_low_results_after_top_k(tmp_path: Path) -> None:
    database = tmp_path / "rag.sqlite3"
    seed(database)

    results = retrieve_chunks([1.0, 0.0], top_k=3, threshold=0.99, db_path=database)

    assert [item.content for item in results] == ["en yüksek"]
