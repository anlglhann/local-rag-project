"""documents.txt okuma, anlamlı parçalama ve embedding önbelleği."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from database import ChunkRecord, index_is_current, replace_index
from settings import (
    DATABASE_PATH,
    DOCUMENT_PATH,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_CHARS,
    MAX_CHUNK_WORDS,
    OVERLAP_MAX_WORDS,
    TARGET_CHUNK_CHARS,
    TARGET_CHUNK_WORDS,
)


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Küçük, çok dilli ve Apple Silicon üzerinde kararlı embedding katmanı."""

    model_name = EMBEDDING_MODEL_NAME

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is None:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer

            # Önbellek varsa ağ kontrolü yapmadan aç; ilk kullanımda bulunamazsa
            # normal indirme yoluna geri dön.
            try:
                cached_path = snapshot_download(
                    self.model_name,
                    local_files_only=True,
                )
                self._model = SentenceTransformer(cached_path, local_files_only=True)
            except (OSError, FileNotFoundError):
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._load_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class IndexStatus:
    rebuilt: bool
    chunk_count: int
    source_hash: str


@dataclass(frozen=True)
class _TextUnit:
    text: str
    paragraph: int


def read_source(path: Path = DOCUMENT_PATH) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{source.name} dosyası bulunamadı.")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("documents.txt boş bırakılamaz.")
    return text


def save_source(text: str, path: Path = DOCUMENT_PATH) -> None:
    if not text.strip():
        raise ValueError("documents.txt boş bırakılamaz.")

    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.strip() + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=source.parent,
        prefix=f".{source.name}.",
        delete=False,
    ) as temporary:
        temporary.write(normalized)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, source)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_paragraph(paragraph: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", paragraph).strip()


def _split_long_unit(text: str, paragraph: int) -> list[_TextUnit]:
    words = text.split()
    units: list[_TextUnit] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])
        if current and (
            len(current) >= MAX_CHUNK_WORDS or len(candidate) > MAX_CHUNK_CHARS
        ):
            units.append(_TextUnit(" ".join(current), paragraph))
            current = [word]
        else:
            current.append(word)

    if current:
        units.append(_TextUnit(" ".join(current), paragraph))
    return units


def _text_units(text: str) -> list[_TextUnit]:
    paragraphs = [
        _normalize_paragraph(item)
        for item in re.split(r"\n\s*\n", text.strip())
        if item.strip()
    ]
    units: list[_TextUnit] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
            if sentence.strip()
        ]
        for sentence in sentences:
            if (
                len(sentence.split()) > MAX_CHUNK_WORDS
                or len(sentence) > MAX_CHUNK_CHARS
            ):
                units.extend(_split_long_unit(sentence, paragraph_index))
            else:
                units.append(_TextUnit(sentence, paragraph_index))
    return units


def _join_units(units: Sequence[_TextUnit]) -> str:
    output = ""
    previous_paragraph: int | None = None
    for unit in units:
        if output:
            output += "\n\n" if unit.paragraph != previous_paragraph else " "
        output += unit.text
        previous_paragraph = unit.paragraph
    return output


def create_chunks(text: str) -> list[str]:
    """Paragraf/cümle sınırlarını koruyarak yaklaşık 200–300 karakterlik parçalar üretir."""

    units = _text_units(text)
    chunks: list[str] = []
    current: list[_TextUnit] = []

    def flush(*, keep_overlap: bool) -> None:
        nonlocal current
        chunk = _join_units(current).strip()
        if chunk and (not chunks or chunks[-1] != chunk):
            chunks.append(chunk)

        overlap: list[_TextUnit] = []
        if keep_overlap and len(current) > 1:
            last = current[-1]
            if len(last.text.split()) <= OVERLAP_MAX_WORDS:
                overlap = [last]
        current = overlap

    for unit in units:
        # Kısa, bağımsız paragrafları sırf hedef boyuta ulaşmak için birbirine
        # yapıştırma; bu, retrieval sırasında farklı konuları bulanıklaştırır.
        if current and unit.paragraph != current[-1].paragraph:
            flush(keep_overlap=False)

        candidate_units = [*current, unit]
        candidate = _join_units(candidate_units)
        candidate_words = len(candidate.split())
        current_text = _join_units(current)
        reached_target = (
            len(current_text) >= TARGET_CHUNK_CHARS
            or len(current_text.split()) >= TARGET_CHUNK_WORDS
        )
        exceeds_limit = (
            len(candidate) > MAX_CHUNK_CHARS or candidate_words > MAX_CHUNK_WORDS
        )

        if current and (reached_target or exceeds_limit):
            flush(keep_overlap=True)
            candidate_units = [*current, unit]
            candidate = _join_units(candidate_units)
            if current and (
                len(candidate) > MAX_CHUNK_CHARS
                or len(candidate.split()) > MAX_CHUNK_WORDS
            ):
                current = []

        current.append(unit)

    if current:
        flush(keep_overlap=False)

    return chunks


def ensure_index(
    *,
    source_path: Path = DOCUMENT_PATH,
    db_path: Path = DATABASE_PATH,
    embedder: Embedder | None = None,
    force: bool = False,
) -> IndexStatus:
    text = read_source(source_path)
    digest = content_hash(text)
    model_name = embedder.model_name if embedder is not None else EMBEDDING_MODEL_NAME

    if not force and index_is_current(digest, model_name, db_path):
        from database import count_chunks

        return IndexStatus(False, count_chunks(db_path), digest)

    chunks = create_chunks(text)
    if not chunks:
        raise ValueError("documents.txt içinden indekslenebilir metin çıkarılamadı.")

    active_embedder = embedder or SentenceTransformerEmbedder()
    embeddings = active_embedder.encode(chunks)
    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding sayısı metin parçası sayısıyla eşleşmiyor.")

    records = [
        ChunkRecord(position=index, content=chunk, embedding=embeddings[index])
        for index, chunk in enumerate(chunks)
    ]
    count = replace_index(
        records,
        source_hash=digest,
        embedding_model=active_embedder.model_name,
        db_path=db_path,
    )
    return IndexStatus(True, count, digest)


def main() -> None:
    status = ensure_index()
    action = "yenilendi" if status.rebuilt else "zaten güncel"
    print(f"documents.txt indeksi {action}: {status.chunk_count} parça.")


if __name__ == "__main__":
    main()
