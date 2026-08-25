"""Uygulamanın tek noktadan değiştirilebilen ayarları."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENT_PATH = PROJECT_ROOT / "documents.txt"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "rag.sqlite3"

# Sunumda farklı bir model denemek için yalnızca bu değeri veya aynı adlı
# ortam değişkenini değiştirmek yeterlidir.
CHAT_MODEL_NAME = os.getenv("LOCAL_RAG_CHAT_MODEL", "qwen2.5-1.5b")
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

FOUNDRY_APP_DATA_DIR = DATA_DIR / "foundry"
FOUNDRY_MODEL_CACHE_DIR = DATA_DIR / "model_cache"
FOUNDRY_LOG_DIR = DATA_DIR / "logs"

TOP_K = 3
# Örnek documents.txt üzerinde doğru soruların en düşük skoru 0.412,
# bağlam dışı "fiyat" ve "sahip" sorularının skorları sırasıyla 0.358/0.329.
SIMILARITY_THRESHOLD = 0.38
LEXICAL_FALLBACK_MIN_SIMILARITY = 0.10
LEXICAL_CANDIDATE_K = 12

TARGET_CHUNK_CHARS = 280
MAX_CHUNK_CHARS = 360
TARGET_CHUNK_WORDS = 55
MAX_CHUNK_WORDS = 70
OVERLAP_MAX_WORDS = 12

FALLBACK_MESSAGE = "Bu bilgi documents.txt dosyasında bulunamadı."
