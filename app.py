"""RAG retrieval ve Microsoft Foundry Local yanıt katmanı."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from database import RetrievedChunk, retrieve_chunks
from ingest import Embedder, SentenceTransformerEmbedder, ensure_index
from settings import (
    CHAT_MODEL_NAME,
    DATABASE_PATH,
    DOCUMENT_PATH,
    FALLBACK_MESSAGE,
    FOUNDRY_APP_DATA_DIR,
    FOUNDRY_LOG_DIR,
    FOUNDRY_MODEL_CACHE_DIR,
    LEXICAL_CANDIDATE_K,
    LEXICAL_FALLBACK_MIN_SIMILARITY,
    SIMILARITY_THRESHOLD,
    TOP_K,
)


SYSTEM_PROMPT = """Sen, yalnızca verilen documents.txt parçalarından bilgi çıkaran bir asistansın.
Kurallar:
- Yalnızca verilen metin parçalarındaki bilgileri kullan.
- Genel bilgini, varsayımını veya tahminini ekleme.
- Cevabın tamamını Türkçe, kısa ve doğal yaz; İngilizce "yes/no" kullanma.
- Uygunsa cevabı metindeki ilgili cümleyi aynen aktararak ver.
- Tarih, saat, gün sayısı ve e-posta adresi gibi değerleri metindeki biçimiyle aynen koru.
- İlk sonuç en ilgili metindir; soru orada yanıtlanıyorsa doğrudan o cümledeki bilgiyi kullan.
- Soru birden fazla bilgi istiyorsa ilgili tüm sonuçları kullan ve sorunun her kısmını cevapla.
- "değildir", "verilmez" ve "paylaşılmaz" gibi olumsuzlukları yok sayma; anlamı aynen koru.
- Talimatları ya da kaynak etiketlerini cevaba kopyalama.
- "Bağlamda" gibi mekanik ifadeler kullanma.
- Bilgi yetersizse hiçbir şey uydurma.
"""


class ChatService(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class UserFacingError(RuntimeError):
    """Arayüzde teknik stack trace olmadan gösterilebilen hata."""


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    chunks: tuple[RetrievedChunk, ...]


_QUESTION_STOP_WORDS = {
    "acaba",
    "bir",
    "bu",
    "da",
    "de",
    "hangi",
    "kaç",
    "için",
    "ile",
    "kim",
    "kimdir",
    "mı",
    "mi",
    "mu",
    "mü",
    "nasıl",
    "ne",
    "nedir",
    "nelerdir",
    "nerede",
    "var",
    "ve",
    "veya",
    "açıkla",
    "açıklar",
    "birlikte",
    "hakkında",
    "misin",
    "mısın",
    "nin",
    "özetle",
    "özetler",
    "söyle",
    "söyler",
}

_GENERIC_SOURCE_TERMS = {
    "bilgi",
    "müşteri",
    "nova",
    "şirket",
    "teknoloji",
    "ürün",
}

_TERM_PREFIXES = (
    ("adres", "adres"),
    ("çalış", "çalışma"),
    ("destek", "destek"),
    ("fiyat", "fiyat"),
    ("gerek", "gerek"),
    ("gün", "gün"),
    ("hizmet", "hizmet"),
    ("iad", "iade"),
    ("koşul", "koşul"),
    ("kurul", "kurul"),
    ("kurucu", "kurucu"),
    ("maliyet", "maliyet"),
    ("müşteri", "müşteri"),
    ("paylaş", "paylaş"),
    ("sahip", "sahip"),
    ("saat", "saat"),
    ("şart", "şart"),
    ("şehir", "şehir"),
    ("şirket", "şirket"),
    ("ulaş", "ulaş"),
    ("ücret", "ücret"),
    ("ürün", "ürün"),
    ("hedef", "hedef"),
    ("yıl", "yıl"),
)

_REQUIRED_EVIDENCE_RULES = (
    (r"\btelefon\b", r"\btelefon\b|(?:\+?\d[\d ()-]{7,}\d)"),
    (r"\bgaranti\b", r"\bgaranti\b"),
    (r"\bkargo\b", r"\bkargo\b"),
    (
        r"\b(?:fiyat|ücret|maliyet)\w*\b|\bkaç\s+(?:para|tl)\b",
        r"\b(?:fiyat|ücret|maliyet)\w*\b|₺|\b(?:tl|try|usd|eur)\b",
    ),
    (
        r"\b(?:çalışan|personel)\w*\s+(?:sayısı|adedi)\b|\bkaç\s+(?:çalışan|personel)\w*\b",
        r"\b(?:çalışan|personel)\w*\s+(?:sayısı|adedi)\b|\b\d+\s+(?:çalışan|personel)\w*\b",
    ),
    (r"\b(?:yurt\s*dış|uluslararası)\w*\b", r"\b(?:yurt\s*dış|uluslararası)\w*\b"),
    (r"\bödeme\w*\b", r"\bödeme\w*\b"),
    (
        r"\b(?:sosyal\s*medya|instagram|linkedin|facebook|twitter)\b",
        r"\b(?:sosyal\s*medya|instagram|linkedin|facebook|twitter)\b",
    ),
    (r"\b(?:sahip|kurucu|genel\s*müdür)\w*\b", r"\b(?:sahip|kurucu|genel\s*müdür)\w*\b"),
    (r"\b(?:ankara|şube|ofis)\w*\b", r"\b(?:ankara|şube|ofis)\w*\b"),
)


def _normalize_term(token: str) -> str:
    lowered = token.casefold()
    for prefix, normalized in _TERM_PREFIXES:
        if lowered.startswith(prefix):
            return normalized
    return lowered


def _content_terms(text: str) -> set[str]:
    raw_tokens = re.findall(r"[0-9a-zçğıöşü@._+-]+", text.casefold())
    normalized = {_normalize_term(token) for token in raw_tokens}
    return normalized - _QUESTION_STOP_WORDS - _GENERIC_SOURCE_TERMS


def _has_required_evidence(question: str, candidates: Sequence[RetrievedChunk]) -> bool:
    normalized_question = question.casefold()
    searchable_text = "\n".join(candidate.content for candidate in candidates).casefold()
    for question_pattern, evidence_pattern in _REQUIRED_EVIDENCE_RULES:
        if re.search(question_pattern, normalized_question) and not re.search(
            evidence_pattern,
            searchable_text,
        ):
            return False
    return True


def _term_overlap(question: str, candidate: RetrievedChunk) -> set[str]:
    overlap = _content_terms(question) & _content_terms(candidate.content)
    asks_for_offerings = bool(
        re.search(r"\bhangi\s+hizmet|\bhizmetlerini\b|\bhizmetleri\b", question.casefold())
    )
    is_availability_statement = bool(
        re.search(r"\bhizmet\s+(?:vermemektedir|verilmez|vermiyor)", candidate.content.casefold())
    )
    if asks_for_offerings and is_availability_statement:
        overlap.discard("hizmet")
    return overlap


def _select_relevant_chunks(
    question: str,
    query_embedding,
    *,
    top_k: int,
    threshold: float,
    db_path: Path,
) -> list[RetrievedChunk]:
    """Cosine eşiğine ek olarak açık konu sözcükleriyle kontrollü geri çağırım yapar."""

    candidates = retrieve_chunks(
        query_embedding,
        top_k=max(top_k, LEXICAL_CANDIDATE_K),
        threshold=-1.0,
        db_path=db_path,
    )
    if not _has_required_evidence(question, candidates):
        return []

    lexical: list[tuple[int, RetrievedChunk]] = []
    semantic: list[RetrievedChunk] = []
    for candidate in candidates:
        if candidate.score >= threshold:
            semantic.append(candidate)
        overlap = _term_overlap(question, candidate)
        if overlap and candidate.score >= LEXICAL_FALLBACK_MIN_SIMILARITY:
            lexical.append((len(overlap), candidate))

    semantic.sort(
        key=lambda candidate: (
            -len(_term_overlap(question, candidate)),
            -candidate.score,
            candidate.position,
        )
    )

    is_multi_topic = bool(
        re.search(r"\b(?:birlikte|özet\w*|açıkla\w*)\b|\sve\s", question.casefold())
    )

    # Normal tek konulu sorularda cosine sonuçlarını kullan. Çok konulu sorularda
    # eşiğin altında kalmış açık sözcük eşleşmelerini de bağlama ekle.
    if semantic:
        if not is_multi_topic:
            return semantic[:top_k]

        lexical.sort(key=lambda item: (-item[0], -item[1].score, item[1].position))
        combined: list[RetrievedChunk] = []
        seen: set[int] = set()
        for candidate in [item[1] for item in lexical] + semantic:
            if candidate.position not in seen:
                combined.append(candidate)
                seen.add(candidate.position)
            if len(combined) == top_k:
                break
        return combined

    lexical.sort(key=lambda item: (-item[0], -item[1].score, item[1].position))
    return [item[1] for item in lexical[:top_k]]


class FoundryChatService:
    """Foundry modelini gerektiğinde indiren/yükleyen tembel sohbet servisi."""

    def __init__(self, model_name: str = CHAT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._manager = None
        self._model = None
        self._client = None
        self._lock = threading.RLock()

    def _initialize_manager(self):
        if self._manager is not None:
            return self._manager

        from foundry_local_sdk import Configuration, FoundryLocalManager

        for directory in (
            FOUNDRY_APP_DATA_DIR,
            FOUNDRY_MODEL_CACHE_DIR,
            FOUNDRY_LOG_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if FoundryLocalManager.instance is None:
            FoundryLocalManager.initialize(
                Configuration(
                    app_name="local_rag_project",
                    app_data_dir=str(FOUNDRY_APP_DATA_DIR),
                    model_cache_dir=str(FOUNDRY_MODEL_CACHE_DIR),
                    logs_dir=str(FOUNDRY_LOG_DIR),
                )
            )
        self._manager = FoundryLocalManager.instance
        return self._manager

    def _ensure_ready(self):
        manager = self._initialize_manager()
        if self._model is None:
            self._model = manager.catalog.get_model(self.model_name)

        if not self._model.is_cached:
            self._model.download()
        if not self._model.is_loaded:
            self._model.load()
            self._client = None
        if self._client is None:
            self._client = self._model.get_chat_client()
            self._client.settings.temperature = 0.0
            self._client.settings.max_tokens = 180
            self._client.settings.top_p = 0.9
        return self._client

    def complete(self, messages: list[dict[str, str]]) -> str:
        with self._lock:
            try:
                client = self._ensure_ready()
                response = client.complete_chat(messages)
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise RuntimeError("Model boş yanıt döndürdü.")
                return content.strip()
            except UserFacingError:
                raise
            except Exception as exc:
                message = str(exc).lower()
                if "cancel" in message or "iptal" in message:
                    friendly = "Foundry Local işlemi iptal edildi. Lütfen yeniden deneyin."
                elif "load" in message or "yük" in message:
                    friendly = (
                        "Foundry Local modeli yüklenemedi. Disk alanını ve model ayarını "
                        "kontrol edip yeniden deneyin."
                    )
                else:
                    friendly = (
                        "Foundry Local yanıt modeli başlatılamadı. Bağlantıyı yalnızca "
                        "ilk model indirmesi için kontrol edip yeniden deneyin."
                    )
                raise UserFacingError(friendly) from exc


def _clean_answer(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    question: str,
) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    normalized = cleaned.casefold().rstrip(".! ")
    answer_words = re.findall(r"[0-9a-zçğıöşü@._+-]+", normalized)
    hedging_markers = {"belki", "muhtemelen", "olmayabilir", "vermeyebilir"}
    invalid_answer_markers = {
        "edilebilir değil",
        "documents.txt bulunmuyor",
        "dosya bulunmuyor",
        "bilgi bulunamadı",
        "kontrol edemeyiz",
        "üzgünüz",
    }
    normalized_question = question.casefold().strip().rstrip("?!.")
    if chunks and (
        len(answer_words) <= 1
        or any(marker in normalized for marker in hedging_markers)
        or any(marker in normalized for marker in invalid_answer_markers)
        or "?" in cleaned
        or normalized_question in normalized
    ):
        return chunks[0].content
    return cleaned


def _build_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[Sonuç {rank} - documents.txt parçası {chunk.position + 1}]\n{chunk.content}"
        for rank, chunk in enumerate(chunks, start=1)
    )
    user_prompt = (
        f"documents.txt parçaları:\n{context}\n\n"
        f"Soru: {question}\n\n"
        "Yalnızca yukarıdaki metinden kısa bir cevap yaz. Sorulan sayı, saat, tarih "
        "veya adres varsa onu değiştirmeden aktar. Birden fazla bilgi isteniyorsa "
        "her birini cevapla."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def answer_query(
    question: str,
    *,
    embedder: Embedder | None = None,
    chat_service: ChatService | None = None,
    source_path: Path = DOCUMENT_PATH,
    db_path: Path = DATABASE_PATH,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> AnswerResult:
    normalized_question = question.strip()
    if not normalized_question:
        raise UserFacingError("Lütfen bir soru yazın.")

    active_embedder = embedder or SentenceTransformerEmbedder()
    try:
        ensure_index(
            source_path=source_path,
            db_path=db_path,
            embedder=active_embedder,
        )
        query_embedding = active_embedder.encode([normalized_question])[0]
        chunks = _select_relevant_chunks(
            normalized_question,
            query_embedding,
            top_k=top_k,
            threshold=threshold,
            db_path=db_path,
        )
    except UserFacingError:
        raise
    except Exception as exc:
        raise UserFacingError(
            "Bilgi metni aranamadı. documents.txt dosyasını kaydedip yeniden deneyin."
        ) from exc

    if not chunks:
        return AnswerResult(FALLBACK_MESSAGE, ())

    service = chat_service or FoundryChatService()
    try:
        answer = _clean_answer(
            service.complete(_build_messages(normalized_question, chunks)),
            chunks,
            normalized_question,
        )
    except UserFacingError:
        raise
    except Exception as exc:
        raise UserFacingError(
            "Foundry Local yanıt üretirken bir sorun oluştu. Lütfen yeniden deneyin."
        ) from exc

    if not answer:
        raise UserFacingError("Foundry Local boş bir yanıt verdi. Lütfen yeniden deneyin.")
    return AnswerResult(answer, tuple(chunks))
