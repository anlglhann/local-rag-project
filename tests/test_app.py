from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app import UserFacingError, answer_query
from settings import FALLBACK_MESSAGE


class MappingEmbedder:
    model_name = "mapping-model"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "çalışma" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "sahibi" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class BorderlinePriceEmbedder:
    """İlgisiz fiyat sorusunu iade metnine 0.36 benzerlikle yaklaştırır."""

    model_name = "borderline-price-model"

    def encode(self, texts):
        vectors = []
        for text in texts:
            if "fiyat" in text.lower():
                vectors.append([0.36, 0.9329523])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class LowScoreReturnEmbedder:
    """İade sorusunu ilgili parçaya ana eşiğin altında, 0.15 skorla bağlar."""

    model_name = "low-score-return-model"

    def encode(self, texts):
        vectors = []
        for text in texts:
            if "şart" in text.lower() or "koşul" in text.lower():
                vectors.append([0.15, 0.988686])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class SemanticPriorityEmbedder:
    """Zayıf lexical adayın güçlü semantic adayın önüne geçmediğini sınar."""

    model_name = "semantic-priority-model"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "web uygulamaları" in lowered or "hangi hizmet" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.2, 0.9797959])
        return np.asarray(vectors, dtype=np.float32)


class RecordingChatService:
    def __init__(self, answer="Hafta içi 09.00 ile 18.00 arasındadır.") -> None:
        self.answer = answer
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.answer


class BrokenChatService:
    def complete(self, messages):
        raise RuntimeError("native model stopped")


def make_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "documents.txt"
    source.write_text(
        "Çalışma saatleri hafta içi 09.00 ile 18.00 arasındadır.\n\n"
        "Destek ekibine e-posta ile ulaşılır.",
        encoding="utf-8",
    )
    return source, tmp_path / "rag.sqlite3"


def test_missing_information_returns_exact_fallback_without_model(tmp_path: Path) -> None:
    source, database = make_source(tmp_path)
    chat = RecordingChatService()

    result = answer_query(
        "Şirketin sahibi kimdir?",
        source_path=source,
        db_path=database,
        embedder=MappingEmbedder(),
        chat_service=chat,
        threshold=0.5,
    )

    assert result.answer == FALLBACK_MESSAGE
    assert result.chunks == ()
    assert chat.calls == 0


def test_foundry_error_becomes_understandable_message(tmp_path: Path) -> None:
    source, database = make_source(tmp_path)

    with pytest.raises(UserFacingError, match="Foundry Local"):
        answer_query(
            "Çalışma saatleri nedir?",
            source_path=source,
            db_path=database,
            embedder=MappingEmbedder(),
            chat_service=BrokenChatService(),
            threshold=0.5,
        )


def test_answer_uses_only_documents_txt_context(tmp_path: Path) -> None:
    source, database = make_source(tmp_path)
    chat = RecordingChatService()

    result = answer_query(
        "Çalışma saatleri nedir?",
        source_path=source,
        db_path=database,
        embedder=MappingEmbedder(),
        chat_service=chat,
        threshold=0.5,
    )

    assert result.answer == "Hafta içi 09.00 ile 18.00 arasındadır."
    assert all(chunk.source_name == "documents.txt" for chunk in result.chunks)
    assert chat.calls == 1


def test_one_word_english_no_is_replaced_with_top_chunk(tmp_path: Path) -> None:
    source, database = make_source(tmp_path)
    chat = RecordingChatService(answer="No")

    result = answer_query(
        "Çalışma saatleri var mı?",
        source_path=source,
        db_path=database,
        embedder=MappingEmbedder(),
        chat_service=chat,
        threshold=0.5,
    )

    assert result.answer.startswith("Çalışma saatleri hafta içi")


def test_one_word_turkish_negative_is_replaced_with_top_chunk(tmp_path: Path) -> None:
    source, database = make_source(tmp_path)
    chat = RecordingChatService(answer="Verilmez.")

    result = answer_query(
        "Çalışma saatleri hafta sonu nasıl?",
        source_path=source,
        db_path=database,
        embedder=MappingEmbedder(),
        chat_service=chat,
        threshold=0.5,
    )

    assert "09.00 ile 18.00" in result.answer


def test_hedged_answer_is_replaced_with_definitive_source(tmp_path: Path) -> None:
    source, database = make_source(tmp_path)
    chat = RecordingChatService(answer="Şirket hafta sonu hizmet vermeyebilir.")

    result = answer_query(
        "Çalışma saatleri nedir?",
        source_path=source,
        db_path=database,
        embedder=MappingEmbedder(),
        chat_service=chat,
        threshold=0.5,
    )

    assert result.answer == "Çalışma saatleri hafta içi 09.00 ile 18.00 arasındadır."


def test_question_echo_is_replaced_with_source_chunk(tmp_path: Path) -> None:
    source, database = make_source(tmp_path)
    chat = RecordingChatService(answer="Hayır, çalışma saatleri nedir?")

    result = answer_query(
        "Çalışma saatleri nedir?",
        source_path=source,
        db_path=database,
        embedder=MappingEmbedder(),
        chat_service=chat,
        threshold=0.5,
    )

    assert result.answer == "Çalışma saatleri hafta içi 09.00 ile 18.00 arasındadır."


def test_malformed_negative_is_replaced_with_return_source(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text(
        "Ürünün kullanılmamış ve hasar görmemiş olması gerekir.",
        encoding="utf-8",
    )
    chat = RecordingChatService(answer="Kullanılmış bir ürün iade edilebilir değil.")

    result = answer_query(
        "Kullanılmış bir ürün iade edilebilir mi?",
        source_path=source,
        db_path=tmp_path / "rag.sqlite3",
        embedder=BorderlinePriceEmbedder(),
        chat_service=chat,
    )

    assert result.answer == "Ürünün kullanılmamış ve hasar görmemiş olması gerekir."


def test_false_missing_source_claim_is_replaced_with_source_chunk(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text(
        "Müşteri bilgileri izin alınmadan üçüncü kişilerle paylaşılmaz.",
        encoding="utf-8",
    )
    chat = RecordingChatService(
        answer="Üzgünüz, documents.txt bulunmuyor; bilgiyi kontrol edemeyiz."
    )

    result = answer_query(
        "Müşteri bilgileri hangi durumda paylaşılmaz?",
        source_path=source,
        db_path=tmp_path / "rag.sqlite3",
        embedder=BorderlinePriceEmbedder(),
        chat_service=chat,
    )

    assert result.answer == "Müşteri bilgileri izin alınmadan üçüncü kişilerle paylaşılmaz."


def test_borderline_price_question_returns_fallback_without_model(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text(
        "Satın alınan ürünler 14 gün içinde iade edilebilir.",
        encoding="utf-8",
    )
    chat = RecordingChatService()

    result = answer_query(
        "Ürünlerin fiyatı ne kadar?",
        source_path=source,
        db_path=tmp_path / "rag.sqlite3",
        embedder=BorderlinePriceEmbedder(),
        chat_service=chat,
    )

    assert result.answer == FALLBACK_MESSAGE
    assert result.chunks == ()
    assert chat.calls == 0


def test_explicit_return_term_rescues_low_similarity_chunk(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text(
        "Satın alınan ürünler 14 gün içinde iade edilebilir. "
        "Ürünün kullanılmamış ve hasar görmemiş olması gerekir.",
        encoding="utf-8",
    )
    chat = RecordingChatService(
        answer=(
            "Ürünün kullanılmamış ve hasar görmemiş olması gerekir; "
            "iade süresi 14 gündür."
        )
    )

    result = answer_query(
        "İade için gereken şartlar nelerdir?",
        source_path=source,
        db_path=tmp_path / "rag.sqlite3",
        embedder=LowScoreReturnEmbedder(),
        chat_service=chat,
    )

    assert "kullanılmamış" in result.answer
    assert result.chunks[0].score == pytest.approx(0.15, abs=0.001)
    assert chat.calls == 1


def test_lexical_fallback_does_not_reorder_semantic_results(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text(
        "Şirket hafta sonu hizmet vermemektedir.\n\n"
        "Şirket web uygulamaları geliştirmektedir.",
        encoding="utf-8",
    )
    chat = RecordingChatService(answer="Şirket web uygulamaları geliştirmektedir.")

    result = answer_query(
        "Şirket hangi hizmetleri sunuyor?",
        source_path=source,
        db_path=tmp_path / "rag.sqlite3",
        embedder=SemanticPriorityEmbedder(),
        chat_service=chat,
    )

    assert result.chunks[0].content == "Şirket web uygulamaları geliştirmektedir."
