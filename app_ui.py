"""Üç bölümlü sade Streamlit arayüzü."""

from __future__ import annotations

import streamlit as st

from app import FoundryChatService, UserFacingError, answer_query
from ingest import SentenceTransformerEmbedder, ensure_index, read_source, save_source
from settings import CHAT_MODEL_NAME, DOCUMENT_PATH, EMBEDDING_MODEL_NAME


st.set_page_config(
    page_title="Yerel RAG Asistanı",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@st.cache_resource(show_spinner=False)
def get_chat_service() -> FoundryChatService:
    return FoundryChatService()


def clear_chat() -> None:
    st.session_state.messages = []


def show_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return
    with st.expander("Kullanılan metin parçaları ve benzerlik puanları"):
        for index, chunk in enumerate(chunks, start=1):
            st.markdown(f"**{index}. parça · benzerlik {chunk['score']:.3f}**")
            st.write(chunk["content"])


def question_answer_page() -> None:
    st.title("Yerel RAG Asistanı")
    st.caption("Yanıtlar yalnızca documents.txt içeriğinden üretilir.")

    if st.button("Yeni sohbet", type="secondary"):
        clear_chat()
        st.rerun()

    if "messages" not in st.session_state:
        clear_chat()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption("Kaynak: documents.txt")
                show_chunks(message.get("chunks", []))

    if question := st.chat_input("Sorunuzu yazın"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("documents.txt aranıyor ve yerel model hazırlanıyor..."):
                result = answer_query(
                    question,
                    embedder=get_embedder(),
                    chat_service=get_chat_service(),
                )
        except UserFacingError as exc:
            st.error(str(exc))
            return

        chunk_data = [
            {"content": chunk.content, "score": chunk.score}
            for chunk in result.chunks
        ]
        with st.chat_message("assistant"):
            st.markdown(result.answer)
            st.caption("Kaynak: documents.txt")
            show_chunks(chunk_data)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "chunks": chunk_data,
            }
        )


def document_page() -> None:
    st.title("Bilgi Metni")
    st.write("Asistanın tek bilgi kaynağını burada düzenleyebilirsiniz.")

    if "document_editor" not in st.session_state:
        st.session_state.document_editor = read_source(DOCUMENT_PATH)

    st.text_area(
        "documents.txt",
        key="document_editor",
        height=420,
        label_visibility="collapsed",
    )

    if st.button("Kaydet ve indeksle", type="primary"):
        source_saved = False
        try:
            save_source(st.session_state.document_editor, DOCUMENT_PATH)
            source_saved = True
            with st.spinner("Metin parçalanıyor ve indeks güncelleniyor..."):
                status = ensure_index(
                    embedder=get_embedder(),
                    force=True,
                )
            clear_chat()
            st.success(
                f"documents.txt kaydedildi; indeks {status.chunk_count} parça ile yenilendi. "
                "Eski sohbet temizlendi."
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            if source_saved:
                st.error(
                    "documents.txt kaydedildi ancak indeks güncellenemedi. "
                    "Embedding modelinin hazır olduğundan emin olup yeniden deneyin."
                )
            else:
                st.error("documents.txt kaydedilemedi. Dosya izinlerini kontrol edin.")


def about_page() -> None:
    st.title("Proje Hakkında")
    st.write(
        "Bu uygulama, documents.txt içindeki ilgili parçaları cosine similarity ile "
        "bulur ve yalnızca bu parçaları Microsoft Foundry Local modeline gönderir."
    )
    st.markdown(
        f"""
**RAG akışı**

documents.txt → anlamlı parçalama → embedding → SQLite → Top-K arama → eşik → yerel yanıt

**Sohbet modeli:** `{CHAT_MODEL_NAME}`

**Embedding modeli:** `{EMBEDDING_MODEL_NAME}`

Metin ve model işlemleri yerel cihazda yapılır. İlk model indirmesi için internet gerekir;
önbelleğe alınan modeller daha sonra yeniden indirilmez.
"""
    )


with st.sidebar:
    st.header("Yerel RAG")
    page = st.radio(
        "Bölüm",
        ["Soru-Cevap", "Bilgi Metni", "Proje Hakkında"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Microsoft Foundry Local · SQLite · Streamlit")

if page == "Soru-Cevap":
    question_answer_page()
elif page == "Bilgi Metni":
    document_page()
else:
    about_page()
