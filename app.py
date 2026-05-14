import os
import tempfile
from pathlib import Path

import streamlit as st
from rag_engine import build_vector_store, query

st.set_page_config(page_title="Mini RAG Assistant", page_icon="🔍", layout="wide")
st.title("🔍 Mini RAG Assistant")
st.caption("Upload documents. Ask questions. Get grounded answers with source citations.")
st.divider()

for key, default in {
    "vector_store": None,
    "chat_history": [],
    "doc_names": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def render_confidence(confidence: float):
    if confidence >= 70:
        st.success(f"Confidence: {confidence}%")
    elif confidence >= 40:
        st.warning(f"Confidence: {confidence}%")
    else:
        st.error(f"Confidence: {confidence}% — low match, answer may be unreliable")


def render_sources(sources: list):
    if not sources:
        st.warning("No matching sources found.")
        return
    with st.expander(f"📚 {len(sources)} source(s) used"):
        for i, doc in enumerate(sources, 1):
            source = Path(doc.metadata.get("source", "unknown")).name
            page = doc.metadata.get("page", "N/A")
            st.markdown(f"**[{i}]** `{source}` — Page {page}")
            st.markdown(f"> {doc.page_content[:400]}...")
            if i < len(sources):
                st.divider()


with st.sidebar:
    st.header("📂 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process Documents", type="primary", use_container_width=True):
        with st.spinner("Indexing documents..."):
            tmp_paths = []
            try:
                for f in uploaded_files:
                    suffix = Path(f.name).suffix
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(f.read())
                    tmp.close()
                    tmp_paths.append(tmp.name)

                st.session_state.vector_store = build_vector_store(tmp_paths)
                st.session_state.doc_names = [f.name for f in uploaded_files]
                st.session_state.chat_history = []
                st.success(f"✅ {len(uploaded_files)} file(s) indexed.")
            except Exception as e:
                st.error(f"Error processing documents: {e}")
            finally:
                for p in tmp_paths:
                    os.unlink(p)

    if st.session_state.doc_names:
        st.divider()
        st.markdown("**Loaded:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- 📄 {name}")

    if st.session_state.vector_store:
        st.divider()
        if st.button("Clear All", use_container_width=True):
            st.session_state.update(
                {"vector_store": None, "chat_history": [], "doc_names": []}
            )
            st.rerun()


for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        render_confidence(entry["confidence"])
        render_sources(entry["sources"])


if st.session_state.vector_store is None:
    st.info("⬅️ Upload and process documents to start.")
else:
    if question := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = query(question, st.session_state.vector_store, st.session_state.chat_history)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.stop()

            st.write(result["answer"])
            render_confidence(result["confidence"])
            render_sources(result["sources"])

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence": result["confidence"],
            }
        )