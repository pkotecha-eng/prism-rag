import os
import tempfile
from typing import Any

import streamlit as st

from prompts import PRISM_SYSTEM_PROMPT
from rag_engine import RAGEngine


st.set_page_config(
    page_title="PRISM — Document Intelligence Assistant",
    page_icon="🔍",
    layout="wide",
)


def _init_session_state() -> None:
    if "rag" not in st.session_state:
        st.session_state["rag"] = RAGEngine()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "doc_name" not in st.session_state:
        st.session_state["doc_name"] = None

    # Keep a copy so it persists across reruns (useful for debugging/UI).
    if "prism_system_prompt" not in st.session_state:
        st.session_state["prism_system_prompt"] = PRISM_SYSTEM_PROMPT


_init_session_state()

rag: RAGEngine = st.session_state["rag"]
doc_name = st.session_state.get("doc_name")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🔍 PRISM")
st.sidebar.subheader("Protocol Research Intelligence & Search Module")
st.sidebar.divider()

uploaded_pdf = st.sidebar.file_uploader("Upload a document", type=["pdf"])

# How many retrieved chunks PRISM will search over for each answer.
if "retrieved_chunks" not in st.session_state:
    st.session_state["retrieved_chunks"] = 5
st.sidebar.slider(
    "How many document sections to search",
    min_value=3,
    max_value=10,
    value=int(st.session_state["retrieved_chunks"]),
    step=1,
    key="retrieved_chunks",
    help="Retrieved chunks",
)

if uploaded_pdf is not None:
    new_doc_name = uploaded_pdf.name

    # If the user uploads a different document, reset the knowledge base.
    if st.session_state.get("doc_name") != new_doc_name:
        try:
            rag.clear_collection()
            st.session_state["messages"] = []
        except Exception as e:
            st.sidebar.error(f"Failed to reset the document index: {e}")
            st.stop()

    with st.spinner("Ingesting document..."):
        tmp_path: str | None = None
        try:
            data = uploaded_pdf.getbuffer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            n_chunks = rag.ingest_pdf(tmp_path, doc_name=new_doc_name)
            st.session_state["doc_name"] = new_doc_name
            st.session_state["messages"] = []
            st.success(f"✅ Document ingested — {n_chunks} chunks indexed")
        except Exception as e:
            st.sidebar.error(f"Failed to ingest the PDF: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

st.sidebar.divider()

if st.sidebar.button("New Session", use_container_width=True):
    try:
        rag.clear_collection()
    except Exception as e:
        st.sidebar.error(f"Failed to clear indexed documents: {e}")
    st.session_state["doc_name"] = None
    st.session_state["messages"] = []
    st.rerun()

if st.session_state.get("doc_name"):
    st.sidebar.write(f"Current document: {st.session_state['doc_name']}")

st.sidebar.info("Answers are based solely on the uploaded document.")


# -----------------------------
# Main area
# -----------------------------
st.header("🔍 PRISM Document Intelligence")
st.subheader("Ask questions about your uploaded document")

if not st.session_state.get("doc_name"):
    st.info("👈 Upload a PDF document in the sidebar to get started")
    st.stop()


suggested_questions = [
    "What is the main purpose of this document?",
    "What are the key requirements or criteria?",
    "What are the risks or concerns mentioned?",
    "Summarize the key findings or recommendations",
]
st.markdown("#### Suggested questions")
for q in suggested_questions:
    st.write(q)


# Render chat history
for msg in st.session_state["messages"]:
    role = msg.get("role")
    content = msg.get("content", "")
    if role in ("user", "assistant"):
        with st.chat_message(role):
            st.markdown(content)


question = st.chat_input(
    "Ask a question about your document...",
    disabled=not bool(st.session_state.get("doc_name")),
)

if question:
    # Show the user message immediately (we append to history after the answer).
    with st.chat_message("user"):
        st.markdown(question)

    retrieved: dict[str, Any] = {}
    with st.spinner("Searching document and generating answer..."):
        try:
            response = rag.answer(
                question,
                st.session_state["messages"],
                n_results=int(st.session_state["retrieved_chunks"]),
            )
            retrieved = rag.last_retrieval
        except Exception as e:
            st.error(f"Sorry, something went wrong while answering: {e}")
            st.stop()

    with st.chat_message("assistant"):
        st.markdown(response)

    with st.expander("📄 Retrieved Context"):
        chunks = retrieved.get("chunks") or []
        metadatas = retrieved.get("metadatas") or []
        distances = retrieved.get("distances") or []

        for idx, chunk in enumerate(chunks):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            dist = distances[idx] if idx < len(distances) else None

            doc = meta.get("doc_name")
            chunk_index = meta.get("chunk_index")
            total_chunks = meta.get("total_chunks")

            header = f"Chunk {chunk_index}/{total_chunks} (doc: {doc})"
            if dist is not None:
                header += f" | distance: {dist}"

            st.markdown(f"**{header}**")
            st.code(chunk, language="text")

    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["messages"].append({"role": "assistant", "content": response})
