import hashlib
import os
import re
from typing import Any, Optional

import anthropic
import chromadb
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from prompts import PRISM_SYSTEM_PROMPT

__all__ = ["RAGEngine"]


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def _safe_collection_id_prefix(doc_name: str) -> str:
    digest = hashlib.sha256(doc_name.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", doc_name)[:80]
    return f"{safe}_{digest}"


class RAGEngine:
    def __init__(self) -> None:
        load_dotenv()
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        self._client = chromadb.EphemeralClient()
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._collection = self._client.get_or_create_collection(name="documents")
        self._anthropic: Optional[anthropic.Anthropic] = None
        self._last_retrieval: dict[str, Any] = {}

    def ingest_pdf(self, file_path: str, doc_name: str) -> int:
        reader = PdfReader(file_path)
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        full_text = "\n".join(parts)
        chunks = _chunk_text(full_text, chunk_size=500, overlap=50)
        if not chunks:
            return 0

        total = len(chunks)
        embeddings = self._embedder.encode(chunks, convert_to_numpy=True)
        id_prefix = _safe_collection_id_prefix(doc_name)
        ids = [f"{id_prefix}_{i}" for i in range(total)]
        metadatas: list[dict[str, Any]] = [
            {"doc_name": doc_name, "chunk_index": i, "total_chunks": total}
            for i in range(total)
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
        )
        return total

    def query(
        self, question: str, n_results: int = 5, distance_threshold: float = 1.5
    ) -> dict:
        q_emb = self._embedder.encode([question], convert_to_numpy=True)
        raw = self._collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=n_results,
        )
        docs = raw.get("documents") or []
        dists = raw.get("distances") or []
        metas = raw.get("metadatas") or []

        if not docs:
            return {"chunks": [], "distances": [], "metadatas": []}

        chunks = docs[0] or []
        dist_list = dists[0] if dists else []
        meta_list = metas[0] if metas else []

        return {
            "chunks": chunks,
            "distances": dist_list,
            "metadatas": meta_list,
        }

    def answer(self, question: str, conversation_history: list, n_results: int = 5) -> str:
        retrieved = self.query(question, n_results=n_results)
        self._last_retrieval = retrieved
        chunks = retrieved["chunks"]
        if not chunks:
            return (
                "I couldn't find relevant information in the document to answer that question. "
                "Try rephrasing or ask something else about the document."
            )
        context = "\n\n---\n\n".join(chunks) if chunks else "(No relevant chunks retrieved.)"

        system_with_context = (
            f"{PRISM_SYSTEM_PROMPT.strip()}\n\n"
            f"<retrieved_context>\n{context}\n</retrieved_context>"
        )

        messages: list[dict[str, str]] = []
        for turn in conversation_history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})

        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic(api_key=self._api_key)

        response = self._anthropic.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=system_with_context,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def clear_collection(self) -> None:
        try:
            self._client.delete_collection("documents")
        except NotFoundError:
            pass
        self._collection = self._client.get_or_create_collection(name="documents")

    @property
    def last_retrieval(self) -> dict[str, Any]:
        return self._last_retrieval
