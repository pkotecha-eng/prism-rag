---
title: PRISM Document Intelligence
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# PRISM — Protocol Research Intelligence & Search Module
Upload any PDF and ask questions about it. PRISM uses RAG (Retrieval-Augmented Generation) 
with ChromaDB + Claude to retrieve relevant document sections and generate grounded answers.

## Sample Document
A fictional VELARA-1 clinical trial protocol is included in sample_docs/ for testing.

## How it works
1. Upload a PDF → document is chunked and embedded into ChromaDB
2. Ask a question → query is embedded and matched against chunks
3. Relevant chunks retrieved → Claude generates a grounded answer

## Verify syntax (indentation, etc.)
From the project root:

```bash
make check
```

This runs `python3 -m py_compile` on `app.py`, `rag_engine.py`, and `prompts.py` without starting Streamlit.