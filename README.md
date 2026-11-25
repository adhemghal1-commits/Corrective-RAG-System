# Corrective RAG System

An end-to-end **Corrective Retrieval-Augmented Generation (CRAG)** system for reliable, document-based question answering. The system evaluates retrieved context quality before generating answers, rewrites queries and re-retrieves when context is weak or irrelevant, cites sources, and handles unanswerable questions without hallucinating.

## Architecture

```text
User Query
   │
   ▼
[Retrieve Top-K Chunks] ◄───────────────────────┐
   │                                            │
   ▼                                            │
[Evaluate Retrieval Quality]                    │
   │                                            │
   ├── STRONG ──► [Generate Verified Answer]    │
   │                   (with sources)           │
   │                                            │
   └── WEAK / IRRELEVANT                        │
          ├── (attempts < max) ──► [Rewrite Query]
          └── (attempts == max) ─► [Best Context or "Cannot Answer"]
```

## Project Structure

- `config.py` — Centralized configuration (chunking, embedding model, thresholds, LLM settings)
- `document_processor.py` — Document ingestion pipeline: upload, extract (`.pdf`, `.txt`, `.docx`, `.md`), clean, and chunk text
- `embedding_store.py` — Sentence-transformer embeddings (`all-MiniLM-L6-v2`) and persistent ChromaDB vector store
- `retrieval_evaluator.py` — Evaluates retrieved chunks (`STRONG`, `WEAK`, `IRRELEVANT`) using similarity thresholds and relevant-chunk ratios
- `query_rewriter.py` — Multi-strategy query rewriting (keyword expansion, simplification, LLM fallback)
- `answer_generator.py` — Grounded answer generation using local HuggingFace (`google/flan-t5-base`) or OpenAI (`gpt-3.5-turbo`), with source attribution and unanswerable detection
- `corrective_rag_pipeline.py` — End-to-end orchestrator tying retrieval, evaluation, rewriting, and generation together
- `app.py` — Interactive Streamlit web application
- `Course files/` — Sample lecture PDFs ready for indexing

## Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional (OpenAI backend):**
   To use OpenAI instead of the local `google/flan-t5-base` model, set `USE_OPENAI = True` in `config.py` and set your API key:
   ```bash
   set OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Features

- **Document Ingestion:** Index existing files in `Course files/` or upload new `.pdf`, `.txt`, `.docx`, or `.md` files from the UI.
- **Retrieval Quality Evaluation:** Scores every retrieved chunk and classifies context as `STRONG`, `WEAK`, or `IRRELEVANT`.
- **Self-Correcting Loop:** Automatically rewrites weak/irrelevant queries and re-retrieves up to `MAX_REWRITE_ATTEMPTS`.
- **Grounded Generation & Sources:** Generates answers strictly from verified chunks, displays source badges, and provides a full retrieval evaluation trace.
- **Unanswerable Question Handling:** Explicitly refuses to hallucinate when documents lack sufficient relevant information.

