"""
Configuration for the Corrective RAG System.
Centralizes all tunable parameters.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "Course files"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "vector_store"

# ──────────────────────────────────────────────
# Document Processing
# ──────────────────────────────────────────────
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between consecutive chunks
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md"]

# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # sentence-transformers model

# ──────────────────────────────────────────────
# Vector Store / Retrieval
# ──────────────────────────────────────────────
COLLECTION_NAME = "corrective_rag_docs"
TOP_K = 5                 # number of chunks to retrieve

# ──────────────────────────────────────────────
# Relevance Evaluation Thresholds
# ──────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.35        # minimum cosine similarity to consider relevant
MIN_RELEVANT_CHUNKS = 2           # minimum number of relevant chunks required
WEAK_CONTEXT_RATIO = 0.4          # if ratio of relevant chunks < this, context is weak

# ──────────────────────────────────────────────
# Query Rewriting
# ──────────────────────────────────────────────
MAX_REWRITE_ATTEMPTS = 2          # maximum query rewrite + re-retrieval cycles

# ──────────────────────────────────────────────
# LLM (Generation)
# ──────────────────────────────────────────────
# Uses a local pipeline via HuggingFace transformers (no API key needed)
# For better quality, set USE_OPENAI=True and provide OPENAI_API_KEY
USE_OPENAI = False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# Local model fallback (runs on CPU — small but functional)
LOCAL_LLM_MODEL = "google/flan-t5-base"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2

