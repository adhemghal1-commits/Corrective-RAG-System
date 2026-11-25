"""
Embedding & Vector Store Module
=================================
Generates embeddings with sentence-transformers and stores them in ChromaDB.
"""

from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    TOP_K,
)


class EmbeddingStore:
    """Manages embedding generation and ChromaDB vector storage."""

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    # ── Lazy initializers ────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded.")
        return self._model

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_PERSIST_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Core operations ──────────────────────────

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add document chunks to the vector store.
        Each chunk dict must have 'text' and 'metadata' keys.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self.generate_embeddings(texts)

        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]

        # Add to ChromaDB in batches (ChromaDB handles large batches fine)
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query.
        Returns list of dicts with 'text', 'metadata', 'distance', 'similarity'.
        """
        query_embedding = self.generate_embeddings([query_text])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance = 1 - cosine_similarity
                similarity = 1.0 - dist
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": round(similarity, 4),
                })

        return retrieved

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic stats about the vector store."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
        }

    def clear_collection(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self._collection = None
        print(f"Collection '{COLLECTION_NAME}' cleared.")

    def has_documents(self) -> bool:
        """Check if the vector store has any documents."""
        return self.collection.count() > 0

