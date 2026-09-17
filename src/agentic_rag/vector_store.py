"""
Vector Store Management Module
==============================
Manages local embedding generation and persistent storage in ChromaDB.
Prioritizes ultra-low-memory ONNX embeddings to fit comfortably within
Render Free Tier (512MB RAM), with automatic fallback to SQLite FTS5 search.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document

from agentic_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages document embedding and persistence in ChromaDB."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        collection_name: str = "financial_reports",
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.persist_directory = str(Path(self.settings.vector_store_dir).resolve())
        self._embeddings: Any = None
        self._vector_store: Any = None

    @property
    def embeddings(self) -> Any:
        """
        Lazily initializes the embedding model.
        Prioritizes OnnxMiniLMEmbeddings (uses ~60MB RAM instead of ~450MB PyTorch).
        Falls back to HuggingFaceEmbeddings only if ONNX dependencies are unavailable.
        """
        if self._embeddings is None:
            try:
                from agentic_rag.embeddings import OnnxMiniLMEmbeddings
                self._embeddings = OnnxMiniLMEmbeddings(
                    model_repo=self.settings.embedding_model,
                )
                # Verify initialization
                self._embeddings._ensure_loaded()
                logger.info("Using OnnxMiniLMEmbeddings for low-memory vector operations.")
            except Exception as e:
                logger.warning(f"Could not load ONNX embeddings ({e}), falling back to HuggingFaceEmbeddings: {e}")
                try:
                    import torch
                    torch.set_num_threads(1)
                except Exception:
                    pass
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
        return self._embeddings

    @property
    def vector_store(self) -> Any:
        """Lazily initializes or connects to the persisted Chroma collection."""
        if self._vector_store is None:
            from langchain_chroma import Chroma
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vector_store

    def index_documents(
        self,
        chunks: List[Document],
        reset_existing: bool = False,
    ) -> Any:
        """
        Embeds and stores document chunks in ChromaDB.

        Args:
            chunks: List of Document chunks to embed.
            reset_existing: If True, deletes existing collection before indexing.

        Returns:
            The populated Chroma vector store instance.
        """
        if not chunks:
            raise ValueError("Cannot index empty chunk list.")

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        if reset_existing:
            self.reset_collection()

        from langchain_chroma import Chroma
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        return self._vector_store

    def get_retriever(self, k: Optional[int] = None) -> Any:
        """Returns a LangChain retriever configured for similarity search."""
        top_k = k if k is not None else self.settings.retriever_k
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Performs a semantic similarity search against the vector database."""
        top_k = k if k is not None else self.settings.retriever_k
        scored = self.similarity_search_with_score(query, k=top_k)
        return [doc for doc, _ in scored]

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Performs vector similarity search with score against ChromaDB.
        Automatically falls back to SQLite Full-Text Search (FTS5) if vector search
        fails or encounters memory constraints.
        """
        top_k = k if k is not None else self.settings.retriever_k
        try:
            return self.vector_store.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            logger.warning(f"Chroma vector search failed ({e}), using SQLite FTS fallback.")
            return self._fts_search(query, k=top_k)

    def _fts_search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Ultra-fast, zero-RAM Full-Text Search directly against Chroma's SQLite database.
        Acts as a reliable safeguard on resource-constrained servers.
        """
        db_path = Path(self.persist_directory) / "chroma.sqlite3"
        if not db_path.is_file():
            return []

        import sqlite3

        # Extract words for FTS5 MATCH
        words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query)
        if not words:
            return []

        fts_query = " OR ".join([f'"{w}"' for w in words[:8]])
        results: List[Tuple[Document, float]] = []

        try:
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT rowid, rank FROM embedding_fulltext_search "
                    "WHERE embedding_fulltext_search MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_query, k),
                )
                matches = cur.fetchall()

                for rowid, rank in matches:
                    cur.execute(
                        "SELECT key, string_value, int_value FROM embedding_metadata WHERE id = ?",
                        (rowid,),
                    )
                    meta = {}
                    doc_content = ""
                    for key, s_val, i_val in cur.fetchall():
                        val = s_val if s_val is not None else i_val
                        if key == "chroma:document":
                            doc_content = str(val or "")
                        else:
                            meta[key] = val

                    # Normalize FTS rank to a distance-like score
                    distance = max(0.1, min(1.2, abs(float(rank)) * 0.08))
                    results.append((Document(page_content=doc_content, metadata=meta), distance))
        except Exception as err:
            logger.error(f"SQLite FTS search error: {err}")

        return results

    def reset_collection(self) -> None:
        """Clears all embeddings from the current collection."""
        if self._vector_store is not None:
            try:
                self._vector_store.delete_collection()
            except Exception:
                pass
            self._vector_store = None

    def get_document_count(self) -> int:
        """Returns the number of indexed records in the vector database."""
        db_path = Path(self.persist_directory) / "chroma.sqlite3"
        if db_path.is_file():
            try:
                import sqlite3
                with sqlite3.connect(str(db_path)) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT count(*) FROM embeddings;")
                    row = cur.fetchone()
                    if row:
                        return int(row[0])
            except Exception:
                pass

        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0
