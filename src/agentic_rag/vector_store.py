"""
Vector Store Management Module
==============================
Manages local embedding generation using Sentence-Transformers and persistent storage in ChromaDB.
"""

from pathlib import Path
from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.config import Settings, get_settings


class VectorStoreManager:
    """Manages document embedding and persistence in ChromaDB."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        collection_name: str = "financial_reports",
    ):
        """
        Initializes the VectorStoreManager.

        Args:
            settings: Optional Settings object. Defaults to get_settings().
            collection_name: Name of the ChromaDB collection.
        """
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.persist_directory = str(Path(self.settings.vector_store_dir).resolve())
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[Chroma] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily initializes the local HuggingFace embedding model with memory optimizations."""
        if self._embeddings is None:
            try:
                import torch
                torch.set_num_threads(1)
                try:
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
            except Exception:
                pass
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def vector_store(self) -> Chroma:
        """Lazily initializes or connects to the persisted Chroma collection."""
        if self._vector_store is None:
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
    ) -> Chroma:
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

        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        return self._vector_store

    def get_retriever(self, k: Optional[int] = None) -> Any:
        """
        Returns a LangChain retriever configured for similarity search.

        Args:
            k: Top-k documents to retrieve. Defaults to settings.retriever_k.

        Returns:
            Configured VectorStoreRetriever instance.
        """
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
        """
        Performs a semantic similarity search against the vector database.

        Args:
            query: Natural language search string.
            k: Number of results to return.

        Returns:
            List of matching Document objects.
        """
        top_k = k if k is not None else self.settings.retriever_k
        return self.vector_store.similarity_search(query, k=top_k)

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
        # Fast path: query sqlite directly to avoid loading heavyweight embedding model into memory
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
