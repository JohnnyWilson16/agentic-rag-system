"""
Agentic RAG Pipeline Coordinator
================================
Main entry point coordinating document ingestion, semantic chunking, vector indexing,
dynamic tool creation, and multi-agent reasoning.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from crewai import Crew, LLM

from agentic_rag.agents import create_financial_analyst_agent
from agentic_rag.config import Settings, get_settings
from agentic_rag.document_loader import DocumentLoader
from agentic_rag.llm_factory import LLMFactory
from agentic_rag.tasks import (
    create_comprehensive_analysis_task,
    create_custom_query_task,
)
from agentic_rag.text_splitter import DocumentSplitter
from agentic_rag.tools import create_search_tool
from agentic_rag.vector_store import VectorStoreManager


class AgenticRAGPipeline:
    """End-to-End pipeline orchestrating ingestion, retrieval, and agentic analysis."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[LLM] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
    ):
        """
        Initializes the Agentic RAG Pipeline.

        Args:
            settings: Application configuration. Defaults to singleton get_settings().
            llm: Optional pre-configured CrewAI LLM instance.
            vector_store_manager: Optional pre-configured VectorStoreManager instance.
        """
        self.settings = settings or get_settings()
        self.vector_store_manager = vector_store_manager or VectorStoreManager(
            settings=self.settings
        )
        self._llm = llm

    @property
    def llm(self) -> LLM:
        """Lazily creates or returns the configured LLM instance."""
        if self._llm is None:
            self._llm = LLMFactory.create_llm(self.settings)
        return self._llm

    def ingest_document(
        self,
        file_path: str | Path,
        reset_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Loads, chunks, and indexes a document into the persistent vector store.

        Args:
            file_path: Path to the target PDF, TXT, or Markdown document.
            reset_existing: Whether to wipe the existing vector index before indexing.

        Returns:
            Dictionary containing ingestion statistics (document count, chunk count, etc.).
        """
        # Step 1: Ingest document
        raw_docs = DocumentLoader.load_document(file_path)

        # Step 2: Split into overlapping semantic chunks
        splitter = DocumentSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)

        # Step 3: Embed and store in ChromaDB
        self.vector_store_manager.index_documents(
            chunks=chunks,
            reset_existing=reset_existing,
        )

        stats = {
            "file_name": Path(file_path).name,
            "raw_pages": len(raw_docs),
            "chunks_created": len(chunks),
            "total_vectors_in_db": self.vector_store_manager.get_document_count(),
            "embedding_model": self.settings.embedding_model,
            "storage_path": self.vector_store_manager.persist_directory,
        }
        return stats

    def query(self, question: str) -> str:
        """
        Executes an agentic Q&A workflow against the indexed document database.

        Args:
            question: Specific question or investigation query.

        Returns:
            Agent's structured, evidence-backed answer string.
        """
        if self.vector_store_manager.get_document_count() == 0:
            raise RuntimeError(
                "Vector database is empty. Please ingest a document using ingest_document() before querying."
            )

        # Build retrieval tool bound to the vector store retriever
        retriever = self.vector_store_manager.get_retriever(k=self.settings.retriever_k)
        search_tool = create_search_tool(retriever)

        # Initialize Analyst Agent
        analyst = create_financial_analyst_agent(
            llm=self.llm,
            tools=[search_tool],
            verbose=self.settings.verbose,
        )

        # Define task
        task = create_custom_query_task(agent=analyst, question=question)

        # Orchestrate execution via Crew
        crew = Crew(
            agents=[analyst],
            tasks=[task],
            verbose=self.settings.verbose,
        )

        result = crew.kickoff()
        return str(result)

    def analyze(self, company_name: Optional[str] = None) -> str:
        """
        Runs an exhaustive end-to-end financial analysis on the indexed report.

        Args:
            company_name: Optional company name for specialized report titling.

        Returns:
            Complete multi-section analytical report.
        """
        if self.vector_store_manager.get_document_count() == 0:
            raise RuntimeError(
                "Vector database is empty. Please ingest a document before running full analysis."
            )

        retriever = self.vector_store_manager.get_retriever(k=self.settings.retriever_k)
        search_tool = create_search_tool(retriever)

        analyst = create_financial_analyst_agent(
            llm=self.llm,
            tools=[search_tool],
            verbose=self.settings.verbose,
        )

        task = create_comprehensive_analysis_task(
            agent=analyst,
            company_name=company_name,
        )

        crew = Crew(
            agents=[analyst],
            tasks=[task],
            verbose=self.settings.verbose,
        )

        result = crew.kickoff()
        return str(result)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current operational status of the pipeline."""
        return {
            "llm_provider": self.settings.llm_provider,
            "llm_model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "indexed_vectors": self.vector_store_manager.get_document_count(),
            "vector_store_dir": self.vector_store_manager.persist_directory,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "retriever_k": self.settings.retriever_k,
        }
