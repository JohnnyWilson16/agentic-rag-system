"""
Unit Tests for Vector Store Manager and ChromaDB Indexing
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from agentic_rag.config import Settings
from agentic_rag.vector_store import VectorStoreManager


@pytest.fixture
def temp_vector_manager(tmp_path: Path):
    """Provides a VectorStoreManager configured with a temporary storage directory."""
    db_dir = tmp_path / "test_chroma_db"
    settings = Settings(
        vector_store_dir=str(db_dir),
        retriever_k=2,
    )
    manager = VectorStoreManager(settings=settings, collection_name="test_reports")
    yield manager
    # Cleanup
    manager.reset_collection()


def test_index_and_similarity_search(temp_vector_manager: VectorStoreManager):
    """Test embedding and semantic similarity search in ChromaDB."""
    sample_chunks = [
        Document(
            page_content="Apex Global reported consolidated revenue of $28.45 billion with 8.2% YoY growth.",
            metadata={"file_name": "report.txt", "page": 1, "chunk_id": "c1"},
        ),
        Document(
            page_content="Operating margin expanded by 60 basis points to 24.8% due to internal automation.",
            metadata={"file_name": "report.txt", "page": 1, "chunk_id": "c2"},
        ),
        Document(
            page_content="Healthcare and Life Sciences vertical surged 14.2% YoY to $3.70 billion.",
            metadata={"file_name": "report.txt", "page": 2, "chunk_id": "c3"},
        ),
        Document(
            page_content="Key risks include macroeconomic slowdown and geopolitical currency volatility.",
            metadata={"file_name": "report.txt", "page": 4, "chunk_id": "c4"},
        ),
    ]

    # Index chunks
    temp_vector_manager.index_documents(sample_chunks, reset_existing=True)

    assert temp_vector_manager.get_document_count() == 4

    # Query for revenue
    results = temp_vector_manager.similarity_search("What was the annual revenue?", k=2)
    assert len(results) == 2
    # The most relevant chunk should be the first document discussing revenue
    assert "28.45 billion" in results[0].page_content or "28.45 billion" in results[1].page_content

    # Query for risk factors
    risk_results = temp_vector_manager.similarity_search("What are the primary business risks?", k=1)
    assert len(risk_results) == 1
    assert "macroeconomic" in risk_results[0].page_content.lower()


def test_get_retriever(temp_vector_manager: VectorStoreManager):
    """Test creating a LangChain retriever and retrieving documents."""
    sample_chunks = [
        Document(
            page_content="Free cash flow reached $5.82 billion for dividend payouts.",
            metadata={"file_name": "report.txt", "page": 1, "chunk_id": "c1"},
        ),
    ]
    temp_vector_manager.index_documents(sample_chunks, reset_existing=True)

    retriever = temp_vector_manager.get_retriever(k=1)
    docs = retriever.invoke("How much was free cash flow?")
    assert len(docs) == 1
    assert "$5.82 billion" in docs[0].page_content


def test_index_empty_list(temp_vector_manager: VectorStoreManager):
    """Test indexing an empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot index empty chunk list"):
        temp_vector_manager.index_documents([])
