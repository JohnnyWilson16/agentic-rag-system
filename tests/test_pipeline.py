"""
Integration Tests for AgenticRAGPipeline
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agentic_rag.config import Settings
from agentic_rag.pipeline import AgenticRAGPipeline
from agentic_rag.vector_store import VectorStoreManager


@pytest.fixture
def temp_pipeline(tmp_path: Path):
    """Provides an AgenticRAGPipeline configured with a temporary vector store directory."""
    db_dir = tmp_path / "pipeline_chroma_db"
    settings = Settings(
        vector_store_dir=str(db_dir),
        llm_provider="groq",
        groq_api_key="mock_key",
        verbose=False,
    )
    vector_manager = VectorStoreManager(settings=settings, collection_name="pipeline_test")
    pipeline = AgenticRAGPipeline(settings=settings, vector_store_manager=vector_manager)
    return pipeline


def test_pipeline_ingest_and_status(temp_pipeline: AgenticRAGPipeline, tmp_path: Path):
    """Test full document ingestion and status reporting."""
    doc_file = tmp_path / "sample_financials.txt"
    doc_file.write_text(
        "Apex Global FY 2025-2026 total revenue reached $28.45 billion.\n"
        "Operating margin was 24.8% and Net Income was $6.12 billion.",
        encoding="utf-8",
    )

    stats = temp_pipeline.ingest_document(doc_file, reset_existing=True)
    assert stats["file_name"] == "sample_financials.txt"
    assert stats["chunks_created"] >= 1
    assert stats["total_vectors_in_db"] >= 1

    status = temp_pipeline.get_status()
    assert status["llm_provider"] == "groq"
    assert status["indexed_vectors"] >= 1


def test_pipeline_query_empty_db(tmp_path: Path):
    """Test pipeline raises error if queried before ingesting documents."""
    empty_settings = Settings(vector_store_dir=str(tmp_path / "empty_db"))
    empty_pipeline = AgenticRAGPipeline(settings=empty_settings)

    with pytest.raises(RuntimeError, match="Vector database is empty"):
        empty_pipeline.query("What is the revenue?")


@patch("agentic_rag.pipeline.Crew")
def test_pipeline_query_mock_execution(mock_crew_class, temp_pipeline: AgenticRAGPipeline, tmp_path: Path):
    """Test query workflow triggers Crew execution properly."""
    doc_file = tmp_path / "mock_report.txt"
    doc_file.write_text("Revenue was $28.45B.", encoding="utf-8")
    temp_pipeline.ingest_document(doc_file, reset_existing=True)

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = "Verified revenue was $28.45B."
    mock_crew_class.return_value = mock_crew_instance

    result = temp_pipeline.query("What was the revenue?")
    assert result == "Verified revenue was $28.45B."
    mock_crew_class.assert_called_once()
    mock_crew_instance.kickoff.assert_called_once()
