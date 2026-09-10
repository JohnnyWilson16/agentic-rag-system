"""
Unit Tests for Document Loading and Text Chunking
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from agentic_rag.document_loader import DocumentLoader
from agentic_rag.text_splitter import DocumentSplitter


def test_load_text_document(tmp_path: Path):
    """Test loading a clean text document."""
    test_file = tmp_path / "report.txt"
    test_file.write_text("Apex Global generated $28.45B in revenue for FY 2025-2026.", encoding="utf-8")

    docs = DocumentLoader.load_document(test_file)
    assert len(docs) == 1
    assert "Apex Global" in docs[0].page_content
    assert docs[0].metadata["file_name"] == "report.txt"
    assert docs[0].metadata["char_count"] > 0


def test_load_missing_file():
    """Test loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load_document("non_existent_file.pdf")


def test_load_empty_file(tmp_path: Path):
    """Test loading an empty file raises ValueError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        DocumentLoader.load_document(empty_file)


def test_load_unsupported_format(tmp_path: Path):
    """Test loading an unsupported extension raises ValueError."""
    bad_file = tmp_path / "unsupported.xyz"
    bad_file.write_text("some content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentLoader.load_document(bad_file)


def test_load_sample_pdf():
    """Test loading the generated sample PDF."""
    pdf_path = Path(__file__).parent.parent / "data" / "annual_report_2025_2026.pdf"
    if pdf_path.exists():
        docs = DocumentLoader.load_document(pdf_path)
        assert any("Annual Report" in doc.page_content or "TCS" in doc.page_content or "Tata" in doc.page_content or "ANNUAL REPORT" in doc.page_content.upper() or "Apex" in doc.page_content for doc in docs)


def test_document_splitter():
    """Test recursive character chunking and metadata enrichment."""
    raw_doc = Document(
        page_content=(
            "Paragraph 1: Revenue grew by 8.2% YoY.\n\n"
            "Paragraph 2: Operating margin was 24.8%.\n\n"
            "Paragraph 3: Net income was $6.12 billion with strong cash conversion.\n\n"
            "Paragraph 4: BFS grew 6.8% and Healthcare grew 14.2%."
        ),
        metadata={"file_name": "annual_report.txt", "page": 1},
    )

    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents([raw_doc])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["file_name"] == "annual_report.txt"
        assert len(chunk.page_content) <= 120  # bounded by chunk_size + margin


def test_document_splitter_empty():
    """Test document splitter with empty list."""
    splitter = DocumentSplitter()
    assert splitter.split_documents([]) == []
