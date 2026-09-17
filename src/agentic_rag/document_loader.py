"""
Document Ingestion and Loading Module
====================================
Handles loading, text extraction, and metadata enrichment for enterprise PDF, Markdown, and TXT documents.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    """Loads and standardizes documents from filesystem paths."""

    @staticmethod
    def load_document(file_path: str | Path) -> List[Document]:
        """
        Loads a document from the specified path and extracts page content with metadata.

        Args:
            file_path: Path to the target PDF, TXT, or Markdown document.

        Returns:
            List of LangChain Document objects containing page content and metadata.

        Raises:
            FileNotFoundError: If the document path does not exist.
            ValueError: If the file format is unsupported or the file is empty.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Document file not found at: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"Document at {path} is empty (0 bytes).")

        ext = path.suffix.lower()

        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(path))
            documents = loader.load()
        elif ext in [".txt", ".md", ".markdown", ".csv", ".json"]:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(str(path), encoding="utf-8")
            documents = loader.load()
        else:
            raise ValueError(
                f"Unsupported document format '{ext}'. Supported formats: .pdf, .txt, .md"
            )

        if not documents:
            raise ValueError(f"No readable content could be extracted from {path}")

        # Standardize metadata across documents
        cleaned_docs: List[Document] = []
        for idx, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue

            metadata = dict(doc.metadata)
            metadata["source"] = str(path)
            metadata["file_name"] = path.name
            if "page" not in metadata:
                metadata["page"] = idx + 1
            metadata["char_count"] = len(content)

            cleaned_docs.append(Document(page_content=content, metadata=metadata))

        if not cleaned_docs:
            raise ValueError(f"Document {path.name} contains no non-empty pages/sections.")

        return cleaned_docs
