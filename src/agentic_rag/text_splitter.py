"""
Text Chunking and Splitting Module
==================================
Splits long documents into semantically coherent, context-preserving chunks with metadata enrichment.
"""

from typing import List, Any
from langchain_core.documents import Document


class DocumentSplitter:
    """Splits documents into overlapping chunks for semantic vector indexing."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: List[str] = None,
    ):
        """
        Initializes the document splitter.

        Args:
            chunk_size: Maximum character length per chunk (default: 800).
            chunk_overlap: Character overlap between consecutive chunks (default: 150).
            separators: Hierarchy of split delimiters (default: paragraphs, sentences, words).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self._splitter_instance: Any = None

    @property
    def splitter(self) -> Any:
        if self._splitter_instance is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter_instance = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len,
                is_separator_regex=False,
            )
        return self._splitter_instance

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into enriched chunks.

        Args:
            documents: Input LangChain Document list.

        Returns:
            List of split Document chunks with enriched metadata (chunk_id, chunk_index).
        """
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        # Enrich chunk metadata for citation and evidence tracking
        enriched_chunks: List[Document] = []
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            file_name = metadata.get("file_name", "doc")
            page = metadata.get("page", 1)

            metadata["chunk_index"] = idx
            metadata["chunk_id"] = f"{file_name}_p{page}_c{idx}"
            metadata["chunk_length"] = len(chunk.page_content)

            enriched_chunks.append(
                Document(page_content=chunk.page_content, metadata=metadata)
            )

        return enriched_chunks
