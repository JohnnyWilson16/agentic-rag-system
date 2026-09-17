"""
Agentic RAG System
==================
An end-to-end Agentic Retrieval-Augmented Generation pipeline built with
LangChain, ChromaDB, and CrewAI for deep financial and enterprise report analysis.
"""
# Global safety patch for Protobuf gencode/runtime version mismatch across mixed environments
try:
    import google.protobuf.runtime_version
    google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except Exception:
    pass

import os
from pathlib import Path

# Ensure local crewai directory to avoid permission errors accessing ~/Library in sandboxes
_local_crewai = Path(__file__).resolve().parent.parent.parent / ".crewai"
_local_crewai.mkdir(parents=True, exist_ok=True)
if "CREWAI_STORAGE_DIR" not in os.environ:
    os.environ["CREWAI_STORAGE_DIR"] = str(_local_crewai)
if "CREWAI_CREDENTIALS_DIR" not in os.environ:
    os.environ["CREWAI_CREDENTIALS_DIR"] = str(_local_crewai)

__version__ = "0.1.0"
__all__ = [
    "AgenticRAGPipeline",
    "Settings",
    "get_settings",
    "DocumentLoader",
    "DocumentSplitter",
    "VectorStoreManager",
    "LLMFactory",
]

def __getattr__(name: str):
    if name == "AgenticRAGPipeline":
        from agentic_rag.pipeline import AgenticRAGPipeline
        return AgenticRAGPipeline
    elif name in ("Settings", "get_settings"):
        from agentic_rag import config
        return getattr(config, name)
    elif name == "DocumentLoader":
        from agentic_rag.document_loader import DocumentLoader
        return DocumentLoader
    elif name == "DocumentSplitter":
        from agentic_rag.text_splitter import DocumentSplitter
        return DocumentSplitter
    elif name == "VectorStoreManager":
        from agentic_rag.vector_store import VectorStoreManager
        return VectorStoreManager
    elif name == "LLMFactory":
        from agentic_rag.llm_factory import LLMFactory
        return LLMFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
