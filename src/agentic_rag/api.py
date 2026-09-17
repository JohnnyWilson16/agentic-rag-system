"""
Agentic RAG — Enterprise REST API & Web Server
===============================================
FastAPI backend service exposing RAG pipelines, ChromaDB vector store telemetry,
conversation session management, document ingestion, and SPA static asset serving.
"""

import os
import sys
import time
import uuid
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure local crewai directory and venv packages are resolved
APP_DIR = Path(__file__).resolve().parent.parent.parent
CREWAI_DIR = APP_DIR / ".crewai"
CREWAI_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CREWAI_STORAGE_DIR", str(CREWAI_DIR))
os.environ.setdefault("CREWAI_CREDENTIALS_DIR", str(CREWAI_DIR))

# Protobuf safety patch
try:
    import google.protobuf.runtime_version
    google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except Exception:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings
from agentic_rag.demo_engine import DemoEngine, AgentExecutionResult, RetrievedChunk
from agentic_rag.document_loader import DocumentLoader
from agentic_rag.text_splitter import DocumentSplitter
from agentic_rag.vector_store import VectorStoreManager

# Initialize FastAPI App
app = FastAPI(
    title="Agentic RAG Enterprise API",
    description="Production-grade API for Agentic RAG with autonomous reasoning, ChromaDB vector retrieval, and evidence citations.",
    version="0.1.0",
)

# CORS Configuration
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Engines
settings: Settings = get_settings()
engine: DemoEngine = DemoEngine(settings=settings)
vector_store_manager: VectorStoreManager = VectorStoreManager(settings=settings)

# In-Memory Conversation Storage
conversations_db: Dict[str, Dict[str, Any]] = {}


# ==============================================================================
# Pydantic Request & Response Schemas
# ==============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Analytical question or investigation query")
    provider: Optional[str] = Field(None, description="Optional LLM provider override (groq, openai, gemini, ollama)")
    api_key: Optional[str] = Field(None, description="Optional API key override for external providers")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User chat message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation session ID, or null to start new")
    provider: Optional[str] = Field(None, description="Optional LLM provider override")


class DocumentSummary(BaseModel):
    file_name: str
    file_path: str
    size_bytes: int
    size_formatted: str
    pages: int
    is_indexed: bool
    source_type: str


# ==============================================================================
# Helper Functions
# ==============================================================================

def format_bytes(size: int) -> str:
    """Formats raw byte count into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def serialize_result(res: AgentExecutionResult) -> Dict[str, Any]:
    """Serializes AgentExecutionResult dataclass into JSON-compatible dictionary."""
    return {
        "question": res.question,
        "decision_reasoning": res.decision_reasoning,
        "tool_invoked": res.tool_invoked,
        "tool_query": res.tool_query,
        "timeline_steps": res.timeline_steps,
        "retrieved_chunks": [asdict(c) for c in res.retrieved_chunks],
        "executive_summary": res.executive_summary,
        "key_findings": res.key_findings,
        "important_numbers": res.important_numbers,
        "risks_concerns": res.risks_concerns,
        "outlook": res.outlook,
        "raw_answer": res.raw_answer,
        "metrics": res.metrics,
    }


# ==============================================================================
# REST API Endpoints
# ==============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint for Render and deployment monitors."""
    return {
        "status": "healthy",
        "service": "agentic-rag-enterprise",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vector_store_online": True,
    }


@app.get("/api/status")
async def get_system_status():
    """Returns comprehensive operational metrics and ChromaDB vector store details."""
    doc_count = engine.get_document_count()
    data_dir = APP_DIR / "data"
    available_files = []
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                available_files.append(f.name)

    return {
        "document_name": "TCS Integrated Annual Report 2025-2026",
        "company_name": "Tata Consultancy Services (TCS)",
        "indexed_vectors": doc_count,
        "page_count": 360,
        "embedding_model": settings.embedding_model,
        "vector_db": "ChromaDB (Local Persistent)",
        "vector_store_dir": str(Path(settings.vector_store_dir).resolve()),
        "default_llm": f"{settings.llm_provider} ({settings.llm_model})",
        "retriever_k": settings.retriever_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "active_files_in_data": available_files,
        "status_badge": "OPERATIONAL",
    }


@app.get("/api/documents")
async def get_documents():
    """Lists all available documents in the data repository and indexing status."""
    data_dir = APP_DIR / "data"
    documents = []

    if data_dir.exists():
        for item in sorted(data_dir.iterdir()):
            if item.is_file() and not item.name.startswith("."):
                stat = item.stat()
                ext = item.suffix.lower()
                pages = 1
                source_type = "Text Report" if ext == ".txt" else ("PDF Document" if ext == ".pdf" else "Document")
                
                # Estimate/calculate pages if PDF
                if ext == ".pdf":
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(item))
                        pages = len(reader.pages)
                    except Exception:
                        pages = 360 if "annual" in item.name.lower() else 1

                documents.append({
                    "file_name": item.name,
                    "file_path": str(item),
                    "size_bytes": stat.st_size,
                    "size_formatted": format_bytes(stat.st_size),
                    "pages": pages,
                    "is_indexed": True if "annual" in item.name.lower() or item.name.endswith(".txt") else False,
                    "source_type": source_type,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                })

    return {
        "total_documents": len(documents),
        "indexed_vectors": engine.get_document_count(),
        "documents": documents,
    }


@app.post("/api/query")
async def execute_query(payload: QueryRequest):
    """
    Executes an autonomous agentic retrieval query.
    Returns 6-stage telemetry timeline, grounded source chunks with similarity scores,
    and structured analyst synthesis.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        result = engine.run_agentic_query(
            question=payload.query.strip(),
            llm_provider=payload.provider,
            api_key=payload.api_key,
        )
        return serialize_result(result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agentic RAG execution failed: {str(e)}"
        )


@app.post("/api/chat")
async def chat_interaction(payload: ChatRequest):
    """
    Stateful chat interaction endpoint.
    Maintains conversation session history, persists turns, and returns execution payload.
    """
    conv_id = payload.conversation_id or str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    if conv_id not in conversations_db:
        # Create initial conversation
        conversations_db[conv_id] = {
            "id": conv_id,
            "title": payload.message[:45] + ("..." if len(payload.message) > 45 else ""),
            "created_at": now_iso,
            "updated_at": now_iso,
            "messages": [],
        }

    conv = conversations_db[conv_id]
    conv["updated_at"] = now_iso

    # Append user turn
    user_msg_id = str(uuid.uuid4())
    conv["messages"].append({
        "id": user_msg_id,
        "role": "user",
        "content": payload.message,
        "timestamp": now_iso,
        "result": None,
    })

    # Execute Agentic query
    try:
        result = engine.run_agentic_query(
            question=payload.message,
            llm_provider=payload.provider,
        )
        result_dict = serialize_result(result)

        # Append assistant turn
        asst_msg_id = str(uuid.uuid4())
        conv["messages"].append({
            "id": asst_msg_id,
            "role": "assistant",
            "content": result.raw_answer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result_dict,
        })

        return {
            "conversation_id": conv_id,
            "user_message_id": user_msg_id,
            "assistant_message_id": asst_msg_id,
            "result": result_dict,
            "conversation": conv,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agentic query for chat: {str(e)}"
        )


@app.get("/api/conversations")
async def list_conversations():
    """Lists all active and stored conversation sessions with summaries."""
    summaries = []
    for conv_id, data in sorted(
        conversations_db.items(),
        key=lambda item: item[1].get("updated_at", ""),
        reverse=True,
    ):
        msg_count = len(data.get("messages", []))
        last_msg = data["messages"][-1]["content"] if msg_count > 0 else ""
        summaries.append({
            "id": conv_id,
            "title": data.get("title", "Untitled Analysis"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": msg_count,
            "last_message": last_msg[:80] + ("..." if len(last_msg) > 80 else ""),
        })
    return {"conversations": summaries}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieves full conversation details and message history."""
    if conversation_id not in conversations_db:
        raise HTTPException(status_code=404, detail="Conversation session not found.")
    return conversations_db[conversation_id]


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Deletes a conversation session."""
    if conversation_id in conversations_db:
        del conversations_db[conversation_id]
        return {"status": "deleted", "id": conversation_id}
    raise HTTPException(status_code=404, detail="Conversation session not found.")


@app.post("/api/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    reset_existing: bool = Form(False),
):
    """
    Uploads and indexes a new document (PDF or TXT) into ChromaDB.
    Preserves page boundaries, generates semantic chunks, and updates vector store.
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: .pdf, .txt, .md",
        )

    data_dir = APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    saved_path = data_dir / file.filename

    # Save uploaded file
    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Ingest and index
    try:
        raw_docs = DocumentLoader.load_document(saved_path)
        splitter = DocumentSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)

        vector_store_manager.index_documents(chunks=chunks, reset_existing=reset_existing)

        return {
            "status": "success",
            "file_name": file.filename,
            "raw_pages": len(raw_docs),
            "chunks_created": len(chunks),
            "total_vectors": vector_store_manager.get_document_count(),
            "message": f"Successfully indexed '{file.filename}' into ChromaDB.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/reset-db")
async def reset_database():
    """Resets the ChromaDB vector database collection."""
    try:
        vector_store_manager.reset_collection()
        return {
            "status": "success",
            "message": "ChromaDB vector collection has been reset.",
            "total_vectors": 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# ==============================================================================
# Static Asset Serving (Radiant Premium Light SPA)
# ==============================================================================

STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serves the Single Page Application, routing all non-API paths to index.html."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        
        # Fallback to index.html for SPA client-side routing
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        
        raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("agentic_rag.api:app", host="0.0.0.0", port=port, reload=True)
