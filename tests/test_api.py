"""
Integration and Unit Tests for Agentic RAG FastAPI Backend
==========================================================
Tests REST endpoints, query execution, health diagnostics, and static asset delivery.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_rag.api import app


@pytest.fixture
def client():
    """Provides a TestClient for testing the FastAPI application."""
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    """Test the /api/health endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "agentic-rag-enterprise"
    assert "timestamp" in data
    assert data["vector_store_online"] is True


def test_status_endpoint(client: TestClient):
    """Test /api/status returns operational metrics and ChromaDB vector details."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "indexed_vectors" in data
    assert "embedding_model" in data
    assert "vector_db" in data
    assert "retriever_k" in data
    assert data["status_badge"] == "OPERATIONAL"


def test_documents_endpoint(client: TestClient):
    """Test /api/documents returns available file list and indexing status."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)
    if len(data["documents"]) > 0:
        doc = data["documents"][0]
        assert "file_name" in doc
        assert "size_bytes" in doc
        assert "source_type" in doc


def test_query_validation(client: TestClient):
    """Test /api/query rejects empty or malformed queries with 422 or 400."""
    response = client.post("/api/query", json={"query": ""})
    assert response.status_code in [400, 422]


def test_query_execution_financial(client: TestClient):
    """Test /api/query executes a financial query and returns grounded 5-section response."""
    response = client.post(
        "/api/query",
        json={"query": "What was TCS operating margin and RoE for FY 2026?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "timeline_steps" in data
    assert len(data["timeline_steps"]) >= 5
    assert "retrieved_chunks" in data
    assert len(data["retrieved_chunks"]) > 0

    # Verify citations and metrics structure
    chunk = data["retrieved_chunks"][0]
    assert "page" in chunk
    assert "relevance_score" in chunk
    assert "content" in chunk

    assert "executive_summary" in data
    assert "key_findings" in data
    assert "important_numbers" in data
    assert "risks_concerns" in data
    assert "outlook" in data
    assert "metrics" in data


def test_chat_interaction_and_conversations(client: TestClient):
    """Test /api/chat stateful interaction and session management."""
    # Send first message in new conversation
    response = client.post(
        "/api/chat",
        json={"message": "How did TCS perform in FY 2026?"},
    )
    assert response.status_code == 200
    chat_data = response.json()
    assert "conversation_id" in chat_data
    conv_id = chat_data["conversation_id"]
    assert "result" in chat_data
    assert "conversation" in chat_data

    # List conversations
    convs_resp = client.get("/api/conversations")
    assert convs_resp.status_code == 200
    convs = convs_resp.json()["conversations"]
    assert any(c["id"] == conv_id for c in convs)

    # Get specific conversation
    get_resp = client.get(f"/api/conversations/{conv_id}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["messages"]) == 2  # user + assistant

    # Delete conversation
    del_resp = client.delete(f"/api/conversations/{conv_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"


def test_static_asset_serving(client: TestClient):
    """Test that static index.html, styles.css, and app.js are served properly."""
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    assert "Agentic RAG" in resp_index.text
    assert "Retrieved Evidence" in resp_index.text

    resp_css = client.get("/static/styles.css")
    assert resp_css.status_code == 200
    assert "--primary: #6366F1;" in resp_css.text

    resp_js = client.get("/static/app.js")
    assert resp_js.status_code == 200
    assert "Agentic RAG" in resp_js.text
