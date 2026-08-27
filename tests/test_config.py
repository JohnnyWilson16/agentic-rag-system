"""
Unit Tests for Configuration & Environment Management
"""

import os
import pytest
from agentic_rag.config import Settings


def test_default_settings():
    """Verify default configurations adhere to system specifications."""
    settings = Settings()
    assert settings.llm_provider in ["groq", "openai", "gemini", "anthropic", "ollama"]
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.retriever_k == 3
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.vector_store_dir == "./company_db"


def test_custom_settings_override():
    """Verify settings can be instantiated with custom parameters."""
    custom = Settings(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        chunk_size=1000,
        chunk_overlap=200,
        retriever_k=5,
        vector_store_dir="./custom_db",
    )
    assert custom.llm_provider == "openai"
    assert custom.llm_model == "gpt-4o-mini"
    assert custom.chunk_size == 1000
    assert custom.chunk_overlap == 200
    assert custom.retriever_k == 5
    assert custom.vector_store_dir == "./custom_db"


def test_validate_provider_setup_missing_key():
    """Verify clear error message is raised when provider API key is missing."""
    # Ensure env vars don't mask test
    old_key = os.environ.pop("GROQ_API_KEY", None)
    try:
        settings = Settings(llm_provider="groq", groq_api_key=None)
        with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
            settings.validate_provider_setup()
    finally:
        if old_key:
            os.environ["GROQ_API_KEY"] = old_key


def test_validate_provider_setup_valid_key():
    """Verify validation passes when valid key is provided."""
    settings = Settings(llm_provider="groq", groq_api_key="gsk_mock_test_key_12345")
    # Should not raise exception
    settings.validate_provider_setup()


def test_validate_ollama_setup():
    """Verify Ollama does not require an API key for local mode."""
    settings = Settings(llm_provider="ollama", llm_model="ollama/llama3.2")
    # Should pass without throwing missing API key error
    settings.validate_provider_setup()
