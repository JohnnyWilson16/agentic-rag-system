"""
Configuration Management for Agentic RAG System
=================================================
Handles environment variables, default parameters, and LLM provider settings.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System settings and environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    llm_provider: Literal["groq", "openai", "gemini", "anthropic", "ollama"] = Field(
        default="groq",
        description="LLM provider: groq, openai, gemini, anthropic, or ollama",
    )
    llm_model: str = Field(
        default="groq/llama-3.3-70b-versatile",
        description="Model identifier compatible with CrewAI LLM abstraction",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for LLM reasoning (lower is more deterministic)",
    )

    # Provider API Keys
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Local Ollama Settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama instance",
    )

    # Embeddings Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for generating semantic vector embeddings",
    )

    # Ingestion & Chunking
    chunk_size: int = Field(
        default=800,
        description="Maximum character length per document chunk",
    )
    chunk_overlap: int = Field(
        default=150,
        description="Character overlap between consecutive chunks",
    )

    # Vector Database
    vector_store_dir: str = Field(
        default="./company_db",
        description="Directory where ChromaDB vector collection is persisted",
    )
    retriever_k: int = Field(
        default=3,
        description="Number of relevant chunks to retrieve per search query",
    )

    # Execution & Verbosity
    verbose: bool = Field(
        default=True,
        description="Whether to display detailed CrewAI agent thought logs",
    )

    def validate_provider_setup(self) -> None:
        """
        Validates that required credentials exist for the selected LLM provider.
        Raises ValueError with clear setup instructions if missing.
        """
        provider = self.llm_provider.lower()

        if provider == "groq":
            key = self.groq_api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Get a free API key at https://console.groq.com "
                    "and add it to your .env file or environment variables."
                )
        elif provider == "openai":
            key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add your OpenAI key to the .env file."
                )
        elif provider == "gemini":
            key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Add your Google Gemini API key to the .env file."
                )
        elif provider == "anthropic":
            key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Add your Anthropic API key to the .env file."
                )
        elif provider == "ollama":
            # Ollama doesn't require an API key, just a running daemon
            pass


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of application settings."""
    return Settings()
