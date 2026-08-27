"""
LLM Factory Module
==================
Initializes CrewAI-compatible LLM instances across cloud and local providers.
Provides seamless fallback from local Ollama to high-speed cloud providers (Groq, OpenAI, Gemini, Anthropic).
"""

import os
from typing import Optional
from crewai import LLM
from agentic_rag.config import Settings, get_settings


class LLMFactory:
    """Factory class for creating and configuring CrewAI LLM instances."""

    @staticmethod
    def create_llm(
        settings: Optional[Settings] = None,
        override_model: Optional[str] = None,
        override_provider: Optional[str] = None,
    ) -> LLM:
        """
        Builds a CrewAI LLM instance based on application settings.

        Args:
            settings: Optional custom Settings instance. Defaults to get_settings().
            override_model: Optional model name to override configured model.
            override_provider: Optional provider name to override configured provider.

        Returns:
            Configured crewai.LLM instance.
        """
        cfg = settings or get_settings()
        provider = (override_provider or cfg.llm_provider).lower()
        model_name = override_model or cfg.llm_model

        # Ensure correct provider prefixing for CrewAI / LiteLLM backend
        if provider == "groq":
            if not model_name.startswith("groq/"):
                model_name = f"groq/{model_name}"
            api_key = cfg.groq_api_key or os.getenv("GROQ_API_KEY")
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
            return LLM(
                model=model_name,
                temperature=cfg.temperature,
                api_key=api_key,
            )

        elif provider == "openai":
            if not model_name.startswith("openai/") and "/" not in model_name:
                # e.g. "gpt-4o-mini" or "gpt-4o"
                pass
            api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            return LLM(
                model=model_name,
                temperature=cfg.temperature,
                api_key=api_key,
            )

        elif provider == "gemini":
            if not model_name.startswith("gemini/"):
                model_name = f"gemini/{model_name}"
            api_key = cfg.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            return LLM(
                model=model_name,
                temperature=cfg.temperature,
                api_key=api_key,
            )

        elif provider == "anthropic":
            if not model_name.startswith("anthropic/"):
                model_name = f"anthropic/{model_name}"
            api_key = cfg.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            return LLM(
                model=model_name,
                temperature=cfg.temperature,
                api_key=api_key,
            )

        elif provider == "ollama":
            if not model_name.startswith("ollama/"):
                model_name = f"ollama/{model_name}"
            return LLM(
                model=model_name,
                base_url=cfg.ollama_base_url,
                temperature=cfg.temperature,
            )

        else:
            # Fallback for custom model string
            return LLM(
                model=model_name,
                temperature=cfg.temperature,
            )
