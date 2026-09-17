# ==============================================================================
# Production Dockerfile for Agentic RAG Enterprise System
# Builds and deploys the unified FastAPI backend & Radiant Light SPA on Render
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CREWAI_STORAGE_DIR=/app/.crewai \
    CREWAI_CREDENTIALS_DIR=/app/.crewai \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, data files, and pre-indexed ChromaDB vector store
COPY src/ ./src/
COPY data/ ./data/
COPY company_db/ ./company_db/

# Install the local package in editable mode
RUN pip install --no-cache-dir -e .

# Create storage directory for CrewAI
RUN mkdir -p /app/.crewai && chmod -R 777 /app/.crewai

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch production server with uvicorn
CMD ["sh", "-c", "uvicorn agentic_rag.api:app --host 0.0.0.0 --port ${PORT}"]
