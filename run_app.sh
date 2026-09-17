#!/usr/bin/env bash
# ==============================================================================
# Agentic RAG — Enterprise Console Launcher
# Launches the Radiant Light Theme Enterprise Web Console & FastAPI Server
# Perfect for local offline execution & confidential private data workflows.
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

export CREWAI_STORAGE_DIR="$SCRIPT_DIR/.crewai"
export CREWAI_CREDENTIALS_DIR="$SCRIPT_DIR/.crewai"
mkdir -p "$CREWAI_STORAGE_DIR"

# Resolve python binary: prefer .venv if present
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "======================================================================"
echo "✨ Launching Agentic RAG Enterprise Console (Radiant Light Theme)"
echo "📂 Working directory: $SCRIPT_DIR"
echo "🧠 CrewAI Storage:   $CREWAI_STORAGE_DIR"
echo "🌐 Local URL:        http://${HOST}:${PORT}"
echo "======================================================================"

# Proactively open browser on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    (sleep 1.2 && open "http://${HOST}:${PORT}") &
fi

exec "$PYTHON_BIN" -m uvicorn agentic_rag.api:app --host "$HOST" --port "$PORT"
