#!/usr/bin/env bash
# ==============================================================================
# Agentic RAG Demo Launcher
# Launches the production-style Streamlit Enterprise AI Intelligence Console
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

export CREWAI_STORAGE_DIR="$SCRIPT_DIR/.crewai"
mkdir -p "$CREWAI_STORAGE_DIR"

echo "================================================================"
echo "⚡ Starting Agentic RAG Enterprise Console..."
echo "📂 Working directory: $SCRIPT_DIR"
echo "🧠 Storage path: $CREWAI_STORAGE_DIR"
echo "================================================================"

# Choose python/streamlit binary
if [ -f "/Users/wilson/opt/anaconda3/bin/streamlit" ]; then
    STREAMLIT_BIN="/Users/wilson/opt/anaconda3/bin/streamlit"
else
    STREAMLIT_BIN="streamlit"
fi

"$STREAMLIT_BIN" run app.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base="dark"
