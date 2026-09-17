"""
Root Application Entrypoint for Render and Production Deployments
==================================================================
Exposes FastAPI app with sys.path properly configured for src/.
"""

import os
import sys
from pathlib import Path

# Add src to sys.path so agentic_rag can always be imported
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure crewai directory is configured
crewai_storage = os.getenv("CREWAI_STORAGE_DIR", "/tmp/.crewai")
Path(crewai_storage).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CREWAI_STORAGE_DIR", crewai_storage)
os.environ.setdefault("CREWAI_CREDENTIALS_DIR", crewai_storage)

from agentic_rag.api import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    print(f"Starting Agentic RAG Server on 0.0.0.0:{port}...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
