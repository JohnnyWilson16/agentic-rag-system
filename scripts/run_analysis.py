#!/usr/bin/env python3
"""
Quickstart Analysis Script
==========================
Demonstrates ingesting a financial report and running both targeted queries
and full comprehensive multi-section agentic audits.
"""

import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.config import get_settings
from agentic_rag.pipeline import AgenticRAGPipeline


def main():
    settings = get_settings()
    pipeline = AgenticRAGPipeline(settings=settings)

    report_path = Path(__file__).parent.parent / "data" / "sample_annual_report.txt"

    print("=" * 80)
    print("STEP 1: INGESTING FINANCIAL REPORT INTO CHROMADB")
    print("=" * 80)
    stats = pipeline.ingest_document(report_path, reset_existing=True)
    print(f"✓ Ingested: {stats['file_name']}")
    print(f"✓ Chunks created: {stats['chunks_created']}")
    print(f"✓ Total vectors in ChromaDB: {stats['total_vectors_in_db']}\n")

    print("=" * 80)
    print("STEP 2: RUNNING AGENTIC FINANCIAL QUERY")
    print("=" * 80)
    question = "What was the total revenue, operating margin, and YoY growth rate for FY 2025-2026?"
    print(f"Question: {question}\n")

    try:
        settings.validate_provider_setup()
        response = pipeline.query(question)
        print("\nAGENT ANSWER:")
        print(response)
    except ValueError as e:
        print(f"[!] API Configuration Note: {e}")
        print("To run LLM agent queries, set GROQ_API_KEY (or OPENAI_API_KEY / GEMINI_API_KEY) in .env")


if __name__ == "__main__":
    main()
