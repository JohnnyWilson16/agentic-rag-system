"""
Command Line Interface (CLI) for Agentic RAG System
===================================================
Provides user commands to ingest documents, query the agent, and run comprehensive financial audits.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_rag.config import get_settings
from agentic_rag.pipeline import AgenticRAGPipeline

console = Console()


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentic-rag",
        description="End-to-End Agentic RAG System for Financial & Enterprise Document Analysis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load, chunk, and index a PDF, TXT, or Markdown document into ChromaDB",
    )
    ingest_parser.add_argument(
        "file_path",
        type=str,
        help="Path to the document file (e.g. ./data/annual_report.pdf)",
    )
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing vector database before indexing new document",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Ask a specific question using autonomous agentic retrieval",
    )
    query_parser.add_argument(
        "question",
        type=str,
        help="Question or query string (e.g. 'What was the Q4 operating margin?')",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run comprehensive financial performance, risk, and outlook analysis",
    )
    analyze_parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Optional company name for report titling",
    )

    # Status command
    subparsers.add_parser(
        "status",
        help="Show vector database status, active LLM provider, and configurations",
    )

    return parser


def handle_ingest(pipeline: AgenticRAGPipeline, args: argparse.Namespace) -> None:
    """Handles document ingestion command."""
    file_path = Path(args.file_path)
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        sys.exit(1)

    with console.status(f"[bold cyan]Ingesting and indexing {file_path.name}...[/bold cyan]"):
        try:
            stats = pipeline.ingest_document(file_path=file_path, reset_existing=args.reset)
        except Exception as e:
            console.print(f"[bold red]Ingestion failed:[/bold red] {str(e)}")
            sys.exit(1)

    table = Table(title="Document Ingestion Summary", border_style="cyan")
    table.add_column("Property", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Document Name", stats["file_name"])
    table.add_row("Raw Pages / Sections", str(stats["raw_pages"]))
    table.add_row("Chunks Created", str(stats["chunks_created"]))
    table.add_row("Total Vectors in DB", str(stats["total_vectors_in_db"]))
    table.add_row("Embedding Model", stats["embedding_model"])
    table.add_row("Persist Directory", stats["storage_path"])

    console.print(table)
    console.print("[bold green]✓ Ingestion completed successfully.[/bold green]\n")


def handle_query(pipeline: AgenticRAGPipeline, args: argparse.Namespace) -> None:
    """Handles single query command."""
    console.print(
        Panel(
            f"[bold yellow]Question:[/bold yellow] {args.question}",
            title="Agentic RAG Query",
            border_style="yellow",
        )
    )

    try:
        pipeline.settings.validate_provider_setup()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)

    try:
        answer = pipeline.query(args.question)
        console.print("\n" + "=" * 80)
        console.print("[bold green]AGENT RESPONSE[/bold green]")
        console.print("=" * 80 + "\n")
        console.print(answer)
    except Exception as e:
        console.print(f"[bold red]Execution error:[/bold red] {str(e)}")
        sys.exit(1)


def handle_analyze(pipeline: AgenticRAGPipeline, args: argparse.Namespace) -> None:
    """Handles full financial report analysis."""
    company_title = args.company or "Indexed Corporate Report"
    console.print(
        Panel(
            f"Running full analytical audit for: [bold cyan]{company_title}[/bold cyan]",
            title="Comprehensive Financial Analysis",
            border_style="cyan",
        )
    )

    try:
        pipeline.settings.validate_provider_setup()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)

    try:
        report = pipeline.analyze(company_name=args.company)
        console.print("\n" + "=" * 80)
        console.print("[bold green]COMPREHENSIVE FINANCIAL AUDIT REPORT[/bold green]")
        console.print("=" * 80 + "\n")
        console.print(report)
    except Exception as e:
        console.print(f"[bold red]Analysis error:[/bold red] {str(e)}")
        sys.exit(1)


def handle_status(pipeline: AgenticRAGPipeline) -> None:
    """Displays current system status."""
    status = pipeline.get_status()

    table = Table(title="Agentic RAG System Status", border_style="blue")
    table.add_column("Setting / Metric", style="bold white")
    table.add_column("Current Value", style="cyan")

    table.add_row("LLM Provider", status["llm_provider"])
    table.add_row("LLM Model", status["llm_model"])
    table.add_row("Embedding Model", status["embedding_model"])
    table.add_row("Indexed Vectors in DB", str(status["indexed_vectors"]))
    table.add_row("Chroma Storage Directory", status["vector_store_dir"])
    table.add_row("Chunk Size / Overlap", f"{status['chunk_size']} / {status['chunk_overlap']}")
    table.add_row("Retriever Top-K", str(status["retriever_k"]))

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    settings = get_settings()
    pipeline = AgenticRAGPipeline(settings=settings)

    if args.command == "ingest":
        handle_ingest(pipeline, args)
    elif args.command == "query":
        handle_query(pipeline, args)
    elif args.command == "analyze":
        handle_analyze(pipeline, args)
    elif args.command == "status":
        handle_status(pipeline)


if __name__ == "__main__":
    main()
