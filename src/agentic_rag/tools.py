"""
CrewAI Retrieval Tools
======================
Exposes vector database retrieval as modular, reusable CrewAI tools for autonomous agent execution.
"""

from typing import Any, Callable
from crewai.tools import tool


def create_search_tool(retriever: Any) -> Any:
    """
    Factory function creating a CrewAI tool bound to the specified vector retriever.

    Args:
        retriever: LangChain retriever instance connected to the vector database.

    Returns:
        Callable CrewAI tool function decorated with @tool.
    """

    @tool("Financial Report Search")
    def search_company_docs(query: str) -> str:
        """
        Search the company's annual financial report and retrieve the most relevant sections,
        tables, risk factors, MD&A discussions, or auditor notes related to the query.

        Args:
            query: The search string or topic to locate in the financial report.

        Returns:
            Formatted string of the most relevant document sections with page references.
        """
        if not query or not query.strip():
            return "Please provide a valid non-empty search query."

        try:
            docs = retriever.invoke(query)
            if not docs:
                return f"No relevant sections found in the annual report for query: '{query}'."

            formatted_results = []
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "N/A")
                file_name = doc.metadata.get("file_name", "document")
                chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")

                header = f"--- [Evidence #{i} | File: {file_name} | Page: {page} | ID: {chunk_id}] ---"
                content = doc.page_content.strip()
                formatted_results.append(f"{header}\n{content}")

            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error executing retrieval search: {str(e)}"

    return search_company_docs
