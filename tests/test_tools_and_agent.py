"""
Unit Tests for CrewAI Retrieval Tools, Agents, and Tasks
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from crewai import LLM
from agentic_rag.tools import create_search_tool
from agentic_rag.agents import create_financial_analyst_agent
from agentic_rag.tasks import create_comprehensive_analysis_task, create_custom_query_task


def test_create_search_tool_formatting():
    """Test retrieval tool formats document evidence blocks with metadata."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Consolidated net profit reached $6.12B, up 9.4% YoY.",
            metadata={"page": 1, "file_name": "annual_report.pdf", "chunk_id": "chunk_0"},
        ),
        Document(
            page_content="BFSI segment grew 6.8% YoY to $8.95B.",
            metadata={"page": 2, "file_name": "annual_report.pdf", "chunk_id": "chunk_1"},
        ),
    ]

    search_tool = create_search_tool(mock_retriever)
    assert search_tool.name == "Financial Report Search"

    # Invoke tool directly
    result = search_tool.run("net profit")

    assert "Consolidated net profit reached $6.12B" in result
    assert "Evidence #1" in result
    assert "Evidence #2" in result
    assert "Page: 1" in result
    assert "Page: 2" in result


def test_search_tool_empty_query():
    """Test retrieval tool handles empty search strings gracefully."""
    mock_retriever = MagicMock()
    search_tool = create_search_tool(mock_retriever)

    result = search_tool.run("")
    assert "Please provide a valid non-empty search query" in result


def test_search_tool_no_results():
    """Test retrieval tool handles empty search results."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    search_tool = create_search_tool(mock_retriever)
    result = search_tool.run("quantum computing algorithms")
    assert "No relevant sections found" in result


def test_agent_and_task_construction():
    """Test constructing Agent and Task objects with proper configuration."""
    mock_retriever = MagicMock()
    search_tool = create_search_tool(mock_retriever)
    test_llm = LLM(model="ollama/llama3.2")

    agent = create_financial_analyst_agent(
        llm=test_llm,
        tools=[search_tool],
        verbose=False,
    )

    assert agent.role == "Senior Financial Report Analysis Expert"
    assert "audited and analyzed hundreds" in agent.backstory

    # Test comprehensive task creation
    full_task = create_comprehensive_analysis_task(agent=agent, company_name="Apex Global")
    assert "Apex Global" in full_task.description
    assert "Executive Financial Summary" in full_task.description
    assert full_task.agent == agent

    # Test custom query task creation
    custom_task = create_custom_query_task(agent=agent, question="What was the CapEx spend?")
    assert "What was the CapEx spend?" in custom_task.description
    assert custom_task.agent == agent
