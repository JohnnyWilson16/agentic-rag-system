"""
Task Definitions Module
=======================
Defines analytical workflows and structured prompt goals for CrewAI agents.
"""

from typing import Optional
from crewai import Agent, Task


def create_comprehensive_analysis_task(
    agent: Agent,
    company_name: Optional[str] = None,
) -> Task:
    """
    Creates a full-spectrum financial analysis task for annual reports.

    Args:
        agent: The CrewAI Agent assigned to execute the task.
        company_name: Optional company identifier for specialized prompts.

    Returns:
        Configured CrewAI Task instance.
    """
    target = f"for {company_name}" if company_name else "from the uploaded report"

    description = f"""
    Conduct an exhaustive, evidence-grounded financial analysis {target}.
    Search and analyze the document to answer the following key areas:

    1. **Executive Financial Summary**:
       - What were the total revenues, net profit, operating margin, and YoY growth rates?
       - How did key business segments or geographies perform?

    2. **Strategic & Business Highlights**:
       - What were the primary operational milestones, key product launches, or major client acquisitions?
       - What investments were made in capital expenditures (CapEx) or R&D?

    3. **Risk Factors & Challenges**:
       - What major operational, market, geopolitical, or regulatory risks did management disclose?
       - Are there any ongoing litigations, supply chain bottlenecks, or currency headwinds?

    4. **Management Outlook & Future Strategy**:
       - What is management's projected guidance or qualitative outlook for the upcoming fiscal years?
       - What strategic priorities (e.g. AI adoption, sustainability, expansion) are highlighted?

    **Mandatory Requirement**: Ground every single metric and finding in facts retrieved directly from
    the document. Cite specific numbers and sections whenever available.
    """

    expected_output = """
    A comprehensive, professionally formatted markdown report with:
    - # Executive Summary & Key Highlights
    - ## 1. Financial Performance & Key Metrics (include clear figures)
    - ## 2. Strategic Business Highlights
    - ## 3. Risk Factors & Vulnerabilities
    - ## 4. Forward-Looking Strategy & Management Guidance
    - ## 5. Conclusion & Investment Summary
    All statements must cite the relevant document context.
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
    )


def create_custom_query_task(
    agent: Agent,
    question: str,
) -> Task:
    """
    Creates a targeted task to answer a specific user question using agentic retrieval.

    Args:
        agent: The CrewAI Agent assigned to execute the task.
        question: The user's specific question or research query.

    Returns:
        Configured CrewAI Task instance.
    """
    description = f"""
    Investigate and answer the following question thoroughly using the document retrieval tool:

    Query: "{question}"

    Instructions:
    1. Break down the query into relevant sub-topics if needed.
    2. Search the vector database using specific keywords and concepts.
    3. Synthesize the findings into a clear, detailed, and directly supported response.
    4. If the document does not contain sufficient information to answer parts of the question, state so explicitly.
    """

    expected_output = """
    A clear, well-structured answer addressing the user's question, supported by exact facts,
    metrics, and citations from the document.
    """

    return Task(
        description=description.strip(),
        expected_output=expected_output.strip(),
        agent=agent,
    )
