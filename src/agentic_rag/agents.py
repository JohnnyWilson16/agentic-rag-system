"""
Agent Definitions Module
========================
Configures specialized CrewAI autonomous agents for financial and enterprise document analysis.
"""

from typing import List, Optional
from crewai import Agent, LLM


def create_financial_analyst_agent(
    llm: LLM,
    tools: List[any],
    verbose: bool = True,
) -> Agent:
    """
    Creates a Senior Financial Analyst agent configured for evidence-grounded document analysis.

    Args:
        llm: Configured CrewAI LLM instance.
        tools: List of CrewAI retrieval tools the agent can invoke.
        verbose: Whether the agent should print internal reasoning steps.

    Returns:
        Configured CrewAI Agent instance.
    """
    return Agent(
        role="Senior Financial Report Analysis Expert",
        goal=(
            "Analyze corporate annual reports, financial statements, and regulatory disclosures "
            "to answer multi-part questions, evaluate financial performance, identify underlying "
            "operational and market risks, and deliver rigorous, evidence-backed insights."
        ),
        backstory=(
            "You are a seasoned financial analyst and equity research specialist. You have audited "
            "and analyzed hundreds of corporate annual reports, 10-K filings, and investor presentations. "
            "You never rely on speculation or hallucinated numbers. Before formulating an answer, you actively "
            "use your search tools to investigate the report—including the Balance Sheet, Income Statement, "
            "Cash Flow Statement, Management Discussion & Analysis (MD&A), Notes to Accounts, and Risk Factors. "
            "Every claim and metric you present is supported by exact numbers and page-level evidence."
        ),
        tools=tools,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )
