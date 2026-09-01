"""LangChain agent entry points for natural-language tool routing."""

from src.agent.agent import get_journal_agent, invoke_journal_agent

__all__ = ["get_journal_agent", "invoke_journal_agent"]
