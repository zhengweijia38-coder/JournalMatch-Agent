"""Read-only high-level capabilities exposed to the journal agent."""

from src.tools.journal_tools import get_journal_details_tool, search_journals_tool
from src.tools.paper_tools import analyze_paper_tool
from src.tools.recommendation_tools import recommend_journals_tool


JOURNAL_AGENT_TOOLS = [
    analyze_paper_tool,
    search_journals_tool,
    get_journal_details_tool,
    recommend_journals_tool,
]

__all__ = [
    "JOURNAL_AGENT_TOOLS",
    "analyze_paper_tool",
    "get_journal_details_tool",
    "recommend_journals_tool",
    "search_journals_tool",
]
