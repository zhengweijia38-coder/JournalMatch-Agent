"""Factory and invocation helpers for the LangChain journal agent."""

from functools import lru_cache
import logging
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.prompt import JOURNAL_AGENT_SYSTEM_PROMPT
from src.models.llm import get_llm
from src.tools import JOURNAL_AGENT_TOOLS


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_journal_agent() -> Any:
    """Create one tool-calling agent with process-local conversation memory."""
    logger.info("Initializing journal agent with %d tools", len(JOURNAL_AGENT_TOOLS))
    return create_agent(
        model=get_llm(),
        tools=JOURNAL_AGENT_TOOLS,
        system_prompt=JOURNAL_AGENT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        name="journal_recommendation_agent",
    )


def invoke_journal_agent(user_input: str, thread_id: str) -> dict[str, Any]:
    """Invoke the current create_agent API for one turn in a stable session."""
    if not user_input or not user_input.strip():
        raise ValueError("Agent user input must not be empty.")
    if not thread_id or not thread_id.strip():
        raise ValueError("Agent thread_id must not be empty.")

    logger.info("Invoking journal agent")
    result = get_journal_agent().invoke(
        {"messages": [{"role": "user", "content": user_input.strip()}]},
        config={"configurable": {"thread_id": thread_id.strip()}},
    )
    if not isinstance(result, dict) or "messages" not in result:
        raise RuntimeError("The journal agent returned an unexpected result shape.")
    logger.info("Journal agent invocation completed")
    return result
