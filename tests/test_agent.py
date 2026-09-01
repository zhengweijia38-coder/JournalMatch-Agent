"""Offline tests for current LangChain create_agent configuration and memory."""

from pathlib import Path
import sys
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.agent.agent as agent_module
from src.agent.prompt import JOURNAL_AGENT_SYSTEM_PROMPT


def test_agent() -> None:
    """Verify model reuse, tool registration, caching, and thread-aware invocation."""
    fake_llm = object()
    fake_checkpointer = object()
    fake_agent = Mock()
    fake_agent.invoke.return_value = {"messages": ["result"]}

    agent_module.get_journal_agent.cache_clear()
    with (
        patch.object(agent_module, "get_llm", return_value=fake_llm) as get_llm,
        patch.object(
            agent_module,
            "InMemorySaver",
            return_value=fake_checkpointer,
        ) as memory,
        patch.object(
            agent_module,
            "create_agent",
            return_value=fake_agent,
        ) as create_agent,
    ):
        first = agent_module.get_journal_agent()
        second = agent_module.get_journal_agent()

    assert first is fake_agent
    assert second is fake_agent
    get_llm.assert_called_once_with()
    memory.assert_called_once_with()
    create_agent.assert_called_once()
    kwargs = create_agent.call_args.kwargs
    assert kwargs["model"] is fake_llm
    assert kwargs["checkpointer"] is fake_checkpointer
    assert kwargs["system_prompt"] == JOURNAL_AGENT_SYSTEM_PROMPT
    assert [tool.name for tool in kwargs["tools"]] == [
        "analyze_paper",
        "search_journals",
        "get_journal_details",
        "recommend_journals",
    ]

    with patch.object(agent_module, "get_journal_agent", return_value=fake_agent):
        result = agent_module.invoke_journal_agent(
            "Find CCF B NLP journals.",
            thread_id="session-123",
        )
    assert result == {"messages": ["result"]}
    fake_agent.invoke.assert_called_once_with(
        {
            "messages": [
                {"role": "user", "content": "Find CCF B NLP journals."}
            ]
        },
        config={"configurable": {"thread_id": "session-123"}},
    )

    for user_input, thread_id in [("", "session"), ("hello", "")]:
        try:
            agent_module.invoke_journal_agent(user_input, thread_id)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty user input or thread_id must be rejected.")

    agent_module.get_journal_agent.cache_clear()
    print("Phase 9 create_agent factory and memory configuration tests passed.")


if __name__ == "__main__":
    try:
        test_agent()
    except Exception as exc:
        print(f"ERROR: Agent factory test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
