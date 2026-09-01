"""Real DeepSeek tool-calling smoke test for authoritative journal lookup."""

from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import invoke_journal_agent


KNOWN_JOURNAL = "Journal of Machine Learning Research"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def main() -> int:
    """Require a real get_journal_details call before accepting the answer."""
    user_input = (
        f"What are the CCF rank, JCR quartile, CAS quartile, and impact factor "
        f"of {KNOWN_JOURNAL}? Use the local journal tool and do not answer from memory."
    )
    try:
        result = invoke_journal_agent(
            user_input,
            thread_id=f"tool-calling-test-{uuid4()}",
        )
    except Exception:
        print(
            "ERROR: Real Agent invocation failed. Check the DeepSeek configuration, "
            "network, and local journal database.",
            file=sys.stderr,
        )
        return 1

    requested_tools = [
        tool_call.get("name")
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    executed_tools = [
        message.name
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    final_answers = [
        _message_text(message.content)
        for message in result["messages"]
        if isinstance(message, AIMessage) and not message.tool_calls
    ]

    print(f"Requested tools: {requested_tools}")
    print(f"Executed tools: {executed_tools}")
    if final_answers:
        print(f"Agent answer:\n{final_answers[-1]}")

    if "get_journal_details" not in requested_tools:
        print(
            "ERROR: DeepSeek answered without requesting get_journal_details.",
            file=sys.stderr,
        )
        return 1
    if "get_journal_details" not in executed_tools:
        print("ERROR: get_journal_details was not executed.", file=sys.stderr)
        return 1

    print("Real DeepSeek get_journal_details tool-calling test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
