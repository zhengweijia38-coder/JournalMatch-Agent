"""Interactive, process-local CLI for the Phase 9 journal agent."""

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import invoke_journal_agent
from src.exceptions import JournalRAGError
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)

USAGE_HINT = (
    "Commands: help shows this message; exit/quit ends the process.\n"
    "Requests: provide a PDF path for analysis/recommendation, ask for a "
    "topic-based journal search, or request details for a journal name.\n"
    "Memory: conversation context lasts only for this process and is not "
    "written to disk."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional tool-trace output for the interactive session."""
    parser = argparse.ArgumentParser(
        description="Run the local journal recommendation Agent CLI."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show tool names and argument/result summaries after each turn.",
    )
    return parser.parse_args(argv)


def _content_to_text(content: Any) -> str:
    """Render current LangChain string or content-block message formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_blocks.append(block["text"])
            elif isinstance(block, str):
                text_blocks.append(block)
        if text_blocks:
            return "\n".join(text_blocks)
    return str(content)


def _latest_assistant_text(result: dict[str, Any]) -> str:
    """Return the newest non-tool-calling assistant message."""
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _content_to_text(message.content)
    return "The agent completed without a final text response."


def _print_debug_trace(result: dict[str, Any]) -> None:
    """Print concise tool activity without environment data or tracebacks."""
    print("\n[Debug tool trace]")
    found = False
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                found = True
                print(f"Tool: {tool_call.get('name', 'unknown')}")
                print(f"Arguments: {tool_call.get('args', {})}")
        elif isinstance(message, ToolMessage):
            found = True
            summary = _content_to_text(message.content).replace("\n", " ")
            print(f"Result ({message.name or 'tool'}): {summary[:300]}")
    if not found:
        print("No tool call was emitted for this turn.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a multi-turn session using one stable in-memory thread ID."""
    args = parse_args(argv)
    configure_logging(debug=args.debug)
    thread_id = f"agent-cli-{uuid4()}"

    print("Journal Recommendation Agent")
    print(USAGE_HINT)
    while True:
        try:
            user_input = input("\nUser:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return 0

        if user_input.casefold() in {"exit", "quit"}:
            print("Session ended.")
            return 0
        if user_input.casefold() in {"help", "?"}:
            print(USAGE_HINT)
            continue
        if not user_input:
            print("Please enter a request or type 'exit'.")
            continue

        try:
            result = invoke_journal_agent(user_input, thread_id=thread_id)
            if args.debug:
                _print_debug_trace(result)
            print("\nAssistant:")
            print(_latest_assistant_text(result))
        except (ValueError, JournalRAGError) as exc:
            if args.debug:
                logger.exception("Agent turn failed")
            else:
                logger.error("%s", exc)
        except Exception:
            if args.debug:
                logger.exception("Unexpected Agent turn failure")
            else:
                logger.error(
                    "The agent failed unexpectedly. Check local data, model "
                    "configuration, and network; use --debug for a traceback."
                )


if __name__ == "__main__":
    raise SystemExit(main())
