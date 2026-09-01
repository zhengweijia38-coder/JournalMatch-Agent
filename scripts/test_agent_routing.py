"""Real integration checks for the five Phase 9 routing and grounding cases."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import invoke_journal_agent


@dataclass(frozen=True, slots=True)
class RoutingCase:
    """One real user request and the high-level tool it must use."""

    name: str
    query: str
    expected_tool: str
    expect_not_found: bool = False


CASES = [
    RoutingCase(
        name="paper analysis",
        query=(
            "Analyze data/papers/test_paper.pdf and tell me its main research "
            "direction. Do not recommend journals."
        ),
        expected_tool="analyze_paper",
    ),
    RoutingCase(
        name="full recommendation",
        query=(
            "Recommend 3 journals for data/papers/test_paper.pdf, only CCF A or B."
        ),
        expected_tool="recommend_journals",
    ),
    RoutingCase(
        name="topic search",
        query=(
            "Find 3 CCF B journals related to retrieval augmented generation."
        ),
        expected_tool="search_journals",
    ),
    RoutingCase(
        name="known journal details",
        query=(
            "What are the JCR quartile and impact factor of Journal of Machine "
            "Learning Research? Use local journal data."
        ),
        expected_tool="get_journal_details",
    ),
    RoutingCase(
        name="unknown journal grounding",
        query=(
            "What are the CCF rank and impact factor of Totally Nonexistent "
            "Journal XYZ? Check the local database and do not guess."
        ),
        expected_tool="get_journal_details",
        expect_not_found=True,
    ),
]


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _run_case(case: RoutingCase) -> None:
    """Invoke one isolated session and enforce actual tool request and execution."""
    result = invoke_journal_agent(
        case.query,
        thread_id=f"routing-{case.name}-{uuid4()}",
    )
    requested_tools = [
        call.get("name")
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    executed_tools = [message.name for message in tool_messages]
    final_answers = [
        _as_text(message.content)
        for message in result["messages"]
        if isinstance(message, AIMessage) and not message.tool_calls
    ]

    print(f"\nCASE: {case.name}")
    print(f"Requested: {requested_tools}")
    print(f"Executed: {executed_tools}")
    if final_answers:
        answer = final_answers[-1]
        print(f"Answer preview: {answer[:1200]}")

    if case.expected_tool not in requested_tools:
        raise RuntimeError(
            f"Model did not request expected tool {case.expected_tool}."
        )
    if case.expected_tool not in executed_tools:
        raise RuntimeError(
            f"Expected tool {case.expected_tool} was not executed."
        )
    if case.expect_not_found:
        matching_contents = [
            _as_text(message.content)
            for message in tool_messages
            if message.name == "get_journal_details"
        ]
        if not any(
            '"found": false' in content.casefold()
            or "'found': false" in content.casefold()
            for content in matching_contents
        ):
            raise RuntimeError(
                "Unknown-journal case did not preserve the database not-found result."
            )


def main() -> int:
    """Run all real routing cases sequentially in isolated memory threads."""
    try:
        for case in CASES:
            _run_case(case)
    except Exception:
        print(
            "\nERROR: Agent routing integration failed. Check the DeepSeek "
            "configuration, local journal data, PDF path, and model cache.",
            file=sys.stderr,
        )
        return 1

    print("\nAll five real Agent routing and grounding cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
