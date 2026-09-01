"""Manual Phase 3 retrieval checks against the formal journals collection."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.semantic_retriever import (
    JournalSearchResult,
    search_journals,
    search_journals_for_paper,
)
from src.schemas.paper import PaperProfile


QUERIES = [
    "deep learning, computer vision, image segmentation, image recognition",
    (
        "retrieval augmented generation, large language models, natural language "
        "processing, information retrieval"
    ),
    "software testing, program analysis, bug detection, software engineering",
]


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_results(query: str, results: list[JournalSearchResult]) -> None:
    print(f"\nQuery: {query}")
    print("Rank | Journal Name | CCF | JCR | CAS | Impact Factor | Raw Retrieval Score")
    for rank, result in enumerate(results, start=1):
        line = (
            f"{rank} | {result.name} | {result.ccf_rank or '-'} | "
            f"{result.jcr_quartile or '-'} | {result.cas_quartile or '-'} | "
            f"{result.impact_factor if result.impact_factor is not None else '-'} | "
            f"{result.retrieval_score:.6f}"
        )
        print(_safe_console_text(line))


def test_semantic_retrieval() -> None:
    """Run three field queries plus a PaperProfile-to-Chroma retrieval."""
    top_name_sets: list[set[str]] = []
    for query in QUERIES:
        results = search_journals(query, k=5)
        assert len(results) == 5
        _print_results(query, results)
        top_name_sets.append({result.name for result in results})

    assert len({frozenset(names) for names in top_name_sets}) == len(QUERIES)

    profile = PaperProfile(
        title="Grounded Large Language Models for Scientific Question Answering",
        keywords=["retrieval augmented generation", "scientific search"],
        research_fields=["Information Retrieval", "Natural Language Processing"],
        research_problem="Grounding generated answers in retrieved scientific papers.",
        methods=["dense retrieval", "large language model"],
        summary="A retrieval-augmented method for grounded scientific question answering.",
    )
    paper_results = search_journals_for_paper(profile, k=5)
    assert len(paper_results) == 5
    _print_results("PaperProfile semantic query", paper_results)

    print("Semantic retrieval query tests passed.")


if __name__ == "__main__":
    try:
        test_semantic_retrieval()
    except Exception as exc:
        print(f"ERROR: Semantic retrieval test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
