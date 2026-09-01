"""Print real Phase 4 candidate sets under progressively stricter filters."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import hybrid_search
from src.schemas.retrieval import HybridCandidate, JournalFilters


QUERY = (
    "Retrieval-Augmented Generation, Large Language Models, Natural Language "
    "Processing, Information Retrieval"
)

CASES = [
    ("A. No structured filters", JournalFilters()),
    ("B. CCF A/B", JournalFilters(ccf_ranks=["A", "B"])),
    (
        "C. CCF A/B + JCR Q1/Q2",
        JournalFilters(
            ccf_ranks=["A", "B"],
            jcr_quartiles=["Q1", "Q2"],
        ),
    ),
    (
        "D. CCF A/B + JCR Q1/Q2 + CAS 1/2 + IF >= 5",
        JournalFilters(
            ccf_ranks=["A", "B"],
            jcr_quartiles=["Q1", "Q2"],
            cas_quartiles=["1", "2"],
            min_impact_factor=5.0,
        ),
    ),
]


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def print_candidates(title: str, candidates: list[HybridCandidate]) -> None:
    print(f"\n{title}")
    print(
        "Rank | Journal Name | CCF | JCR | CAS | Impact Factor | "
        "Semantic Rank | Raw Retrieval Score"
    )
    if not candidates:
        print("No journals satisfy all filters.")
        return

    for rank, candidate in enumerate(candidates, start=1):
        journal = candidate.journal
        line = (
            f"{rank} | {journal.name} | {journal.ccf_rank or '-'} | "
            f"{journal.jcr_quartile or '-'} | {journal.cas_quartile or '-'} | "
            f"{journal.impact_factor if journal.impact_factor is not None else '-'} | "
            f"{candidate.semantic_rank} | "
            f"{candidate.retrieval_score:.6f}"
        )
        print(_safe_console_text(line))


def main() -> int:
    try:
        for title, filters in CASES:
            candidates = hybrid_search(QUERY, filters=filters, k=10)
            print_candidates(title, candidates)
    except Exception as exc:
        print(f"ERROR: Hybrid retrieval manual test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
