"""Phase 4 integration tests against the formal SQLite and Chroma data."""

from pathlib import Path
import sys
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.retrieval.hybrid_retriever as hybrid_module
from src.retrieval.semantic_retriever import search_journals
from src.schemas.paper import PaperProfile
from src.schemas.retrieval import JournalFilters


QUERY = (
    "retrieval augmented generation, large language models, natural language "
    "processing, information retrieval"
)


def test_hybrid_retrieval() -> None:
    """Validate real filtering, progressive over-fetch, order, and strict behavior."""
    unfiltered = hybrid_module.hybrid_search(QUERY, k=5)
    assert len(unfiltered) == 5

    ccf_a = hybrid_module.hybrid_search(
        QUERY,
        filters=JournalFilters(ccf_ranks=["A"]),
        k=5,
    )
    assert len(ccf_a) == 5
    assert all(candidate.journal.ccf_rank == "A" for candidate in ccf_a)

    rank_and_jcr_filters = JournalFilters(
        ccf_ranks=["A", "B"],
        jcr_quartiles=["Q1", "Q2"],
    )
    rank_and_jcr = hybrid_module.hybrid_search(
        QUERY,
        filters=rank_and_jcr_filters,
        k=10,
    )
    assert len(rank_and_jcr) == 10
    assert all(
        candidate.journal.ccf_rank in {"A", "B"}
        and candidate.journal.jcr_quartile in {"Q1", "Q2"}
        for candidate in rank_and_jcr
    )

    min_if = hybrid_module.hybrid_search(
        QUERY,
        filters=JournalFilters(min_impact_factor=8.0),
        k=5,
    )
    assert len(min_if) == 5
    assert all(
        candidate.journal.impact_factor is not None
        and candidate.journal.impact_factor >= 8.0
        for candidate in min_if
    )

    impossible = hybrid_module.hybrid_search(
        QUERY,
        filters=JournalFilters(min_impact_factor=1000.0),
        k=10,
    )
    assert impossible == []

    fetch_sizes: list[int] = []
    original_search = hybrid_module.search_journals

    def tracking_search(query: str, k: int):
        fetch_sizes.append(k)
        return original_search(query, k=k)

    with patch.object(
        hybrid_module,
        "search_journals",
        side_effect=tracking_search,
    ):
        progressive = hybrid_module.hybrid_search(
            QUERY,
            filters=JournalFilters(ccf_ranks=["A"]),
            k=5,
            initial_fetch_k=5,
        )
    assert len(progressive) == 5
    assert len(fetch_sizes) > 1
    assert fetch_sizes == sorted(fetch_sizes)

    semantic_results = search_journals(QUERY, k=291)
    semantic_rank_by_id = {
        result.journal_id: rank
        for rank, result in enumerate(semantic_results, start=1)
    }
    semantic_score_by_id = {
        result.journal_id: result.retrieval_score for result in semantic_results
    }
    semantic_ranks = [candidate.semantic_rank for candidate in rank_and_jcr]
    assert semantic_ranks == sorted(semantic_ranks)
    assert all(
        candidate.semantic_rank
        == semantic_rank_by_id[candidate.journal.journal_id]
        and candidate.retrieval_score
        == semantic_score_by_id[candidate.journal.journal_id]
        for candidate in rank_and_jcr
    )
    assert len(
        {candidate.journal.journal_id for candidate in rank_and_jcr}
    ) == len(rank_and_jcr)

    profile = PaperProfile(
        title="Retrieval-Augmented Generation for Scientific Search",
        research_fields=["Information Retrieval", "Natural Language Processing"],
        keywords=["large language models", "retrieval augmented generation"],
        research_problem="Grounding generated text with retrieved documents.",
        methods=["dense retrieval", "transformer language model"],
        summary="A grounded generation method using scientific document retrieval.",
    )
    paper_candidates = hybrid_module.hybrid_search_for_paper(profile, k=5)
    assert len(paper_candidates) == 5

    print(f"Progressive fetch sizes: {fetch_sizes}")
    print("Real SQLite + Chroma hybrid retrieval tests passed.")


if __name__ == "__main__":
    try:
        test_hybrid_retrieval()
    except Exception as exc:
        print(f"ERROR: Hybrid retrieval test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
