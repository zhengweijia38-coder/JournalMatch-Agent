"""Progressive semantic retrieval followed by exact SQLite filtering."""

import logging

from src.journal.repository import count_journals, get_journals_by_ids
from src.retrieval.filters import matches_filters
from src.retrieval.query_builder import build_paper_query
from src.retrieval.semantic_retriever import search_journals
from src.schemas.paper import PaperProfile
from src.schemas.retrieval import HybridCandidate, JournalFilters


logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    filters: JournalFilters | None = None,
    k: int = 10,
    initial_fetch_k: int = 50,
) -> list[HybridCandidate]:
    """Progressively over-fetch semantic candidates and filter current SQLite data."""
    if not query or not query.strip():
        raise ValueError("Hybrid journal query must not be empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if initial_fetch_k <= 0:
        raise ValueError("initial_fetch_k must be a positive integer.")

    active_filters = filters or JournalFilters()
    total_journal_count = count_journals()
    if total_journal_count == 0:
        return []

    logger.info(
        "Starting hybrid retrieval (target=%d, initial_fetch=%d, corpus=%d)",
        k,
        initial_fetch_k,
        total_journal_count,
    )

    fetch_k = min(max(initial_fetch_k, k), total_journal_count)
    seen_journal_ids: set[int] = set()
    filtered_candidates: list[HybridCandidate] = []

    while True:
        semantic_results = search_journals(query, k=fetch_k)
        unseen_results = [
            result
            for result in semantic_results
            if result.journal_id not in seen_journal_ids
        ]
        journals = get_journals_by_ids(
            [result.journal_id for result in unseen_results]
        )
        journals_by_id = {
            journal.journal_id: journal
            for journal in journals
            if journal.journal_id is not None
        }

        for semantic_rank, result in enumerate(semantic_results, start=1):
            if result.journal_id in seen_journal_ids:
                continue
            seen_journal_ids.add(result.journal_id)
            journal = journals_by_id.get(result.journal_id)
            if journal is None or not matches_filters(journal, active_filters):
                continue
            filtered_candidates.append(
                HybridCandidate(
                    journal=journal,
                    semantic_rank=semantic_rank,
                    retrieval_score=result.retrieval_score,
                )
            )

        if len(filtered_candidates) >= k:
            break
        if fetch_k >= total_journal_count or len(semantic_results) < fetch_k:
            break
        fetch_k = min(fetch_k * 2, total_journal_count)

    results = filtered_candidates[:k]
    logger.info("Hybrid retrieval returned %d candidates", len(results))
    return results


def hybrid_search_for_paper(
    profile: PaperProfile,
    filters: JournalFilters | None = None,
    k: int = 10,
    initial_fetch_k: int = 50,
) -> list[HybridCandidate]:
    """Reuse the Phase 3 PaperProfile query and run progressive hybrid retrieval."""
    return hybrid_search(
        query=build_paper_query(profile),
        filters=filters,
        k=k,
        initial_fetch_k=initial_fetch_k,
    )
