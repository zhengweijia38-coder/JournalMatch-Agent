"""Hard-constraint leakage evaluation for Phase 4 hybrid retrieval."""

from collections.abc import Callable, Sequence
from typing import Any

from src.evaluation.dataset import EvaluationDataset
from src.retrieval.filters import matches_filters
from src.schemas.retrieval import HybridCandidate, JournalFilters


HybridSearchFunction = Callable[..., Sequence[HybridCandidate]]


def _has_constraints(filters: JournalFilters) -> bool:
    return bool(
        filters.ccf_ranks
        or filters.jcr_quartiles
        or filters.cas_quartiles
        or filters.min_impact_factor is not None
        or filters.max_impact_factor is not None
    )


def evaluate_hybrid_filtering(
    dataset: EvaluationDataset,
    *,
    search_function: HybridSearchFunction | None = None,
    k: int = 20,
) -> dict[str, Any]:
    """Measure whether every returned item obeys each case's structured filters."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if search_function is None:
        from src.retrieval.hybrid_retriever import hybrid_search

        search_function = hybrid_search

    per_case: list[dict[str, Any]] = []
    total_results = 0
    satisfied_results = 0
    leakage_count = 0
    for case in dataset.cases:
        if not _has_constraints(case.filters):
            continue
        results = list(search_function(case.query, filters=case.filters, k=k))
        leaked_ids = [
            candidate.journal.journal_id
            for candidate in results
            if not matches_filters(candidate.journal, case.filters)
        ]
        case_satisfied = len(results) - len(leaked_ids)
        total_results += len(results)
        satisfied_results += case_satisfied
        leakage_count += len(leaked_ids)
        per_case.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "requested_k": k,
                "returned_count": len(results),
                "returned_journal_ids": [
                    candidate.journal.journal_id for candidate in results
                ],
                "filter_leakage_count": len(leaked_ids),
                "leaked_journal_ids": leaked_ids,
                "constraint_satisfaction_rate": (
                    case_satisfied / len(results) if results else 1.0
                ),
            }
        )

    return {
        "evaluated_case_count": len(per_case),
        "returned_result_count": total_results,
        "satisfied_result_count": satisfied_results,
        "filter_leakage_count": leakage_count,
        "constraint_satisfaction_rate": (
            satisfied_results / total_results if total_results else 1.0
        ),
        "per_case": per_case,
    }
