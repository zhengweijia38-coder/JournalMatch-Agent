"""Batch evaluation for Phase 3 semantic journal retrieval."""

from collections.abc import Callable, Sequence
from typing import Any

from src.evaluation.dataset import EvaluationDataset
from src.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


SearchFunction = Callable[..., Sequence[Any]]


def _journal_id(result: Any) -> int:
    if isinstance(result, int):
        return result
    journal_id = getattr(result, "journal_id", None)
    if journal_id is None and hasattr(result, "journal"):
        journal_id = getattr(result.journal, "journal_id", None)
    if journal_id is None:
        raise ValueError("A retrieval result does not contain a journal_id.")
    return int(journal_id)


def _first_relevant_rank(ranking: list[int], gold: dict[int, int]) -> int | None:
    for rank, journal_id in enumerate(ranking, start=1):
        if gold.get(journal_id, 0) > 0:
            return rank
    return None


def evaluate_semantic_retrieval(
    dataset: EvaluationDataset,
    *,
    search_function: SearchFunction | None = None,
    max_k: int = 20,
) -> dict[str, Any]:
    """Evaluate Phase 3 on every case and retain aggregate and failure-analysis data."""
    if max_k < 20:
        raise ValueError("max_k must be at least 20 for the required Recall@20 metric.")
    if search_function is None:
        from src.retrieval.semantic_retriever import search_journals

        search_function = search_journals

    per_case: list[dict[str, Any]] = []
    metric_names = (
        "hit_at_5",
        "hit_at_10",
        "precision_at_5",
        "precision_at_10",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "reciprocal_rank",
        "ndcg_at_5",
        "ndcg_at_10",
    )
    for case in dataset.cases:
        results = search_function(case.query, k=max_k)
        ranking = [_journal_id(result) for result in results]
        gold = case.graded_relevance
        metrics = {
            "hit_at_5": hit_at_k(ranking, gold, 5),
            "hit_at_10": hit_at_k(ranking, gold, 10),
            "precision_at_5": precision_at_k(ranking, gold, 5),
            "precision_at_10": precision_at_k(ranking, gold, 10),
            "recall_at_5": recall_at_k(ranking, gold, 5),
            "recall_at_10": recall_at_k(ranking, gold, 10),
            "recall_at_20": recall_at_k(ranking, gold, 20),
            "reciprocal_rank": reciprocal_rank(ranking, gold),
            "ndcg_at_5": ndcg_at_k(ranking, gold, 5),
            "ndcg_at_10": ndcg_at_k(ranking, gold, 10),
        }
        per_case.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "gold_journals": gold,
                "retrieved_top_k": ranking[:max_k],
                "first_relevant_rank": _first_relevant_rank(ranking, gold),
                "metrics": metrics,
            }
        )

    aggregate = {
        name: sum(case["metrics"][name] for case in per_case) / len(per_case)
        for name in metric_names
    }
    aggregate["mrr"] = aggregate.pop("reciprocal_rank")
    return {
        "case_count": len(per_case),
        "max_k": max_k,
        "aggregate": aggregate,
        "per_case": per_case,
    }
