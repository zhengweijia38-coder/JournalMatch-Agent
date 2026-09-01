"""Before/after evaluation for Phase 5 cross-encoder reranking."""

from collections.abc import Callable, Sequence
from typing import Any

from src.evaluation.dataset import EvaluationDataset
from src.evaluation.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


RetrieveFunction = Callable[..., Sequence[HybridCandidate]]
RerankFunction = Callable[..., Sequence[RerankedCandidate]]


def _candidate_id(candidate: HybridCandidate | RerankedCandidate) -> int:
    journal_id = candidate.journal.journal_id
    if journal_id is None:
        raise ValueError(f"Candidate journal '{candidate.journal.name}' has no journal_id.")
    return journal_id


def _ranking_metrics(ranking: list[int], gold: dict[int, int]) -> dict[str, float]:
    return {
        "hit_at_5": hit_at_k(ranking, gold, 5),
        "mrr": reciprocal_rank(ranking, gold),
        "ndcg_at_5": ndcg_at_k(ranking, gold, 5),
        "ndcg_at_10": ndcg_at_k(ranking, gold, 10),
    }


def _first_relevant_rank(ranking: list[int], gold: dict[int, int]) -> int | None:
    for rank, journal_id in enumerate(ranking, start=1):
        if gold.get(journal_id, 0) > 0:
            return rank
    return None


def evaluate_reranking(
    dataset: EvaluationDataset,
    *,
    retrieve_function: RetrieveFunction | None = None,
    rerank_function: RerankFunction | None = None,
    candidate_k: int = 20,
) -> dict[str, Any]:
    """Compare identical case queries before and after reranking without optimism."""
    if candidate_k < 10:
        raise ValueError("candidate_k must be at least 10 for nDCG@10.")
    if retrieve_function is None:
        from src.retrieval.hybrid_retriever import hybrid_search

        retrieve_function = hybrid_search
    if rerank_function is None:
        from src.retrieval.reranker import rerank_candidates

        rerank_function = rerank_candidates

    per_case: list[dict[str, Any]] = []
    metric_names = ("hit_at_5", "mrr", "ndcg_at_5", "ndcg_at_10")
    for case in dataset.cases:
        candidates = list(
            retrieve_function(
                case.query,
                filters=case.filters,
                k=candidate_k,
            )
        )
        before_ranking = [_candidate_id(candidate) for candidate in candidates]
        reranked = list(
            rerank_function(
                case.query,
                candidates,
                top_k=candidate_k,
            )
        )
        after_ranking = [_candidate_id(candidate) for candidate in reranked]
        before = _ranking_metrics(before_ranking, case.graded_relevance)
        after = _ranking_metrics(after_ranking, case.graded_relevance)
        delta = {name: after[name] - before[name] for name in metric_names}
        per_case.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "gold_journals": case.graded_relevance,
                "before_top_k": before_ranking,
                "after_top_k": after_ranking,
                "first_relevant_rank_before": _first_relevant_rank(
                    before_ranking, case.graded_relevance
                ),
                "first_relevant_rank_after": _first_relevant_rank(
                    after_ranking, case.graded_relevance
                ),
                "before": before,
                "after": after,
                "delta": delta,
            }
        )

    before_aggregate = {
        name: sum(case["before"][name] for case in per_case) / len(per_case)
        for name in metric_names
    }
    after_aggregate = {
        name: sum(case["after"][name] for case in per_case) / len(per_case)
        for name in metric_names
    }
    delta_aggregate = {
        name: after_aggregate[name] - before_aggregate[name]
        for name in metric_names
    }
    return {
        "case_count": len(per_case),
        "candidate_k": candidate_k,
        "before": before_aggregate,
        "after": after_aggregate,
        "delta": delta_aggregate,
        "per_case": per_case,
    }
