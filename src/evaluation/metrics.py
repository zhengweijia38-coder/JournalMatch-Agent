"""Small, dependency-free information retrieval metrics with graded relevance."""

from collections.abc import Hashable, Mapping, Sequence
import math
from typing import TypeVar


Identifier = TypeVar("Identifier", bound=Hashable)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")


def _relevant_ids(gold: Mapping[Identifier, int]) -> set[Identifier]:
    return {identifier for identifier, grade in gold.items() if grade > 0}


def hit_at_k(
    ranked_ids: Sequence[Identifier],
    gold: Mapping[Identifier, int],
    k: int,
) -> float:
    """Return 1 when Top-K contains at least one positive-grade gold item, else 0."""
    _validate_k(k)
    relevant = _relevant_ids(gold)
    return float(bool(relevant.intersection(ranked_ids[:k])))


def precision_at_k(
    ranked_ids: Sequence[Identifier],
    gold: Mapping[Identifier, int],
    k: int,
) -> float:
    """Return the fraction of K ranking positions occupied by labeled relevant items."""
    _validate_k(k)
    relevant = _relevant_ids(gold)
    hits = sum(identifier in relevant for identifier in ranked_ids[:k])
    return hits / k


def recall_at_k(
    ranked_ids: Sequence[Identifier],
    gold: Mapping[Identifier, int],
    k: int,
) -> float:
    """Return the fraction of all labeled relevant items recovered in Top-K."""
    _validate_k(k)
    relevant = _relevant_ids(gold)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def reciprocal_rank(
    ranked_ids: Sequence[Identifier],
    gold: Mapping[Identifier, int],
    k: int | None = None,
) -> float:
    """Return inverse rank of the first labeled relevant result, or 0 when absent."""
    if k is not None:
        _validate_k(k)
    relevant = _relevant_ids(gold)
    ranking = ranked_ids if k is None else ranked_ids[:k]
    for rank, identifier in enumerate(ranking, start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def mrr(
    rankings: Sequence[Sequence[Identifier]],
    gold_sets: Sequence[Mapping[Identifier, int]],
    k: int | None = None,
) -> float:
    """Return mean reciprocal rank across aligned rankings and gold mappings."""
    if len(rankings) != len(gold_sets):
        raise ValueError("rankings and gold_sets must have the same length.")
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(ranking, gold, k=k)
        for ranking, gold in zip(rankings, gold_sets, strict=True)
    ) / len(rankings)


def dcg_at_k(
    ranked_ids: Sequence[Identifier],
    gold: Mapping[Identifier, int],
    k: int,
) -> float:
    """Compute graded DCG@K using (2^relevance - 1) / log2(rank + 1)."""
    _validate_k(k)
    return sum(
        (2 ** max(gold.get(identifier, 0), 0) - 1) / math.log2(rank + 1)
        for rank, identifier in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(
    ranked_ids: Sequence[Identifier],
    gold: Mapping[Identifier, int],
    k: int,
) -> float:
    """Normalize graded DCG@K by the ideal ordering of all supplied gold grades."""
    _validate_k(k)
    ideal_grades = sorted((grade for grade in gold.values() if grade > 0), reverse=True)
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades[:k], start=1)
    )
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, gold, k) / ideal_dcg
