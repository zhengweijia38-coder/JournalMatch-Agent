"""Cross-encoder reranking for Phase 4 hybrid journal candidates."""

from collections.abc import Sequence
import logging
from numbers import Real

from src.journal.document_builder import journal_to_document
from src.exceptions import RerankerError
from src.models.reranker import get_reranker
from src.retrieval.query_builder import build_paper_query
from src.schemas.paper import PaperProfile
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


RERANK_MAX_LENGTH = 512
RERANK_BATCH_SIZE = 16
logger = logging.getLogger(__name__)


def _validate_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer or None.")


def _as_float_scores(raw_scores: Real | Sequence[Real]) -> list[float]:
    """Normalize FlagEmbedding's scalar/list return shapes to plain floats."""
    if isinstance(raw_scores, Real):
        return [float(raw_scores)]
    return [float(score) for score in raw_scores]


def rerank_candidates(
    query: str,
    candidates: list[HybridCandidate],
    top_k: int | None = None,
) -> list[RerankedCandidate]:
    """Score query/passage pairs in one batch and return descending raw scores."""
    if not query or not query.strip():
        raise ValueError("Reranker query must not be empty.")
    _validate_top_k(top_k)
    if not candidates:
        return []

    pairs = [
        [query.strip(), journal_to_document(candidate.journal).page_content]
        for candidate in candidates
    ]
    raw_scores = get_reranker().compute_score(
        pairs,
        batch_size=min(RERANK_BATCH_SIZE, len(pairs)),
        max_length=RERANK_MAX_LENGTH,
        normalize=False,
    )
    scores = _as_float_scores(raw_scores)
    if len(scores) != len(candidates):
        raise RerankerError(
            "Reranker returned a different number of scores than input candidates."
        )

    ranked_items = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    if top_k is not None:
        ranked_items = ranked_items[:top_k]

    results = [
        RerankedCandidate(
            journal=candidate.journal,
            semantic_rank=candidate.semantic_rank,
            retrieval_score=candidate.retrieval_score,
            rerank_rank=rerank_rank,
            rerank_score=score,
        )
        for rerank_rank, (candidate, score) in enumerate(ranked_items, start=1)
    ]
    logger.info("Reranked %d candidates into %d results", len(candidates), len(results))
    return results


def rerank_candidates_for_paper(
    profile: PaperProfile,
    candidates: list[HybridCandidate],
    top_k: int | None = None,
) -> list[RerankedCandidate]:
    """Build the existing PaperProfile query and rerank hybrid candidates."""
    return rerank_candidates(
        query=build_paper_query(profile),
        candidates=candidates,
        top_k=top_k,
    )
