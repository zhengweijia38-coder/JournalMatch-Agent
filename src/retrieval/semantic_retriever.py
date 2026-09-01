"""Semantic Top-K journal retrieval backed by the Chroma index."""

from dataclasses import dataclass
import logging

from src.exceptions import VectorStoreError
from src.retrieval.query_builder import build_paper_query
from src.retrieval.vector_store import (
    close_vector_store,
    get_indexed_count,
    get_vector_store,
)
from src.schemas.paper import PaperProfile


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JournalSearchResult:
    """One Chroma retrieval result with a raw cosine-distance score."""

    journal_id: int
    name: str
    ccf_rank: str | None
    jcr_quartile: str | None
    cas_quartile: str | None
    impact_factor: float | None
    retrieval_score: float


def _optional_text(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value is not None else None


def _optional_float(metadata: dict, key: str) -> float | None:
    value = metadata.get(key)
    return float(value) if value is not None else None


def search_journals(query: str, k: int = 10) -> list[JournalSearchResult]:
    """Return Top-K journals and raw Chroma cosine distances for a text query."""
    if not query or not query.strip():
        raise ValueError("Semantic journal query must not be empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    vector_store = get_vector_store()
    try:
        if get_indexed_count(vector_store) == 0:
            raise VectorStoreError(
                "The journals Chroma collection is empty. Run "
                "'python scripts/build_vector_store.py' first."
            )
        raw_results = vector_store.similarity_search_with_score(query.strip(), k=k)
    finally:
        close_vector_store(vector_store)
    results: list[JournalSearchResult] = []
    for document, score in raw_results:
        metadata = document.metadata
        journal_id = metadata.get("journal_id")
        name = metadata.get("name")
        if journal_id is None or name is None:
            raise VectorStoreError(
                "A Chroma journal Document is missing journal_id or name metadata."
            )
        results.append(
            JournalSearchResult(
                journal_id=int(journal_id),
                name=str(name),
                ccf_rank=_optional_text(metadata, "ccf_rank"),
                jcr_quartile=_optional_text(metadata, "jcr_quartile"),
                cas_quartile=_optional_text(metadata, "cas_quartile"),
                impact_factor=_optional_float(metadata, "impact_factor"),
                retrieval_score=float(score),
            )
        )
    logger.info("Semantic retrieval returned %d journals", len(results))
    return results


def search_journals_for_paper(
    profile: PaperProfile,
    k: int = 10,
) -> list[JournalSearchResult]:
    """Build a deterministic paper query and retrieve Top-K journals."""
    return search_journals(build_paper_query(profile), k=k)
