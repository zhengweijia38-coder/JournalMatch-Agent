"""Phase 8 end-to-end orchestration for journal recommendation."""

from pathlib import Path
import logging
import sqlite3
from time import perf_counter

import chromadb

from src.assessment.assessor import assess_paper_quality
from src.config import get_settings
from src.exceptions import JournalDatabaseError, RetrievalError, VectorStoreError
from src.paper.analyzer import analyze_paper
from src.paper.loader import combine_documents, load_pdf
from src.recommendation.recommender import generate_recommendations
from src.retrieval.hybrid_retriever import hybrid_search_for_paper
from src.retrieval.reranker import rerank_candidates_for_paper
from src.retrieval.vector_store import COLLECTION_NAME
from src.schemas.pipeline import PipelineResult, PipelineTimings
from src.schemas.retrieval import JournalFilters


logger = logging.getLogger(__name__)


class NoMatchingJournalsError(RetrievalError):
    """Raised when strict retrieval filters leave no journal candidates."""


def _validate_positive_integer(name: str, value: int) -> None:
    """Reject booleans, non-integers, and non-positive stage limits."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _ensure_sqlite_is_ready(database_path: Path) -> None:
    """Inspect the journal database read-only so this check cannot create it."""
    if not database_path.is_file():
        raise JournalDatabaseError(
            "Journal database is not available. Run:\n"
            "python scripts/init_journals.py data/journals/journals.xlsx"
        )

    connection: sqlite3.Connection | None = None
    try:
        database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True)
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'journals'"
        ).fetchone()
        if table is None:
            raise JournalDatabaseError(
                "Journal database does not contain the journals table. Run:\n"
                "python scripts/init_journals.py data/journals/journals.xlsx"
            )
        journal_count = int(
            connection.execute("SELECT COUNT(*) FROM journals").fetchone()[0]
        )
        if journal_count == 0:
            raise JournalDatabaseError(
                "Journal database contains no journals. Run:\n"
                "python scripts/init_journals.py data/journals/journals.xlsx"
            )
    except JournalDatabaseError:
        raise
    except sqlite3.Error as exc:
        raise JournalDatabaseError(
            f"Journal database could not be read: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


def _ensure_chroma_is_ready(persist_directory: Path) -> None:
    """Open an existing Chroma collection without creating or rebuilding one."""
    build_message = (
        "Journal vector index is not available. Run:\n"
        "python scripts/build_vector_store.py"
    )
    if not persist_directory.is_dir() or not any(persist_directory.iterdir()):
        raise VectorStoreError(build_message)

    try:
        client = chromadb.PersistentClient(path=str(persist_directory))
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            raise VectorStoreError(build_message)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(build_message) from exc


def _ensure_runtime_data_ready() -> None:
    """Verify that Phase 2 and Phase 3 artifacts exist before model work begins."""
    settings = get_settings()
    _ensure_sqlite_is_ready(settings.sqlite_db_path)
    _ensure_chroma_is_ready(settings.chroma_persist_dir)


def run_recommendation_pipeline(
    pdf_path: str | Path,
    filters: JournalFilters | None = None,
    candidate_k: int = 20,
    rerank_k: int = 10,
    recommendation_k: int = 5,
    skip_recommendation: bool = False,
) -> PipelineResult:
    """Run PDF analysis, hybrid retrieval, reranking, and optional recommendation."""
    _validate_positive_integer("candidate_k", candidate_k)
    _validate_positive_integer("rerank_k", rerank_k)
    _validate_positive_integer("recommendation_k", recommendation_k)
    if rerank_k > candidate_k:
        raise ValueError("rerank_k must not exceed candidate_k.")
    if not skip_recommendation and recommendation_k > rerank_k:
        raise ValueError(
            "recommendation_k must not exceed rerank_k when recommendation is enabled."
        )

    total_started = perf_counter()
    logger.info(
        "Starting recommendation pipeline (candidate_k=%d, rerank_k=%d, "
        "recommendation_k=%d, skip_recommendation=%s)",
        candidate_k,
        rerank_k,
        recommendation_k,
        skip_recommendation,
    )

    stage_started = perf_counter()
    documents = load_pdf(pdf_path)
    paper_text = combine_documents(documents)
    pdf_loading_seconds = perf_counter() - stage_started
    logger.debug("PDF loading completed in %.3fs", pdf_loading_seconds)

    # Check local artifacts before making the first paid API call. The check is
    # deliberately read-only and never creates an empty database or Chroma index.
    _ensure_runtime_data_ready()

    stage_started = perf_counter()
    paper_profile = analyze_paper(paper_text)
    paper_analysis_seconds = perf_counter() - stage_started
    logger.info("Paper analysis completed in %.3fs", paper_analysis_seconds)

    stage_started = perf_counter()
    paper_quality_assessment = assess_paper_quality(paper_profile)
    paper_assessment_seconds = perf_counter() - stage_started
    logger.info(
        "Paper quality assessment completed in %.3fs",
        paper_assessment_seconds,
    )

    stage_started = perf_counter()
    hybrid_candidates = hybrid_search_for_paper(
        paper_profile,
        filters=filters,
        k=candidate_k,
        initial_fetch_k=max(50, candidate_k),
    )
    hybrid_retrieval_seconds = perf_counter() - stage_started
    logger.info(
        "Hybrid retrieval returned %d candidates in %.3fs",
        len(hybrid_candidates),
        hybrid_retrieval_seconds,
    )
    if not hybrid_candidates:
        raise NoMatchingJournalsError(
            "No journals satisfy the current retrieval and filter conditions. "
            "The supplied filters were not relaxed."
        )

    stage_started = perf_counter()
    reranked_candidates = rerank_candidates_for_paper(
        paper_profile,
        hybrid_candidates,
        top_k=min(rerank_k, len(hybrid_candidates)),
    )
    reranking_seconds = perf_counter() - stage_started
    logger.info(
        "Reranking returned %d candidates in %.3fs",
        len(reranked_candidates),
        reranking_seconds,
    )
    if not reranked_candidates:
        raise NoMatchingJournalsError(
            "Reranking produced no journal candidates, so recommendation was not run."
        )

    recommendation_report = None
    recommendation_seconds = 0.0
    if not skip_recommendation:
        stage_started = perf_counter()
        recommendation_report = generate_recommendations(
            paper_profile,
            reranked_candidates,
            top_k=min(recommendation_k, len(reranked_candidates)),
            quality_assessment=paper_quality_assessment,
        )
        recommendation_seconds = perf_counter() - stage_started
        logger.info("Recommendation completed in %.3fs", recommendation_seconds)

    timings = PipelineTimings(
        pdf_loading_seconds=pdf_loading_seconds,
        paper_analysis_seconds=paper_analysis_seconds,
        paper_assessment_seconds=paper_assessment_seconds,
        hybrid_retrieval_seconds=hybrid_retrieval_seconds,
        reranking_seconds=reranking_seconds,
        recommendation_seconds=recommendation_seconds,
        total_seconds=perf_counter() - total_started,
    )
    result = PipelineResult(
        paper_profile=paper_profile,
        paper_quality_assessment=paper_quality_assessment,
        hybrid_candidates=hybrid_candidates,
        reranked_candidates=reranked_candidates,
        recommendation_report=recommendation_report,
        timings=timings,
    )
    logger.info("Recommendation pipeline completed in %.3fs", timings.total_seconds)
    return result
