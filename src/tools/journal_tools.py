"""Read-only journal lookup and topic-search tools."""

from difflib import get_close_matches
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings
from src.journal.repository import get_all_journals, get_journal_by_name
from src.pipeline import _ensure_runtime_data_ready, _ensure_sqlite_is_ready
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank_candidates
from src.schemas.journal import Journal
from src.schemas.retrieval import JournalFilters, RerankedCandidate


class JournalDetailsInput(BaseModel):
    """Validated input for one journal metadata lookup."""

    model_config = ConfigDict(extra="forbid")

    journal_name: str = Field(
        min_length=1,
        description="Full or likely journal name to look up in the local SQLite database.",
    )


class SearchJournalsInput(BaseModel):
    """Validated topic query and optional structured constraints."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="Research topic, keywords, methods, or scope used for journal search.",
    )
    ccf_ranks: list[str] | None = Field(
        default=None,
        description="Allowed CCF ranks such as A or B; null means unrestricted.",
    )
    jcr_quartiles: list[str] | None = Field(
        default=None,
        description="Allowed JCR quartiles such as Q1 or Q2; null means unrestricted.",
    )
    cas_quartiles: list[str] | None = Field(
        default=None,
        description="Allowed CAS zones such as 1 or 2; null means unrestricted.",
    )
    min_impact_factor: float | None = Field(
        default=None,
        description="Inclusive minimum impact factor, or null when unrestricted.",
    )
    max_impact_factor: float | None = Field(
        default=None,
        description="Inclusive maximum impact factor, or null when unrestricted.",
    )
    top_k: int = Field(
        default=10,
        gt=0,
        description="Maximum number of reranked journals to return.",
    )


def _journal_facts(journal: Journal) -> dict[str, Any]:
    """Return only SQLite-backed journal facts required by the agent."""
    return {
        "journal_id": journal.journal_id,
        "name": journal.name,
        "abbreviation": journal.abbreviation,
        "research_fields": journal.research_fields,
        "keywords": journal.keywords,
        "aims_scope": journal.aims_scope,
        "ccf_rank": journal.ccf_rank,
        "jcr_quartile": journal.jcr_quartile,
        "cas_quartile": journal.cas_quartile,
        "impact_factor": journal.impact_factor,
        "homepage": journal.homepage,
        "updated_at": journal.updated_at,
    }


def _reranked_result(candidate: RerankedCandidate) -> dict[str, Any]:
    """Combine ranking provenance with current SQLite facts."""
    return {
        **_journal_facts(candidate.journal),
        "semantic_rank": candidate.semantic_rank,
        "retrieval_score": candidate.retrieval_score,
        "rerank_rank": candidate.rerank_rank,
        "rerank_score": candidate.rerank_score,
    }


def _safe_journal_error(exc: Exception, operation: str) -> str:
    """Describe expected local failures without exposing environment details."""
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, RuntimeError):
        return str(exc).split("Original error:", maxsplit=1)[0].strip()
    return f"{operation} failed because an unexpected local error occurred."


@tool("get_journal_details", args_schema=JournalDetailsInput)
def get_journal_details_tool(journal_name: str) -> dict[str, Any]:
    """Look up authoritative metadata for a named journal in local SQLite.

    Use this tool whenever the user asks about a journal's CCF rank, JCR
    quartile, CAS quartile, impact factor, research fields, keywords, or scope.
    Never fill missing fields from model memory.
    """
    try:
        query = journal_name.strip()
        _ensure_sqlite_is_ready(get_settings().sqlite_db_path)
        journal = get_journal_by_name(query)
        if journal is not None:
            return {"ok": True, "found": True, "journal": _journal_facts(journal)}

        journals = get_all_journals()
        names = [item.name for item in journals]
        query_casefold = query.casefold()
        partial_matches = [
            name for name in names if query_casefold in name.casefold()
        ][:5]
        candidate_names = partial_matches or get_close_matches(
            query,
            names,
            n=5,
            cutoff=0.55,
        )
        return {
            "ok": True,
            "found": False,
            "query": query,
            "candidate_names": candidate_names,
            "message": (
                "No exact journal name was found in the local SQLite database. "
                "Do not invent journal metadata."
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": _safe_journal_error(exc, "Journal lookup"),
        }


@tool("search_journals", args_schema=SearchJournalsInput)
def search_journals_tool(
    query: str,
    ccf_ranks: list[str] | None = None,
    jcr_quartiles: list[str] | None = None,
    cas_quartiles: list[str] | None = None,
    min_impact_factor: float | None = None,
    max_impact_factor: float | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search and rerank journals by topic and optional exact constraints.

    Use this tool when the user wants to discover journals by topic, keywords,
    methods, CCF/JCR/CAS constraints, or impact factor without supplying a PDF
    for a full recommendation. It performs Phase 4 hybrid retrieval followed
    by Phase 5 local reranking; it does not run final DeepSeek recommendation.
    """
    try:
        _ensure_runtime_data_ready()
        filters = JournalFilters(
            ccf_ranks=ccf_ranks or [],
            jcr_quartiles=jcr_quartiles or [],
            cas_quartiles=cas_quartiles or [],
            min_impact_factor=min_impact_factor,
            max_impact_factor=max_impact_factor,
        )
        candidate_k = max(20, top_k * 2)
        hybrid_candidates = hybrid_search(
            query=query,
            filters=filters,
            k=candidate_k,
            initial_fetch_k=max(50, candidate_k),
        )
        if not hybrid_candidates:
            return {
                "ok": True,
                "count": 0,
                "journals": [],
                "message": (
                    "No journals satisfy the current topic and filters. "
                    "The filters were not relaxed."
                ),
            }

        reranked = rerank_candidates(
            query=query,
            candidates=hybrid_candidates,
            top_k=min(top_k, len(hybrid_candidates)),
        )
        return {
            "ok": True,
            "count": len(reranked),
            "journals": [_reranked_result(item) for item in reranked],
            "note": (
                "retrieval_score and rerank_score are ranking signals, not "
                "acceptance probabilities."
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": _safe_journal_error(exc, "Journal search"),
        }
