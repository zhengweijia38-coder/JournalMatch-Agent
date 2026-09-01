"""High-level full recommendation tool backed by the Phase 8 pipeline."""

from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from src.pipeline import run_recommendation_pipeline
from src.schemas.pipeline import PipelineResult
from src.schemas.retrieval import JournalFilters


class RecommendJournalsInput(BaseModel):
    """Validated inputs for complete PDF-to-journal recommendation."""

    model_config = ConfigDict(extra="forbid")

    pdf_path: str = Field(
        min_length=1,
        description="Path to the specific text-based PDF supplied by the user.",
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
        default=5,
        gt=0,
        description="Maximum number of final recommendations to return.",
    )


def _agent_recommendation_result(result: PipelineResult) -> dict[str, Any]:
    """Return a concise, grounded projection while retaining result internally."""
    report = result.recommendation_report
    if report is None:
        raise RuntimeError("The full pipeline returned no recommendation report.")

    candidates_by_id = {
        candidate.journal.journal_id: candidate
        for candidate in result.reranked_candidates
    }
    recommendations: list[dict[str, Any]] = []
    for recommendation in sorted(
        report.recommendations,
        key=lambda item: item.final_rank,
    ):
        candidate = candidates_by_id.get(recommendation.journal_id)
        if candidate is None:
            raise RuntimeError(
                "A recommendation is missing from the verified candidate set."
            )
        journal = candidate.journal
        recommendations.append(
            {
                "journal_id": journal.journal_id,
                "journal_name": journal.name,
                "final_rank": recommendation.final_rank,
                "recommendation_tier": recommendation.recommendation_tier,
                "ccf_rank": journal.ccf_rank,
                "jcr_quartile": journal.jcr_quartile,
                "cas_quartile": journal.cas_quartile,
                "impact_factor": journal.impact_factor,
                "research_fields": journal.research_fields,
                "keywords": journal.keywords,
                "aims_scope": journal.aims_scope,
                "topic_fit": recommendation.topic_fit,
                "method_fit": recommendation.method_fit,
                "scope_fit": recommendation.scope_fit,
                "reasons": recommendation.reasons,
                "concerns": recommendation.concerns,
            }
        )

    profile = result.paper_profile
    return {
        "ok": True,
        "paper_profile": {
            "title": profile.title,
            "research_fields": profile.research_fields,
            "keywords": profile.keywords,
            "summary": profile.summary,
        },
        "paper_assessment": report.paper_assessment.model_dump(mode="json"),
        "paper_quality_assessment": (
            result.paper_quality_assessment.model_dump(mode="json")
            if result.paper_quality_assessment is not None
            else None
        ),
        "recommendations": recommendations,
        "overall_advice": report.overall_advice,
        "note": (
            "Journal metadata comes from the SQLite-backed candidate objects. "
            "No acceptance probability is provided."
        ),
    }


def _safe_recommendation_error(exc: Exception) -> str:
    """Return a useful failure message without tracebacks or sensitive details."""
    if isinstance(exc, (FileNotFoundError, ValueError, RuntimeError)):
        return str(exc).split("Original error:", maxsplit=1)[0].strip()
    return "Full journal recommendation failed because an unexpected error occurred."


@tool("recommend_journals", args_schema=RecommendJournalsInput)
def recommend_journals_tool(
    pdf_path: str,
    ccf_ranks: list[str] | None = None,
    jcr_quartiles: list[str] | None = None,
    cas_quartiles: list[str] | None = None,
    min_impact_factor: float | None = None,
    max_impact_factor: float | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the complete deterministic pipeline for a user-supplied paper.

    Use this tool when the user asks for journal recommendations for a PDF.
    It reuses the Phase 8 pipeline for paper analysis, hybrid retrieval, strict
    filtering, local reranking, and evidence-grounded final recommendations.
    """
    try:
        filters = JournalFilters(
            ccf_ranks=ccf_ranks or [],
            jcr_quartiles=jcr_quartiles or [],
            cas_quartiles=cas_quartiles or [],
            min_impact_factor=min_impact_factor,
            max_impact_factor=max_impact_factor,
        )
        result = run_recommendation_pipeline(
            pdf_path=pdf_path,
            filters=filters,
            candidate_k=max(20, top_k),
            rerank_k=max(10, top_k),
            recommendation_k=top_k,
        )
        return _agent_recommendation_result(result)
    except Exception as exc:
        return {"ok": False, "error": _safe_recommendation_error(exc)}
