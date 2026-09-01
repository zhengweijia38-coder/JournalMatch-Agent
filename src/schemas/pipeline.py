"""Structured result and timing models for the Phase 8 pipeline."""

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.assessment import PaperQualityAssessment
from src.schemas.paper import PaperProfile
from src.schemas.recommendation import RecommendationReport
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


class PipelineTimings(BaseModel):
    """Wall-clock time spent in each end-to-end pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    pdf_loading_seconds: float = Field(ge=0, description="PDF loading and text-combination time.")
    paper_analysis_seconds: float = Field(ge=0, description="DeepSeek paper-analysis time.")
    paper_assessment_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Evidence-based paper quality assessment time.",
    )
    hybrid_retrieval_seconds: float = Field(ge=0, description="Hybrid retrieval time.")
    reranking_seconds: float = Field(ge=0, description="Local cross-encoder reranking time.")
    recommendation_seconds: float = Field(
        ge=0,
        description="DeepSeek recommendation time; zero when recommendation is skipped.",
    )
    total_seconds: float = Field(ge=0, description="Total pipeline wall-clock time.")


class PipelineResult(BaseModel):
    """All inspectable intermediate and final outputs from one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    paper_profile: PaperProfile = Field(description="Structured profile extracted from the PDF.")
    paper_quality_assessment: PaperQualityAssessment | None = Field(
        default=None,
        description=(
            "Five-dimensional rubric assessment, or None only when assessment was "
            "explicitly omitted by a compatibility/debug caller."
        ),
    )
    hybrid_candidates: list[HybridCandidate] = Field(
        default_factory=list,
        description="Candidates returned by semantic retrieval and exact filtering.",
    )
    reranked_candidates: list[RerankedCandidate] = Field(
        default_factory=list,
        description="Hybrid candidates reordered by the local cross-encoder.",
    )
    recommendation_report: RecommendationReport | None = Field(
        default=None,
        description="Final recommendation report, or None when explicitly skipped.",
    )
    timings: PipelineTimings | None = Field(
        default=None,
        description="Optional wall-clock timings for the complete run.",
    )
