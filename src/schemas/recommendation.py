"""Structured, evidence-grounded outputs for Phase 6 recommendations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PaperAssessment(BaseModel):
    """Qualitative paper assessment grounded only in the PaperProfile evidence."""

    model_config = ConfigDict(extra="forbid")

    innovation_level: Literal["incremental", "moderate", "strong"] = Field(
        description="Qualitative innovation level supported by contributions and evidence."
    )
    experimental_completeness: Literal["weak", "moderate", "strong"] = Field(
        description="Qualitative completeness of the reported experimental evidence."
    )
    paper_maturity: Literal["early", "solid", "mature"] = Field(
        description="Qualitative paper maturity based on methods, results, and limitations."
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Specific strengths supported by the PaperProfile.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Specific weaknesses or limitations supported by the PaperProfile.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="PaperProfile facts that justify the qualitative assessment.",
    )


class JournalRecommendation(BaseModel):
    """LLM-generated analysis for one journal already present in the candidates."""

    model_config = ConfigDict(extra="forbid")

    journal_id: int = Field(
        gt=0,
        description="Stable SQLite journal ID copied from the candidate payload.",
    )
    journal_name: str = Field(
        min_length=1,
        description="Journal name copied from the same candidate payload.",
    )
    final_rank: int = Field(
        ge=1,
        description="One-based final recommendation rank within this report.",
    )
    recommendation_tier: Literal["Strong Match", "Good Match", "Backup"] = Field(
        description="Qualitative recommendation tier, not an acceptance likelihood."
    )
    topic_fit: str = Field(
        min_length=1,
        description="Evidence-based explanation of topic alignment.",
    )
    method_fit: str = Field(
        min_length=1,
        description="Evidence-based explanation of method alignment.",
    )
    scope_fit: str = Field(
        min_length=1,
        description="Evidence-based explanation of journal scope alignment.",
    )
    reasons: list[str] = Field(
        min_length=1,
        description="Concrete reasons for recommending this candidate journal.",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Potential fit or maturity concerns supported by supplied evidence.",
    )


class RecommendationReport(BaseModel):
    """Final Phase 6 structured report over a retrieved candidate set."""

    model_config = ConfigDict(extra="forbid")

    paper_assessment: PaperAssessment = Field(
        description="Paper assessment made independently from journal prestige."
    )
    recommendations: list[JournalRecommendation] = Field(
        min_length=1,
        description="Ranked recommendations selected only from the supplied candidates.",
    )
    overall_advice: str = Field(
        min_length=1,
        description="Concise submission-positioning advice without probability claims.",
    )

    @model_validator(mode="after")
    def validate_recommendation_identity_and_ranks(self) -> "RecommendationReport":
        """Require unique journals and a complete one-based rank sequence."""
        journal_ids = [item.journal_id for item in self.recommendations]
        if len(journal_ids) != len(set(journal_ids)):
            raise ValueError("recommendations must not contain duplicate journal_id values.")

        ranks = [item.final_rank for item in self.recommendations]
        if len(ranks) != len(set(ranks)):
            raise ValueError("recommendations must not contain duplicate final_rank values.")
        expected_ranks = set(range(1, len(ranks) + 1))
        if set(ranks) != expected_ranks:
            raise ValueError(
                "final_rank values must form a continuous sequence from 1 to the "
                "number of recommendations."
            )
        return self
