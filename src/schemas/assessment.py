"""Rubric-constrained, evidence-based paper quality assessment schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AssessmentLevel = Literal[
    "Very Weak",
    "Weak",
    "Moderate",
    "Strong",
    "Very Strong",
]
OverallMaturity = Literal["Early", "Developing", "Solid", "Mature", "Strong"]

_LEVEL_BY_SCORE: dict[int, str] = {
    1: "Very Weak",
    2: "Weak",
    3: "Moderate",
    4: "Strong",
    5: "Very Strong",
}


class DimensionAssessment(BaseModel):
    """One quality dimension scored against an explicit five-level rubric."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(
        ge=1,
        le=5,
        description="Discrete rubric score from 1 (Very Weak) to 5 (Very Strong).",
    )
    level: AssessmentLevel = Field(
        description="Human-readable level that must exactly match the rubric score."
    )
    evidence: list[str] = Field(
        min_length=1,
        description=(
            "Specific PaperProfile facts supporting the score; use "
            "'Insufficient evidence.' when the profile lacks the required facts."
        ),
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Evidence-grounded limitations, missing support, or uncertainty.",
    )

    @field_validator("evidence", "concerns")
    @classmethod
    def reject_blank_items(cls, values: list[str]) -> list[str]:
        """Prevent apparently populated evidence lists containing blank strings."""
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("evidence and concerns must not contain blank strings.")
        return cleaned

    @model_validator(mode="after")
    def validate_level_matches_score(self) -> "DimensionAssessment":
        """Keep numeric and qualitative rubric representations consistent."""
        expected_level = _LEVEL_BY_SCORE[self.score]
        if self.level != expected_level:
            raise ValueError(
                f"level must be '{expected_level}' when score is {self.score}."
            )
        return self


class PaperQualityAssessment(BaseModel):
    """Five-dimensional assessment grounded only in a supplied PaperProfile."""

    model_config = ConfigDict(extra="forbid")

    novelty: DimensionAssessment = Field(
        description="Originality and demonstrated distinction from existing work."
    )
    methodology: DimensionAssessment = Field(
        description="Coherence, completeness, and rigor of the technical design."
    )
    dataset_quality: DimensionAssessment = Field(
        description="Adequacy and validity of the reported data usage for the task."
    )
    experimental_quality: DimensionAssessment = Field(
        description="Completeness and rigor of the reported experimental validation."
    )
    conclusion_support: DimensionAssessment = Field(
        description="Degree to which reported evidence supports the paper's conclusions."
    )
    overall_maturity: OverallMaturity = Field(
        description=(
            "Holistic maturity judgment based on the five evidence-bearing dimensions; "
            "it is not a numeric average or acceptance prediction."
        )
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Cross-dimensional strengths directly supported by the profile.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Cross-dimensional weaknesses or evidence gaps in the profile.",
    )

    @field_validator("strengths", "weaknesses")
    @classmethod
    def reject_blank_summary_items(cls, values: list[str]) -> list[str]:
        """Keep assessment summaries non-blank when supplied."""
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("strengths and weaknesses must not contain blank strings.")
        return cleaned
