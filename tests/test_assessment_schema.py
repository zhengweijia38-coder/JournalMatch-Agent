"""Validation tests for the evidence-based paper assessment schemas."""

import pytest
from pydantic import ValidationError

from src.schemas.assessment import DimensionAssessment, PaperQualityAssessment


def _dimension(score: int = 3, level: str = "Moderate") -> dict[str, object]:
    return {
        "score": score,
        "level": level,
        "evidence": ["The PaperProfile explicitly reports one supporting result."],
        "concerns": [],
    }


def _assessment_payload() -> dict[str, object]:
    return {
        "novelty": _dimension(),
        "methodology": _dimension(),
        "dataset_quality": _dimension(),
        "experimental_quality": _dimension(),
        "conclusion_support": _dimension(),
        "overall_maturity": "Solid",
        "strengths": [],
        "weaknesses": [],
    }


def test_dimension_score_boundaries_and_level_mapping() -> None:
    """Scores are discrete 1-5 values with one canonical qualitative level."""
    assert DimensionAssessment.model_validate(_dimension(1, "Very Weak")).score == 1
    assert DimensionAssessment.model_validate(_dimension(5, "Very Strong")).score == 5

    for invalid_score in (0, 6):
        with pytest.raises(ValidationError):
            DimensionAssessment.model_validate(
                _dimension(invalid_score, "Moderate")
            )

    with pytest.raises(ValidationError):
        DimensionAssessment.model_validate(_dimension(4, "Excellent"))
    with pytest.raises(ValidationError):
        DimensionAssessment.model_validate(_dimension(4, "Moderate"))


def test_assessment_validation_and_empty_list_serialization() -> None:
    """A complete assessment validates and optional empty lists remain JSON arrays."""
    assessment = PaperQualityAssessment.model_validate(_assessment_payload())
    assert assessment.novelty.evidence
    assert assessment.strengths == []
    assert assessment.weaknesses == []
    serialized = assessment.model_dump(mode="json")
    assert serialized["strengths"] == []
    assert serialized["novelty"]["concerns"] == []

    no_evidence = _assessment_payload()
    no_evidence["dataset_quality"] = {
        **_dimension(),
        "evidence": [],
    }
    with pytest.raises(ValidationError):
        PaperQualityAssessment.model_validate(no_evidence)
