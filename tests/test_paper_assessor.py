"""Offline structured-output tests for the PaperProfile assessor."""

from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import src.assessment.assessor as assessor_module
from src.schemas.assessment import PaperQualityAssessment
from src.schemas.paper import PaperProfile


def _response() -> dict[str, object]:
    dimension = {
        "score": 3,
        "level": "Moderate",
        "evidence": ["The profile reports results on one named benchmark."],
        "concerns": ["The profile contains no statistical significance test."],
    }
    return {
        "novelty": dimension,
        "methodology": dimension,
        "dataset_quality": dimension,
        "experimental_quality": dimension,
        "conclusion_support": dimension,
        "overall_maturity": "Solid",
        "strengths": ["The method and reported result are explicit."],
        "weaknesses": ["Several validation details are absent from the profile."],
    }


class FakeAssessmentLLM:
    """Return deterministic structured data through the LangChain Runnable API."""

    def __init__(self, response: object | None = None) -> None:
        self.rendered_prompt = ""
        self.response = _response() if response is None else response
        self.calls = 0

    def with_structured_output(
        self,
        schema: type[PaperQualityAssessment],
    ) -> RunnableLambda:
        assert schema is PaperQualityAssessment

        def invoke(prompt_value: object) -> object:
            self.calls += 1
            self.rendered_prompt = getattr(prompt_value, "to_string")()
            if isinstance(self.response, tuple):
                return self.response[min(self.calls - 1, len(self.response) - 1)]
            return self.response

        return RunnableLambda(invoke)


def test_assessor_returns_validated_evidence_from_profile() -> None:
    """PaperProfile becomes a typed assessment without a PDF or network call."""
    profile = PaperProfile(
        title="Grounded Test Paper",
        research_fields=["Information Retrieval"],
        methods=["dense retrieval"],
        datasets=["Example Benchmark"],
        experimental_results=["The method improves recall on Example Benchmark."],
        summary="A retrieval method evaluated on one benchmark.",
    )
    fake = FakeAssessmentLLM()
    with patch.object(assessor_module, "get_llm", return_value=fake):
        assessment = assessor_module.assess_paper_quality(profile)

    assert isinstance(assessment, PaperQualityAssessment)
    assert assessment.experimental_quality.evidence
    assert assessment.novelty.concerns
    assert assessment.strengths
    assert assessment.weaknesses
    assert "Grounded Test Paper" in fake.rendered_prompt
    assert "Example Benchmark" in fake.rendered_prompt
    assert "acceptance probability" in fake.rendered_prompt
    assert fake.calls == 1


def test_assessor_canonicalizes_unsupported_dataset_status_claim() -> None:
    """External dataset reputation becomes a conservative profile-only statement."""
    unsupported = _response()
    unsupported["dataset_quality"] = {
        "score": 4,
        "level": "Strong",
        "evidence": [
            "Example Dataset is a standard large-scale machine translation benchmark."
        ],
        "concerns": [],
    }
    fake = FakeAssessmentLLM(unsupported)
    profile = PaperProfile(
        datasets=["Example Dataset"],
        summary="The paper reports a result on Example Dataset.",
    )
    with patch.object(assessor_module, "get_llm", return_value=fake):
        assessment = assessor_module.assess_paper_quality(profile)

    assert isinstance(assessment, PaperQualityAssessment)
    assert fake.calls == 1
    assert assessment.dataset_quality.score == 3
    assert assessment.dataset_quality.level == "Moderate"
    evidence_text = " ".join(assessment.dataset_quality.evidence).casefold()
    assert "standard" not in evidence_text
    assert "benchmark" not in evidence_text
    assert "large-scale" not in evidence_text
    assert any(
        "does not establish dataset source" in concern
        for concern in assessment.dataset_quality.concerns
    )
