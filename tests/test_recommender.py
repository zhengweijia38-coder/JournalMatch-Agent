"""Phase 6 unit tests with a fake structured LLM and no network requests."""

from pathlib import Path
import sys
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.recommendation.recommender as recommender_module
from src.recommendation.recommender import InvalidRecommendationError
from src.schemas.assessment import PaperQualityAssessment
from src.schemas.journal import Journal
from src.schemas.paper import PaperProfile
from src.schemas.recommendation import RecommendationReport
from src.schemas.retrieval import RerankedCandidate


class FakeStructuredLLM:
    """Expose LangChain's Runnable interface while returning deterministic data."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0
        self.rendered_prompt = ""

    def with_structured_output(
        self,
        schema: type[RecommendationReport],
    ) -> RunnableLambda:
        assert schema is RecommendationReport

        def invoke(prompt_value: object) -> object:
            self.calls += 1
            to_string = getattr(prompt_value, "to_string")
            self.rendered_prompt = to_string()
            if isinstance(self.response, tuple):
                return self.response[min(self.calls - 1, len(self.response) - 1)]
            return self.response

        return RunnableLambda(invoke)


def _profile() -> PaperProfile:
    return PaperProfile(
        title="Evidence-Grounded Retrieval-Augmented Generation",
        research_fields=["Information Retrieval", "Natural Language Processing"],
        keywords=["retrieval augmented generation", "large language models"],
        research_problem="Ground language-model output in retrieved evidence.",
        methods=["dense retrieval", "transformer language model"],
        main_contributions=["An evidence-grounded generation pipeline."],
        claimed_innovations=["Joint retrieval and grounded generation analysis."],
        experimental_results=["The method improves factuality on two benchmarks."],
        limitations=["Only two benchmarks are evaluated."],
        summary="The paper retrieves evidence before generating grounded responses.",
    )


def _quality_assessment() -> PaperQualityAssessment:
    dimension = {
        "score": 4,
        "level": "Strong",
        "evidence": ["The profile reports improvements on two benchmarks."],
        "concerns": ["Only two benchmarks are recorded in the profile."],
    }
    return PaperQualityAssessment.model_validate(
        {
            "novelty": dimension,
            "methodology": dimension,
            "dataset_quality": {
                "score": 3,
                "level": "Moderate",
                "evidence": ["The profile reports two benchmarks."],
                "concerns": ["No external validation is recorded."],
            },
            "experimental_quality": dimension,
            "conclusion_support": dimension,
            "overall_maturity": "Mature",
            "strengths": ["The profile records concrete benchmark improvements."],
            "weaknesses": ["Validation breadth is limited in the profile."],
        }
    )


def _candidates(count: int = 5) -> list[RerankedCandidate]:
    candidates: list[RerankedCandidate] = []
    for index in range(1, count + 1):
        candidates.append(
            RerankedCandidate(
                journal=Journal(
                    journal_id=index,
                    name=f"Retrieved Journal {index}",
                    research_fields=["Information Retrieval"],
                    keywords=["retrieval", "language models"],
                    aims_scope="Publishes research on information access and NLP.",
                    ccf_rank="A" if index == 1 else "B",
                    jcr_quartile="Q1",
                    cas_quartile="1区" if index == 1 else "2区",
                    impact_factor=10.0 - index,
                ),
                semantic_rank=index,
                retrieval_score=index / 10,
                rerank_rank=index,
                rerank_score=5.0 - index,
            )
        )
    return candidates


def _recommendation(journal_id: int, rank: int) -> dict[str, object]:
    return {
        "journal_id": journal_id,
        "journal_name": f"Retrieved Journal {journal_id}",
        "final_rank": rank,
        "recommendation_tier": "Strong Match" if rank == 1 else "Good Match",
        "topic_fit": "The candidate covers retrieval and language models.",
        "method_fit": "The supplied scope is compatible with retrieval methods.",
        "scope_fit": "The contribution aligns with the supplied aims and scope.",
        "reasons": ["The research fields and keywords align."],
        "concerns": ["The limited benchmark coverage should be explained."],
    }


def _response(recommendations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "paper_assessment": {
            "innovation_level": "moderate",
            "experimental_completeness": "moderate",
            "paper_maturity": "solid",
            "strengths": ["The contribution and method are clearly stated."],
            "weaknesses": ["Only two benchmarks are evaluated."],
            "evidence": ["PaperProfile reports improved factuality."],
        },
        "recommendations": recommendations,
        "overall_advice": "Emphasize the grounded retrieval contribution.",
    }


def _generate_with_fake(
    fake: FakeStructuredLLM,
    candidates: list[RerankedCandidate],
    top_k: int,
) -> RecommendationReport:
    with patch.object(recommender_module, "get_llm", return_value=fake):
        return recommender_module.generate_recommendations(
            profile=_profile(),
            candidates=candidates,
            top_k=top_k,
            quality_assessment=_quality_assessment(),
        )


def test_recommender() -> None:
    """Cover candidate boundaries, top_k, ordering, and database-fact protection."""
    candidates = _candidates(5)
    normal = FakeStructuredLLM(
        _response(
            [
                _recommendation(2, 2),
                {**_recommendation(1, 1), "journal_name": "retrieved journal 1"},
                _recommendation(3, 3),
            ]
        )
    )
    report = _generate_with_fake(normal, candidates, top_k=3)
    assert normal.calls == 1
    assert len(report.recommendations) == 3
    assert [item.final_rank for item in report.recommendations] == [1, 2, 3]
    assert report.recommendations[0].journal_name == "Retrieved Journal 1"
    assert '"ccf_rank": "A"' in normal.rendered_prompt
    assert '"rerank_score": 4.0' in normal.rendered_prompt
    assert '"overall_maturity": "Mature"' in normal.rendered_prompt
    assert "acceptance probability" in normal.rendered_prompt
    assert candidates[0].journal.ccf_rank == "A"

    invalid_then_valid = FakeStructuredLLM(
        (
            _response(
                [
                    {
                        **_recommendation(1, 1),
                        "recommendation_tier_typo": "Strong Match",
                    }
                ]
            ),
            _response([_recommendation(1, 1)]),
        )
    )
    recovered_report = _generate_with_fake(invalid_then_valid, candidates, top_k=1)
    assert invalid_then_valid.calls == 2
    assert len(recovered_report.recommendations) == 1
    assert "failed Pydantic validation" in invalid_then_valid.rendered_prompt

    three_candidates = _candidates(3)
    top_five_fake = FakeStructuredLLM(
        _response([_recommendation(index, index) for index in range(1, 4)])
    )
    limited_report = _generate_with_fake(top_five_fake, three_candidates, top_k=5)
    assert len(limited_report.recommendations) == 3
    assert "at most 3 recommendations" in top_five_fake.rendered_prompt

    empty_fake = FakeStructuredLLM(_response([_recommendation(1, 1)]))
    with patch.object(recommender_module, "get_llm", return_value=empty_fake):
        try:
            recommender_module.generate_recommendations(_profile(), [], top_k=3)
        except ValueError as exc:
            assert "empty" in str(exc)
        else:
            raise AssertionError("Empty candidates should raise ValueError.")
    assert empty_fake.calls == 0

    invalid_responses = [
        _response(
            [
                {
                    **_recommendation(99, 1),
                    "journal_name": "Hallucinated Journal",
                }
            ]
        ),
        _response([_recommendation(1, 1), _recommendation(2, 1)]),
        _response([_recommendation(1, 1), _recommendation(2, 3)]),
        _response([_recommendation(index, index) for index in range(1, 5)]),
        _response(
            [
                {
                    **_recommendation(1, 1),
                    "ccf_rank": "C",
                }
            ]
        ),
    ]
    for index, response in enumerate(invalid_responses):
        request_top_k = 3 if index == 3 else 5
        try:
            _generate_with_fake(
                FakeStructuredLLM(response),
                candidates,
                top_k=request_top_k,
            )
        except InvalidRecommendationError:
            pass
        else:
            raise AssertionError("Invalid structured recommendation should be rejected.")

    for invalid_top_k in (0, -1, 1.5, True):
        try:
            _generate_with_fake(
                FakeStructuredLLM(_response([_recommendation(1, 1)])),
                candidates,
                top_k=invalid_top_k,  # type: ignore[arg-type]
            )
        except ValueError as exc:
            assert "top_k" in str(exc)
        else:
            raise AssertionError("Invalid top_k should raise ValueError.")

    print("Evidence-grounded recommender boundary tests passed.")


if __name__ == "__main__":
    try:
        test_recommender()
    except Exception as exc:
        print(f"ERROR: Recommender test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
