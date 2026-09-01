"""Phase 6 validation tests for structured recommendation schemas."""

from pathlib import Path
import sys

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.recommendation import (
    JournalRecommendation,
    PaperAssessment,
    RecommendationReport,
)


def _assessment() -> PaperAssessment:
    return PaperAssessment(
        innovation_level="moderate",
        experimental_completeness="strong",
        paper_maturity="mature",
        strengths=["Clear technical contribution."],
        weaknesses=["Evaluation covers a limited domain."],
        evidence=["Multiple benchmark results are reported."],
    )


def _recommendation(journal_id: int, rank: int) -> JournalRecommendation:
    return JournalRecommendation(
        journal_id=journal_id,
        journal_name=f"Candidate Journal {journal_id}",
        final_rank=rank,
        recommendation_tier="Strong Match" if rank == 1 else "Good Match",
        topic_fit="The journal covers the paper's research topic.",
        method_fit="The journal publishes work using the reported methods.",
        scope_fit="The contribution is consistent with the supplied aims and scope.",
        reasons=["Research fields and keywords align."],
        concerns=["The paper should state the limitation clearly."],
    )


def test_recommendation_schema() -> None:
    """Accept valid reports and reject unsupported levels, fields, and ranks."""
    report = RecommendationReport(
        paper_assessment=_assessment(),
        recommendations=[_recommendation(1, 1), _recommendation(2, 2)],
        overall_advice="Position the submission around its demonstrated contribution.",
    )
    assert len(report.recommendations) == 2

    invalid_cases = [
        lambda: PaperAssessment(
            innovation_level="87",
            experimental_completeness="strong",
            paper_maturity="mature",
        ),
        lambda: JournalRecommendation(
            **{
                **_recommendation(1, 1).model_dump(),
                "recommendation_tier": "Guaranteed Acceptance",
            }
        ),
        lambda: JournalRecommendation(
            **{
                **_recommendation(1, 1).model_dump(),
                "acceptance_probability": 0.9,
            }
        ),
        lambda: RecommendationReport(
            paper_assessment=_assessment(),
            recommendations=[_recommendation(1, 1), _recommendation(2, 1)],
            overall_advice="Advice.",
        ),
        lambda: RecommendationReport(
            paper_assessment=_assessment(),
            recommendations=[_recommendation(1, 1), _recommendation(2, 3)],
            overall_advice="Advice.",
        ),
    ]
    for build_invalid in invalid_cases:
        try:
            build_invalid()
        except ValidationError:
            pass
        else:
            raise AssertionError("Invalid recommendation data should fail validation.")

    print("Recommendation schema validation tests passed.")


if __name__ == "__main__":
    try:
        test_recommendation_schema()
    except Exception as exc:
        print(f"ERROR: Recommendation schema test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
