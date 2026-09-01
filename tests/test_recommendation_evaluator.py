"""Fast hard-metric tests for Phase 6 recommendation evaluation."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.recommendation_evaluator import (
    evaluate_recommendation_hard_metrics,
)
from src.schemas.journal import Journal
from src.schemas.recommendation import (
    JournalRecommendation,
    PaperAssessment,
    RecommendationReport,
)
from src.schemas.retrieval import RerankedCandidate


def _candidate(journal_id: int, ccf_rank: str = "A") -> RerankedCandidate:
    return RerankedCandidate(
        journal=Journal(
            journal_id=journal_id,
            name=f"Journal {journal_id}",
            ccf_rank=ccf_rank,
            jcr_quartile="Q1",
            cas_quartile="1区",
            impact_factor=5.0,
        ),
        semantic_rank=journal_id,
        retrieval_score=0.1,
        rerank_rank=journal_id,
        rerank_score=1.0,
    )


def _report(journal_id: int = 1) -> RecommendationReport:
    return RecommendationReport(
        paper_assessment=PaperAssessment(
            innovation_level="moderate",
            experimental_completeness="moderate",
            paper_maturity="solid",
        ),
        recommendations=[
            JournalRecommendation(
                journal_id=journal_id,
                journal_name=f"Journal {journal_id}",
                final_rank=1,
                recommendation_tier="Good Match",
                topic_fit="Topic evidence aligns.",
                method_fit="Method evidence aligns.",
                scope_fit="Scope evidence aligns.",
                reasons=["Supplied fields overlap."],
            )
        ],
        overall_advice="Use the supplied evidence when positioning the paper.",
    )


def test_recommendation_evaluator() -> None:
    """Check containment, schema, count, and current SQLite metadata comparisons."""
    candidates = [_candidate(1), _candidate(2, "B")]

    def matching_lookup(ids: list[int]) -> list[Journal]:
        return [candidates[journal_id - 1].journal for journal_id in ids]

    valid = evaluate_recommendation_hard_metrics(
        _report(1),
        candidates,
        top_k=1,
        journal_lookup=matching_lookup,
    )
    assert valid["structured_output_validity"] is True
    assert valid["candidate_containment"] is True
    assert valid["metadata_faithfulness"] is True
    assert valid["recommendation_count_validity"] is True

    outside = evaluate_recommendation_hard_metrics(
        _report(3),
        candidates,
        top_k=1,
        journal_lookup=lambda _: [],
    )
    assert outside["candidate_containment"] is False
    assert outside["metadata_faithfulness"] is False

    mismatch = evaluate_recommendation_hard_metrics(
        _report(1),
        candidates,
        top_k=1,
        journal_lookup=lambda _: [Journal(journal_id=1, name="Journal 1", ccf_rank="C")],
    )
    assert mismatch["metadata_faithfulness"] is False
    assert "ccf_rank" in mismatch["metadata_mismatches"][0]["differing_fields"]

    invalid_schema = evaluate_recommendation_hard_metrics(
        {"not": "a report"},
        candidates,
        top_k=1,
        journal_lookup=matching_lookup,
    )
    assert invalid_schema["structured_output_validity"] is False

    print("Recommendation hard-metric evaluator tests passed.")


if __name__ == "__main__":
    try:
        test_recommendation_evaluator()
    except Exception as exc:
        print(f"ERROR: Recommendation evaluator test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
