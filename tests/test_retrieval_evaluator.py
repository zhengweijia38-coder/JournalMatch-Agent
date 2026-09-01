"""Fast fake-result tests for semantic and hybrid evaluators."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dataset import EvaluationCase, EvaluationDataset
from src.evaluation.hybrid_evaluator import evaluate_hybrid_filtering
from src.evaluation.retrieval_evaluator import evaluate_semantic_retrieval
from src.schemas.journal import Journal
from src.schemas.retrieval import HybridCandidate, JournalFilters


def test_retrieval_evaluator() -> None:
    """Check exact aggregate retrieval values and filter-leakage detection."""
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                case_id="first",
                query="first query",
                graded_relevance={1: 3, 3: 1},
            ),
            EvaluationCase(
                case_id="second",
                query="second query",
                graded_relevance={30: 3},
            ),
        ]
    )

    rankings = {
        "first query": [1, 2, 3] + list(range(4, 21)),
        "second query": [20, 30] + list(range(31, 49)),
    }

    def fake_semantic(query: str, k: int) -> list[int]:
        return rankings[query][:k]

    result = evaluate_semantic_retrieval(dataset, search_function=fake_semantic)
    assert result["case_count"] == 2
    assert result["aggregate"]["hit_at_5"] == 1.0
    assert result["aggregate"]["mrr"] == 0.75
    assert result["per_case"][0]["first_relevant_rank"] == 1
    assert result["per_case"][1]["first_relevant_rank"] == 2
    assert result["per_case"][0]["metrics"]["recall_at_5"] == 1.0

    filtered_dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                case_id="filtered",
                query="filtered query",
                graded_relevance={1: 3},
                filters=JournalFilters(ccf_ranks=["A"], min_impact_factor=5.0),
            )
        ]
    )
    valid = HybridCandidate(
        journal=Journal(
            journal_id=1,
            name="Valid Journal",
            ccf_rank="A",
            impact_factor=6.0,
        ),
        semantic_rank=1,
        retrieval_score=0.1,
    )
    leaked = HybridCandidate(
        journal=Journal(
            journal_id=2,
            name="Leaked Journal",
            ccf_rank="B",
            impact_factor=6.0,
        ),
        semantic_rank=2,
        retrieval_score=0.2,
    )

    def valid_hybrid(*_: object, **__: object) -> list[HybridCandidate]:
        return [valid]

    valid_result = evaluate_hybrid_filtering(
        filtered_dataset,
        search_function=valid_hybrid,
        k=5,
    )
    assert valid_result["constraint_satisfaction_rate"] == 1.0
    assert valid_result["filter_leakage_count"] == 0

    def leaking_hybrid(*_: object, **__: object) -> list[HybridCandidate]:
        return [valid, leaked]

    leaked_result = evaluate_hybrid_filtering(
        filtered_dataset,
        search_function=leaking_hybrid,
        k=5,
    )
    assert leaked_result["constraint_satisfaction_rate"] == 0.5
    assert leaked_result["filter_leakage_count"] == 1

    print("Retrieval and hybrid evaluator fake-result tests passed.")


if __name__ == "__main__":
    try:
        test_retrieval_evaluator()
    except Exception as exc:
        print(f"ERROR: Retrieval evaluator test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
