"""Fast before/after tests for the Phase 5 rerank evaluator."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dataset import EvaluationCase, EvaluationDataset
from src.evaluation.rerank_evaluator import evaluate_reranking
from src.schemas.journal import Journal
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


def _hybrid(journal_id: int, rank: int) -> HybridCandidate:
    return HybridCandidate(
        journal=Journal(journal_id=journal_id, name=f"Journal {journal_id}"),
        semantic_rank=rank,
        retrieval_score=rank / 10,
    )


def test_rerank_evaluator() -> None:
    """Prove the evaluator reports both improvements and regressions unchanged."""
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(case_id="improves", query="improves", graded_relevance={1: 3}),
            EvaluationCase(case_id="declines", query="declines", graded_relevance={3: 3}),
        ]
    )
    before = {
        "improves": [_hybrid(2, 1), _hybrid(1, 2)],
        "declines": [_hybrid(3, 1), _hybrid(4, 2)],
    }

    def fake_retrieve(query: str, **_: object) -> list[HybridCandidate]:
        return before[query]

    def fake_rerank(
        query: str,
        candidates: list[HybridCandidate],
        top_k: int,
    ) -> list[RerankedCandidate]:
        ordered = list(reversed(candidates))[:top_k]
        return [
            RerankedCandidate(
                journal=candidate.journal,
                semantic_rank=candidate.semantic_rank,
                retrieval_score=candidate.retrieval_score,
                rerank_rank=rank,
                rerank_score=float(len(ordered) - rank),
            )
            for rank, candidate in enumerate(ordered, start=1)
        ]

    result = evaluate_reranking(
        dataset,
        retrieve_function=fake_retrieve,
        rerank_function=fake_rerank,
        candidate_k=10,
    )
    assert result["per_case"][0]["delta"]["mrr"] == 0.5
    assert result["per_case"][1]["delta"]["mrr"] == -0.5
    assert result["delta"]["mrr"] == 0.0
    assert result["before"]["hit_at_5"] == 1.0
    assert result["after"]["hit_at_5"] == 1.0
    assert result["per_case"][0]["first_relevant_rank_before"] == 2
    assert result["per_case"][0]["first_relevant_rank_after"] == 1

    print("Before/after rerank evaluator tests passed.")


if __name__ == "__main__":
    try:
        test_rerank_evaluator()
    except Exception as exc:
        print(f"ERROR: Rerank evaluator test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
