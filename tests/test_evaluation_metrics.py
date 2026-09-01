"""Hand-calculated tests for Phase 7 information retrieval metrics."""

import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    dcg_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_evaluation_metrics() -> None:
    """Compare every metric with explicit manually derived expected values."""
    ranking = ["a", "x", "b", "c"]
    gold = {"a": 3, "b": 2, "c": 1, "d": 2}

    assert hit_at_k(ranking, gold, 1) == 1.0
    assert hit_at_k(["x"], gold, 1) == 0.0
    assert precision_at_k(ranking, gold, 2) == 0.5
    assert precision_at_k(["a"], gold, 5) == 0.2
    assert recall_at_k(ranking, gold, 3) == 0.5
    assert recall_at_k(ranking, gold, 20) == 0.75
    assert reciprocal_rank(ranking, gold) == 1.0
    assert reciprocal_rank(["x", "b", "a"], gold) == 0.5
    assert mrr([ranking, ["x", "b"]], [gold, gold]) == 0.75

    expected_dcg = 7.0 + 3.0 / math.log2(4)
    assert math.isclose(dcg_at_k(ranking, gold, 3), expected_dcg)
    ideal_dcg = 7.0 + 3.0 / math.log2(3) + 3.0 / math.log2(4)
    assert math.isclose(ndcg_at_k(ranking, gold, 3), expected_dcg / ideal_dcg)

    assert hit_at_k([], gold, 5) == 0.0
    assert precision_at_k([], gold, 5) == 0.0
    assert recall_at_k([], gold, 5) == 0.0
    assert recall_at_k(ranking, {}, 5) == 0.0
    assert reciprocal_rank([], gold) == 0.0
    assert mrr([], []) == 0.0
    assert dcg_at_k([], gold, 5) == 0.0
    assert ndcg_at_k(ranking, {}, 5) == 0.0

    for metric in (hit_at_k, precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k):
        try:
            metric(ranking, gold, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Metric should reject non-positive k.")

    print("Hand-calculated IR metric tests passed.")


if __name__ == "__main__":
    try:
        test_evaluation_metrics()
    except Exception as exc:
        print(f"ERROR: Evaluation metric test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
