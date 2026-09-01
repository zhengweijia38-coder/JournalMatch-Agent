"""Evaluate Phase 3 semantic retrieval against a manually labeled JSONL dataset."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dataset import load_evaluation_dataset, validate_gold_journals
from src.evaluation.retrieval_evaluator import evaluate_semantic_retrieval


DEFAULT_REPORT = PROJECT_ROOT / "reports" / "evaluation" / "retrieval_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 semantic retrieval.")
    parser.add_argument("dataset_path", type=Path, help="Evaluation JSONL path")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_evaluation_dataset(args.dataset_path)
    validate_gold_journals(dataset, strict=True)
    result = evaluate_semantic_retrieval(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    aggregate = result["aggregate"]
    print(f"Case count: {result['case_count']}")
    for metric in (
        "hit_at_5",
        "hit_at_10",
        "precision_at_5",
        "precision_at_10",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ):
        print(f"{metric}: {aggregate[metric]:.6f}")
    print(f"Detailed report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: Retrieval evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
