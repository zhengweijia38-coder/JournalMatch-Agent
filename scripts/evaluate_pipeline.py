"""Run the Phase 7 evaluation pipeline; LLM recommendation is opt-in only."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dataset import load_evaluation_dataset, validate_gold_journals
from src.evaluation.hybrid_evaluator import evaluate_hybrid_filtering
from src.evaluation.recommendation_evaluator import (
    evaluate_recommendation_hard_metrics,
)
from src.evaluation.rerank_evaluator import evaluate_reranking
from src.evaluation.retrieval_evaluator import evaluate_semantic_retrieval
from src.recommendation.recommender import generate_recommendations
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank_candidates
from src.schemas.paper import PaperProfile


REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval, filtering, and reranking. DeepSeek is disabled "
            "unless --include-recommendation is explicitly supplied."
        )
    )
    parser.add_argument("dataset_path", type=Path, help="Evaluation JSONL path")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--recommendation-candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--include-recommendation",
        action="store_true",
        help="Opt in to one paid DeepSeek recommendation call per evaluation case.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _evaluate_recommendations(
    dataset: Any,
    *,
    candidate_k: int,
    top_k: int,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for case in dataset.cases:
        hybrid_candidates = hybrid_search(
            case.query,
            filters=case.filters,
            k=candidate_k,
        )
        reranked = rerank_candidates(
            case.query,
            hybrid_candidates,
            top_k=min(candidate_k, len(hybrid_candidates)),
        )
        profile = PaperProfile(
            research_fields=case.research_fields,
            research_problem=case.query,
            summary=case.query,
        )
        report = generate_recommendations(profile, reranked, top_k=top_k)
        hard_metrics = evaluate_recommendation_hard_metrics(
            report,
            reranked,
            top_k=top_k,
        )
        per_case.append(
            {
                "case_id": case.case_id,
                "hard_metrics": hard_metrics,
                "report": report.model_dump(),
            }
        )

    metric_names = (
        "candidate_containment",
        "metadata_faithfulness",
        "structured_output_validity",
        "recommendation_count_validity",
    )
    aggregate = {
        name: sum(bool(case["hard_metrics"][name]) for case in per_case)
        / len(per_case)
        for name in metric_names
    }
    return {"case_count": len(per_case), "aggregate": aggregate, "per_case": per_case}


def main() -> int:
    args = parse_args()
    if args.candidate_k < 20:
        raise ValueError("--candidate-k must be at least 20.")
    if args.recommendation_candidate_k <= 0 or args.top_k <= 0:
        raise ValueError("Recommendation candidate count and top_k must be positive.")

    started_at = perf_counter()
    dataset = load_evaluation_dataset(args.dataset_path)
    validate_gold_journals(dataset, strict=True)

    semantic = evaluate_semantic_retrieval(dataset)
    hybrid = evaluate_hybrid_filtering(dataset, k=args.candidate_k)
    reranking = evaluate_reranking(dataset, candidate_k=args.candidate_k)
    recommendation = None
    if args.include_recommendation:
        print(
            "WARNING: --include-recommendation is enabled; DeepSeek will be called "
            f"once for each of {len(dataset.cases)} cases."
        )
        recommendation = _evaluate_recommendations(
            dataset,
            candidate_k=args.recommendation_candidate_k,
            top_k=args.top_k,
        )

    duration = perf_counter() - started_at
    pipeline_result = {
        "dataset_size": len(dataset.cases),
        "semantic_retrieval": semantic,
        "hybrid_filtering": hybrid,
        "reranking": reranking,
        "recommendation_hard_metrics": recommendation,
        "recommendation_included": args.include_recommendation,
        "runtime_seconds": duration,
    }
    summary = {
        "dataset_size": len(dataset.cases),
        "semantic_retrieval_metrics": semantic["aggregate"],
        "hybrid_filtering_metrics": {
            "constraint_satisfaction_rate": hybrid[
                "constraint_satisfaction_rate"
            ],
            "filter_leakage_count": hybrid["filter_leakage_count"],
        },
        "reranking_before_metrics": reranking["before"],
        "reranking_after_metrics": reranking["after"],
        "reranking_delta": reranking["delta"],
        "recommendation_hard_metrics": (
            recommendation["aggregate"] if recommendation is not None else None
        ),
        "recommendation_included": args.include_recommendation,
        "runtime_seconds": duration,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "pipeline_results.json", pipeline_result)
    _write_json(args.output_dir / "evaluation_summary.json", summary)
    _write_json(args.output_dir / "retrieval_results.json", semantic)
    _write_json(args.output_dir / "rerank_results.json", reranking)

    print(f"Dataset size: {len(dataset.cases)}")
    print(
        "Hybrid constraint satisfaction: "
        f"{hybrid['constraint_satisfaction_rate']:.2%}"
    )
    print(f"Hybrid filter leakage count: {hybrid['filter_leakage_count']}")
    print("Metric          Before     After      Delta")
    for metric in ("hit_at_5", "mrr", "ndcg_at_5", "ndcg_at_10"):
        print(
            f"{metric:<15} {reranking['before'][metric]:<10.6f} "
            f"{reranking['after'][metric]:<10.6f} "
            f"{reranking['delta'][metric]:+.6f}"
        )
    print(f"Recommendation included: {args.include_recommendation}")
    print(f"Runtime: {duration:.2f}s")
    print(f"Reports: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: Pipeline evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
