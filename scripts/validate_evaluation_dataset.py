"""Validate a built evaluation JSONL against schemas and current SQLite IDs."""

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data_preparation import (
    EvaluationDataError,
    validate_evaluation_jsonl,
)
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)
DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate evaluation JSONL structure, labels, and SQLite IDs."
    )
    parser.add_argument("dataset_path", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(debug=args.debug)
    try:
        _, report = validate_evaluation_jsonl(args.dataset_path)
    except (FileNotFoundError, EvaluationDataError, OSError) as exc:
        if args.debug:
            logger.exception("Evaluation dataset validation failed")
        else:
            logger.error("%s", exc)
        return 1

    print(f"Total Cases: {report.total_cases}")
    print(f"Total Gold Labels: {report.total_gold_labels}")
    print(f"Relevance 3 Count: {report.relevance_counts[3]}")
    print(f"Relevance 2 Count: {report.relevance_counts[2]}")
    print(f"Relevance 1 Count: {report.relevance_counts[1]}")
    print(
        "Average Gold Journals Per Case: "
        f"{report.average_gold_journals_per_case:.2f}"
    )
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    print("Evaluation Dataset Validation Passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
