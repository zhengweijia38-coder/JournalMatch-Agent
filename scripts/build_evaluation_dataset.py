"""Combine fixed PaperProfiles and human Gold labels into Phase 7 JSONL."""

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
    EvaluationDataReport,
    build_evaluation_dataset,
)
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build retrieval_cases.jsonl from fixed profiles and exclusively "
            "human-authored relevance labels."
        )
    )
    parser.add_argument("--profiles", type=Path, default=DEFAULT_EVALUATION_DIR / "profiles")
    parser.add_argument(
        "--gold-labels",
        type=Path,
        default=DEFAULT_EVALUATION_DIR / "annotations" / "gold_labels.csv",
    )
    parser.add_argument(
        "--paper-metadata",
        type=Path,
        default=DEFAULT_EVALUATION_DIR / "annotations" / "paper_metadata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVALUATION_DIR / "retrieval_cases.jsonl",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def _print_report(report: EvaluationDataReport) -> None:
    print(f"Total Cases: {report.total_cases}")
    print(f"Total Gold Labels: {report.total_gold_labels}")
    print(f"Relevance 3 Count: {report.relevance_counts[3]}")
    print(f"Relevance 2 Count: {report.relevance_counts[2]}")
    print(f"Relevance 1 Count: {report.relevance_counts[1]}")
    print(
        "Average Gold Journals Per Case: "
        f"{report.average_gold_journals_per_case:.2f}"
    )
    print(
        "Profiles Without Gold Labels: "
        f"{len(report.profiles_without_gold_labels)}"
    )
    print(
        "Gold Case IDs Missing Profiles: "
        f"{len(report.gold_case_ids_missing_profiles)}"
    )
    print(f"Unknown Journal IDs: {len(report.unknown_journal_ids)}")
    print(f"Duplicate Labels: {report.duplicate_labels}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if report.output_path is not None:
        print(f"Output: {report.output_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(debug=args.debug)
    try:
        report = build_evaluation_dataset(
            profiles_dir=args.profiles,
            gold_labels_path=args.gold_labels,
            paper_metadata_path=args.paper_metadata,
            output_path=args.output,
        )
    except (FileNotFoundError, EvaluationDataError, OSError) as exc:
        if args.debug:
            logger.exception("Evaluation dataset build failed")
        else:
            logger.error("%s", exc)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
