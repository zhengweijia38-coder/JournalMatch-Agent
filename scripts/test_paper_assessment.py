"""Real PDF -> PaperProfile -> PaperQualityAssessment integration check."""

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assessment.assessor import assess_paper_quality
from src.exceptions import JournalRAGError
from src.logging_config import configure_logging
from src.paper.analyzer import analyze_paper
from src.paper.loader import combine_documents, load_pdf


logger = logging.getLogger(__name__)
DEFAULT_PDF = PROJECT_ROOT / "data" / "papers" / "test_paper.pdf"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one optional local PDF path and debug switch."""
    parser = argparse.ArgumentParser(
        description="Run the real evidence-based paper quality assessment flow."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help="Text-based PDF path (default: data/papers/test_paper.pdf)",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print both source profile and assessment for manual grounding review."""
    args = parse_args(argv)
    configure_logging(debug=args.debug)
    try:
        documents = load_pdf(args.pdf_path)
        profile = analyze_paper(combine_documents(documents))
        assessment = assess_paper_quality(profile)
    except (FileNotFoundError, ValueError, JournalRAGError) as exc:
        if args.debug:
            logger.exception("Paper assessment integration test failed")
        else:
            logger.error("%s", exc)
        return 1
    except Exception:
        if args.debug:
            logger.exception("Unexpected paper assessment integration failure")
        else:
            logger.error(
                "Unexpected assessment failure; re-run with --debug for a traceback."
            )
        return 1

    print("\n=== SOURCE PAPER PROFILE ===")
    print(profile.model_dump_json(indent=2))
    print("\n=== PAPER QUALITY ASSESSMENT ===")
    print(assessment.model_dump_json(indent=2))
    print("\n=== GROUNDING REVIEW ===")
    print(
        "Verify that positive claims about datasets, ablations, SOTA comparisons, "
        "statistical tests, and external validation appear in SOURCE PAPER PROFILE."
    )
    print("This is a rubric-based assessment, not an acceptance probability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
