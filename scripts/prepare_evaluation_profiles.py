"""Generate stable PaperProfile artifacts from evaluation PDFs only."""

import argparse
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data_preparation import (
    CASE_ID_PATTERN,
    EvaluationProfileArtifact,
    load_paper_metadata,
    possible_venue_leakage,
)
from src.exceptions import JournalRAGError
from src.logging_config import configure_logging
from src.paper.analyzer import analyze_paper
from src.paper.loader import combine_documents, load_pdf


logger = logging.getLogger(__name__)
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "evaluation" / "profiles"
DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "evaluation" / "annotations" / "paper_metadata.csv"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed PaperProfiles from evaluation PDFs. This script never "
            "generates journal labels or recommendations."
        )
    )
    parser.add_argument("papers_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--paper-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly regenerate profiles that already exist",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def _discover_pdfs(papers_dir: Path) -> list[Path]:
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"Evaluation papers directory does not exist: {papers_dir}")
    pdfs = sorted(
        (path for path in papers_dir.iterdir() if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    seen: dict[str, str] = {}
    for pdf_path in pdfs:
        case_id = pdf_path.stem
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(
                f"Invalid PDF case_id '{case_id}' from '{pdf_path.name}'. Use only "
                "letters, digits, underscores, or hyphens."
            )
        folded = case_id.casefold()
        if folded in seen:
            raise ValueError(
                f"Duplicate PDF case_id values: '{seen[folded]}' and '{case_id}'."
            )
        seen[folded] = case_id
    return pdfs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(debug=args.debug)
    try:
        pdfs = _discover_pdfs(args.papers_dir)
        metadata = load_paper_metadata(args.paper_metadata)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    generated = 0
    skipped = 0
    failed = 0
    failures: list[tuple[str, str, str]] = []
    for pdf_path in pdfs:
        case_id = pdf_path.stem
        output_path = args.output_dir / f"{case_id}.json"
        if output_path.exists() and not args.overwrite:
            logger.info("Skipping existing fixed profile: %s", output_path.name)
            skipped += 1
            continue
        try:
            documents = load_pdf(pdf_path)
            profile = analyze_paper(combine_documents(documents))
            artifact = EvaluationProfileArtifact(
                case_id=case_id,
                source_pdf=pdf_path.name,
                paper_profile=profile,
            )
            output_path.write_text(
                json.dumps(
                    artifact.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            generated += 1

            metadata_values = metadata.get(case_id, {})
            leakage_warnings = possible_venue_leakage(
                profile,
                str(metadata_values.get("published_journal_name") or "") or None,
            )
            for warning in leakage_warnings:
                logger.warning(
                    "possible_venue_leakage: case_id=%s, source_pdf=%s, reason=%s",
                    case_id,
                    pdf_path.name,
                    warning,
                )
        except (OSError, ValueError, JournalRAGError) as exc:
            failed += 1
            failures.append((case_id, pdf_path.name, str(exc)))
            logger.error(
                "Profile generation failed: case_id=%s, source_pdf=%s, error=%s",
                case_id,
                pdf_path.name,
                exc,
            )
        except Exception:
            failed += 1
            failures.append((case_id, pdf_path.name, "unexpected processing error"))
            if args.debug:
                logger.exception(
                    "Unexpected profile generation failure for case_id=%s",
                    case_id,
                )
            else:
                logger.error(
                    "Profile generation failed: case_id=%s, source_pdf=%s, "
                    "error=unexpected processing error; use --debug for traceback",
                    case_id,
                    pdf_path.name,
                )

    print(f"Total PDFs: {len(pdfs)}")
    print(f"Generated: {generated}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    for case_id, source_pdf, error in failures:
        print(f"FAILED case_id={case_id}, source_pdf={source_pdf}, error={error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
