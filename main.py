"""Command-line interface for the Phase 8 journal recommendation pipeline."""

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path
import sys

from pydantic import ValidationError

from src.exceptions import JournalRAGError
from src.logging_config import configure_logging
from src.schemas.assessment import DimensionAssessment
from src.schemas.pipeline import PipelineResult
from src.schemas.retrieval import JournalFilters, RerankedCandidate


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one PDF path and optional exact journal filters."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a computer science paper and recommend journals from the "
            "local SQLite and Chroma knowledge base."
        )
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a text-based paper PDF")
    parser.add_argument("--ccf", nargs="+", metavar="RANK", help="Allowed CCF ranks, e.g. A B")
    parser.add_argument("--jcr", nargs="+", metavar="QUARTILE", help="Allowed JCR quartiles")
    parser.add_argument("--cas", nargs="+", metavar="ZONE", help="Allowed CAS zones")
    parser.add_argument("--min-if", type=float, dest="min_if", help="Minimum impact factor")
    parser.add_argument("--max-if", type=float, dest="max_if", help="Maximum impact factor")
    parser.add_argument("--candidate-k", type=int, default=20, help="Hybrid candidate count")
    parser.add_argument("--rerank-k", type=int, default=10, help="Reranked candidate count")
    parser.add_argument("--top-k", type=int, default=5, help="Final recommendation count")
    parser.add_argument(
        "--skip-recommendation",
        action="store_true",
        help=(
            "Stop after paper assessment and local reranking; do not call DeepSeek "
            "for final recommendations"
        ),
    )
    parser.add_argument("--output", type=Path, help="Write the complete PipelineResult as JSON")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed application logs and chained tracebacks on failure",
    )
    return parser.parse_args(argv)


def build_filters(args: argparse.Namespace) -> JournalFilters:
    """Build the existing Phase 4 filter model from CLI arguments."""
    return JournalFilters(
        ccf_ranks=args.ccf or [],
        jcr_quartiles=args.jcr or [],
        cas_quartiles=args.cas or [],
        min_impact_factor=args.min_if,
        max_impact_factor=args.max_if,
    )


def _run_pipeline(**kwargs: object) -> PipelineResult:
    """Import the model-facing pipeline only after CLI arguments are valid."""
    from src.pipeline import run_recommendation_pipeline

    return run_recommendation_pipeline(**kwargs)  # type: ignore[arg-type]


def _format_list(values: list[str]) -> str:
    return "; ".join(values) if values else "Not available"


def _print_paper_profile(result: PipelineResult) -> None:
    """Print a compact paper profile suitable for terminal use."""
    profile = result.paper_profile
    print("\n=== Paper Profile ===")
    print(f"Title: {profile.title or 'Not available'}")
    print(f"Research fields: {_format_list(profile.research_fields)}")
    print(f"Keywords: {_format_list(profile.keywords)}")
    print(f"Research problem: {profile.research_problem or 'Not available'}")
    print(f"Methods: {_format_list(profile.methods)}")
    print(f"Summary: {profile.summary}")


def _print_candidate(candidate: RerankedCandidate) -> None:
    """Print SQLite metadata and ranking signals for one verified candidate."""
    journal = candidate.journal
    impact_factor = (
        str(journal.impact_factor) if journal.impact_factor is not None else "N/A"
    )
    retrieval_score = (
        f"{candidate.retrieval_score:.6f}"
        if candidate.retrieval_score is not None
        else "N/A"
    )
    print(
        f"{candidate.rerank_rank}. {journal.name} "
        f"(CCF: {journal.ccf_rank or 'N/A'}, JCR: {journal.jcr_quartile or 'N/A'}, "
        f"CAS: {journal.cas_quartile or 'N/A'}, IF: {impact_factor})"
    )
    print(
        f"   semantic_rank={candidate.semantic_rank}, "
        f"retrieval_score={retrieval_score}, "
        f"rerank_score={candidate.rerank_score:.6f}"
    )


def _print_dimension(name: str, dimension: DimensionAssessment) -> None:
    """Print one rubric dimension with its evidence and concerns."""
    print(f"\n{name}:")
    print(f"Score: {dimension.score}/5")
    print(f"Level: {dimension.level}")
    print("Evidence:")
    for item in dimension.evidence:
        print(f"- {item}")
    print("Concerns:")
    if dimension.concerns:
        for item in dimension.concerns:
            print(f"- {item}")
    else:
        print("- None identified from the available PaperProfile evidence.")


def _print_quality_assessment(result: PipelineResult) -> None:
    """Print the five-dimensional decision-support assessment."""
    assessment = result.paper_quality_assessment
    if assessment is None:
        return

    print("\n==================================================")
    print("PAPER QUALITY ASSESSMENT")
    print("==================================================")
    _print_dimension("Novelty", assessment.novelty)
    _print_dimension("Methodology", assessment.methodology)
    _print_dimension("Dataset Quality", assessment.dataset_quality)
    _print_dimension("Experimental Quality", assessment.experimental_quality)
    _print_dimension("Conclusion Support", assessment.conclusion_support)
    print(f"\nOverall Maturity: {assessment.overall_maturity}")
    print("Strengths:")
    if assessment.strengths:
        for item in assessment.strengths:
            print(f"- {item}")
    else:
        print("- None identified from the available PaperProfile evidence.")
    print("Weaknesses:")
    if assessment.weaknesses:
        for item in assessment.weaknesses:
            print(f"- {item}")
    else:
        print("- None identified from the available PaperProfile evidence.")
    print(
        "\nThis is a rubric-based assessment, not an acceptance probability."
    )


def _print_result(result: PipelineResult, skipped: bool) -> None:
    """Render either the final report or the local reranking result."""
    _print_paper_profile(result)
    _print_quality_assessment(result)

    if skipped:
        print("\n=== Reranked Candidates ===")
        for candidate in result.reranked_candidates:
            _print_candidate(candidate)
        print(
            "\nFinal DeepSeek recommendation was skipped; "
            "no final LLM report was created."
        )
    else:
        report = result.recommendation_report
        if report is None:
            raise RuntimeError("The pipeline completed without a recommendation report.")

        assessment = report.paper_assessment
        print("\n=== Paper Assessment ===")
        print(f"Innovation: {assessment.innovation_level}")
        print(f"Experiments: {assessment.experimental_completeness}")
        print(f"Maturity: {assessment.paper_maturity}")
        print(f"Strengths: {_format_list(assessment.strengths)}")
        print(f"Weaknesses: {_format_list(assessment.weaknesses)}")

        candidates_by_id = {
            candidate.journal.journal_id: candidate
            for candidate in result.reranked_candidates
        }
        print("\n=== Journal Recommendations ===")
        for recommendation in sorted(
            report.recommendations, key=lambda item: item.final_rank
        ):
            candidate = candidates_by_id.get(recommendation.journal_id)
            if candidate is None:
                raise RuntimeError(
                    "A recommendation could not be mapped to its verified candidate."
                )
            journal = candidate.journal
            print(
                f"{recommendation.final_rank}. {journal.name} "
                f"[{recommendation.recommendation_tier}]"
            )
            print(
                f"   CCF: {journal.ccf_rank or 'N/A'} | "
                f"JCR: {journal.jcr_quartile or 'N/A'} | "
                f"CAS: {journal.cas_quartile or 'N/A'} | "
                f"IF: {journal.impact_factor if journal.impact_factor is not None else 'N/A'}"
            )
            print(f"   Topic fit: {recommendation.topic_fit}")
            print(f"   Method fit: {recommendation.method_fit}")
            print(f"   Scope fit: {recommendation.scope_fit}")
            print(f"   Reasons: {_format_list(recommendation.reasons)}")
            if recommendation.concerns:
                print(f"   Concerns: {_format_list(recommendation.concerns)}")
        print(f"\nOverall advice: {report.overall_advice}")

    if result.timings is not None:
        timings = result.timings
        print("\n=== Timings ===")
        print(
            "PDF {:.2f}s | Analysis {:.2f}s | Assessment {:.2f}s | Retrieval {:.2f}s | "
            "Reranking {:.2f}s | Recommendation {:.2f}s | Total {:.2f}s".format(
                timings.pdf_loading_seconds,
                timings.paper_analysis_seconds,
                timings.paper_assessment_seconds,
                timings.hybrid_retrieval_seconds,
                timings.reranking_seconds,
                timings.recommendation_seconds,
                timings.total_seconds,
            )
        )


def _write_json_output(result: PipelineResult, output_path: Path) -> None:
    """Create parent directories and write UTF-8 structured output."""
    resolved_path = output_path.expanduser()
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write JSON output to '{resolved_path}': {exc}"
        ) from exc
    print(f"\nJSON result written to: {resolved_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the end-to-end pipeline and translate failures into CLI messages."""
    args = parse_args(argv)
    configure_logging(debug=args.debug)

    try:
        filters = build_filters(args)
        result = _run_pipeline(
            pdf_path=args.pdf_path,
            filters=filters,
            candidate_k=args.candidate_k,
            rerank_k=args.rerank_k,
            recommendation_k=args.top_k,
            skip_recommendation=args.skip_recommendation,
        )
        _print_result(result, skipped=args.skip_recommendation)
        if args.output is not None:
            _write_json_output(result, args.output)
    except (FileNotFoundError, ValueError, JournalRAGError, ValidationError) as exc:
        if args.debug:
            logger.exception("Pipeline command failed")
        else:
            logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("ERROR: Operation cancelled by user.", file=sys.stderr)
        return 130
    except Exception:
        if args.debug:
            logger.exception("Unexpected pipeline command failure")
        else:
            logger.error(
                "Unexpected pipeline failure. Re-run with --debug for a traceback."
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
