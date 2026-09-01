"""Run the complete Phase 1-6 recommendation pipeline on a real PDF."""

import argparse
from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper.analyzer import analyze_paper
from src.paper.loader import combine_documents, load_pdf
from src.recommendation.recommender import generate_recommendations
from src.retrieval.hybrid_retriever import hybrid_search_for_paper
from src.retrieval.reranker import rerank_candidates_for_paper
from src.schemas.recommendation import PaperAssessment, RecommendationReport
from src.schemas.retrieval import RerankedCandidate


DEFAULT_PDF = PROJECT_ROOT / "data" / "papers" / "test_paper.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evidence-grounded journal recommendation for one PDF."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help="Text-based paper PDF (default: data/papers/test_paper.pdf)",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=20,
        help="Phase 4 hybrid candidates before reranking (default: 20)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Phase 5 reranked candidates supplied to DeepSeek (default: 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum final recommendations (default: 5)",
    )
    return parser.parse_args()


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_items(title: str, items: list[str]) -> None:
    print(f"{title}:")
    if not items:
        print("- None stated")
        return
    for item in items:
        print(_safe_console_text(f"- {item}"))


def _print_assessment(assessment: PaperAssessment) -> None:
    print("\n==============================")
    print("PAPER ASSESSMENT")
    print("==============================")
    print(f"Innovation: {assessment.innovation_level}")
    print(f"Experimental Completeness: {assessment.experimental_completeness}")
    print(f"Paper Maturity: {assessment.paper_maturity}")
    print()
    _print_items("Strengths", assessment.strengths)
    print()
    _print_items("Weaknesses", assessment.weaknesses)
    print()
    _print_items("Evidence", assessment.evidence)


def _print_recommendations(
    report: RecommendationReport,
    candidates: list[RerankedCandidate],
) -> None:
    candidates_by_id = {
        candidate.journal.journal_id: candidate for candidate in candidates
    }
    print("\n==============================")
    print("RECOMMENDATIONS")
    print("==============================")

    for recommendation in report.recommendations:
        candidate = candidates_by_id[recommendation.journal_id]
        journal = candidate.journal
        print(_safe_console_text(f"\n#{recommendation.final_rank} {journal.name}"))
        print(f"Tier: {recommendation.recommendation_tier}")
        print(f"CCF: {journal.ccf_rank or 'unknown'}")
        print(f"JCR: {journal.jcr_quartile or 'unknown'}")
        print(_safe_console_text(f"CAS: {journal.cas_quartile or 'unknown'}"))
        impact_factor = (
            journal.impact_factor
            if journal.impact_factor is not None
            else "unknown"
        )
        print(f"Impact Factor: {impact_factor}")
        print(f"Semantic Rank: {candidate.semantic_rank}")
        print(f"Rerank Rank: {candidate.rerank_rank}")
        print(f"Topic Fit: {_safe_console_text(recommendation.topic_fit)}")
        print(f"Method Fit: {_safe_console_text(recommendation.method_fit)}")
        print(f"Scope Fit: {_safe_console_text(recommendation.scope_fit)}")
        _print_items("Reasons", recommendation.reasons)
        _print_items("Concerns", recommendation.concerns)

    print("\nOverall Advice:")
    print(_safe_console_text(report.overall_advice))


def main() -> int:
    args = parse_args()
    for name, value in (
        ("retrieval-k", args.retrieval_k),
        ("candidate-k", args.candidate_k),
        ("top-k", args.top_k),
    ):
        if value <= 0:
            raise ValueError(f"--{name} must be a positive integer.")

    started_at = perf_counter()
    documents = load_pdf(args.pdf_path)
    profile = analyze_paper(combine_documents(documents))
    analysis_finished_at = perf_counter()

    hybrid_candidates = hybrid_search_for_paper(
        profile,
        k=args.retrieval_k,
    )
    reranked_candidates = rerank_candidates_for_paper(
        profile,
        hybrid_candidates,
        top_k=min(args.candidate_k, len(hybrid_candidates)),
    )
    retrieval_finished_at = perf_counter()

    report = generate_recommendations(
        profile,
        reranked_candidates,
        top_k=args.top_k,
    )
    recommendation_finished_at = perf_counter()

    print(_safe_console_text(f"Paper: {profile.title or args.pdf_path.name}"))
    print(f"Hybrid candidates: {len(hybrid_candidates)}")
    print(f"Reranked candidates sent to DeepSeek: {len(reranked_candidates)}")
    print(f"Final recommendations: {len(report.recommendations)}")
    print(f"Paper analysis duration: {analysis_finished_at - started_at:.2f}s")
    print(
        "Retrieval + reranking duration: "
        f"{retrieval_finished_at - analysis_finished_at:.2f}s"
    )
    print(
        "Recommendation duration: "
        f"{recommendation_finished_at - retrieval_finished_at:.2f}s"
    )
    _print_assessment(report.paper_assessment)
    _print_recommendations(report, reranked_candidates)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: Phase 6 recommendation test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
