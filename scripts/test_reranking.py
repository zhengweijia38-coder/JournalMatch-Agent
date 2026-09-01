"""Run real Phase 5 reranking for three representative research queries."""

from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.models.reranker import get_reranker, get_reranker_device
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank_candidates
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


QUERIES = [
    (
        "Retrieval-Augmented Generation",
        (
            "Retrieval-Augmented Generation for large language models, natural "
            "language processing, information retrieval, and grounded generation."
        ),
    ),
    (
        "Computer Vision",
        (
            "Computer vision and deep learning for medical image segmentation, "
            "image recognition, and visual analysis."
        ),
    ),
    (
        "Software Engineering",
        (
            "Software engineering, automated program repair, software testing, "
            "program analysis, and software quality."
        ),
    ),
]


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_semantic_top(candidates: list[HybridCandidate]) -> None:
    print("Semantic Top-10 before reranking:")
    for candidate in candidates[:10]:
        print(
            _safe_console_text(
                f"  S{candidate.semantic_rank:02d} | {candidate.journal.name}"
            )
        )


def _print_reranked_top(candidates: list[RerankedCandidate]) -> None:
    print("Reranked Top-10:")
    print(
        "  New | Old | Change | Journal | CCF | JCR | CAS | IF | "
        "Raw retrieval score | Raw rerank score"
    )
    for candidate in candidates:
        journal = candidate.journal
        change = candidate.semantic_rank - candidate.rerank_rank
        change_text = f"{change:+d}"
        retrieval_score = (
            f"{candidate.retrieval_score:.6f}"
            if candidate.retrieval_score is not None
            else "-"
        )
        print(
            _safe_console_text(
                f"  R{candidate.rerank_rank:02d} | S{candidate.semantic_rank:02d} | "
                f"{change_text:>6} | {journal.name} | {journal.ccf_rank or '-'} | "
                f"{journal.jcr_quartile or '-'} | {journal.cas_quartile or '-'} | "
                f"{journal.impact_factor if journal.impact_factor is not None else '-'} | "
                f"{retrieval_score} | {candidate.rerank_score:.6f}"
            )
        )


def main() -> int:
    settings = get_settings()
    print(f"Reranker model: {settings.reranker_model_name}")
    print(f"Device: {get_reranker_device()}")

    load_started_at = perf_counter()
    get_reranker()
    print(f"Model load duration: {perf_counter() - load_started_at:.2f}s")

    for title, query in QUERIES:
        print(f"\n===== {title} =====")
        candidates = hybrid_search(query=query, k=20)
        _print_semantic_top(candidates)

        rerank_started_at = perf_counter()
        reranked = rerank_candidates(query, candidates, top_k=10)
        rerank_duration = perf_counter() - rerank_started_at
        _print_reranked_top(reranked)
        print(f"Candidate count: {len(candidates)}")
        print(f"Rerank duration: {rerank_duration:.2f}s")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: Real reranking test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
