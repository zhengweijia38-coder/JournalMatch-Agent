"""Fast integration smoke test with optional LLM and reranker stages."""

import argparse
from collections.abc import Sequence
import logging
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_environment import print_checks, run_checks
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)
SMOKE_QUERY = "retrieval augmented generation and information retrieval"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional paid/slow smoke-test stages."""
    parser = argparse.ArgumentParser(
        description="Check SQLite, Chroma, semantic retrieval, and optional models."
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Do not call the DeepSeek API; local retrieval still runs",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run local hybrid retrieval and cross-encoder reranking",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug traceback")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a small real component chain without full PDF recommendation."""
    args = parse_args(argv)
    configure_logging(debug=args.debug)
    if args.skip_llm:
        # This smoke mode is explicitly offline and must reuse existing HF cache.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        checks = run_checks(require_api_key=not args.skip_llm)
        print_checks(checks)
        if not all(check.ok for check in checks):
            print("\nSmoke test stopped because the environment is not ready.")
            return 1

        from src.retrieval.semantic_retriever import search_journals

        semantic_results = search_journals(SMOKE_QUERY, k=1)
        if not semantic_results:
            raise RuntimeError("Semantic retrieval returned no journals.")
        print(
            "Semantic retrieval:              OK "
            f"({semantic_results[0].name})"
        )

        if args.full:
            from src.retrieval.hybrid_retriever import hybrid_search
            from src.retrieval.reranker import rerank_candidates

            candidates = hybrid_search(SMOKE_QUERY, k=3)
            reranked = rerank_candidates(
                SMOKE_QUERY,
                candidates,
                top_k=min(3, len(candidates)),
            )
            if not reranked:
                raise RuntimeError("Full smoke reranking returned no journals.")
            print(f"Local reranker:                  OK ({len(reranked)} results)")

        if not args.skip_llm:
            from src.models.llm import get_llm

            response = get_llm().invoke("Reply with exactly: smoke-ok")
            if str(response.content).strip() != "smoke-ok":
                raise RuntimeError("DeepSeek returned an unexpected smoke response.")
            print("DeepSeek API:                    OK")
    except Exception:
        if args.debug:
            logger.exception("Smoke test failed")
        else:
            logger.error("Smoke test failed. Re-run with --debug for a traceback.")
        return 1

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
