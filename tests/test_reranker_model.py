"""Phase 5 smoke test for the real local BGE reranker model."""

from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.models.reranker import get_reranker, get_reranker_device


def test_reranker_model() -> None:
    """Confirm local inference and a basic computer-vision relevance ordering."""
    query = "Computer vision for medical image segmentation."
    passages = [
        (
            "Computer vision and medical imaging research on deep-learning image "
            "segmentation, recognition, and visual analysis."
        ),
        "Software engineering research on program repair and software testing.",
        "Computer networks research on routing protocols and network security.",
    ]
    pairs = [[query, passage] for passage in passages]

    started_at = perf_counter()
    reranker = get_reranker()
    scores = reranker.compute_score(
        pairs,
        batch_size=len(pairs),
        max_length=512,
        normalize=False,
    )
    duration = perf_counter() - started_at
    numeric_scores = [float(score) for score in scores]

    assert len(numeric_scores) == len(pairs)
    assert numeric_scores[0] > numeric_scores[1]
    assert numeric_scores[0] > numeric_scores[2]
    assert get_reranker() is reranker

    print(f"Reranker model: {get_settings().reranker_model_name}")
    print(f"Device: {get_reranker_device()}")
    print(f"Scores: {numeric_scores}")
    print(f"Duration (including first load): {duration:.2f}s")
    print("Local BGE reranker model test passed.")


if __name__ == "__main__":
    try:
        test_reranker_model()
    except Exception as exc:
        print(f"ERROR: Local reranker model test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
