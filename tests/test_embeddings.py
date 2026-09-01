"""Minimal local BGE-M3 embedding test."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embeddings import get_embeddings


TEXTS = [
    "Retrieval augmented generation for large language models.",
    "Computer vision for medical image segmentation.",
]


def test_local_embeddings() -> None:
    """Embed two texts locally and validate the vector shapes."""
    try:
        vectors = get_embeddings().embed_documents(TEXTS)
    except Exception as exc:
        raise RuntimeError(
            "BGE-M3 embedding test failed. On the first run, check that Hugging "
            "Face is reachable and that the model download has enough disk space. "
            f"Original error: {exc}"
        ) from exc

    if len(vectors) != len(TEXTS):
        raise AssertionError(
            f"Expected {len(TEXTS)} embeddings, but received {len(vectors)}."
        )
    if not vectors or not vectors[0]:
        raise AssertionError("The embedding model returned an empty vector.")
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise AssertionError("The returned embeddings have inconsistent dimensions.")

    print(f"Embedding count: {len(vectors)}")
    print(f"Embedding dimension: {len(vectors[0])}")
    print("BGE-M3 local embedding test passed.")


if __name__ == "__main__":
    try:
        test_local_embeddings()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
