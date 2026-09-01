"""Fully rebuild the persistent Chroma journal semantic index."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.vector_store import build_journal_index


def main() -> int:
    """Build the configured journals collection and print its final count."""
    try:
        result = build_journal_index()
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected vector index build failure: {exc}", file=sys.stderr)
        return 1

    print(result)
    print("Vector index built successfully.")
    print(f"Indexed journals: {result.indexed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
