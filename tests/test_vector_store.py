"""Phase 3 isolated BGE-M3 and temporary Chroma integration test."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.journal.repository import upsert_journal
from src.retrieval.vector_store import (
    build_journal_index,
    close_vector_store,
    get_indexed_count,
    get_vector_store,
)
from src.schemas.journal import Journal


def test_vector_store() -> None:
    """Build twice in temporary storage and run one semantic search."""
    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        database_path = temporary_path / "journals.db"
        chroma_path = temporary_path / "chroma"

        with patch.dict(
            os.environ,
            {
                "SQLITE_DB_PATH": str(database_path),
                "CHROMA_PERSIST_DIR": str(chroma_path),
            },
        ):
            get_settings.cache_clear()
            journals = [
                Journal(
                    name="Computer Vision Research",
                    research_fields=["Computer Vision"],
                    keywords=["image segmentation", "image recognition"],
                    aims_scope="Deep learning for visual understanding and images.",
                ),
                Journal(
                    name="Language and Retrieval Review",
                    research_fields=["Natural Language Processing", "Information Retrieval"],
                    keywords=["large language models", "retrieval augmented generation"],
                    aims_scope="Language understanding, search, and text retrieval.",
                ),
                Journal(
                    name="Software Engineering Methods",
                    research_fields=["Software Engineering"],
                    keywords=["software testing", "bug detection", "program analysis"],
                    aims_scope="Methods for reliable software development and testing.",
                ),
            ]
            for journal in journals:
                assert upsert_journal(journal) == "imported"

            first_build = build_journal_index()
            second_build = build_journal_index()
            assert first_build.indexed_count == 3
            assert second_build.indexed_count == 3

            vector_store = get_vector_store()
            assert get_indexed_count(vector_store) == 3
            results = vector_store.similarity_search(
                "deep learning computer vision image segmentation",
                k=1,
            )
            assert len(results) == 1
            assert results[0].metadata["name"] == "Computer Vision Research"

            vector_store.delete_collection()
            close_vector_store(vector_store)

        get_settings.cache_clear()

    print("Temporary Chroma full-rebuild and semantic-search test passed.")


if __name__ == "__main__":
    try:
        test_vector_store()
    except Exception as exc:
        print(f"ERROR: Vector store test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
