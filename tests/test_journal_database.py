"""Phase 2 test for SQLite initialization and Journal persistence."""

from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.journal.database import initialize_database
from src.journal.repository import get_journal_by_name, upsert_journal
from src.schemas.journal import Journal


def test_journal_database() -> None:
    """Create an isolated database, insert one Journal, and read it back."""
    with TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "nested" / "journals.db"
        with patch.dict(os.environ, {"SQLITE_DB_PATH": str(database_path)}):
            get_settings.cache_clear()
            created_path = initialize_database()
            assert created_path == database_path.resolve()
            assert created_path.exists()

            with closing(sqlite3.connect(created_path)) as connection:
                table = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'journals'"
                ).fetchone()
            assert table is not None

            journal = Journal(
                name="Journal of Reliable Systems",
                issn="1234-5678",
                research_fields=["Software Engineering"],
                keywords=["reliability", "testing"],
                ccf_rank="B",
                impact_factor=4.2,
            )
            assert upsert_journal(journal) == "imported"

            loaded = get_journal_by_name("  journal of reliable systems ")
            assert loaded == journal

        get_settings.cache_clear()

    print("Journal database test passed.")


if __name__ == "__main__":
    try:
        test_journal_database()
    except Exception as exc:
        print(f"ERROR: Journal database test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
