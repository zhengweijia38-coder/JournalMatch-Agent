"""Phase 2 end-to-end tests for CSV/XLSX import and repository queries."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.journal.importer import import_journals
from src.journal.repository import (
    get_all_journals,
    get_journal_by_name,
    get_journals_by_ccf_rank,
    get_journals_by_research_field,
)


def test_journal_importer() -> None:
    """Import CSV then XLSX data into an isolated SQLite database."""
    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        database_path = temporary_path / "journals.db"
        csv_path = temporary_path / "journals.csv"
        xlsx_path = temporary_path / "journals.xlsx"

        csv_data = pd.DataFrame(
            [
                {
                    " Name ": "Journal of AI Research",
                    "ISSN": "1111-2222",
                    "Research Fields": "Artificial Intelligence;Machine Learning",
                    "Keywords": "reasoning;planning",
                    "CCF Rank": "a",
                    "Impact Factor": "4.5",
                },
                {
                    " Name ": "Computer Vision Letters",
                    "Research Fields": "Computer Vision",
                    "Keywords": "segmentation;recognition",
                    "CCF Rank": "A 类",
                    "Impact Factor": 3.1,
                },
                {
                    " Name ": "Broken Metrics Journal",
                    "Research Fields": "Databases",
                    "Impact Factor": "not-a-number",
                },
            ]
        )
        csv_data.to_csv(csv_path, index=False, encoding="utf-8-sig")

        xlsx_data = pd.DataFrame(
            [
                {
                    "name": "Journal of AI Research",
                    "issn": "11112222",
                    "research_fields": "Artificial Intelligence;Machine Learning",
                    "keywords": "reasoning;planning",
                    "ccf_rank": "A",
                    "impact_factor": 5.0,
                },
                {
                    "name": "Database Systems Review",
                    "research_fields": "Databases;Information Retrieval",
                    "keywords": "query processing;indexing",
                    "ccf_rank": "B",
                    "impact_factor": 2.8,
                },
                {
                    "name": "Database Systems Review",
                    "research_fields": "Databases;Information Retrieval",
                    "keywords": "query processing;indexing",
                    "ccf_rank": "B",
                    "impact_factor": 2.8,
                },
            ]
        )
        xlsx_data.to_excel(xlsx_path, index=False)

        with patch.dict(os.environ, {"SQLITE_DB_PATH": str(database_path)}):
            get_settings.cache_clear()

            csv_report = import_journals(csv_path)
            assert csv_report.total_rows == 3
            assert csv_report.imported == 2
            assert csv_report.failed == 1
            assert "Row 4 failed" in csv_report.errors[0]

            xlsx_report = import_journals(xlsx_path)
            assert xlsx_report.total_rows == 3
            assert xlsx_report.imported == 1
            assert xlsx_report.updated == 1
            assert xlsx_report.skipped == 1
            assert xlsx_report.failed == 0

            journals = get_all_journals()
            assert len(journals) == 3
            assert len(get_journals_by_ccf_rank("a")) == 2
            assert len(get_journals_by_research_field("Databases")) == 1

            updated = get_journal_by_name("journal   of ai research")
            assert updated is not None
            assert updated.impact_factor == 5.0
            assert updated.research_fields == [
                "Artificial Intelligence",
                "Machine Learning",
            ]

            with closing(sqlite3.connect(database_path)) as connection:
                stored_json = connection.execute(
                    "SELECT research_fields FROM journals WHERE name = ?",
                    ("Journal of AI Research",),
                ).fetchone()[0]
            assert json.loads(stored_json) == updated.research_fields

        get_settings.cache_clear()

    print("CSV and XLSX journal importer tests passed.")


if __name__ == "__main__":
    try:
        test_journal_importer()
    except Exception as exc:
        print(f"ERROR: Journal importer test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
