"""Offline JSONL format, uniqueness, and SQLite validation tests."""

import json
from pathlib import Path
import sqlite3

import pytest

from src.evaluation.data_preparation import (
    EvaluationDataError,
    validate_evaluation_jsonl,
)


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE journals (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO journals (id, name) VALUES (?, ?)",
            [(1, "Journal One"), (2, "Journal Two")],
        )


def _record(
    case_id: str = "case_1",
    *,
    relevance: int = 3,
    journal_id: str = "1",
    published_journal_id: int | None = 2,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "source_pdf": f"{case_id}.pdf",
        "published_journal_id": published_journal_id,
        "paper_profile": {
            "title": f"Paper {case_id}",
            "research_fields": ["Information Retrieval"],
            "keywords": ["retrieval"],
            "methods": ["dense retrieval"],
            "summary": "A fixed profile for validation.",
        },
        "graded_relevance": {journal_id: relevance},
        "filters": {
            "ccf_ranks": [],
            "jcr_quartiles": [],
            "cas_quartiles": [],
            "min_impact_factor": None,
            "max_impact_factor": None,
        },
    }


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_validator_accepts_round_trip_and_warns_for_published_not_gold(
    tmp_path: Path,
) -> None:
    database = tmp_path / "journals.db"
    _create_database(database)
    dataset_path = tmp_path / "cases.jsonl"
    _write_records(dataset_path, [_record()])

    dataset, report = validate_evaluation_jsonl(
        dataset_path,
        database_path=database,
    )
    assert len(dataset.cases) == 1
    assert dataset.cases[0].graded_relevance == {1: 3}
    assert report.total_gold_labels == 1
    assert any("Published journal is not included" in item for item in report.warnings)


def test_validator_rejects_duplicate_case_and_journal_keys(tmp_path: Path) -> None:
    database = tmp_path / "journals.db"
    _create_database(database)
    dataset_path = tmp_path / "cases.jsonl"
    duplicate = _record()
    _write_records(dataset_path, [duplicate, duplicate])
    with pytest.raises(EvaluationDataError, match="duplicate case_id"):
        validate_evaluation_jsonl(dataset_path, database_path=database)

    raw = json.dumps(_record())
    raw = raw.replace('"graded_relevance": {"1": 3}', '"graded_relevance": {"1": 3, "1": 2}')
    dataset_path.write_text(raw + "\n", encoding="utf-8")
    with pytest.raises(EvaluationDataError, match="Duplicate JSON key '1'"):
        validate_evaluation_jsonl(dataset_path, database_path=database)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_record(relevance=0), "Gold relevance"),
        (_record(relevance=4), "Gold relevance"),
        (_record(journal_id="99"), "do not exist in SQLite"),
        (_record(published_journal_id=99), "do not exist in SQLite"),
        ({**_record(), "graded_relevance": {}}, "at least 1 item"),
    ],
)
def test_validator_rejects_invalid_labels_and_ids(
    tmp_path: Path,
    record: dict[str, object],
    message: str,
) -> None:
    database = tmp_path / "journals.db"
    _create_database(database)
    dataset_path = tmp_path / "cases.jsonl"
    _write_records(dataset_path, [record])
    with pytest.raises(EvaluationDataError, match=message):
        validate_evaluation_jsonl(dataset_path, database_path=database)
