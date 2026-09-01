"""Offline tests for fixed profiles plus manually authored Gold CSV building."""

import csv
import json
from pathlib import Path
import sqlite3

import pytest

from src.evaluation.data_preparation import (
    EvaluationDataError,
    build_evaluation_dataset,
)
from src.evaluation.dataset import load_evaluation_dataset
from src.schemas.paper import PaperProfile


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE journals (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO journals (id, name) VALUES (?, ?)",
            [
                (1, "Journal One"),
                (2, "Journal Two"),
                (3, "Journal Three"),
                (4, "Published Journal"),
            ],
        )


def _write_profile(directory: Path, case_id: str) -> None:
    profile = PaperProfile(
        title=f"Paper {case_id}",
        research_fields=["Information Retrieval"],
        keywords=["retrieval"],
        methods=["dense retrieval"],
        summary="A fixed profile for retrieval evaluation.",
    )
    payload = {
        "case_id": case_id,
        "source_pdf": f"{case_id}.pdf",
        "paper_profile": profile.model_dump(mode="json"),
    }
    (directory / f"{case_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _gold_row(
    *,
    case_id: str = "case_1",
    journal_id: object = 1,
    journal_name: str = "Journal One",
    relevance: object = 3,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "journal_id": journal_id,
        "journal_name": journal_name,
        "relevance": relevance,
        "note": "Human scope annotation.",
    }


def _prepare_paths(tmp_path: Path) -> dict[str, Path]:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    database = tmp_path / "journals.db"
    _create_database(database)
    _write_profile(profiles, "case_1")
    _write_profile(profiles, "case_2")
    return {
        "profiles": profiles,
        "gold": annotations / "gold_labels.csv",
        "metadata": annotations / "paper_metadata.csv",
        "database": database,
        "output": tmp_path / "retrieval_cases.jsonl",
    }


def _write_empty_metadata(path: Path) -> None:
    _write_csv(
        path,
        [
            "case_id",
            "source_pdf",
            "published_journal_id",
            "published_journal_name",
            "notes",
        ],
        [],
    )


def _build(paths: dict[str, Path]):
    return build_evaluation_dataset(
        profiles_dir=paths["profiles"],
        gold_labels_path=paths["gold"],
        paper_metadata_path=paths["metadata"],
        output_path=paths["output"],
        database_path=paths["database"],
    )


def test_build_profiles_and_manual_gold_into_phase7_jsonl(tmp_path: Path) -> None:
    """Build a fixed-profile case and retain publication/profile warnings."""
    paths = _prepare_paths(tmp_path)
    _write_csv(
        paths["gold"],
        ["case_id", "journal_id", "journal_name", "relevance", "note"],
        [
            _gold_row(journal_id="journal-1", relevance=3),
            _gold_row(journal_id=2, journal_name="journal two", relevance=2),
            _gold_row(journal_id=3, journal_name="Reviewer Display Name", relevance=1),
        ],
    )
    _write_csv(
        paths["metadata"],
        [
            "case_id",
            "source_pdf",
            "published_journal_id",
            "published_journal_name",
            "notes",
        ],
        [
            {
                "case_id": "case_1",
                "source_pdf": "case_1.pdf",
                "published_journal_id": 4,
                "published_journal_name": "Published Journal",
                "notes": "Human-provided publication metadata.",
            }
        ],
    )

    report = _build(paths)
    assert report.total_cases == 1
    assert report.total_gold_labels == 3
    assert report.relevance_counts == {3: 1, 2: 1, 1: 1}
    assert report.profiles_without_gold_labels == ["case_2"]
    assert any("Published journal is not included" in item for item in report.warnings)
    assert any("case_2" in item for item in report.warnings)
    assert any("differs from SQLite name 'Journal Three'" in item for item in report.warnings)

    lines = paths["output"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    raw_record = json.loads(lines[0])
    assert raw_record["published_journal_id"] == 4
    assert raw_record["paper_profile"]["title"] == "Paper case_1"
    assert raw_record["graded_relevance"] == {"1": 3, "2": 2, "3": 1}
    assert raw_record["filters"]["ccf_ranks"] == []

    dataset = load_evaluation_dataset(paths["output"])
    assert len(dataset.cases) == 1
    assert dataset.cases[0].paper_profile is not None
    assert "Paper case_1" in dataset.cases[0].query


@pytest.mark.parametrize("relevance", [0, 4])
def test_builder_rejects_invalid_relevance(tmp_path: Path, relevance: int) -> None:
    paths = _prepare_paths(tmp_path)
    _write_csv(
        paths["gold"],
        ["case_id", "journal_id", "journal_name", "relevance", "note"],
        [_gold_row(relevance=relevance)],
    )
    _write_empty_metadata(paths["metadata"])
    with pytest.raises(EvaluationDataError, match="relevance must be 1, 2, or 3"):
        _build(paths)


def test_builder_rejects_unknown_duplicate_and_missing_profile(tmp_path: Path) -> None:
    paths = _prepare_paths(tmp_path)
    _write_empty_metadata(paths["metadata"])
    fields = ["case_id", "journal_id", "journal_name", "relevance", "note"]

    _write_csv(
        paths["gold"],
        fields,
        [_gold_row(journal_id=99, journal_name="Unknown Journal")],
    )
    with pytest.raises(EvaluationDataError, match="Unknown journal IDs"):
        _build(paths)

    _write_csv(
        paths["gold"],
        fields,
        [_gold_row(), _gold_row(relevance=2)],
    )
    with pytest.raises(EvaluationDataError, match="Duplicate gold label"):
        _build(paths)

    _write_csv(
        paths["gold"],
        fields,
        [_gold_row(case_id="missing_case")],
    )
    with pytest.raises(EvaluationDataError, match="missing profile"):
        _build(paths)
