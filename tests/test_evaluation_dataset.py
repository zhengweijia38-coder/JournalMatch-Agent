"""Unit tests for JSONL evaluation dataset loading and validation."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.evaluation.dataset as dataset_module
from src.schemas.journal import Journal


def _record(case_id: str = "case_1") -> dict[str, object]:
    return {
        "case_id": case_id,
        "query": "medical image segmentation",
        "research_fields": ["Computer Vision"],
        "graded_relevance": {"1": 3, "2": 1},
        "filters": {
            "ccf_ranks": ["A", "B"],
            "jcr_quartiles": [],
            "cas_quartiles": [],
            "min_impact_factor": None,
            "max_impact_factor": None,
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_evaluation_dataset() -> None:
    """Validate parsing, duplicate IDs, empty queries, grades, and SQLite IDs."""
    with TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir) / "cases.jsonl"
        _write_jsonl(dataset_path, [_record("case_1"), _record("case_2")])
        dataset = dataset_module.load_evaluation_dataset(dataset_path)
        assert len(dataset.cases) == 2
        assert dataset.cases[0].graded_relevance == {1: 3, 2: 1}
        assert dataset.cases[0].filters.ccf_ranks == ["A", "B"]

        with patch.object(
            dataset_module,
            "get_journals_by_ids",
            return_value=[Journal(journal_id=1, name="Journal One")],
        ):
            try:
                dataset_module.validate_gold_journals(dataset, strict=True)
            except ValueError as exc:
                assert "2" in str(exc)
            else:
                raise AssertionError("Missing SQLite gold IDs should be rejected.")

        invalid_sets = [
            [_record("duplicate"), _record("duplicate")],
            [{**_record(), "query": "   "}],
            [{**_record(), "graded_relevance": {"1": -1}}],
            [{**_record(), "graded_relevance": {"1": 4}}],
        ]
        for records in invalid_sets:
            _write_jsonl(dataset_path, records)
            try:
                dataset_module.load_evaluation_dataset(dataset_path)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid evaluation dataset should be rejected.")

    print("Evaluation dataset validation tests passed.")


if __name__ == "__main__":
    try:
        test_evaluation_dataset()
    except Exception as exc:
        print(f"ERROR: Evaluation dataset test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
