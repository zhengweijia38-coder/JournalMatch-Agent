"""Build and validate fixed-profile, manually labeled evaluation datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import get_settings
from src.evaluation.dataset import EvaluationCase, EvaluationDataset
from src.schemas.paper import PaperProfile
from src.schemas.retrieval import JournalFilters


CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEFAULT_FILTERS = JournalFilters()


class EvaluationDataError(ValueError):
    """Raised when manually prepared evaluation data is unsafe to build."""


class EvaluationProfileArtifact(BaseModel):
    """Stable on-disk wrapper around the existing PaperProfile schema."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, pattern=CASE_ID_PATTERN.pattern)
    source_pdf: str = Field(min_length=1)
    paper_profile: PaperProfile


@dataclass(slots=True)
class EvaluationDataReport:
    """Validation statistics and non-mutating annotation warnings."""

    total_cases: int = 0
    total_gold_labels: int = 0
    relevance_counts: dict[int, int] = field(
        default_factory=lambda: {3: 0, 2: 0, 1: 0}
    )
    average_gold_journals_per_case: float = 0.0
    profiles_without_gold_labels: list[str] = field(default_factory=list)
    gold_case_ids_missing_profiles: list[str] = field(default_factory=list)
    unknown_journal_ids: list[int] = field(default_factory=list)
    duplicate_labels: int = 0
    warnings: list[str] = field(default_factory=list)
    output_path: Path | None = None


def parse_journal_id(value: str, *, context: str) -> int:
    """Accept the current integer ID and the human-friendly journal-12 form."""
    normalized = value.strip()
    match = re.fullmatch(r"(?:journal-)?([0-9]+)", normalized, re.IGNORECASE)
    if match is None or int(match.group(1)) <= 0:
        raise EvaluationDataError(
            f"{context}: journal_id must be a positive SQLite integer or journal-<id>."
        )
    return int(match.group(1))


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _read_csv_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read a UTF-8 CSV and enforce its manual annotation contract."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotation CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        actual_columns = set(reader.fieldnames or [])
        missing = sorted(required_columns - actual_columns)
        if missing:
            raise EvaluationDataError(
                f"{path.name} is missing required columns: {', '.join(missing)}"
            )
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def load_journal_catalog(database_path: Path | None = None) -> dict[int, str]:
    """Read journal IDs and names without creating or mutating SQLite."""
    resolved_path = (database_path or get_settings().sqlite_db_path).resolve()
    if not resolved_path.is_file():
        raise EvaluationDataError(f"Journal SQLite database does not exist: {resolved_path}")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{resolved_path.as_uri()}?mode=ro",
            uri=True,
        )
        rows = connection.execute("SELECT id, name FROM journals ORDER BY id").fetchall()
    except sqlite3.Error as exc:
        raise EvaluationDataError(
            f"Journal SQLite database could not be read: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if not rows:
        raise EvaluationDataError("Journal SQLite database contains no journals.")
    return {int(row[0]): str(row[1]) for row in rows}


def load_profile_artifacts(profiles_dir: Path) -> dict[str, EvaluationProfileArtifact]:
    """Load profiles in stable filename order and reject ambiguous case IDs."""
    if not profiles_dir.is_dir():
        raise EvaluationDataError(f"Profiles directory does not exist: {profiles_dir}")
    profiles: dict[str, EvaluationProfileArtifact] = {}
    seen_casefold: dict[str, str] = {}
    for profile_path in sorted(profiles_dir.glob("*.json"), key=lambda path: path.name.casefold()):
        try:
            artifact = EvaluationProfileArtifact.model_validate_json(
                profile_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise EvaluationDataError(
                f"Invalid profile file '{profile_path.name}': {exc}"
            ) from exc
        if artifact.case_id != profile_path.stem:
            raise EvaluationDataError(
                f"Profile '{profile_path.name}' contains case_id '{artifact.case_id}'; "
                "case_id must exactly match the filename stem."
            )
        folded = artifact.case_id.casefold()
        if folded in seen_casefold:
            raise EvaluationDataError(
                "Duplicate profile case_id values: "
                f"{seen_casefold[folded]} and {artifact.case_id}."
            )
        seen_casefold[folded] = artifact.case_id
        profiles[artifact.case_id] = artifact
    return profiles


def load_gold_labels(
    path: Path,
) -> tuple[dict[str, dict[int, int]], dict[tuple[str, int], str], int]:
    """Load exclusively human-authored labels and reject duplicate decisions."""
    rows = _read_csv_rows(
        path,
        {"case_id", "journal_id", "journal_name", "relevance", "note"},
    )
    labels: dict[str, dict[int, int]] = {}
    names: dict[tuple[str, int], str] = {}
    seen: set[tuple[str, int]] = set()
    duplicate_count = 0
    for row_number, row in enumerate(rows, start=2):
        case_id = row["case_id"]
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise EvaluationDataError(
                f"gold_labels.csv row {row_number}: invalid case_id '{case_id}'."
            )
        journal_id = parse_journal_id(
            row["journal_id"],
            context=f"gold_labels.csv row {row_number}",
        )
        key = (case_id, journal_id)
        if key in seen:
            duplicate_count += 1
            raise EvaluationDataError(
                "Duplicate gold label for "
                f"case_id='{case_id}', journal_id={journal_id}."
            )
        seen.add(key)
        try:
            relevance = int(row["relevance"])
        except ValueError as exc:
            raise EvaluationDataError(
                f"gold_labels.csv row {row_number}: relevance must be 1, 2, or 3."
            ) from exc
        if relevance not in {1, 2, 3}:
            raise EvaluationDataError(
                f"gold_labels.csv row {row_number}: relevance must be 1, 2, or 3."
            )
        if not row["journal_name"]:
            raise EvaluationDataError(
                f"gold_labels.csv row {row_number}: journal_name is required for review."
            )
        labels.setdefault(case_id, {})[journal_id] = relevance
        names[key] = row["journal_name"]
    return labels, names, duplicate_count


def load_paper_metadata(path: Path | None) -> dict[str, dict[str, str | int | None]]:
    """Load optional human-authored publication metadata without inferring values."""
    if path is None or not path.exists():
        return {}
    rows = _read_csv_rows(
        path,
        {
            "case_id",
            "source_pdf",
            "published_journal_id",
            "published_journal_name",
            "notes",
        },
    )
    metadata: dict[str, dict[str, str | int | None]] = {}
    for row_number, row in enumerate(rows, start=2):
        case_id = row["case_id"]
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise EvaluationDataError(
                f"paper_metadata.csv row {row_number}: invalid case_id '{case_id}'."
            )
        if case_id in metadata:
            raise EvaluationDataError(
                f"paper_metadata.csv contains duplicate case_id '{case_id}'."
            )
        journal_id = (
            parse_journal_id(
                row["published_journal_id"],
                context=f"paper_metadata.csv row {row_number}",
            )
            if row["published_journal_id"]
            else None
        )
        metadata[case_id] = {
            "source_pdf": row["source_pdf"],
            "published_journal_id": journal_id,
            "published_journal_name": row["published_journal_name"],
            "notes": row["notes"],
        }
    return metadata


def _check_annotation_ids_and_names(
    labels: dict[str, dict[int, int]],
    label_names: dict[tuple[str, int], str],
    metadata: dict[str, dict[str, str | int | None]],
    journal_catalog: dict[int, str],
) -> tuple[list[str], list[int]]:
    warnings: list[str] = []
    unknown: set[int] = set()
    for case_id, case_labels in labels.items():
        for journal_id in case_labels:
            actual_name = journal_catalog.get(journal_id)
            csv_name = label_names[(case_id, journal_id)]
            if actual_name is None:
                unknown.add(journal_id)
                continue
            if _normalize_name(csv_name) != _normalize_name(actual_name):
                warnings.append(
                    f"case_id={case_id}, journal_id={journal_id}: CSV journal_name "
                    f"'{csv_name}' differs from SQLite name '{actual_name}'."
                )
    for case_id, values in metadata.items():
        published_id = values["published_journal_id"]
        if not isinstance(published_id, int):
            continue
        actual_name = journal_catalog.get(published_id)
        if actual_name is None:
            unknown.add(published_id)
            continue
        metadata_name = str(values["published_journal_name"] or "")
        if metadata_name and _normalize_name(metadata_name) != _normalize_name(actual_name):
            warnings.append(
                f"case_id={case_id}, published_journal_id={published_id}: metadata "
                f"name '{metadata_name}' differs from SQLite name '{actual_name}'."
            )
    return warnings, sorted(unknown)


def _build_report(
    cases: list[EvaluationCase],
    *,
    warnings: list[str],
    profiles_without_gold: list[str] | None = None,
    output_path: Path | None = None,
) -> EvaluationDataReport:
    relevance_counts = {3: 0, 2: 0, 1: 0}
    total_labels = 0
    for case in cases:
        for relevance in case.graded_relevance.values():
            relevance_counts[relevance] += 1
            total_labels += 1
    return EvaluationDataReport(
        total_cases=len(cases),
        total_gold_labels=total_labels,
        relevance_counts=relevance_counts,
        average_gold_journals_per_case=(
            total_labels / len(cases) if cases else 0.0
        ),
        profiles_without_gold_labels=profiles_without_gold or [],
        warnings=warnings,
        output_path=output_path,
    )


def build_evaluation_dataset(
    *,
    profiles_dir: Path,
    gold_labels_path: Path,
    paper_metadata_path: Path | None,
    output_path: Path,
    database_path: Path | None = None,
) -> EvaluationDataReport:
    """Combine fixed profiles and manual labels without generating any relevance."""
    profiles = load_profile_artifacts(profiles_dir)
    labels, label_names, duplicate_count = load_gold_labels(gold_labels_path)
    metadata = load_paper_metadata(paper_metadata_path)
    catalog = load_journal_catalog(database_path)

    missing_profiles = sorted(set(labels) - set(profiles))
    if missing_profiles:
        raise EvaluationDataError(
            "Gold case IDs are missing profile files: " + ", ".join(missing_profiles)
        )
    metadata_missing_profiles = sorted(set(metadata) - set(profiles))
    if metadata_missing_profiles:
        raise EvaluationDataError(
            "Metadata case IDs are missing profile files: "
            + ", ".join(metadata_missing_profiles)
        )

    warnings, unknown_ids = _check_annotation_ids_and_names(
        labels,
        label_names,
        metadata,
        catalog,
    )
    if unknown_ids:
        details = []
        for case_id, case_labels in labels.items():
            for journal_id in case_labels:
                if journal_id in unknown_ids:
                    details.append(
                        f"case_id={case_id}, journal_id={journal_id}, "
                        f"journal_name='{label_names[(case_id, journal_id)]}', "
                        "reason=not found in SQLite"
                    )
        for case_id, values in metadata.items():
            published_id = values["published_journal_id"]
            if isinstance(published_id, int) and published_id in unknown_ids:
                details.append(
                    f"case_id={case_id}, journal_id={published_id}, "
                    f"journal_name='{values['published_journal_name']}', "
                    "reason=published journal not found in SQLite"
                )
        raise EvaluationDataError("Unknown journal IDs:\n" + "\n".join(details))

    profiles_without_gold = sorted(set(profiles) - set(labels))
    for case_id in profiles_without_gold:
        warnings.append(f"Profile case_id={case_id} has no manually curated gold labels.")

    cases: list[EvaluationCase] = []
    output_records: list[dict[str, Any]] = []
    for case_id in sorted(labels, key=str.casefold):
        artifact = profiles[case_id]
        case_labels = labels[case_id]
        label_count = len(case_labels)
        if label_count < 3:
            warnings.append(
                f"case_id={case_id}: Gold set may be too small for reliable "
                "multi-label retrieval evaluation."
            )
        if label_count > 20:
            warnings.append(
                f"case_id={case_id}: Gold set is unusually large; verify annotation quality."
            )
        metadata_values = metadata.get(case_id, {})
        published_id = metadata_values.get("published_journal_id")
        if isinstance(published_id, int) and published_id not in case_labels:
            warnings.append(
                f"case_id={case_id}: Published journal is not included in the "
                "manually curated gold labels."
            )
        case = EvaluationCase(
            case_id=case_id,
            source_pdf=artifact.source_pdf,
            published_journal_id=(published_id if isinstance(published_id, int) else None),
            paper_profile=artifact.paper_profile,
            graded_relevance=case_labels,
            filters=DEFAULT_FILTERS,
        )
        cases.append(case)
        output_records.append(
            {
                "case_id": case.case_id,
                "source_pdf": case.source_pdf,
                "published_journal_id": case.published_journal_id,
                "paper_profile": artifact.paper_profile.model_dump(mode="json"),
                "graded_relevance": {
                    str(journal_id): relevance
                    for journal_id, relevance in sorted(case_labels.items())
                },
                "filters": DEFAULT_FILTERS.model_dump(mode="json"),
            }
        )

    if not cases:
        raise EvaluationDataError(
            "No evaluation cases can be built. Add manual rows to gold_labels.csv."
        )
    EvaluationDataset(cases=cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in output_records
    ) + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    report = _build_report(
        cases,
        warnings=warnings,
        profiles_without_gold=profiles_without_gold,
        output_path=output_path,
    )
    report.duplicate_labels = duplicate_count
    return report


class DuplicateJSONKeyError(ValueError):
    """Raised before json.loads can silently overwrite a duplicate key."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"Duplicate JSON key '{key}'.")
        result[key] = value
    return result


def validate_evaluation_jsonl(
    path: Path,
    *,
    database_path: Path | None = None,
) -> tuple[EvaluationDataset, EvaluationDataReport]:
    """Validate JSONL structure, Pydantic cases, uniqueness, and SQLite IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            data = json.loads(
                raw_line,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            cases.append(EvaluationCase.model_validate(data))
        except (json.JSONDecodeError, DuplicateJSONKeyError, ValidationError) as exc:
            raise EvaluationDataError(
                f"Invalid evaluation JSONL line {line_number}: {exc}"
            ) from exc
    try:
        dataset = EvaluationDataset(cases=cases)
    except ValidationError as exc:
        raise EvaluationDataError(f"Invalid evaluation dataset: {exc}") from exc

    catalog = load_journal_catalog(database_path)
    unknown_gold = sorted(
        {
            journal_id
            for case in cases
            for journal_id in case.graded_relevance
            if journal_id not in catalog
        }
    )
    unknown_published = sorted(
        {
            case.published_journal_id
            for case in cases
            if case.published_journal_id is not None
            and case.published_journal_id not in catalog
        }
    )
    unknown_ids = sorted(set(unknown_gold) | set(unknown_published))
    if unknown_ids:
        raise EvaluationDataError(
            "Evaluation journal_id values do not exist in SQLite: "
            + ", ".join(map(str, unknown_ids))
        )

    warnings: list[str] = []
    for case in cases:
        if (
            case.published_journal_id is not None
            and case.published_journal_id not in case.graded_relevance
        ):
            warnings.append(
                f"case_id={case.case_id}: Published journal is not included in the "
                "manually curated gold labels."
            )
    report = _build_report(cases, warnings=warnings)
    return dataset, report


def possible_venue_leakage(
    profile: PaperProfile,
    published_journal_name: str | None,
) -> list[str]:
    """Warn about visible venue clues without modifying the fixed profile."""
    visible_text = "\n".join(
        value
        for value in (profile.title, profile.abstract, profile.summary)
        if value
    ).casefold()
    warnings: list[str] = []
    if published_journal_name and published_journal_name.strip().casefold() in visible_text:
        warnings.append(
            "published_journal_name appears directly in title/abstract/summary"
        )
    if re.search(r"\b(?:published|appeared)\s+in\b|\bdoi\s*:", visible_text):
        warnings.append("publication or DOI wording appears in title/abstract/summary")
    return warnings
