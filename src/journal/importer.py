"""Import cleaned journal metadata from CSV or XLSX into SQLite."""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.journal.database import initialize_database
from src.journal.normalization import normalize_ccf_rank
from src.journal.repository import upsert_journal
from src.schemas.journal import Journal


logger = logging.getLogger(__name__)


SUPPORTED_COLUMNS = {
    "name",
    "abbreviation",
    "publication_type",
    "publisher",
    "issn",
    "eissn",
    "research_fields",
    "keywords",
    "aims_scope",
    "ccf_rank",
    "jcr_quartile",
    "cas_quartile",
    "impact_factor",
    "oa_type",
    "apc",
    "homepage",
    "source_url",
    "updated_at",
}

TEXT_COLUMNS = {
    "name",
    "abbreviation",
    "publisher",
    "issn",
    "eissn",
    "aims_scope",
    "jcr_quartile",
    "cas_quartile",
    "oa_type",
    "homepage",
    "source_url",
    "updated_at",
}


@dataclass(slots=True)
class ImportReport:
    """Counters and row-level errors produced by one import operation."""

    total_rows: int
    imported: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format the required human-readable import summary."""
        return "\n".join(
            [
                f"Total rows: {self.total_rows}",
                f"Imported: {self.imported}",
                f"Updated: {self.updated}",
                f"Skipped: {self.skipped}",
                f"Failed: {self.failed}",
            ]
        )


def _standardize_column_name(column: object) -> str:
    """Trim and convert a source column name to lowercase snake_case."""
    normalized = str(column).strip().casefold()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    return normalized.strip("_")


def _read_dataframe(source_path: Path) -> pd.DataFrame:
    """Read the supported tabular source format into a DataFrame."""
    try:
        if source_path.suffix.lower() == ".xlsx":
            return pd.read_excel(source_path, dtype=object)
        return pd.read_csv(source_path, dtype=object, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read journal data file '{source_path}': {exc}"
        ) from exc


def _is_missing(value: Any) -> bool:
    """Recognize None, pandas NaN/NA, and blank strings as missing values."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_optional_text(value: Any) -> str | None:
    """Convert a non-empty scalar to trimmed text, otherwise return None."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _to_string_list(value: Any) -> list[str]:
    """Split semicolon-delimited fields and remove empty or duplicate entries."""
    if _is_missing(value):
        return []

    raw_items = value if isinstance(value, (list, tuple, set)) else re.split(
        r"[;；]", str(value)
    )
    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = str(raw_item).strip()
        comparison_key = item.casefold()
        if item and comparison_key not in seen:
            seen.add(comparison_key)
            items.append(item)
    return items


def _to_optional_float(value: Any, field_name: str) -> float | None:
    """Safely convert a numeric cell or raise a clear row-level data error."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not a boolean value.")

    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} value {value!r} cannot be safely converted to float."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}.")
    return number


def _row_to_journal(row: pd.Series) -> Journal:
    """Clean one DataFrame row and validate it as a Journal."""
    data: dict[str, Any] = {}
    for column in TEXT_COLUMNS:
        if column in row.index:
            data[column] = _to_optional_text(row[column])

    if "publication_type" in row.index:
        publication_type = _to_optional_text(row["publication_type"])
        if publication_type is not None:
            data["publication_type"] = publication_type

    data["research_fields"] = _to_string_list(row.get("research_fields"))
    data["keywords"] = _to_string_list(row.get("keywords"))
    data["ccf_rank"] = normalize_ccf_rank(
        _to_optional_text(row.get("ccf_rank"))
    )
    data["impact_factor"] = _to_optional_float(
        row.get("impact_factor"), "impact_factor"
    )
    data["apc"] = _to_optional_float(row.get("apc"), "apc")

    return Journal.model_validate(data)


def import_journals(path: str | Path) -> ImportReport:
    """Import journal rows independently and return a complete import report."""
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Journal data file does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Journal data path is not a file: {source_path}")
    if source_path.suffix.lower() not in {".xlsx", ".csv"}:
        raise ValueError(
            f"Unsupported journal data format '{source_path.suffix}'. "
            "Only .xlsx and .csv files are supported."
        )

    dataframe = _read_dataframe(source_path)
    standardized_columns = [
        _standardize_column_name(column) for column in dataframe.columns
    ]
    duplicates = sorted(
        {
            column
            for column in standardized_columns
            if standardized_columns.count(column) > 1
        }
    )
    if duplicates:
        raise ValueError(
            "Column names become duplicates after standardization: "
            + ", ".join(duplicates)
        )
    dataframe.columns = standardized_columns

    if "name" not in dataframe.columns:
        raise ValueError(
            "Journal data must contain a 'name' column after column standardization."
        )

    ignored_columns = set(dataframe.columns) - SUPPORTED_COLUMNS
    if ignored_columns:
        logger.warning(
            "Ignoring unsupported columns: %s",
            ", ".join(sorted(ignored_columns)),
        )

    initialize_database()
    report = ImportReport(total_rows=len(dataframe))

    for row_number, (_, row) in enumerate(dataframe.iterrows(), start=2):
        try:
            journal = _row_to_journal(row)
            result = upsert_journal(journal)
            if result == "imported":
                report.imported += 1
            elif result == "updated":
                report.updated += 1
            else:
                report.skipped += 1
        except (ValidationError, ValueError, RuntimeError) as exc:
            message = f"Row {row_number} failed: {exc}"
            report.failed += 1
            report.errors.append(message)
            logger.warning(message)
        except Exception as exc:
            message = f"Row {row_number} failed with an unexpected error: {exc}"
            report.failed += 1
            report.errors.append(message)
            logger.error(message)

    return report
