"""Store and query Journal objects without exposing raw SQLite rows."""

import json
import sqlite3
from typing import Literal

from src.journal.database import (
    database_connection,
    initialize_database,
    normalize_issn,
    normalize_journal_name,
)
from src.schemas.journal import Journal


SaveResult = Literal["imported", "updated", "skipped"]

JOURNAL_COLUMNS = (
    "name",
    "normalized_name",
    "abbreviation",
    "publication_type",
    "publisher",
    "issn",
    "normalized_issn",
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
)


def _journal_to_record(journal: Journal) -> dict[str, str | float | None]:
    """Convert a Journal into SQLite-compatible scalar values."""
    data = journal.model_dump()
    return {
        "name": journal.name,
        "normalized_name": normalize_journal_name(journal.name),
        "abbreviation": journal.abbreviation,
        "publication_type": journal.publication_type,
        "publisher": journal.publisher,
        "issn": journal.issn,
        "normalized_issn": normalize_issn(journal.issn),
        "eissn": journal.eissn,
        "research_fields": json.dumps(
            data["research_fields"], ensure_ascii=False
        ),
        "keywords": json.dumps(data["keywords"], ensure_ascii=False),
        "aims_scope": journal.aims_scope,
        "ccf_rank": journal.ccf_rank,
        "jcr_quartile": journal.jcr_quartile,
        "cas_quartile": journal.cas_quartile,
        "impact_factor": journal.impact_factor,
        "oa_type": journal.oa_type,
        "apc": journal.apc,
        "homepage": journal.homepage,
        "source_url": journal.source_url,
        "updated_at": journal.updated_at,
    }


def _row_to_journal(row: sqlite3.Row) -> Journal:
    """Convert a SQLite row into a validated Journal object."""
    try:
        research_fields = json.loads(row["research_fields"])
        keywords = json.loads(row["keywords"])
        return Journal(
            journal_id=row["id"],
            name=row["name"],
            abbreviation=row["abbreviation"],
            publication_type=row["publication_type"],
            publisher=row["publisher"],
            issn=row["issn"],
            eissn=row["eissn"],
            research_fields=research_fields,
            keywords=keywords,
            aims_scope=row["aims_scope"],
            ccf_rank=row["ccf_rank"],
            jcr_quartile=row["jcr_quartile"],
            cas_quartile=row["cas_quartile"],
            impact_factor=row["impact_factor"],
            oa_type=row["oa_type"],
            apc=row["apc"],
            homepage=row["homepage"],
            source_url=row["source_url"],
            updated_at=row["updated_at"],
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Journal row {row['id']} contains invalid stored data: {exc}"
        ) from exc


def _find_existing_journal(
    connection: sqlite3.Connection,
    record: dict[str, str | float | None],
) -> sqlite3.Row | None:
    """Find a duplicate by ISSN first, then safely fall back to normalized name."""
    normalized_issn = record["normalized_issn"]
    name_row = connection.execute(
        "SELECT * FROM journals WHERE normalized_name = ?",
        (record["normalized_name"],),
    ).fetchone()

    if normalized_issn:
        issn_row = connection.execute(
            "SELECT * FROM journals WHERE normalized_issn = ?",
            (normalized_issn,),
        ).fetchone()
        if issn_row is not None:
            if name_row is not None and name_row["id"] != issn_row["id"]:
                raise ValueError(
                    "The ISSN and normalized journal name match different existing "
                    "records. Resolve the source data conflict before importing."
                )
            return issn_row

        if name_row is not None:
            existing_issn = name_row["normalized_issn"]
            if existing_issn and existing_issn != normalized_issn:
                raise ValueError(
                    f"Journal name '{record['name']}' already exists with a different ISSN."
                )
            return name_row
        return None

    return name_row


def upsert_journal(journal: Journal) -> SaveResult:
    """Insert, update, or skip one journal using ISSN/name duplicate rules."""
    initialize_database()
    record = _journal_to_record(journal)

    with database_connection() as connection:
        existing = _find_existing_journal(connection, record)
        try:
            if existing is None:
                columns = ", ".join(JOURNAL_COLUMNS)
                placeholders = ", ".join(f":{column}" for column in JOURNAL_COLUMNS)
                cursor = connection.execute(
                    f"INSERT INTO journals ({columns}) VALUES ({placeholders})",
                    record,
                )
                journal.journal_id = cursor.lastrowid
                return "imported"

            if all(existing[column] == record[column] for column in JOURNAL_COLUMNS):
                journal.journal_id = existing["id"]
                return "skipped"

            assignments = ", ".join(
                f"{column} = :{column}" for column in JOURNAL_COLUMNS
            )
            record_with_id = {**record, "id": existing["id"]}
            connection.execute(
                f"UPDATE journals SET {assignments} WHERE id = :id",
                record_with_id,
            )
            journal.journal_id = existing["id"]
            return "updated"
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Journal '{journal.name}' conflicts with an existing name or ISSN: {exc}"
            ) from exc


def get_all_journals() -> list[Journal]:
    """Return all journals ordered by name."""
    initialize_database()
    with database_connection() as connection:
        rows = connection.execute("SELECT * FROM journals ORDER BY name").fetchall()
    return [_row_to_journal(row) for row in rows]


def get_journal_by_name(name: str) -> Journal | None:
    """Return one journal by case-insensitive normalized name."""
    initialize_database()
    normalized_name = normalize_journal_name(name)
    with database_connection() as connection:
        row = connection.execute(
            "SELECT * FROM journals WHERE normalized_name = ?",
            (normalized_name,),
        ).fetchone()
    return _row_to_journal(row) if row is not None else None


def get_journal_by_id(journal_id: int) -> Journal | None:
    """Return the latest SQLite Journal for one primary key."""
    initialize_database()
    with database_connection() as connection:
        row = connection.execute(
            "SELECT * FROM journals WHERE id = ?",
            (journal_id,),
        ).fetchone()
    return _row_to_journal(row) if row is not None else None


def get_journals_by_ids(journal_ids: list[int]) -> list[Journal]:
    """Return current SQLite Journals in the caller's stable ID order."""
    unique_ids = list(dict.fromkeys(journal_ids))
    if not unique_ids:
        return []

    initialize_database()
    placeholders = ", ".join("?" for _ in unique_ids)
    with database_connection() as connection:
        rows = connection.execute(
            f"SELECT * FROM journals WHERE id IN ({placeholders})",
            unique_ids,
        ).fetchall()
    journals_by_id = {
        journal.journal_id: journal for journal in map(_row_to_journal, rows)
    }
    return [
        journals_by_id[journal_id]
        for journal_id in unique_ids
        if journal_id in journals_by_id
    ]


def count_journals() -> int:
    """Return the current number of journal records in SQLite."""
    initialize_database()
    with database_connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM journals").fetchone()
    return int(row["count"])


def get_journals_by_ccf_rank(rank: str) -> list[Journal]:
    """Return journals matching a CCF rank without case sensitivity."""
    initialize_database()
    normalized_rank = rank.strip().upper()
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM journals
            WHERE UPPER(TRIM(ccf_rank)) = ?
            ORDER BY name
            """,
            (normalized_rank,),
        ).fetchall()
    return [_row_to_journal(row) for row in rows]


def get_journals_by_research_field(field: str) -> list[Journal]:
    """Return journals whose decoded research field list contains the field."""
    normalized_field = field.strip().casefold()
    if not normalized_field:
        return []
    return [
        journal
        for journal in get_all_journals()
        if any(
            research_field.casefold() == normalized_field
            for research_field in journal.research_fields
        )
    ]
