"""SQLite connection and schema initialization for journal metadata."""

from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
from collections.abc import Iterator

from src.config import get_settings


CREATE_JOURNALS_TABLE = """
CREATE TABLE IF NOT EXISTS journals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    abbreviation TEXT,
    publication_type TEXT NOT NULL DEFAULT 'journal',
    publisher TEXT,
    issn TEXT,
    normalized_issn TEXT,
    eissn TEXT,
    research_fields TEXT NOT NULL DEFAULT '[]',
    keywords TEXT NOT NULL DEFAULT '[]',
    aims_scope TEXT,
    ccf_rank TEXT,
    jcr_quartile TEXT,
    cas_quartile TEXT,
    impact_factor REAL,
    oa_type TEXT,
    apc REAL,
    homepage TEXT,
    source_url TEXT,
    updated_at TEXT
)
"""


def normalize_journal_name(name: str) -> str:
    """Build a case-insensitive, whitespace-normalized journal name key."""
    return " ".join(name.casefold().split())


def normalize_issn(issn: str | None) -> str | None:
    """Build an ISSN comparison key while preserving the original stored value."""
    if not issn:
        return None
    normalized = re.sub(r"[\s-]+", "", issn).upper()
    return normalized or None


def get_database_path() -> Path:
    """Return the configured SQLite path without hardcoding a location."""
    return get_settings().sqlite_db_path


def get_connection() -> sqlite3.Connection:
    """Open a configured SQLite connection whose rows support name lookup."""
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    """Yield a transactional connection and always close it on Windows."""
    connection = get_connection()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> Path:
    """Create the database directory, journals table, and uniqueness indexes."""
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with database_connection() as connection:
        connection.execute(CREATE_JOURNALS_TABLE)
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_journals_normalized_name
            ON journals(normalized_name)
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_journals_normalized_issn
            ON journals(normalized_issn)
            WHERE normalized_issn IS NOT NULL
            """
        )

    return database_path
