"""Read-only readiness checks that do not load models or call external APIs."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ENV_FILE, Settings, get_settings
from src.constants import JOURNAL_COLLECTION_NAME
from src.logging_config import configure_logging


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    """One safe readiness result and an optional recovery action."""

    label: str
    ok: bool
    status: str
    remediation: str | None = None


def _api_key_is_configured(settings: Settings) -> bool:
    """Check only presence/placeholder state; never return or print the secret."""
    key = settings.deepseek_api_key
    return bool(key and key != "your_deepseek_api_key")


def _inspect_sqlite(database_path: Path) -> tuple[bool, int | None, str | None]:
    """Inspect the journals table via a read-only SQLite connection."""
    if not database_path.is_file():
        return False, None, "database file is missing"

    connection: sqlite3.Connection | None = None
    try:
        uri = f"{database_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='journals'"
        ).fetchone()
        if table is None:
            return False, None, "journals table is missing"
        count = int(connection.execute("SELECT COUNT(*) FROM journals").fetchone()[0])
        if count == 0:
            return False, 0, "journals table is empty"
        return True, count, None
    except sqlite3.Error:
        return False, None, "database cannot be read"
    finally:
        if connection is not None:
            connection.close()


def _inspect_chroma(persist_directory: Path) -> tuple[bool, int | None, str | None]:
    """Inspect an existing collection without embedding initialization or writes."""
    if not persist_directory.is_dir() or not any(persist_directory.iterdir()):
        return False, None, "persist directory is missing or empty"

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_directory))
        collection = client.get_collection(JOURNAL_COLLECTION_NAME)
        count = int(collection.count())
        if count == 0:
            return False, 0, "journals collection is empty"
        return True, count, None
    except Exception:
        return False, None, "journals collection is unavailable"


def run_checks(require_api_key: bool = True) -> list[EnvironmentCheck]:
    """Return model-free environment checks suitable for scripts and smoke tests."""
    settings = get_settings()
    checks: list[EnvironmentCheck] = []

    python_ok = sys.version_info[:2] == (3, 11)
    checks.append(
        EnvironmentCheck(
            "Python 3.11",
            python_ok,
            "OK" if python_ok else f"found {sys.version.split()[0]}",
            None if python_ok else "Create and activate a Python 3.11 Conda environment.",
        )
    )
    checks.append(
        EnvironmentCheck(
            ".env file",
            ENV_FILE.is_file(),
            "OK" if ENV_FILE.is_file() else "missing",
            None if ENV_FILE.is_file() else "Run: Copy-Item .env.example .env",
        )
    )

    key_configured = _api_key_is_configured(settings)
    key_ok = key_configured or not require_api_key
    key_status = "configured" if key_configured else (
        "missing (not required)" if not require_api_key else "missing"
    )
    checks.append(
        EnvironmentCheck(
            "DeepSeek API Key",
            key_ok,
            key_status,
            None if key_ok else "Set DEEPSEEK_API_KEY in .env.",
        )
    )
    checks.extend(
        [
            EnvironmentCheck("DeepSeek Model", True, settings.deepseek_model),
            EnvironmentCheck("BGE Model Configuration", True, settings.bge_model_name),
            EnvironmentCheck(
                "Reranker Model Configuration",
                True,
                settings.reranker_model_name,
            ),
            EnvironmentCheck("SQLite DB Path", True, str(settings.sqlite_db_path)),
        ]
    )

    sqlite_ok, journal_count, sqlite_error = _inspect_sqlite(settings.sqlite_db_path)
    checks.append(
        EnvironmentCheck(
            "Journal Database",
            sqlite_ok,
            "OK" if sqlite_ok else str(sqlite_error),
            None
            if sqlite_ok
            else "Run: python scripts/init_journals.py data/journals/journals.xlsx",
        )
    )
    checks.append(
        EnvironmentCheck(
            "Journal Count",
            sqlite_ok and journal_count is not None,
            str(journal_count) if journal_count is not None else "unavailable",
        )
    )
    checks.append(
        EnvironmentCheck(
            "Chroma Persist Directory",
            settings.chroma_persist_dir.is_dir(),
            str(settings.chroma_persist_dir),
            None
            if settings.chroma_persist_dir.is_dir()
            else "Run: python scripts/build_vector_store.py",
        )
    )
    chroma_ok, chroma_count, chroma_error = _inspect_chroma(
        settings.chroma_persist_dir
    )
    if chroma_ok and journal_count is not None and chroma_count != journal_count:
        chroma_ok = False
        chroma_error = (
            f"count mismatch: SQLite={journal_count}, Chroma={chroma_count}"
        )
    checks.append(
        EnvironmentCheck(
            "Chroma Journal Index",
            chroma_ok,
            f"OK ({chroma_count} records)" if chroma_ok else str(chroma_error),
            None if chroma_ok else "Run: python scripts/build_vector_store.py",
        )
    )
    return checks


def print_checks(checks: list[EnvironmentCheck]) -> None:
    """Print aligned statuses and actionable recovery steps without secrets."""
    label_width = max(len(check.label) for check in checks)
    for check in checks:
        print(f"{check.label + ':':<{label_width + 2}} {check.status}")
    failures = [check for check in checks if not check.ok]
    if failures:
        print("\nRequired fixes:")
        for check in failures:
            if check.remediation:
                print(f"- {check.label}: {check.remediation}")


def main() -> int:
    """Run model-free checks and return a shell-friendly status code."""
    configure_logging(debug=False)
    try:
        checks = run_checks(require_api_key=True)
    except Exception:
        print(
            "Environment check could not load project settings. Verify .env and "
            ".env.example.",
            file=sys.stderr,
        )
        return 1

    print_checks(checks)
    if all(check.ok for check in checks):
        print("\nEnvironment check passed.")
        return 0
    print("\nEnvironment check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
