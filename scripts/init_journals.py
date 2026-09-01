"""Initialize SQLite and import journal metadata from CSV or XLSX."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.journal.database import initialize_database
from src.journal.importer import import_journals


def parse_args() -> argparse.Namespace:
    """Parse the journal data file path."""
    parser = argparse.ArgumentParser(
        description="Import journal metadata from an XLSX or CSV file into SQLite."
    )
    parser.add_argument("data_path", type=Path, help="Path to journals.xlsx or CSV")
    return parser.parse_args()


def main() -> int:
    """Initialize the configured database and print the final import report."""
    args = parse_args()
    try:
        database_path = initialize_database()
        report = import_journals(args.data_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected journal import failure: {exc}", file=sys.stderr)
        return 1

    print(f"Database: {database_path}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
