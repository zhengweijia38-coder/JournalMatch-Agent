"""Phase 4 unit tests for exact JournalFilters semantics."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.filters import filter_journals, matches_filters
from src.schemas.journal import Journal
from src.schemas.retrieval import JournalFilters


def make_journal(
    name: str,
    ccf: str | None,
    jcr: str | None,
    cas: str | None,
    impact_factor: float | None,
) -> Journal:
    return Journal(
        name=name,
        ccf_rank=ccf,
        jcr_quartile=jcr,
        cas_quartile=cas,
        impact_factor=impact_factor,
    )


def test_filters() -> None:
    """Verify within-field OR, cross-field AND, boundaries, and missing values."""
    journal_a = make_journal("Journal A", "A", "Q1", "1区-Top", 5.0)
    journal_b = make_journal("Journal B", "B", "Q2", "2区", 7.5)
    journal_c = make_journal("Journal C", "C", "Q3", "3区", 4.9)
    journal_missing = make_journal("Missing IF", "A", "Q1", "1区-Top", None)

    assert matches_filters(journal_a, JournalFilters())
    assert matches_filters(journal_a, JournalFilters(ccf_ranks=["a"]))
    assert not matches_filters(journal_b, JournalFilters(ccf_ranks=["A类"]))

    ccf_ab = JournalFilters(ccf_ranks=["A", "b"])
    assert matches_filters(journal_a, ccf_ab)
    assert matches_filters(journal_b, ccf_ab)
    assert not matches_filters(journal_c, ccf_ab)

    combined = JournalFilters(
        ccf_ranks=["A", "B"],
        jcr_quartiles=["q1", "Q2"],
        cas_quartiles=["1", "2区"],
        min_impact_factor=5.0,
    )
    assert matches_filters(journal_a, combined)
    assert matches_filters(journal_b, combined)
    assert not matches_filters(journal_c, combined)

    boundary = JournalFilters(min_impact_factor=5.0, max_impact_factor=7.5)
    assert matches_filters(journal_a, boundary)
    assert matches_filters(journal_b, boundary)
    assert not matches_filters(journal_c, boundary)
    assert not matches_filters(journal_missing, boundary)
    assert not matches_filters(
        journal_missing, JournalFilters(min_impact_factor=5.0)
    )

    filtered = filter_journals(
        [journal_c, journal_a, journal_b],
        JournalFilters(ccf_ranks=["A", "B"]),
    )
    assert filtered == [journal_a, journal_b]

    print("JournalFilters AND/OR, boundary, and missing-value tests passed.")


if __name__ == "__main__":
    try:
        test_filters()
    except Exception as exc:
        print(f"ERROR: Filter test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
