"""Pure Python structured filtering for current SQLite Journal records."""

from collections.abc import Iterable
from collections.abc import Callable

from src.journal.normalization import (
    normalize_cas_quartile,
    normalize_ccf_rank,
    normalize_jcr_quartile,
)
from src.schemas.journal import Journal
from src.schemas.retrieval import JournalFilters


def _normalized_allowed(
    values: list[str],
    normalizer: Callable[[str | None], str | None],
) -> set[str]:
    """Normalize requested values and discard only empty inputs."""
    normalized_values = {
        normalized.casefold()
        for value in values
        if (normalized := normalizer(value)) is not None
    }
    return normalized_values


def _matches_allowed(
    journal_value: str | None,
    allowed_values: set[str],
    normalizer: Callable[[str | None], str | None],
) -> bool:
    """Apply one unrestricted-or-OR field condition."""
    if not allowed_values:
        return True
    normalized_journal_value = normalizer(journal_value)
    if normalized_journal_value is None:
        return False
    return normalized_journal_value.casefold() in allowed_values


def matches_filters(journal: Journal, filters: JournalFilters) -> bool:
    """Apply OR within fields and AND across all structured filter fields."""
    allowed_ccf = _normalized_allowed(filters.ccf_ranks, normalize_ccf_rank)
    allowed_jcr = _normalized_allowed(filters.jcr_quartiles, normalize_jcr_quartile)
    allowed_cas = _normalized_allowed(filters.cas_quartiles, normalize_cas_quartile)

    if not _matches_allowed(journal.ccf_rank, allowed_ccf, normalize_ccf_rank):
        return False
    if not _matches_allowed(
        journal.jcr_quartile, allowed_jcr, normalize_jcr_quartile
    ):
        return False
    if not _matches_allowed(
        journal.cas_quartile, allowed_cas, normalize_cas_quartile
    ):
        return False

    if filters.min_impact_factor is not None:
        if (
            journal.impact_factor is None
            or journal.impact_factor < filters.min_impact_factor
        ):
            return False
    if filters.max_impact_factor is not None:
        if (
            journal.impact_factor is None
            or journal.impact_factor > filters.max_impact_factor
        ):
            return False
    return True


def filter_journals(
    journals: Iterable[Journal],
    filters: JournalFilters,
) -> list[Journal]:
    """Filter journals without changing their existing order."""
    return [journal for journal in journals if matches_filters(journal, filters)]
