"""Automatically verifiable hard metrics for Phase 6 recommendation reports."""

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from src.journal.repository import get_journals_by_ids
from src.schemas.journal import Journal
from src.schemas.recommendation import RecommendationReport
from src.schemas.retrieval import RerankedCandidate


JournalLookup = Callable[[list[int]], list[Journal]]
METADATA_FIELDS = ("name", "ccf_rank", "jcr_quartile", "cas_quartile", "impact_factor")


def evaluate_recommendation_hard_metrics(
    report: RecommendationReport | dict[str, Any],
    candidates: list[RerankedCandidate],
    *,
    top_k: int,
    journal_lookup: JournalLookup = get_journals_by_ids,
) -> dict[str, Any]:
    """Check schema, containment, count limits, and SQLite metadata faithfulness."""
    try:
        validated_report = RecommendationReport.model_validate(report)
    except ValidationError as exc:
        return {
            "structured_output_validity": False,
            "candidate_containment": False,
            "metadata_faithfulness": False,
            "recommendation_count_validity": False,
            "errors": [str(exc)],
        }

    candidate_by_id = {
        candidate.journal.journal_id: candidate
        for candidate in candidates
        if candidate.journal.journal_id is not None
    }
    recommendation_ids = [
        recommendation.journal_id
        for recommendation in validated_report.recommendations
    ]
    outside_ids = [
        journal_id for journal_id in recommendation_ids if journal_id not in candidate_by_id
    ]
    count_valid = len(recommendation_ids) <= min(top_k, len(candidates))

    sqlite_journals = journal_lookup(recommendation_ids)
    sqlite_by_id = {journal.journal_id: journal for journal in sqlite_journals}
    metadata_mismatches: list[dict[str, Any]] = []
    for journal_id in recommendation_ids:
        candidate = candidate_by_id.get(journal_id)
        sqlite_journal = sqlite_by_id.get(journal_id)
        if candidate is None or sqlite_journal is None:
            metadata_mismatches.append(
                {"journal_id": journal_id, "reason": "missing candidate or SQLite row"}
            )
            continue
        differing_fields = [
            field
            for field in METADATA_FIELDS
            if getattr(candidate.journal, field) != getattr(sqlite_journal, field)
        ]
        if differing_fields:
            metadata_mismatches.append(
                {"journal_id": journal_id, "differing_fields": differing_fields}
            )

    return {
        "structured_output_validity": True,
        "candidate_containment": not outside_ids,
        "metadata_faithfulness": not metadata_mismatches,
        "recommendation_count_validity": count_valid,
        "outside_candidate_journal_ids": outside_ids,
        "metadata_mismatches": metadata_mismatches,
        "recommendation_count": len(recommendation_ids),
        "top_k": top_k,
        "candidate_count": len(candidates),
    }
