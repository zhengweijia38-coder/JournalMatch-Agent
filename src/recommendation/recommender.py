"""Generate and validate DeepSeek recommendations over Phase 5 candidates."""

import json
import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.models.llm import get_llm
from src.exceptions import RecommendationError
from src.recommendation.prompt import RECOMMENDATION_PROMPT
from src.schemas.assessment import PaperQualityAssessment
from src.schemas.paper import PaperProfile
from src.schemas.recommendation import RecommendationReport
from src.schemas.retrieval import RerankedCandidate


logger = logging.getLogger(__name__)


class InvalidRecommendationError(RecommendationError):
    """Raised when structured output violates candidate or ranking constraints."""


def _known(value: Any) -> Any:
    """Represent absent candidate facts explicitly without asking the LLM to guess."""
    return "unknown" if value is None else value


def _build_candidate_payload(
    candidates: list[RerankedCandidate],
) -> list[dict[str, Any]]:
    """Serialize only candidate facts and retrieval provenance required by Phase 6."""
    payload: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for candidate in candidates:
        journal = candidate.journal
        if journal.journal_id is None:
            raise ValueError(
                f"Candidate journal '{journal.name}' has no stable journal_id."
            )
        if journal.journal_id in seen_ids:
            raise ValueError(
                f"Candidate list contains duplicate journal_id {journal.journal_id}."
            )
        seen_ids.add(journal.journal_id)
        payload.append(
            {
                "journal_id": journal.journal_id,
                "name": journal.name,
                "research_fields": journal.research_fields,
                "keywords": journal.keywords,
                "aims_scope": _known(journal.aims_scope),
                "ccf_rank": _known(journal.ccf_rank),
                "jcr_quartile": _known(journal.jcr_quartile),
                "cas_quartile": _known(journal.cas_quartile),
                "impact_factor": _known(journal.impact_factor),
                "semantic_rank": candidate.semantic_rank,
                "rerank_rank": candidate.rerank_rank,
                "retrieval_score": _known(candidate.retrieval_score),
                "rerank_score": candidate.rerank_score,
            }
        )
    return payload


def _validate_and_canonicalize_report(
    report: RecommendationReport,
    candidates: list[RerankedCandidate],
    maximum_recommendations: int,
) -> RecommendationReport:
    """Reject hallucinated identities and restore display names from SQLite data."""
    if len(report.recommendations) > maximum_recommendations:
        raise InvalidRecommendationError(
            "Structured recommendation returned more journals than the allowed top_k."
        )

    candidates_by_id = {
        candidate.journal.journal_id: candidate for candidate in candidates
    }
    canonical_recommendations = []
    for recommendation in report.recommendations:
        candidate = candidates_by_id.get(recommendation.journal_id)
        if candidate is None:
            raise InvalidRecommendationError(
                "Structured recommendation contains a journal outside the Phase 5 "
                f"candidate list: id={recommendation.journal_id}, "
                f"name='{recommendation.journal_name}'."
            )

        database_name = candidate.journal.name
        if recommendation.journal_name.strip().casefold() != database_name.casefold():
            raise InvalidRecommendationError(
                "Structured recommendation journal_id/name do not match the same "
                f"candidate: id={recommendation.journal_id}."
            )
        canonical_recommendations.append(
            recommendation.model_copy(update={"journal_name": database_name})
        )

    canonical_recommendations.sort(key=lambda item: item.final_rank)
    return report.model_copy(update={"recommendations": canonical_recommendations})


def generate_recommendations(
    profile: PaperProfile,
    candidates: list[RerankedCandidate],
    top_k: int = 5,
    quality_assessment: PaperQualityAssessment | None = None,
) -> RecommendationReport:
    """Ask DeepSeek to analyze only supplied candidates and enforce its boundaries."""
    if not candidates:
        raise ValueError(
            "Cannot generate recommendations because the Phase 5 candidate list is empty."
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    effective_top_k = min(top_k, len(candidates))
    candidate_payload = _build_candidate_payload(candidates)
    structured_llm = get_llm().with_structured_output(RecommendationReport)
    chain = RECOMMENDATION_PROMPT | structured_llm

    prompt_inputs = {
        "top_k": effective_top_k,
        "paper_profile_json": profile.model_dump_json(indent=2),
        "paper_quality_assessment_json": (
            quality_assessment.model_dump_json(indent=2)
            if quality_assessment is not None
            else "Not supplied by this compatibility caller."
        ),
        "candidate_journals_json": json.dumps(
            candidate_payload,
            ensure_ascii=False,
            indent=2,
        ),
        "validation_feedback": "No previous response; follow the output schema exactly.",
    }
    report: RecommendationReport | None = None
    last_validation_error: OutputParserException | ValidationError | None = None
    for attempt in range(2):
        try:
            raw_report = chain.invoke(prompt_inputs)
            report = (
                raw_report
                if isinstance(raw_report, RecommendationReport)
                else RecommendationReport.model_validate(raw_report)
            )
            break
        except (OutputParserException, ValidationError) as exc:
            last_validation_error = exc
            if attempt == 0:
                prompt_inputs["validation_feedback"] = (
                    "The previous response failed Pydantic validation. Correct the "
                    "schema errors below without changing candidate identities or "
                    f"inventing facts:\n{exc}"
                )
                continue
        except Exception as exc:
            raise RecommendationError(
                "DeepSeek API call failed during journal recommendation. Check the API "
                "configuration, network, and account status."
            ) from exc

    if report is None:
        raise InvalidRecommendationError(
            "DeepSeek returned an invalid structured recommendation after one "
            f"correction attempt: {last_validation_error}"
        ) from last_validation_error

    validated_report = _validate_and_canonicalize_report(
        report=report,
        candidates=candidates,
        maximum_recommendations=effective_top_k,
    )
    logger.info(
        "Generated %d grounded journal recommendations",
        len(validated_report.recommendations),
    )
    return validated_report
