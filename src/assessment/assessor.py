"""Create a structured paper quality assessment from an existing PaperProfile."""

import logging
import re

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.assessment.prompt import ASSESSMENT_PROMPT
from src.exceptions import PaperAssessmentError
from src.models.llm import get_llm
from src.schemas.assessment import DimensionAssessment, PaperQualityAssessment
from src.schemas.paper import PaperProfile


logger = logging.getLogger(__name__)

_UNSUPPORTED_DATASET_STATUS_PHRASES = (
    "benchmark",
    "large-scale",
    "standard reported",
    "standard dataset",
    "standard benchmark",
    "standard machine translation",
    "well-established",
    "widely recognized",
    "public benchmark",
    "representative dataset",
)

_DATASET_STATUS_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        r"(?:standard\s+)?(?:large-scale\s+)?machine translation benchmarks?",
        "machine translation datasets/tasks named in PaperProfile",
    ),
    (r"standard machine translation benchmarks?", "machine translation tasks named in PaperProfile"),
    (r"a standard benchmark", "a dataset/task named in PaperProfile"),
    (r"(?:standard|well-established|public) benchmarks?", "datasets/tasks named in PaperProfile"),
    (r"widely recognized", "named in PaperProfile"),
    (r"representative datasets?", "datasets named in PaperProfile"),
    (r"large-scale", "reported"),
    (r"benchmarks?", "reported datasets/tasks"),
    (r"standard reported", "profile-named"),
    (r"standard datasets?", "datasets named in PaperProfile"),
)

_DATASET_ADEQUACY_INDICATORS = (
    "benchmark",
    "representative",
    "public dataset",
    "training split",
    "validation split",
    "test split",
    "preprocessing",
    "data source",
    "external validation",
    "cross-domain",
)


def _iter_assessment_text(assessment: PaperQualityAssessment) -> list[str]:
    """Flatten all generated explanatory strings for grounding checks."""
    texts = [*assessment.strengths, *assessment.weaknesses]
    for dimension in (
        assessment.novelty,
        assessment.methodology,
        assessment.dataset_quality,
        assessment.experimental_quality,
        assessment.conclusion_support,
    ):
        texts.extend(dimension.evidence)
        texts.extend(dimension.concerns)
    return texts


def _validate_profile_grounding(
    assessment: PaperQualityAssessment,
    profile: PaperProfile,
) -> None:
    """Reject common dataset-status claims that are absent from PaperProfile."""
    profile_text = profile.model_dump_json().casefold()
    unsupported: set[str] = set()
    for statement in _iter_assessment_text(assessment):
        normalized = statement.casefold()
        for phrase in _UNSUPPORTED_DATASET_STATUS_PHRASES:
            if phrase in normalized and phrase not in profile_text:
                unsupported.add(phrase)
    if unsupported:
        phrases = ", ".join(sorted(unsupported))
        raise ValueError(
            "The assessment added dataset-status knowledge absent from PaperProfile: "
            f"{phrases}. Dataset names alone do not establish those properties."
        )


def _neutralize_unsupported_status(text: str, profile_text: str) -> tuple[str, bool]:
    """Replace unsupported dataset-status language with a profile-only description."""
    updated = text
    changed = False
    for pattern, replacement in _DATASET_STATUS_REPLACEMENTS:
        if re.search(pattern, profile_text, flags=re.IGNORECASE):
            continue
        updated, count = re.subn(
            pattern,
            replacement,
            updated,
            flags=re.IGNORECASE,
        )
        changed = changed or count > 0
    return updated, changed


def _canonicalize_dataset_grounding(
    assessment: PaperQualityAssessment,
    profile: PaperProfile,
) -> PaperQualityAssessment:
    """Neutralize unsupported dataset reputation and conservatively cap its score."""
    profile_text = profile.model_dump_json()
    any_changed = False

    def canonicalize_dimension(
        dimension: DimensionAssessment,
    ) -> DimensionAssessment:
        nonlocal any_changed
        evidence: list[str] = []
        concerns: list[str] = []
        for value in dimension.evidence:
            updated, changed = _neutralize_unsupported_status(value, profile_text)
            evidence.append(updated)
            any_changed = any_changed or changed
        for value in dimension.concerns:
            updated, changed = _neutralize_unsupported_status(value, profile_text)
            concerns.append(updated)
            any_changed = any_changed or changed
        return dimension.model_copy(
            update={"evidence": evidence, "concerns": concerns}
        )

    updates = {
        "novelty": canonicalize_dimension(assessment.novelty),
        "methodology": canonicalize_dimension(assessment.methodology),
        "dataset_quality": canonicalize_dimension(assessment.dataset_quality),
        "experimental_quality": canonicalize_dimension(assessment.experimental_quality),
        "conclusion_support": canonicalize_dimension(assessment.conclusion_support),
    }
    strengths: list[str] = []
    weaknesses: list[str] = []
    for value in assessment.strengths:
        updated, changed = _neutralize_unsupported_status(value, profile_text)
        strengths.append(updated)
        any_changed = any_changed or changed
    for value in assessment.weaknesses:
        updated, changed = _neutralize_unsupported_status(value, profile_text)
        weaknesses.append(updated)
        any_changed = any_changed or changed

    dataset_quality = updates["dataset_quality"]
    has_adequacy_evidence = any(
        indicator in profile_text.casefold()
        for indicator in _DATASET_ADEQUACY_INDICATORS
    )
    if dataset_quality.score > 3 and not has_adequacy_evidence:
        concern = (
            "PaperProfile does not establish dataset source, split quality, "
            "preprocessing, representativeness, or external validation."
        )
        concerns = list(dataset_quality.concerns)
        if concern not in concerns:
            concerns.append(concern)
        dataset_quality = dataset_quality.model_copy(
            update={"score": 3, "level": "Moderate", "concerns": concerns}
        )
        updates["dataset_quality"] = dataset_quality
        any_changed = True

    canonical = assessment.model_copy(
        update={**updates, "strengths": strengths, "weaknesses": weaknesses}
    )
    validated = PaperQualityAssessment.model_validate(canonical.model_dump())
    if any_changed:
        logger.warning(
            "Canonicalized unsupported dataset-status language using PaperProfile boundaries"
        )
    return validated


def assess_paper_quality(profile: PaperProfile) -> PaperQualityAssessment:
    """Assess a PaperProfile with the cached LLM and Pydantic structured output."""
    structured_llm = get_llm().with_structured_output(PaperQualityAssessment)
    chain = ASSESSMENT_PROMPT | structured_llm

    prompt_inputs = {
        "paper_profile_json": profile.model_dump_json(indent=2),
        "validation_feedback": "No previous attempt; follow the schema and grounding rules.",
    }
    assessment: PaperQualityAssessment | None = None
    last_validation_error: OutputParserException | ValidationError | ValueError | None = None
    logger.info("Assessing paper quality from PaperProfile evidence")
    for attempt in range(3):
        try:
            raw_assessment = chain.invoke(prompt_inputs)
            candidate = (
                raw_assessment
                if isinstance(raw_assessment, PaperQualityAssessment)
                else PaperQualityAssessment.model_validate(raw_assessment)
            )
            candidate = _canonicalize_dataset_grounding(candidate, profile)
            _validate_profile_grounding(candidate, profile)
            assessment = candidate
            break
        except (OutputParserException, ValidationError, ValueError) as exc:
            last_validation_error = exc
            logger.debug("Assessment rejection reason: %s", exc)
            if attempt < 2:
                logger.warning(
                    "Assessment validation failed; requesting a complete grounded correction"
                )
                prompt_inputs["validation_feedback"] = (
                    "The previous response was rejected. Return a brand-new COMPLETE "
                    "PaperQualityAssessment, not a partial patch. It must contain all "
                    "five dimensions, overall_maturity, strengths, and weaknesses. "
                    "Correct the following issue without adding new PaperProfile facts:\n"
                    f"{exc}"
                )
                continue
        except Exception as exc:
            raise PaperAssessmentError(
                "DeepSeek API call failed during paper quality assessment. Check the "
                "API configuration, network, account status, and PaperProfile content."
            ) from exc

    if assessment is None:
        raise PaperAssessmentError(
            "DeepSeek returned an invalid or ungrounded PaperQualityAssessment after "
            "two correction attempts. Run with --debug for the chained error."
        ) from last_validation_error

    logger.info("Paper quality assessment produced five validated dimensions")
    return assessment
