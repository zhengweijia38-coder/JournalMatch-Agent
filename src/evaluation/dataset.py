"""Pydantic models and JSONL loading for offline evaluation datasets."""

import json
from pathlib import Path
import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.journal.repository import get_journals_by_ids
from src.retrieval.query_builder import build_paper_query
from src.schemas.paper import PaperProfile
from src.schemas.retrieval import JournalFilters


class EvaluationCase(BaseModel):
    """One manually labeled query and its graded relevant journal IDs."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    case_id: str = Field(min_length=1, description="Unique evaluation case identifier.")
    source_pdf: str | None = Field(
        default=None,
        description="Original PDF filename used to create the fixed PaperProfile.",
    )
    published_journal_id: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional actual published journal ID for analysis only; it is not "
            "automatically added to graded relevance."
        ),
    )
    paper_profile: PaperProfile | None = Field(
        default=None,
        description="Fixed Phase 1 profile used unchanged across retrieval experiments.",
    )
    query: str = Field(
        default="",
        description=(
            "Query used unchanged across stages; deterministically derived from "
            "paper_profile when omitted."
        ),
    )
    research_fields: list[str] = Field(
        default_factory=list,
        description="Human-provided research fields for analysis and reporting.",
    )
    graded_relevance: dict[int, int] = Field(
        min_length=1,
        description="Stable SQLite journal_id to manually assigned grade 1, 2, or 3.",
    )
    filters: JournalFilters = Field(
        default_factory=JournalFilters,
        description="Optional Phase 4 structured constraints for this case.",
    )

    @field_validator("graded_relevance")
    @classmethod
    def validate_graded_relevance(cls, value: dict[int, int]) -> dict[int, int]:
        for journal_id, relevance in value.items():
            if journal_id <= 0:
                raise ValueError("Gold journal_id values must be positive integers.")
            if relevance not in {1, 2, 3}:
                raise ValueError("Gold relevance must be one of 1, 2, or 3.")
        return value

    @model_validator(mode="after")
    def derive_fixed_query_from_profile(self) -> "EvaluationCase":
        """Reuse the production query builder while preserving legacy query cases."""
        if not self.query.strip():
            if self.paper_profile is None:
                raise ValueError("Evaluation case requires query or paper_profile.")
            self.query = build_paper_query(self.paper_profile)
        if not self.research_fields and self.paper_profile is not None:
            self.research_fields = list(self.paper_profile.research_fields)
        return self


class EvaluationDataset(BaseModel):
    """Validated collection of uniquely identified evaluation cases."""

    model_config = ConfigDict(extra="forbid")

    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "EvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation dataset contains duplicate case_id values.")
        return self


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load non-empty JSONL records and return a validated EvaluationDataset."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")
    if dataset_path.suffix.casefold() != ".jsonl":
        raise ValueError("Evaluation dataset must be a .jsonl file.")

    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            cases.append(EvaluationCase.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on evaluation dataset line {line_number}: {exc}"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: {exc}"
            ) from exc
    return EvaluationDataset(cases=cases)


def validate_gold_journals(
    dataset: EvaluationDataset,
    *,
    strict: bool = True,
) -> list[int]:
    """Check that every positive-grade gold journal ID exists in current SQLite."""
    gold_ids = sorted(
        {
            journal_id
            for case in dataset.cases
            for journal_id, grade in case.graded_relevance.items()
            if grade > 0
        }
    )
    existing_ids = {
        journal.journal_id
        for journal in get_journals_by_ids(gold_ids)
        if journal.journal_id is not None
    }
    missing_ids = [journal_id for journal_id in gold_ids if journal_id not in existing_ids]
    if missing_ids:
        message = (
            "Evaluation gold journal_id values do not exist in SQLite: "
            + ", ".join(map(str, missing_ids))
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    return missing_ids
