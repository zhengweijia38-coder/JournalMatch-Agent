"""High-level, read-only paper analysis tool."""

from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from src.assessment.assessor import assess_paper_quality
from src.paper.analyzer import analyze_paper
from src.paper.loader import combine_documents, load_pdf


class AnalyzePaperInput(BaseModel):
    """Validated input for paper-only analysis."""

    model_config = ConfigDict(extra="forbid")

    pdf_path: str = Field(
        min_length=1,
        description="Path to the specific text-based PDF supplied by the user.",
    )
    include_quality_assessment: bool = Field(
        default=True,
        description=(
            "Whether to include the five-dimensional evidence-based quality "
            "assessment; defaults to true."
        ),
    )


def _safe_paper_error(exc: Exception) -> str:
    """Return an actionable error without propagating tracebacks or API details."""
    if isinstance(exc, (FileNotFoundError, ValueError)):
        return str(exc)
    if isinstance(exc, RuntimeError):
        return str(exc).split("Original error:", maxsplit=1)[0].strip()
    return "Paper analysis failed because an unexpected local error occurred."


@tool("analyze_paper", args_schema=AnalyzePaperInput)
def analyze_paper_tool(
    pdf_path: str,
    include_quality_assessment: bool = True,
) -> dict[str, Any]:
    """Analyze a user-supplied PDF without recommending journals.

    Use this tool when the user wants to understand a paper, its direction,
    methods, contributions, experiments, or limitations without requesting
    journal recommendations.
    """
    try:
        documents = load_pdf(pdf_path)
        paper_text = combine_documents(documents)
        profile = analyze_paper(paper_text)
        result: dict[str, Any] = {
            "ok": True,
            "paper_profile": profile.model_dump(mode="json"),
        }
        if include_quality_assessment:
            assessment = assess_paper_quality(profile)
            result["paper_quality_assessment"] = assessment.model_dump(mode="json")
            result["assessment_note"] = (
                "Rubric-based decision support, not an acceptance probability. "
                "Preserve evidence boundaries: dataset names do not imply public, "
                "standard, benchmark, representative, or large-scale status."
            )
        return result
    except Exception as exc:
        return {"ok": False, "error": _safe_paper_error(exc)}
