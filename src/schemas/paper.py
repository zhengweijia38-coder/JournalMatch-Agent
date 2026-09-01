"""Structured information extracted from a research paper."""

from pydantic import BaseModel, Field


class PaperProfile(BaseModel):
    """Evidence-based profile of a computer science paper."""

    title: str | None = Field(
        default=None,
        description=(
            "The paper title exactly as stated in the provided text; use None when "
            "the title cannot be identified."
        ),
    )
    abstract: str | None = Field(
        default=None,
        description=(
            "The paper abstract as stated or faithfully extracted from the text; "
            "use None when no abstract is available."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Keywords explicitly listed by the paper or clearly used as its core "
            "technical topics; do not invent unrelated keywords."
        ),
    )
    research_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Standard computer science research fields directly supported by the "
            "paper text, such as Information Retrieval or Computer Vision."
        ),
    )
    research_problem: str | None = Field(
        default=None,
        description=(
            "The specific research problem or question addressed by the paper; use "
            "None if it is not clear from the text."
        ),
    )
    methods: list[str] = Field(
        default_factory=list,
        description=(
            "Methods, models, algorithms, or technical procedures actually used in "
            "the paper."
        ),
    )
    datasets: list[str] = Field(
        default_factory=list,
        description="Datasets explicitly named or described in the paper.",
    )
    main_contributions: list[str] = Field(
        default_factory=list,
        description=(
            "The paper's central contributions, supported by its explicit claims "
            "or direct statements."
        ),
    )
    claimed_innovations: list[str] = Field(
        default_factory=list,
        description=(
            "Innovations claimed by the authors or explicitly demonstrated in the "
            "paper; these are not subjective innovation scores or judgments."
        ),
    )
    experimental_results: list[str] = Field(
        default_factory=list,
        description=(
            "Experimental findings, measurements, or conclusions explicitly "
            "reported in the paper."
        ),
    )
    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Limitations explicitly discussed or directly extractable from the "
            "paper's statements; do not speculate."
        ),
    )
    summary: str = Field(
        description=(
            "A concise, factual summary of the entire paper based only on the "
            "provided text."
        )
    )
