"""Structured filters and results used by hybrid journal retrieval."""

from pydantic import BaseModel, Field, model_validator

from src.schemas.journal import Journal


class JournalFilters(BaseModel):
    """Exact SQLite filters applied after semantic retrieval."""

    ccf_ranks: list[str] = Field(
        default_factory=list,
        description="Allowed CCF ranks; multiple values use OR and an empty list is unrestricted.",
    )
    jcr_quartiles: list[str] = Field(
        default_factory=list,
        description="Allowed JCR quartiles; multiple values use OR and an empty list is unrestricted.",
    )
    cas_quartiles: list[str] = Field(
        default_factory=list,
        description="Allowed CAS zones; multiple values use OR and an empty list is unrestricted.",
    )
    min_impact_factor: float | None = Field(
        default=None,
        description="Inclusive minimum impact factor; missing journal values do not match.",
    )
    max_impact_factor: float | None = Field(
        default=None,
        description="Inclusive maximum impact factor; missing journal values do not match.",
    )

    @model_validator(mode="after")
    def validate_impact_factor_range(self) -> "JournalFilters":
        """Reject an impossible minimum/maximum range early."""
        if (
            self.min_impact_factor is not None
            and self.max_impact_factor is not None
            and self.min_impact_factor > self.max_impact_factor
        ):
            raise ValueError("min_impact_factor must not exceed max_impact_factor.")
        return self


class HybridCandidate(BaseModel):
    """A current SQLite Journal plus its original semantic retrieval position."""

    journal: Journal = Field(
        description="The latest structured journal record loaded from SQLite."
    )
    semantic_rank: int = Field(
        ge=1,
        description="The journal's original one-based rank from vector retrieval.",
    )
    retrieval_score: float | None = Field(
        default=None,
        description="The raw Chroma vector-search distance, not a probability.",
    )


class RerankedCandidate(BaseModel):
    """A HybridCandidate enriched with its cross-encoder reranking result."""

    journal: Journal = Field(
        description="The current structured journal record from the input candidate."
    )
    semantic_rank: int = Field(
        ge=1,
        description="The journal's original one-based Phase 3 semantic rank.",
    )
    retrieval_score: float | None = Field(
        default=None,
        description="The original raw Chroma distance, preserved for traceability.",
    )
    rerank_rank: int = Field(
        ge=1,
        description="The one-based position after sorting by raw reranker score.",
    )
    rerank_score: float = Field(
        description=(
            "The raw BGE reranker relevance score; it is not a probability or "
            "acceptance likelihood."
        ),
    )
