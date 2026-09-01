"""Structured journal metadata stored in SQLite."""

from pydantic import BaseModel, ConfigDict, Field


class Journal(BaseModel):
    """Validated metadata for one academic journal or publication venue."""

    model_config = ConfigDict(str_strip_whitespace=True)

    journal_id: int | None = Field(
        default=None,
        description=(
            "The SQLite primary key populated by the repository for stable indexing; "
            "it is None before a new journal is stored."
        ),
    )
    name: str = Field(
        min_length=1,
        description="The full official or commonly used name of the journal.",
    )
    abbreviation: str | None = Field(
        default=None,
        description="The journal's standard abbreviation, when available.",
    )
    publication_type: str = Field(
        default="journal",
        min_length=1,
        description=(
            "The publication venue type, such as journal; defaults to journal."
        ),
    )
    publisher: str | None = Field(
        default=None,
        description="The organization or company that publishes the journal.",
    )
    issn: str | None = Field(
        default=None,
        description="The journal's print or primary ISSN, when available.",
    )
    eissn: str | None = Field(
        default=None,
        description="The journal's electronic ISSN, when available.",
    )
    research_fields: list[str] = Field(
        default_factory=list,
        description="Computer science research fields covered by the journal.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Topic keywords that describe the journal's publication scope.",
    )
    aims_scope: str | None = Field(
        default=None,
        description="The journal's stated aims and scope text.",
    )
    ccf_rank: str | None = Field(
        default=None,
        description="The CCF recommended rank, normally A, B, or C when known.",
    )
    jcr_quartile: str | None = Field(
        default=None,
        description="The journal's JCR quartile, such as Q1, when known.",
    )
    cas_quartile: str | None = Field(
        default=None,
        description="The journal's Chinese Academy of Sciences quartile or zone.",
    )
    impact_factor: float | None = Field(
        default=None,
        description="The journal's reported impact factor, when available.",
    )
    oa_type: str | None = Field(
        default=None,
        description="The journal's open-access type, such as gold, hybrid, or closed.",
    )
    apc: float | None = Field(
        default=None,
        description="The article processing charge as a numeric value, when known.",
    )
    homepage: str | None = Field(
        default=None,
        description="The journal's official homepage URL.",
    )
    source_url: str | None = Field(
        default=None,
        description="The URL from which this journal metadata was collected.",
    )
    updated_at: str | None = Field(
        default=None,
        description="The source data's update date or timestamp, when available.",
    )
