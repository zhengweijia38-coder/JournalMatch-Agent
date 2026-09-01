"""Build deterministic semantic search text from a PaperProfile."""

from src.schemas.paper import PaperProfile


def build_paper_query(profile: PaperProfile) -> str:
    """Select research-scope fields from a PaperProfile for semantic retrieval."""
    sections: list[str] = []
    if profile.title:
        sections.append(f"Title:\n{profile.title}")
    if profile.research_fields:
        sections.append("Research Fields:\n" + "; ".join(profile.research_fields))
    if profile.keywords:
        sections.append("Keywords:\n" + "; ".join(profile.keywords))
    if profile.research_problem:
        sections.append(f"Research Problem:\n{profile.research_problem}")
    if profile.methods:
        sections.append("Methods:\n" + "; ".join(profile.methods))
    if profile.summary and profile.summary.strip():
        sections.append(f"Summary:\n{profile.summary.strip()}")

    if not sections:
        raise ValueError(
            "PaperProfile does not contain any fields suitable for semantic retrieval."
        )
    return "\n\n".join(sections)
