"""Convert structured Journal objects into semantic LangChain Documents."""

from langchain_core.documents import Document

from src.schemas.journal import Journal


def journal_to_document(journal: Journal) -> Document:
    """Build embedding text and scalar metadata for one stored journal."""
    if journal.journal_id is None:
        raise ValueError(
            f"Journal '{journal.name}' has no journal_id. Load or store it through "
            "the repository before building a vector index."
        )

    sections = [f"Journal:\n{journal.name}"]
    if journal.research_fields:
        sections.append("Research Fields:\n" + "; ".join(journal.research_fields))
    if journal.keywords:
        sections.append("Keywords:\n" + "; ".join(journal.keywords))
    if journal.aims_scope:
        sections.append(f"Aims and Scope:\n{journal.aims_scope}")

    metadata: dict[str, str | int | float] = {
        "journal_id": journal.journal_id,
        "name": journal.name,
    }
    optional_metadata: dict[str, str | float | None] = {
        "ccf_rank": journal.ccf_rank,
        "jcr_quartile": journal.jcr_quartile,
        "cas_quartile": journal.cas_quartile,
        "impact_factor": journal.impact_factor,
    }
    metadata.update(
        {key: value for key, value in optional_metadata.items() if value is not None}
    )

    return Document(page_content="\n\n".join(sections), metadata=metadata)


def journals_to_documents(journals: list[Journal]) -> list[Document]:
    """Convert stored journals to Documents while preserving their order."""
    return [journal_to_document(journal) for journal in journals]
