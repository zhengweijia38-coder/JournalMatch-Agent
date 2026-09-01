"""Persistent Chroma vector store and full journal index construction."""

from dataclasses import dataclass
import logging
from pathlib import Path

from langchain_chroma import Chroma

from src.config import get_settings
from src.constants import JOURNAL_COLLECTION_NAME
from src.exceptions import VectorStoreError
from src.journal.document_builder import journals_to_documents
from src.journal.repository import get_all_journals
from src.models.embeddings import get_embeddings


COLLECTION_NAME = JOURNAL_COLLECTION_NAME
COLLECTION_METADATA = {"hnsw:space": "cosine"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Summary of one full journal index rebuild."""

    journal_count: int
    document_count: int
    indexed_count: int
    persist_directory: Path
    collection_name: str

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Journal count from SQLite: {self.journal_count}",
                f"Document count: {self.document_count}",
                f"Indexed count: {self.indexed_count}",
                f"Persist directory: {self.persist_directory}",
                f"Collection name: {self.collection_name}",
            ]
        )


def get_vector_store() -> Chroma:
    """Open the configured persistent journals collection."""
    persist_directory = get_settings().chroma_persist_dir
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_metadata=COLLECTION_METADATA,
    )


def close_vector_store(vector_store: Chroma) -> None:
    """Release the underlying Chroma client and Windows file handles."""
    client = getattr(vector_store, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def get_indexed_count(vector_store: Chroma | None = None) -> int:
    """Return the number of stable IDs currently stored in the collection."""
    owns_store = vector_store is None
    store = vector_store or get_vector_store()
    try:
        stored = store.get(include=[])
        return len(stored["ids"])
    finally:
        if owns_store:
            close_vector_store(store)


def build_journal_index() -> IndexBuildResult:
    """Fully rebuild the journals collection from SQLite source-of-truth data."""
    journals = get_all_journals()
    if not journals:
        raise VectorStoreError(
            "SQLite contains no journals. Import journal data before building Chroma."
        )

    documents = journals_to_documents(journals)
    logger.info("Building Chroma journal index from %d journals", len(journals))
    document_ids = [
        f"journal-{document.metadata['journal_id']}" for document in documents
    ]

    vector_store = get_vector_store()
    try:
        vector_store.delete_collection()
    finally:
        close_vector_store(vector_store)
    vector_store = get_vector_store()

    try:
        indexed_ids = vector_store.add_documents(documents, ids=document_ids)
        indexed_count = get_indexed_count(vector_store)
    except Exception as exc:
        raise VectorStoreError(
            "Failed to build the Chroma journal index. Run with --debug for the "
            "chained Chroma error."
        ) from exc
    finally:
        close_vector_store(vector_store)
    if indexed_count != len(documents) or len(indexed_ids) != len(documents):
        raise VectorStoreError(
            "Chroma index count mismatch: "
            f"expected {len(documents)}, stored {indexed_count}."
        )

    settings = get_settings()
    result = IndexBuildResult(
        journal_count=len(journals),
        document_count=len(documents),
        indexed_count=indexed_count,
        persist_directory=settings.chroma_persist_dir,
        collection_name=COLLECTION_NAME,
    )
    logger.info("Chroma journal index contains %d records", result.indexed_count)
    return result
