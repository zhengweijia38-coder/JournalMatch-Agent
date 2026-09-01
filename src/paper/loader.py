"""Load text-based PDF papers and combine their page content."""

from pathlib import Path
import logging
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.exceptions import PaperProcessingError


logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> list[Document]:
    """Load non-empty text pages from a PDF using PyPDFLoader."""
    pdf_path = Path(path).expanduser()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, but received: {pdf_path}")

    try:
        documents = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:
        raise PaperProcessingError(
            f"Failed to read PDF '{pdf_path}'. The file may be damaged, encrypted, "
            "or unsupported."
        ) from exc

    non_empty_documents = [
        document
        for document in documents
        if document.page_content and document.page_content.strip()
    ]
    if not non_empty_documents:
        raise PaperProcessingError(
            f"No extractable text was found in PDF '{pdf_path}'. The PDF may be a "
            "scanned image; Phase 1 does not support OCR."
        )

    logger.info("Loaded %d non-empty PDF pages", len(non_empty_documents))
    return non_empty_documents


def _clean_page_text(text: str) -> str:
    """Normalize redundant whitespace without rewriting page content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def combine_documents(documents: list[Document]) -> str:
    """Combine PDF pages into one text with visible page separators."""
    if not documents:
        raise ValueError("Cannot combine documents because the page list is empty.")

    page_sections: list[str] = []
    for page_number, document in enumerate(documents, start=1):
        cleaned_text = _clean_page_text(document.page_content)
        if cleaned_text:
            page_sections.append(
                f"===== Page {page_number} =====\n\n{cleaned_text}"
            )

    if not page_sections:
        raise ValueError("No extractable page text is available to combine.")

    return "\n\n".join(page_sections)
