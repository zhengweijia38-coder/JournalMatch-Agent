"""Manual Phase 1 test for PDF loading and page combination."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper.loader import combine_documents, load_pdf


TEST_PDF = PROJECT_ROOT / "data" / "papers" / "01paper.pdf"


def _safe_console_text(text: str) -> str:
    """Replace characters unsupported by the active Windows console encoding."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def test_paper_loader() -> None:
    """Load the configured test PDF and print a short manual inspection preview."""
    documents = load_pdf(TEST_PDF)
    combined_text = combine_documents(documents)

    print("PDF loaded successfully.")
    print(f"Page count: {len(documents)}")
    print("First page preview:")
    print(_safe_console_text(documents[0].page_content[:500]))
    print(f"Combined text length: {len(combined_text)}")


if __name__ == "__main__":
    if not TEST_PDF.exists():
        print("Please place a test PDF at data/papers/test_paper.pdf")
        raise SystemExit(1)

    try:
        test_paper_loader()
    except Exception as exc:
        print(f"ERROR: PDF loader test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
