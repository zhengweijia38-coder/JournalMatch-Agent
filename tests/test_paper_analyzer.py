"""Manual end-to-end Phase 1 paper analysis test."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper.analyzer import analyze_paper
from src.paper.loader import combine_documents, load_pdf


TEST_PDF = PROJECT_ROOT / "data" / "papers" / "01paper.pdf"


def test_paper_analyzer() -> None:
    """Run the complete Phase 1 pipeline and print structured JSON."""
    documents = load_pdf(TEST_PDF)
    paper_text = combine_documents(documents)
    profile = analyze_paper(paper_text)

    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    if not TEST_PDF.exists():
        print("Please place a test PDF at data/papers/test_paper.pdf")
        raise SystemExit(1)

    try:
        test_paper_analyzer()
    except Exception as exc:
        print(f"ERROR: Paper analyzer test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
