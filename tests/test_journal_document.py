"""Phase 3 tests for Journal Document and PaperProfile query construction."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.journal.document_builder import journal_to_document
from src.retrieval.query_builder import build_paper_query
from src.schemas.journal import Journal
from src.schemas.paper import PaperProfile


def test_journal_document() -> None:
    """Verify semantic text and scalar metadata stay on their proper sides."""
    journal = Journal(
        journal_id=42,
        name="Journal of Computer Vision Systems",
        research_fields=["Computer Vision", "Machine Learning"],
        keywords=["image segmentation", "image recognition"],
        aims_scope="Publishes research on visual understanding and image analysis.",
        ccf_rank="A",
        jcr_quartile="Q1",
        cas_quartile="1区",
        impact_factor=9.9,
    )
    document = journal_to_document(journal)

    assert journal.name in document.page_content
    assert "Computer Vision" in document.page_content
    assert "image segmentation" in document.page_content
    assert journal.aims_scope in document.page_content
    assert "CCF" not in document.page_content
    assert "JCR" not in document.page_content
    assert "Impact Factor" not in document.page_content
    assert "9.9" not in document.page_content

    assert document.metadata == {
        "journal_id": 42,
        "name": journal.name,
        "ccf_rank": "A",
        "jcr_quartile": "Q1",
        "cas_quartile": "1区",
        "impact_factor": 9.9,
    }
    assert all(
        isinstance(value, (str, int, float, bool))
        for value in document.metadata.values()
    )

    profile = PaperProfile(
        title="Retrieval-Augmented Generation for Scientific Search",
        keywords=["retrieval augmented generation", "large language models"],
        research_fields=["Information Retrieval", "Natural Language Processing"],
        research_problem="Grounding generated answers in scientific documents.",
        methods=["dense retrieval", "transformer language model"],
        experimental_results=["A reported benchmark improvement."],
        limitations=["Only one benchmark was evaluated."],
        summary="The paper combines retrieval and generation for scientific search.",
    )
    query = build_paper_query(profile)
    assert profile.title in query
    assert "Information Retrieval" in query
    assert "retrieval augmented generation" in query
    assert profile.research_problem in query
    assert "dense retrieval" in query
    assert profile.summary in query
    assert profile.experimental_results[0] not in query
    assert profile.limitations[0] not in query

    print("Journal Document and PaperProfile query tests passed.")


if __name__ == "__main__":
    try:
        test_journal_document()
    except Exception as exc:
        print(f"ERROR: Journal Document test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
