"""Phase 5 unit tests for candidate reranking behavior."""

from pathlib import Path
import sys
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.retrieval.reranker as reranker_module
from src.retrieval.query_builder import build_paper_query
from src.schemas.journal import Journal
from src.schemas.paper import PaperProfile
from src.schemas.retrieval import HybridCandidate


class FakeReranker:
    """Return deterministic scores while recording the batch passed by the service."""

    def __init__(self) -> None:
        self.pairs: list[list[str]] = []

    def compute_score(self, pairs: list[list[str]], **_: object) -> list[float]:
        self.pairs = pairs
        return [0.2, 0.9, -0.1]


def _candidates() -> list[HybridCandidate]:
    journals = [
        Journal(
            journal_id=1,
            name="Software Systems Journal",
            research_fields=["Software Engineering"],
            keywords=["software testing"],
            aims_scope="Research on software systems and quality assurance.",
            ccf_rank="A",
        ),
        Journal(
            journal_id=2,
            name="Medical Vision Journal",
            research_fields=["Computer Vision", "Medical Imaging"],
            keywords=["image segmentation", "deep learning"],
            aims_scope="Visual analysis and segmentation of medical images.",
            ccf_rank="B",
        ),
        Journal(
            journal_id=3,
            name="Computer Networks Journal",
            research_fields=["Computer Networks"],
            keywords=["routing", "network security"],
            aims_scope="Communication networks and routing protocols.",
            ccf_rank="C",
        ),
    ]
    return [
        HybridCandidate(
            journal=journal,
            semantic_rank=rank,
            retrieval_score=rank / 10,
        )
        for rank, journal in enumerate(journals, start=1)
    ]


def test_reranking() -> None:
    """Check ordering, rank provenance, truncation, reuse, and validation."""
    candidates = _candidates()
    fake_reranker = FakeReranker()
    query = "Computer vision for medical image segmentation."

    with patch.object(
        reranker_module,
        "get_reranker",
        return_value=fake_reranker,
    ) as factory:
        results = reranker_module.rerank_candidates(query, candidates)

    assert factory.call_count == 1
    assert len(results) == 3
    assert [result.rerank_score for result in results] == [0.9, 0.2, -0.1]
    assert [result.rerank_rank for result in results] == [1, 2, 3]
    assert [result.semantic_rank for result in results] == [2, 1, 3]
    assert [result.retrieval_score for result in results] == [0.2, 0.1, 0.3]
    assert results[0].journal.name == "Medical Vision Journal"
    assert len(fake_reranker.pairs) == 3
    assert all(pair[0] == query for pair in fake_reranker.pairs)
    assert "Aims and Scope" in fake_reranker.pairs[0][1]
    assert "CCF" not in fake_reranker.pairs[0][1]

    with patch.object(
        reranker_module,
        "get_reranker",
        return_value=FakeReranker(),
    ):
        top_two = reranker_module.rerank_candidates(query, candidates, top_k=2)
    assert len(top_two) == 2
    assert [result.rerank_rank for result in top_two] == [1, 2]

    profile = PaperProfile(
        title="Medical Image Segmentation",
        research_fields=["Computer Vision", "Medical Imaging"],
        keywords=["image segmentation"],
        research_problem="Segment anatomical structures in medical images.",
        methods=["deep learning"],
        summary="A vision model for accurate medical image segmentation.",
    )
    paper_fake = FakeReranker()
    with patch.object(
        reranker_module,
        "get_reranker",
        return_value=paper_fake,
    ):
        reranker_module.rerank_candidates_for_paper(profile, candidates, top_k=1)
    assert paper_fake.pairs[0][0] == build_paper_query(profile)

    with patch.object(reranker_module, "get_reranker") as empty_factory:
        assert reranker_module.rerank_candidates(query, []) == []
        empty_factory.assert_not_called()

    for invalid_top_k in (0, -1):
        try:
            reranker_module.rerank_candidates(
                query,
                candidates,
                top_k=invalid_top_k,
            )
        except ValueError as exc:
            assert "top_k" in str(exc)
        else:
            raise AssertionError("Invalid top_k should raise ValueError.")

    print("Candidate reranking behavior tests passed.")


if __name__ == "__main__":
    try:
        test_reranking()
    except Exception as exc:
        print(f"ERROR: Candidate reranking test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
