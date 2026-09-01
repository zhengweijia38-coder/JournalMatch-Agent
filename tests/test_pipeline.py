"""Phase 8 pipeline orchestration tests using only deterministic fakes."""

from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.pipeline as pipeline_module
from src.schemas.assessment import PaperQualityAssessment
from src.schemas.journal import Journal
from src.schemas.paper import PaperProfile
from src.schemas.recommendation import RecommendationReport
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


def _profile() -> PaperProfile:
    return PaperProfile(
        title="Test RAG Paper",
        research_fields=["Information Retrieval"],
        keywords=["retrieval augmented generation"],
        research_problem="Ground generated answers in retrieved evidence.",
        methods=["dense retrieval"],
        main_contributions=["A grounded retrieval pipeline."],
        summary="A test paper about retrieval-augmented generation.",
    )


def _quality_assessment() -> PaperQualityAssessment:
    dimension = {
        "score": 3,
        "level": "Moderate",
        "evidence": ["The profile states one supported contribution."],
        "concerns": [],
    }
    return PaperQualityAssessment.model_validate(
        {
            "novelty": dimension,
            "methodology": dimension,
            "dataset_quality": dimension,
            "experimental_quality": dimension,
            "conclusion_support": dimension,
            "overall_maturity": "Solid",
            "strengths": [],
            "weaknesses": [],
        }
    )


def _hybrid_candidates(count: int) -> list[HybridCandidate]:
    return [
        HybridCandidate(
            journal=Journal(
                journal_id=index,
                name=f"Journal {index}",
                research_fields=["Information Retrieval"],
                keywords=["retrieval"],
                aims_scope="Information retrieval research.",
                ccf_rank="A",
                jcr_quartile="Q1",
                cas_quartile="1",
                impact_factor=10.0,
            ),
            semantic_rank=index,
            retrieval_score=index / 10,
        )
        for index in range(1, count + 1)
    ]


def _reranked(candidates: list[HybridCandidate], top_k: int) -> list[RerankedCandidate]:
    return [
        RerankedCandidate(
            journal=candidate.journal,
            semantic_rank=candidate.semantic_rank,
            retrieval_score=candidate.retrieval_score,
            rerank_rank=rank,
            rerank_score=float(len(candidates) - rank),
        )
        for rank, candidate in enumerate(candidates[:top_k], start=1)
    ]


def _report(candidate: RerankedCandidate) -> RecommendationReport:
    journal = candidate.journal
    assert journal.journal_id is not None
    return RecommendationReport.model_validate(
        {
            "paper_assessment": {
                "innovation_level": "moderate",
                "experimental_completeness": "moderate",
                "paper_maturity": "solid",
                "strengths": ["Clear retrieval contribution."],
                "weaknesses": ["Limited test coverage."],
                "evidence": ["The profile describes a grounded retrieval pipeline."],
            },
            "recommendations": [
                {
                    "journal_id": journal.journal_id,
                    "journal_name": journal.name,
                    "final_rank": 1,
                    "recommendation_tier": "Strong Match",
                    "topic_fit": "The topics align.",
                    "method_fit": "The methods align.",
                    "scope_fit": "The scope aligns.",
                    "reasons": ["Matching retrieval focus."],
                    "concerns": [],
                }
            ],
            "overall_advice": "Emphasize retrieval evidence.",
        }
    )


def _run_with_fakes(
    candidate_count: int,
    *,
    candidate_k: int = 20,
    rerank_k: int = 10,
    recommendation_k: int = 5,
    skip_recommendation: bool = False,
) -> tuple[object, Mock, Mock]:
    candidates = _hybrid_candidates(candidate_count)
    rerank_mock = Mock(
        side_effect=lambda profile, values, top_k: _reranked(values, top_k)
    )
    recommendation_mock = Mock(
        side_effect=lambda profile, values, top_k, quality_assessment: _report(values[0])
    )

    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline_module, "load_pdf", return_value=[object()]))
        stack.enter_context(patch.object(pipeline_module, "combine_documents", return_value="paper text"))
        stack.enter_context(patch.object(pipeline_module, "_ensure_runtime_data_ready"))
        stack.enter_context(patch.object(pipeline_module, "analyze_paper", return_value=_profile()))
        stack.enter_context(
            patch.object(
                pipeline_module,
                "assess_paper_quality",
                return_value=_quality_assessment(),
            )
        )
        stack.enter_context(
            patch.object(
                pipeline_module,
                "hybrid_search_for_paper",
                return_value=candidates,
            )
        )
        stack.enter_context(
            patch.object(
                pipeline_module,
                "rerank_candidates_for_paper",
                rerank_mock,
            )
        )
        stack.enter_context(
            patch.object(
                pipeline_module,
                "generate_recommendations",
                recommendation_mock,
            )
        )
        result = pipeline_module.run_recommendation_pipeline(
            "fake.pdf",
            candidate_k=candidate_k,
            rerank_k=rerank_k,
            recommendation_k=recommendation_k,
            skip_recommendation=skip_recommendation,
        )
    return result, rerank_mock, recommendation_mock


def test_pipeline() -> None:
    """Cover full/skip/short/empty paths, limits, intermediates, and timings."""
    full_result, rerank_mock, recommendation_mock = _run_with_fakes(20)
    assert len(full_result.hybrid_candidates) == 20
    assert len(full_result.reranked_candidates) == 10
    assert full_result.recommendation_report is not None
    assert full_result.paper_profile.title == "Test RAG Paper"
    assert full_result.paper_quality_assessment == _quality_assessment()
    rerank_mock.assert_called_once()
    recommendation_mock.assert_called_once()

    skipped_result, _, skipped_recommendation = _run_with_fakes(
        20, skip_recommendation=True
    )
    assert skipped_result.recommendation_report is None
    assert skipped_result.timings is not None
    assert skipped_result.timings.recommendation_seconds == 0
    skipped_recommendation.assert_not_called()

    short_result, short_reranker, short_recommender = _run_with_fakes(3)
    assert len(short_result.reranked_candidates) == 3
    assert short_reranker.call_args.kwargs["top_k"] == 3
    assert short_recommender.call_args.kwargs["top_k"] == 3

    no_rerank = Mock()
    no_recommendation = Mock()
    with (
        patch.object(pipeline_module, "load_pdf", return_value=[object()]),
        patch.object(pipeline_module, "combine_documents", return_value="paper text"),
        patch.object(pipeline_module, "_ensure_runtime_data_ready"),
        patch.object(pipeline_module, "analyze_paper", return_value=_profile()),
        patch.object(
            pipeline_module,
            "assess_paper_quality",
            return_value=_quality_assessment(),
        ),
        patch.object(pipeline_module, "hybrid_search_for_paper", return_value=[]),
        patch.object(pipeline_module, "rerank_candidates_for_paper", no_rerank),
        patch.object(pipeline_module, "generate_recommendations", no_recommendation),
    ):
        try:
            pipeline_module.run_recommendation_pipeline("fake.pdf")
        except pipeline_module.NoMatchingJournalsError as exc:
            assert "filters were not relaxed" in str(exc)
        else:
            raise AssertionError("Empty retrieval results must stop the pipeline.")
    no_rerank.assert_not_called()
    no_recommendation.assert_not_called()

    invalid_calls = [
        {"candidate_k": 0},
        {"candidate_k": True},
        {"candidate_k": 5, "rerank_k": 6},
        {"candidate_k": 10, "rerank_k": 5, "recommendation_k": 6},
    ]
    for kwargs in invalid_calls:
        try:
            pipeline_module.run_recommendation_pipeline("fake.pdf", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid limits should fail: {kwargs}")

    assert full_result.timings is not None
    for value in full_result.timings.model_dump().values():
        assert value >= 0
    assert full_result.model_dump()["hybrid_candidates"]
    assert full_result.model_dump()["reranked_candidates"]

    with TemporaryDirectory() as temporary_directory:
        missing_index = Path(temporary_directory) / "missing_chroma"
        try:
            pipeline_module._ensure_chroma_is_ready(missing_index)
        except pipeline_module.VectorStoreError as exc:
            assert "python scripts/build_vector_store.py" in str(exc)
        else:
            raise AssertionError("A missing Chroma index must fail clearly.")
        assert not missing_index.exists()

        missing_database = Path(temporary_directory) / "missing.db"
        try:
            pipeline_module._ensure_sqlite_is_ready(missing_database)
        except pipeline_module.JournalDatabaseError as exc:
            assert "python scripts/init_journals.py" in str(exc)
        else:
            raise AssertionError("A missing SQLite database must fail clearly.")
        assert not missing_database.exists()
    print("Phase 8 fake pipeline tests passed.")


if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as exc:
        print(f"ERROR: Pipeline test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
