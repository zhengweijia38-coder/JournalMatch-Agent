"""Offline grounding tests for all four Phase 9 high-level tools."""

from pathlib import Path
import sys
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.tools.journal_tools as journal_tools
import src.tools.paper_tools as paper_tools
import src.tools.recommendation_tools as recommendation_tools
from src.schemas.assessment import PaperQualityAssessment
from src.schemas.journal import Journal
from src.schemas.paper import PaperProfile
from src.schemas.pipeline import PipelineResult
from src.schemas.recommendation import RecommendationReport
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


def _profile() -> PaperProfile:
    return PaperProfile(
        title="Grounded RAG Paper",
        research_fields=["Information Retrieval"],
        keywords=["retrieval augmented generation"],
        methods=["dense retrieval"],
        summary="A grounded retrieval paper.",
    )


def _quality_assessment() -> PaperQualityAssessment:
    dimension = {
        "score": 3,
        "level": "Moderate",
        "evidence": ["The profile describes a grounded retrieval method."],
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


def _journal() -> Journal:
    return Journal(
        journal_id=7,
        name="Journal of Grounded Retrieval",
        abbreviation="JGR",
        research_fields=["Information Retrieval"],
        keywords=["retrieval", "ranking"],
        aims_scope="Publishes information retrieval research.",
        ccf_rank="B",
        jcr_quartile="Q1",
        cas_quartile="2",
        impact_factor=7.5,
    )


def _hybrid() -> HybridCandidate:
    return HybridCandidate(
        journal=_journal(),
        semantic_rank=2,
        retrieval_score=0.2,
    )


def _reranked() -> RerankedCandidate:
    return RerankedCandidate(
        journal=_journal(),
        semantic_rank=2,
        retrieval_score=0.2,
        rerank_rank=1,
        rerank_score=3.2,
    )


def _report() -> RecommendationReport:
    return RecommendationReport.model_validate(
        {
            "paper_assessment": {
                "innovation_level": "moderate",
                "experimental_completeness": "moderate",
                "paper_maturity": "solid",
                "strengths": ["Clear retrieval contribution."],
                "weaknesses": ["Limited benchmarks."],
                "evidence": ["The supplied profile describes dense retrieval."],
            },
            "recommendations": [
                {
                    "journal_id": 7,
                    "journal_name": "Journal of Grounded Retrieval",
                    "final_rank": 1,
                    "recommendation_tier": "Strong Match",
                    "topic_fit": "The topic aligns.",
                    "method_fit": "The method aligns.",
                    "scope_fit": "The scope aligns.",
                    "reasons": ["Grounded retrieval focus."],
                    "concerns": ["Benchmark coverage is limited."],
                }
            ],
            "overall_advice": "Emphasize the retrieval contribution.",
        }
    )


def test_agent_tools() -> None:
    """Ensure every tool delegates to existing phases and preserves SQLite facts."""
    profile = _profile()
    with (
        patch.object(paper_tools, "load_pdf", return_value=[object()]) as load,
        patch.object(
            paper_tools,
            "combine_documents",
            return_value="paper text",
        ) as combine,
        patch.object(paper_tools, "analyze_paper", return_value=profile) as analyze,
        patch.object(
            paper_tools,
            "assess_paper_quality",
            return_value=_quality_assessment(),
        ) as assess,
    ):
        paper_result = paper_tools.analyze_paper_tool.invoke(
            {"pdf_path": "paper.pdf"}
        )
    assert paper_result["ok"] is True
    assert paper_result["paper_profile"]["title"] == profile.title
    assert paper_result["paper_quality_assessment"]["overall_maturity"] == "Solid"
    assert "not an acceptance probability" in paper_result["assessment_note"]
    load.assert_called_once_with("paper.pdf")
    combine.assert_called_once()
    analyze.assert_called_once_with("paper text")
    assess.assert_called_once_with(profile)

    with (
        patch.object(journal_tools, "_ensure_sqlite_is_ready") as ready,
        patch.object(
            journal_tools,
            "get_journal_by_name",
            return_value=_journal(),
        ) as lookup,
    ):
        details_result = journal_tools.get_journal_details_tool.invoke(
            {"journal_name": "Journal of Grounded Retrieval"}
        )
    assert details_result["ok"] is True
    assert details_result["journal"]["ccf_rank"] == "B"
    assert details_result["journal"]["impact_factor"] == 7.5
    ready.assert_called_once()
    lookup.assert_called_once_with("Journal of Grounded Retrieval")

    hybrid_candidate = _hybrid()
    reranked_candidate = _reranked()
    with (
        patch.object(journal_tools, "_ensure_runtime_data_ready") as runtime_ready,
        patch.object(
            journal_tools,
            "hybrid_search",
            return_value=[hybrid_candidate],
        ) as hybrid_search,
        patch.object(
            journal_tools,
            "rerank_candidates",
            return_value=[reranked_candidate],
        ) as rerank,
    ):
        search_result = journal_tools.search_journals_tool.invoke(
            {
                "query": "retrieval augmented generation",
                "ccf_ranks": ["B"],
                "top_k": 5,
            }
        )
    assert search_result["ok"] is True
    assert search_result["journals"][0]["name"] == _journal().name
    assert search_result["journals"][0]["ccf_rank"] == "B"
    runtime_ready.assert_called_once()
    assert hybrid_search.call_args.kwargs["filters"].ccf_ranks == ["B"]
    rerank.assert_called_once()

    pipeline_result = PipelineResult(
        paper_profile=profile,
        paper_quality_assessment=_quality_assessment(),
        hybrid_candidates=[hybrid_candidate],
        reranked_candidates=[reranked_candidate],
        recommendation_report=_report(),
    )
    with patch.object(
        recommendation_tools,
        "run_recommendation_pipeline",
        return_value=pipeline_result,
    ) as pipeline:
        recommendation_result = (
            recommendation_tools.recommend_journals_tool.invoke(
                {
                    "pdf_path": "paper.pdf",
                    "ccf_ranks": ["A", "B"],
                    "top_k": 5,
                }
            )
        )
    assert recommendation_result["ok"] is True
    assert recommendation_result["paper_quality_assessment"]["overall_maturity"] == "Solid"
    final_item = recommendation_result["recommendations"][0]
    assert final_item["journal_name"] == _journal().name
    assert final_item["ccf_rank"] == "B"
    assert final_item["impact_factor"] == 7.5
    assert pipeline.call_args.kwargs["filters"].ccf_ranks == ["A", "B"]
    assert pipeline.call_args.kwargs["candidate_k"] == 20
    assert pipeline.call_args.kwargs["rerank_k"] == 10
    assert pipeline.call_args.kwargs["recommendation_k"] == 5

    print("Phase 9 high-level tool delegation and grounding tests passed.")


if __name__ == "__main__":
    try:
        test_agent_tools()
    except Exception as exc:
        print(f"ERROR: Agent tool test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
