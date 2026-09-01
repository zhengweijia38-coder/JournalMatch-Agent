"""CLI argument and JSON-output tests that never initialize model factories."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as cli
from src.schemas.assessment import PaperQualityAssessment
from src.schemas.journal import Journal
from src.schemas.paper import PaperProfile
from src.schemas.pipeline import PipelineResult, PipelineTimings
from src.schemas.retrieval import HybridCandidate, RerankedCandidate


def _result() -> PipelineResult:
    profile = PaperProfile(summary="CLI 中文 fake profile")
    journal = Journal(journal_id=1, name="CLI Journal", ccf_rank="A")
    hybrid = HybridCandidate(journal=journal, semantic_rank=1, retrieval_score=0.1)
    reranked = RerankedCandidate(
        journal=journal,
        semantic_rank=1,
        retrieval_score=0.1,
        rerank_rank=1,
        rerank_score=1.5,
    )
    dimension = {
        "score": 3,
        "level": "Moderate",
        "evidence": ["The profile contains one explicit method."],
        "concerns": [],
    }
    quality_assessment = PaperQualityAssessment.model_validate(
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
    return PipelineResult(
        paper_profile=profile,
        paper_quality_assessment=quality_assessment,
        hybrid_candidates=[hybrid],
        reranked_candidates=[reranked],
        recommendation_report=None,
        timings=PipelineTimings(
            pdf_loading_seconds=0,
            paper_analysis_seconds=0,
            hybrid_retrieval_seconds=0,
            reranking_seconds=0,
            recommendation_seconds=0,
            total_seconds=0,
        ),
    )


def test_cli() -> None:
    """Verify every Phase 8 flag, filter mapping, and UTF-8 JSON export."""
    args = cli.parse_args(
        [
            "paper.pdf",
            "--ccf",
            "A",
            "B",
            "--jcr",
            "Q1",
            "Q2",
            "--cas",
            "1",
            "2",
            "--min-if",
            "5",
            "--max-if",
            "20",
            "--candidate-k",
            "30",
            "--rerank-k",
            "12",
            "--top-k",
            "4",
            "--skip-recommendation",
            "--output",
            "result.json",
        ]
    )
    filters = cli.build_filters(args)
    assert args.pdf_path == Path("paper.pdf")
    assert filters.ccf_ranks == ["A", "B"]
    assert filters.jcr_quartiles == ["Q1", "Q2"]
    assert filters.cas_quartiles == ["1", "2"]
    assert filters.min_impact_factor == 5
    assert filters.max_impact_factor == 20
    assert (args.candidate_k, args.rerank_k, args.top_k) == (30, 12, 4)
    assert args.skip_recommendation is True
    assert args.debug is False

    with TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "nested" / "result.json"
        with patch.object(cli, "_run_pipeline", return_value=_result()) as run:
            exit_code = cli.main(
                [
                    "paper.pdf",
                    "--skip-recommendation",
                    "--output",
                    str(output_path),
                ]
            )
        assert exit_code == 0
        run.assert_called_once()
        assert output_path.is_file()
        json_text = output_path.read_text(encoding="utf-8")
        assert '"paper_profile"' in json_text
        assert '"paper_quality_assessment"' in json_text
        assert '"overall_maturity": "Solid"' in json_text
        assert '"recommendation_report": null' in json_text
        assert "中文" in json_text

    print("Phase 8 CLI argument tests passed without loading models.")


if __name__ == "__main__":
    try:
        test_cli()
    except Exception as exc:
        print(f"ERROR: CLI test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
