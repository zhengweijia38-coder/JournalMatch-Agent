"""Pytest categories keep the default suite fast and offline."""

from pathlib import Path

import pytest


API_TESTS = {
    "test_llm.py",
    "test_paper_analyzer.py",
}
SLOW_TESTS = {
    "test_embeddings.py",
    "test_hybrid_retrieval.py",
    "test_paper_analyzer.py",
    "test_reranker_model.py",
    "test_semantic_retrieval.py",
    "test_vector_store.py",
}
INTEGRATION_TESTS = {
    "test_hybrid_retrieval.py",
    "test_journal_database.py",
    "test_journal_importer.py",
    "test_paper_loader.py",
    "test_semantic_retrieval.py",
    "test_vector_store.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add explicit opt-ins for tests with external resources or large models."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that use real local files, SQLite, or Chroma",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests that load local BGE models",
    )
    parser.addoption(
        "--run-api",
        action="store_true",
        default=False,
        help="run tests that call the real DeepSeek API",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Mark known real tests and skip them unless their category is enabled."""
    run_integration = config.getoption("--run-integration")
    run_slow = config.getoption("--run-slow")
    run_api = config.getoption("--run-api")

    for item in items:
        filename = Path(str(item.path)).name
        required_options: list[tuple[bool, str]] = []
        if filename in INTEGRATION_TESTS:
            item.add_marker(pytest.mark.integration)
            required_options.append((run_integration, "--run-integration"))
        if filename in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
            required_options.append((run_slow, "--run-slow"))
        if filename in API_TESTS:
            item.add_marker(pytest.mark.api)
            required_options.append((run_api, "--run-api"))

        missing_options = [option for enabled, option in required_options if not enabled]
        if missing_options:
            item.add_marker(
                pytest.mark.skip(
                    reason="requires " + " and ".join(missing_options)
                )
            )
