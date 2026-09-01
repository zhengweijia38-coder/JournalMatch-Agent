"""Fast offline tests for Phase 10 logging, exceptions, and model caching."""

import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_environment import EnvironmentCheck, print_checks
from src.exceptions import (
    ConfigurationError,
    JournalRAGError,
    RerankerError,
    VectorStoreError,
)
from src.logging_config import configure_logging
import src.models.embeddings as embeddings_module


def test_engineering_hardening(capsys: object) -> None:
    """Verify log levels, safe status output, exception hierarchy, and cache reuse."""
    configure_logging(debug=False)
    assert logging.getLogger().level == logging.INFO
    configure_logging(debug=True)
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore2").level == logging.WARNING

    assert issubclass(ConfigurationError, JournalRAGError)
    assert issubclass(VectorStoreError, JournalRAGError)
    assert issubclass(RerankerError, JournalRAGError)

    secret = "never-print-this-real-secret"
    print_checks([EnvironmentCheck("DeepSeek API Key", True, "configured")])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "configured" in captured.out
    assert secret not in captured.out

    fake_settings = SimpleNamespace(
        bge_model_name="example/bge",
        huggingface_cache_dir=Path("cache"),
    )
    fake_embeddings = object()
    embeddings_module.get_embeddings.cache_clear()
    with (
        patch.object(
            embeddings_module,
            "get_settings",
            return_value=fake_settings,
        ),
        patch.object(
            embeddings_module,
            "HuggingFaceEmbeddings",
            return_value=fake_embeddings,
        ) as constructor,
    ):
        first = embeddings_module.get_embeddings()
        second = embeddings_module.get_embeddings()
    assert first is fake_embeddings
    assert second is fake_embeddings
    constructor.assert_called_once()
    embeddings_module.get_embeddings.cache_clear()
    # configure_logging(force=True) intentionally owns the application stream.
    # Remove pytest's temporary capture stream before the next test logs.
    logging.getLogger().handlers.clear()
