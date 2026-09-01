"""Centralized application configuration loaded from the project .env file."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

from src.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _get_text(name: str, default: str | None = None) -> str | None:
    """Read and trim one environment variable, using the default if it is blank."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _get_path(name: str, default: str) -> Path:
    """Read a path setting and resolve relative paths from the project root."""
    raw_path = _get_text(name, default)
    if raw_path is None:
        raise ConfigurationError(f"Configuration error: {name} must not be empty.")

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Typed settings shared by the project's model and storage factories."""

    deepseek_api_key: str | None
    deepseek_model: str
    bge_model_name: str
    huggingface_cache_dir: Path
    reranker_model_name: str
    chroma_persist_dir: Path
    sqlite_db_path: Path

    def require_deepseek_api_key(self) -> str:
        """Return the API key or raise an actionable configuration error."""
        if not self.deepseek_api_key:
            raise ConfigurationError(
                "DEEPSEEK_API_KEY is missing. Copy .env.example to .env and "
                "replace the placeholder with your DeepSeek API Key."
            )
        if self.deepseek_api_key == "your_deepseek_api_key":
            raise ConfigurationError(
                "DEEPSEEK_API_KEY still contains the example placeholder. "
                "Set a real DeepSeek API Key in .env before running the LLM test."
            )
        return self.deepseek_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load .env once and return the application's centralized settings."""
    load_dotenv(dotenv_path=ENV_FILE)

    deepseek_model = _get_text("DEEPSEEK_MODEL", "deepseek-v4-flash")
    bge_model_name = _get_text("BGE_MODEL_NAME", "BAAI/bge-m3")
    reranker_model_name = _get_text(
        "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
    )
    if (
        deepseek_model is None
        or bge_model_name is None
        or reranker_model_name is None
    ):
        raise ConfigurationError("Model names must not be empty.")

    return Settings(
        deepseek_api_key=_get_text("DEEPSEEK_API_KEY"),
        deepseek_model=deepseek_model,
        bge_model_name=bge_model_name,
        huggingface_cache_dir=_get_path("HF_HOME", "./storage/huggingface_cache"),
        reranker_model_name=reranker_model_name,
        chroma_persist_dir=_get_path(
            "CHROMA_PERSIST_DIR", "./storage/chroma_db"
        ),
        sqlite_db_path=_get_path("SQLITE_DB_PATH", "./storage/journals.db"),
    )
