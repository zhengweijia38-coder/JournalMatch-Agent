"""Local Hugging Face embedding model factory."""

from functools import lru_cache
import logging

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings
from src.exceptions import VectorStoreError


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache one configured local BGE embedding model per process."""
    settings = get_settings()
    logger.info("Loading local embedding model: %s", settings.bge_model_name)
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.bge_model_name,
            cache_folder=str(settings.huggingface_cache_dir),
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise VectorStoreError(
            "Failed to load the local embedding model. Check BGE_MODEL_NAME, "
            "HF_HOME, model cache availability, and disk space."
        ) from exc
