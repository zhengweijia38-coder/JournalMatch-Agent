"""Cached factory for the local BGE cross-encoder reranker."""

from functools import lru_cache
import logging

import torch
from FlagEmbedding import FlagReranker

from src.config import get_settings
from src.exceptions import RerankerError


logger = logging.getLogger(__name__)


def get_reranker_device() -> str:
    """Return the device selected for reranking without loading model weights."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_reranker() -> FlagReranker:
    """Load and cache the configured local BGE reranker for process-wide reuse."""
    settings = get_settings()
    use_cuda = torch.cuda.is_available()
    devices = ["cuda:0"] if use_cuda else ["cpu"]
    logger.info(
        "Loading local reranker model: %s on %s",
        settings.reranker_model_name,
        devices[0],
    )
    try:
        return FlagReranker(
            settings.reranker_model_name,
            use_fp16=use_cuda,
            devices=devices,
            cache_dir=str(settings.huggingface_cache_dir),
        )
    except Exception as exc:
        raise RerankerError(
            "Failed to load the local reranker. Check RERANKER_MODEL_NAME, "
            "HF_HOME, model cache availability, memory, and device support."
        ) from exc
