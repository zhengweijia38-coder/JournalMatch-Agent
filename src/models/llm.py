"""DeepSeek chat model factory."""

from functools import lru_cache
import logging

from langchain_deepseek import ChatDeepSeek

from src.config import get_settings


logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_llm() -> ChatDeepSeek:
    """Create and cache one deterministic DeepSeek chat model."""

    settings = get_settings()
    logger.info("Creating DeepSeek chat client for model: %s", settings.deepseek_model)

    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.require_deepseek_api_key(),
        temperature=0,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )
