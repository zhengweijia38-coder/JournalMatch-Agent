"""Minimal live DeepSeek API connection test."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.llm import get_llm


EXPECTED_REPLY = "DeepSeek connection successful."


def test_deepseek_connection() -> None:
    """Send one small request and validate the returned text."""
    try:
        response = get_llm().invoke(f"Reply with exactly: {EXPECTED_REPLY}")
        content = str(response.content).strip()
    except Exception as exc:
        raise RuntimeError(
            "DeepSeek connection test failed. Check DEEPSEEK_API_KEY, "
            "DEEPSEEK_MODEL, network access, and your API account status. "
            f"Original error: {exc}"
        ) from exc

    print(f"DeepSeek response: {content}")
    if content != EXPECTED_REPLY:
        raise AssertionError(
            f"DeepSeek returned an unexpected response. Expected "
            f"{EXPECTED_REPLY!r}, got {content!r}."
        )
    print("DeepSeek connection test passed.")


if __name__ == "__main__":
    try:
        test_deepseek_connection()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
