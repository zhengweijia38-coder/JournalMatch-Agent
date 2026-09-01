"""Standard-library logging configuration shared by project entry points."""

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(debug: bool = False) -> None:
    """Configure concise application logs without enabling sensitive HTTP traces."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )

    # Debug mode is for application flow and tracebacks, not request headers.
    for logger_name in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
        "urllib3",
        "huggingface_hub",
        "chromadb",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
