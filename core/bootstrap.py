"""Runtime bootstrap helpers: logging and LangSmith tracing."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from config.settings import Settings


def configure_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure application logging once and return the app logger."""
    logger = logging.getLogger("portfolio_assistant")
    if logger.handlers:  # already configured
        return logger

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
    )

    file_handler = logging.FileHandler(path / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def configure_tracing(settings: Settings) -> None:
    """Enable LangSmith tracing when an API key is available."""
    if not settings.langchain_api_key:
        return
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
