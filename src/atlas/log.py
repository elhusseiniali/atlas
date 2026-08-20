"""Loguru setup, shared by the orchestrator, workers, and piped subprocess output.

Import the configured logger as ``from atlas.log import logger`` everywhere
else in the package.
"""

import sys
from pathlib import Path

from loguru import logger

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[worker]}</cyan> | "
    "<level>{message}</level>"
)

DEFAULT_LOG_FILE = Path("atlas.log")


def configure_logging(
    level: str = "INFO", log_file: str | Path | None = DEFAULT_LOG_FILE
) -> None:
    """Configure the shared loguru sinks.

    Parameters
    ----------
    level : str, optional
        Minimum log level to emit, by default "INFO".
    log_file : str | pathlib.Path, optional
        Path to append logs to, in addition to stderr, by default
        `atlas.log` in the current working directory. Pass None to
        disable file logging.
    """
    logger.remove()
    logger.configure(extra={"worker": "main"})
    logger.add(sys.stderr, format=_FORMAT, level=level, colorize=True)
    if log_file is not None:
        logger.add(log_file, format=_FORMAT, level=level, colorize=False)


__all__ = ["configure_logging", "logger"]
