"""Library-wide logging utilities for bforbuntyai."""
import logging
import sys
from typing import Optional

_LIBRARY_NAME = "bforbuntyai"


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under bforbuntyai.<name>."""
    return logging.getLogger(f"{_LIBRARY_NAME}.{name}")


def setup_logging(
    level: str = "INFO",
    fmt: Optional[str] = None,
    file: Optional[str] = None,
) -> None:
    """Configure bforbuntyai library logging.

    Args:
        level: "DEBUG", "INFO", "WARNING", or "ERROR". Default "INFO".
        fmt:   Custom log format string. None uses the library default.
        file:  Optional path to also write logs to a file.
    """
    root = logging.getLogger(_LIBRARY_NAME)
    root.setLevel(getattr(logging, level.upper()))
    root.propagate = False

    for h in list(root.handlers):
        h.close()
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt or "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
