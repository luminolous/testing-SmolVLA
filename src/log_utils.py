"""Shared logging setup.

`logging.basicConfig(level=INFO)` configures the **root** logger, so every
third-party library starts printing its INFO records too. In this stack that buries
the run's own output under HuggingFace cache-validation requests and a repeated
five-traceback dump from a `torchcodec` probe that is expected to fail on Windows
and is entirely harmless here -- the dataset is PNG, so no video decoder is ever
used.

:func:`configure_logging` keeps the project's own loggers at INFO and pins the known
noisy ones above it.
"""

from __future__ import annotations

import logging
from pathlib import Path

# Chatty at INFO, and none of it is about this project.
_NOISY_INFO = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "datasets",
    "fsspec",
    "PIL",
    "matplotlib",
)

# Warns at import that torchcodec cannot be loaded, quoting the full multi-version
# traceback. Expected on Windows and harmless with `use_videos=False`; documented in
# docs/phase-3/result.md.
_NOISY_WARNING = (
    "lerobot.utils.import_utils",
    "torchcodec",
)


def configure_logging(
    log_file: Path | None = None,
    level: int = logging.INFO,
    console: bool = True,
    quiet_third_party: bool = True,
) -> None:
    """Configure logging for a script.

    Args:
        log_file: Also write records here, if given.
        level: Level for the root logger and therefore for this project's loggers.
        console: Echo to stderr.
        quiet_third_party: Pin the known noisy libraries above `level`. Set False
            when actually debugging a download or a decoder.
    """
    handlers: list[logging.Handler] = []
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    if not quiet_third_party:
        return

    for name in _NOISY_INFO:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _NOISY_WARNING:
        logging.getLogger(name).setLevel(logging.ERROR)
