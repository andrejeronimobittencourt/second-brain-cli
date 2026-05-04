"""Configure Python logging from UserConfig settings."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str = '', log_file: str = '') -> None:
    """Set up the root ``brain`` logger with an optional file handler."""
    logger = logging.getLogger('brain')
    effective = (level.strip().upper() or 'WARNING')
    numeric = getattr(logging, effective, logging.WARNING)
    logger.setLevel(numeric)

    fmt = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(name)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if log_file.strip():
        target = Path(log_file).expanduser().resolve()
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == target
            for h in logger.handlers
        )
        if not already:
            target.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(target, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
