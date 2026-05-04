"""Normalize Ollama message objects to plain dicts."""

from __future__ import annotations

from typing import Any


def msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an Ollama message (dict or SDK model) to a plain dict."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, 'model_dump'):
        return msg.model_dump()
    if hasattr(msg, 'dict'):
        return msg.dict()
    try:
        return vars(msg)
    except TypeError:
        return dict(msg)
