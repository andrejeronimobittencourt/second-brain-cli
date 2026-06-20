"""Normalize Ollama message objects to plain dicts."""

from __future__ import annotations

import json
from typing import Any


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Coerce tool-call arguments into a plain dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


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
