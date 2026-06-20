"""Structured agent events for UI subscribers and future TUI frontends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

AgentEventType = Literal[
    'turn_start',
    'turn_end',
    'tool_start',
    'tool_end',
    'compact_start',
    'compact_end',
    'error',
]


@dataclass(frozen=True)
class AgentEvent:
    """One observable step in the agent loop."""

    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[AgentEvent], None]


def emit_agent_event(
    on_event: EventHandler | None,
    event_type: AgentEventType,
    **data: Any,
) -> None:
    """Invoke ``on_event`` with a structured :class:`AgentEvent` when a handler is set."""
    if on_event is not None:
        on_event(AgentEvent(type=event_type, data=data))
