"""Runtime context appended to the system prompt (date/time, vault root)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def system_context_footer(*, vault_path: Path, now: datetime | None = None) -> str:
    """
    Build volatile footer lines for the system message.

    Always appended last so custom prompts still receive current clock and vault path.
    """
    clock = now or datetime.now().astimezone()
    time_str = clock.strftime('%Y-%m-%d %H:%M:%S')
    tz = clock.strftime('%Z')
    offset = clock.strftime('%z')
    tz_part = tz if tz else offset
    if tz and offset and tz != offset:
        tz_part = f'{tz}, {offset}'
    root = vault_path.resolve()
    return (
        f'Current local time: {time_str} ({tz_part})\n'
        f'Vault root: {root.as_posix()}'
    )
