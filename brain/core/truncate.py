"""Truncate tool output before it is sent back to the model."""


def truncate_for_model(
    text: str,
    *,
    max_lines: int,
    max_bytes: int,
) -> str:
    """
    Keep the head of *text* within line and byte limits.

    Whichever limit is hit first wins. Never returns partial lines except when
    the first line alone exceeds ``max_bytes``.
    """
    if not text:
        return text

    raw_bytes = text.encode('utf-8')
    total_bytes = len(raw_bytes)
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return text

    kept: list[str] = []
    used_bytes = 0
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = len(line.encode('utf-8'))
        if used_bytes + line_bytes > max_bytes:
            break
        kept.append(line)
        used_bytes += line_bytes

    if not kept and lines:
        prefix = raw_bytes[:max_bytes].decode('utf-8', errors='ignore')
        omitted = total_bytes - len(prefix.encode('utf-8'))
        return (
            f'{prefix}\n\n'
            f'…[truncated for model context; {omitted:,} bytes omitted]'
        )

    body = ''.join(kept)
    omitted_lines = max(0, total_lines - len(kept))
    omitted_bytes = max(0, total_bytes - used_bytes)
    parts: list[str] = [body.rstrip('\n')]
    if omitted_lines or omitted_bytes:
        parts.append(
            f'…[truncated for model context; '
            f'{omitted_lines:,} lines and {omitted_bytes:,} bytes omitted]',
        )
    return '\n\n'.join(parts)
