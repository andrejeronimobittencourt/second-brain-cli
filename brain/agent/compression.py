"""Context trimming, history persistence, and rolling summarization."""

from __future__ import annotations

import json
import logging
from typing import Any

from brain.ui import render as ui
from brain.core.context import get_context
from brain.core.context_messages import (
    split_messages_pair_aware,
    truncate_messages_pair_aware,
)
from brain.core.events import EventHandler, emit_agent_event
from brain.core.messages import msg_to_dict, parse_tool_arguments
from brain.core.retry import retry_call
from brain.core.sanitize import sanitize_assistant_text
from brain.core.session import load_history, save_history

logger = logging.getLogger(__name__)

_READ_NOTE_TOOLS = frozenset({
    'read_note',
    'list_directory',
    'get_backlinks',
    'search_by_tag',
})

_MODIFY_NOTE_TOOLS = frozenset({
    'create_note',
    'edit_note',
    'move_note',
    'rename_note',
    'delete_note',
    'update_frontmatter',
    'create_folder',
    'delete_folder',
})


def _tool_calls_brief(tool_calls: Any) -> str:
    """Short JSON-ish preview of tool calls for summarizer input."""
    if not tool_calls:
        return ''
    try:
        return json.dumps(tool_calls, ensure_ascii=False)[:500]
    except (TypeError, ValueError):
        return str(tool_calls)[:500]


def _note_ref_from_tool(name: str, args: dict[str, Any]) -> str | None:
    """Extract a vault-relative note or folder reference from a tool call."""
    if name == 'read_note':
        return str(args.get('note_title', '')).strip() or None
    if name == 'get_backlinks':
        return str(args.get('note_title', '')).strip() or None
    if name == 'list_directory':
        path = str(args.get('relative_path', '')).strip()
        return f'{path}/' if path else '(vault root)/'
    if name == 'create_note':
        title = str(args.get('note_title', '')).strip()
        folder = str(args.get('folder', '')).strip()
        if not title:
            return None
        return f'{folder}/{title}' if folder else title
    if name in {'edit_note', 'delete_note', 'update_frontmatter'}:
        return str(args.get('note_title', '')).strip() or None
    if name == 'move_note':
        title = str(args.get('note_title', '')).strip()
        dest = str(args.get('destination_folder', '')).strip()
        if title and dest:
            return f'{title} → {dest}/'
        return title or None
    if name == 'rename_note':
        old = str(args.get('note_title', '')).strip()
        new = str(args.get('new_title', '')).strip()
        if old and new:
            return f'{old} → {new}'
        return old or new or None
    if name in {'create_folder', 'delete_folder'}:
        return str(args.get('relative_path', '')).strip() or None
    return None


def vault_activity_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Collect note/folder paths read vs changed from assistant tool_calls."""
    reads: list[str] = []
    modified: list[str] = []
    seen_r: set[str] = set()
    seen_m: set[str] = set()
    for msg in messages:
        if msg.get('role') != 'assistant':
            continue
        raw_tc = msg.get('tool_calls')
        if not raw_tc:
            continue
        for tc in raw_tc if isinstance(raw_tc, list) else []:
            fn = (tc.get('function') if isinstance(tc, dict) else {}) or {}
            tname = str(fn.get('name', ''))
            targs = parse_tool_arguments(fn.get('arguments', {}))
            ref = _note_ref_from_tool(tname, targs)
            if not ref:
                continue
            if tname in _READ_NOTE_TOOLS and ref not in seen_m:
                if ref not in seen_r:
                    reads.append(ref)
                    seen_r.add(ref)
            elif tname in _MODIFY_NOTE_TOOLS:
                if ref not in seen_m:
                    modified.append(ref)
                    seen_m.add(ref)
    return reads, modified


def _format_dropped_transcript(
    dropped: list[dict[str, Any]],
    per_message: int,
) -> str:
    """Flatten dropped chat messages into plain text for the summarizer."""
    blocks: list[str] = []
    for msg in dropped:
        role = str(msg.get('role', '?'))
        name = msg.get('name')
        name_s = f' ({name})' if name else ''
        content = msg.get('content')
        text = content if isinstance(content, str) else ''
        if len(text) > per_message:
            text = text[:per_message] + '\n…[truncated]'
        extra = ''
        tc = msg.get('tool_calls')
        if tc:
            extra = f'\n[tool_calls: {_tool_calls_brief(tc)}]'
        blocks.append(f'{role}{name_s}:\n{text}{extra}')
    reads, modified = vault_activity_from_messages(dropped)
    if reads:
        blocks.append('Notes read:\n' + '\n'.join(f'- {p}' for p in reads))
    if modified:
        blocks.append('Notes changed:\n' + '\n'.join(f'- {p}' for p in modified))
    return '\n\n---\n\n'.join(blocks)


def summarize_dropped_turns(dropped: list[dict[str, Any]]) -> str | None:
    """
    Ask the chat model (no tools, non-streaming) to compress *dropped* turns.

    Returns stripped summary text, or ``None`` if the call fails or yields
    empty content.
    """
    ctx = get_context()
    cc = ctx.defaults.context_compression
    transcript = _format_dropped_transcript(
        dropped,
        cc.max_transcript_chars_per_message,
    )
    user_turn = (
        f'Summarize the following prior conversation transcript in at most '
        f'{cc.max_summary_lines} short lines of plain text (no fenced code '
        'blocks). Output ONLY the summary — do not continue the conversation '
        'or answer any questions from the transcript.\n\n'
        f'<conversation>\n{transcript}\n</conversation>'
    )
    resp = retry_call(
        lambda: ctx.ollama_client.chat(
            model=ctx.user.ollama_model,
            messages=[
                {'role': 'system', 'content': cc.summarizer_system_prompt},
                {'role': 'user', 'content': user_turn},
            ],
            stream=False,
        ),
        ctx.defaults.retry,
    )
    raw_msg = resp.get('message') if isinstance(resp, dict) else getattr(resp, 'message', resp)
    parsed = msg_to_dict(raw_msg)
    body = (parsed.get('content') or '').strip()
    return body or None


def load_agent_history() -> list[dict[str, Any]]:
    ctx = get_context()
    return load_history(
        ctx.history_path,
        max_messages=ctx.defaults.limits.max_history_messages,
        truncate_fn=truncate_messages_pair_aware,
        on_error=ui.print_error,
    )


def save_agent_history(messages: list[dict[str, Any]]) -> None:
    ctx = get_context()
    save_history(
        ctx.history_path,
        messages,
        max_messages=ctx.defaults.limits.max_history_messages,
        truncate_fn=truncate_messages_pair_aware,
        on_error=ui.print_error,
    )


def trim_context(
    messages: list[dict[str, Any]],
    *,
    force: bool = False,
    on_event: EventHandler | None = None,
) -> list[dict[str, Any]]:
    """
    Trim non-system context, optionally compressing evicted turns into a summary.

    **Trigger logic** (two independent signals, either fires compression):

    1. *Token fill* — when Ollama reported ``num_ctx`` at startup and the last
       response included ``prompt_eval_count``, compress when the prompt consumed
       ≥ ``context_fill_ratio`` of the model's token budget.
    2. *Message count* — fallback cap (``max_context_messages``) always enforced;
       used as the sole trigger when token counts are unavailable.

    When rolling summary is enabled, evicted messages are summarized into one
    synthetic assistant message placed before the retained tail.  On failure,
    falls back to plain truncation.
    """
    ctx = get_context()
    cc = ctx.defaults.context_compression
    cap = ctx.defaults.limits.max_context_messages
    system_msgs = [m for m in messages if m.get('role') == 'system']
    rest = [m for m in messages if m.get('role') != 'system']

    token_pressure = False
    if ctx.num_ctx > 0 and ctx.last_prompt_tokens > 0:
        fill = ctx.last_prompt_tokens / ctx.num_ctx
        token_pressure = fill >= cc.context_fill_ratio
        logger.debug(
            'Token fill: %d / %d = %.1f%% (threshold %.0f%%)',
            ctx.last_prompt_tokens,
            ctx.num_ctx,
            fill * 100,
            cc.context_fill_ratio * 100,
        )

    dropped, kept = split_messages_pair_aware(rest, cap)

    if force and len(rest) > 1:
        split_cap = max(1, len(rest) // 2)
        dropped, kept = split_messages_pair_aware(rest, split_cap)

    if not dropped and not token_pressure and not force:
        return system_msgs + kept

    if not dropped and not force:
        return system_msgs + kept

    if not cc.enabled or cap < 2:
        return system_msgs + kept

    if not dropped and token_pressure:
        split_cap = max(1, len(rest) // 2)
        dropped, kept = split_messages_pair_aware(rest, split_cap)
        if not dropped:
            return system_msgs + kept

    summary_body: str | None = None
    emit_agent_event(on_event, 'compact_start', reason='force' if force else 'threshold')
    wait = ui.generation_wait_start(ctx.defaults.ui.cli_compression_wait_message)
    try:
        summary_body = summarize_dropped_turns(dropped)
    except Exception as exc:
        logger.warning('Rolling context summary failed: %s', exc, exc_info=True)
        ui.print_error(
            f'Could not compress prior context ({exc}); oldest messages were dropped.',
        )
    finally:
        wait.finish()

    emit_agent_event(on_event, 'compact_end', success=bool(summary_body))

    if not summary_body:
        return system_msgs + kept

    summary_body = sanitize_assistant_text(summary_body, ctx)
    label = cc.summary_message_label.strip()
    summary_msg: dict[str, Any] = {
        'role': 'assistant',
        'content': f'{label}\n\n{summary_body}' if label else summary_body,
    }

    tail_cap = cap - 1
    kept_tail = truncate_messages_pair_aware(kept, tail_cap) if tail_cap >= 1 else []
    return system_msgs + [summary_msg] + kept_tail
