"""Ollama streaming chat, message sanitization, and assistant UI finalization."""

from __future__ import annotations

import logging
import sys
from typing import Any

from brain.ui import render as ui
from brain.core.context import get_context
from brain.core.messages import msg_to_dict
from brain.core.retry import retry_call
from brain.core.sanitize import extract_display_thinking, sanitize_assistant_text
from brain.core.streaming import merge_stream_content
from brain.core.truncate import truncate_for_model
from brain.repl.display_port import (
    PlainTtyDisplayPort,
    ReplDisplayPort,
    RichConsoleDisplayPort,
    active_display_port,
    set_display_port,
)
from brain.vault.tools import TOOL_LIST

logger = logging.getLogger(__name__)


def _chunk_message(chunk: Any) -> Any:
    if isinstance(chunk, dict):
        return chunk.get('message') or {}
    return getattr(chunk, 'message', None) or {}


def _stream_text_deltas(msg: Any) -> tuple[str, str]:
    """Return ``(content_delta, thinking_delta)`` from one streamed message object."""
    if msg is None:
        return '', ''
    if hasattr(msg, 'content'):
        c_raw = msg.content
    elif isinstance(msg, dict):
        c_raw = msg.get('content')
    else:
        c_raw = ''
    c = c_raw if isinstance(c_raw, str) else (str(c_raw) if c_raw is not None else '')
    if hasattr(msg, 'thinking'):
        t_raw = getattr(msg, 'thinking', None)
    elif isinstance(msg, dict):
        t_raw = msg.get('thinking')
    else:
        t_raw = ''
    t = t_raw if isinstance(t_raw, str) else (str(t_raw) if t_raw is not None else '')
    return c, t


def _raw_tool_calls(msg: Any) -> Any:
    """Return ``tool_calls`` from a chunk message, or ``None`` if absent / empty."""
    if msg is None:
        return None
    if isinstance(msg, dict):
        tc = msg.get('tool_calls')
    else:
        tc = getattr(msg, 'tool_calls', None)
    if not tc:
        return None
    return tc


def reset_thinking_display_state() -> None:
    """Clear in-flight reasoning UI before a new model round (keep turn commit flag)."""
    port = active_display_port()
    port.cancel_thinking()
    port.last_thinking_streamed_to_tty = False
    set_display_port(port)


def discard_thinking_display() -> None:
    """Drop buffered reasoning without keeping a panel (tool-planning rounds)."""
    active_display_port().discard_thinking()


def commit_thinking_display(raw_thinking: str, *, think: bool, print_mode: bool) -> None:
    """Commit filtered reasoning once for the final answer (never tool rounds)."""
    if not think or print_mode or not raw_thinking.strip():
        return
    port = active_display_port()
    if port.has_committed_thinking():
        return
    extracted = extract_display_thinking(raw_thinking, streaming=False)
    if not extracted:
        discard_thinking_display()
        return
    text = sanitize_assistant_text(extracted, get_context())
    port = active_display_port()
    if isinstance(port, ReplDisplayPort):
        port.commit_thinking(raw_thinking, final_content=text)
        return
    if ui.thinking_stream_is_active():
        ui.end_thinking_stream(final_content=text)
    elif not port.last_thinking_streamed_to_tty:
        port.print_thinking_panel(text)


def finalize_assistant_answer(
    visible: str,
    empty_fallback: str,
    *,
    print_mode: bool,
    was_streamed: bool,
    content_echoed_to_tty: bool,
) -> None:
    """Commit streamed or buffered assistant text to the UI."""
    port = active_display_port()
    if print_mode:
        port.last_answer_streamed_to_tty = content_echoed_to_tty
        return
    if not was_streamed:
        ui.print_agent(visible if visible else empty_fallback)
        return
    if (
        content_echoed_to_tty
        or ui.answer_stream_is_active()
        or port.has_live_answer()
    ):
        ui.end_answer_stream(final_content=visible if visible else None)
        return
    if not visible:
        ui.cancel_answer_stream()
        ui.print_agent(empty_fallback)
        return
    ui.print_agent(visible)


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip lone surrogates from message content strings before JSON encoding.

    Lone surrogates (e.g. ``\\udcc2``) arise when file bytes invalid in the
    declared encoding are read under the ``'surrogateescape'`` handler; they
    propagate silently until ``httpx`` tries to JSON-encode them.
    ``_read_vault_text`` avoids this at the source; this is the safety net.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, str) and any('\uD800' <= ch <= '\uDFFF' for ch in content):
            msg = {**msg, 'content': content.encode('utf-8', errors='replace').decode('utf-8')}
        out.append(msg)
    return out


def truncate_tool_result(text: str) -> str:
    ctx = get_context()
    lim = ctx.defaults.limits
    return truncate_for_model(
        text,
        max_lines=lim.max_tool_result_lines,
        max_bytes=lim.max_tool_result_bytes,
    )


def _turn_has_tool_results(messages: list[dict[str, Any]]) -> bool:
    """Return True when the current user turn already includes tool output."""
    for msg in reversed(messages):
        role = msg.get('role')
        if role == 'user':
            return False
        if role == 'tool':
            return True
    return False


def _present_thinking_before_answer(raw_thinking: str) -> None:
    """Stream filtered reasoning immediately before the answer panel."""
    ui.reveal_thinking_before_answer(raw_thinking)


def agent_chat(
    messages: list[dict[str, Any]],
    think: bool,
    *,
    stream_answers: bool = False,
    print_mode: bool = False,
    live_thinking: bool | None = None,
) -> tuple[Any, bool, bool]:
    """Call Ollama with streaming.

    Returns ``(response, was_streamed, assistant_content_echoed_to_stdout)``.

    When ``stream_answers`` is True, assistant ``content`` deltas are written
    live as plain text. Otherwise Rich mode buffers until ``print_agent``.

    The wait indicator starts **before** ``Client.chat`` returns: some Ollama
    stacks answer with a complete dict (non-streaming) while ``stream=True``,
    especially after long tool turns — that call blocks with no chunk iterator,
    so the spinner must cover the whole HTTP wait, not only the stream loop.
    """
    ctx = get_context()
    kwargs: dict[str, Any] = {
        'model': ctx.user.ollama_model,
        'messages': _sanitize_messages(messages),
        'tools': TOOL_LIST,
        'stream': True,
    }
    if think:
        kwargs['think'] = True

    port = active_display_port()
    repl_active = ctx.repl_display is not None
    use_rich = repl_active or isinstance(port, (ReplDisplayPort, RichConsoleDisplayPort))
    live_stream = stream_answers and (use_rich or print_mode or repl_active)
    if live_thinking is None:
        live_thinking = False
    wait_msg = (
        ctx.defaults.ui.cli_generation_wait_message_think
        if think
        else ctx.defaults.ui.cli_generation_wait_message
    )
    wait = ui.generation_wait_start(wait_msg)
    wait_open = True
    final: Any = None
    streamed_any = False
    content_echoed_to_tty = False
    content_running = ''
    thinking_running = ''
    thinking_displayed_prefix = ''
    last_tool_calls: Any = None
    tool_round = False
    visible_stream_started = False
    reset_thinking_display_state()
    try:
        def _start_chat() -> Any:
            try:
                return ctx.ollama_client.chat(**kwargs)
            except TypeError:
                if think:
                    ui.print_error(
                        '--think was ignored: upgrade the `ollama` Python package so '
                        'Client.chat() accepts think= (needs a recent ollama package).',
                    )
                    kwargs.pop('think', None)
                    return ctx.ollama_client.chat(**kwargs)
                raise

        resp = retry_call(_start_chat, ctx.defaults.retry)

        if isinstance(resp, dict):
            return resp, False, False

        for chunk in resp:
            final = chunk
            msg = _chunk_message(chunk)
            c_piece, t_piece = _stream_text_deltas(msg)
            tc = _raw_tool_calls(msg)
            if tc:
                last_tool_calls = tc
                tool_round = True
                if ui.answer_stream_is_active():
                    ui.cancel_answer_stream()
                discard_thinking_display()
            if c_piece:
                content_running = merge_stream_content(content_running, c_piece)
                if live_stream and not tool_round and content_running.strip():
                    if wait_open:
                        wait.finish()
                        wait_open = False
                    visible_stream_started = True
                    if (
                        think
                        and not print_mode
                        and not port.has_committed_thinking()
                        and thinking_running.strip()
                        and not ui.thinking_stream_is_active()
                    ):
                        _present_thinking_before_answer(thinking_running)
                    elif (
                        think
                        and not port.has_committed_thinking()
                        and thinking_running.strip()
                        and ui.thinking_stream_is_active()
                    ):
                        commit_thinking_display(
                            thinking_running,
                            think=think,
                            print_mode=print_mode,
                        )
                    if not ui.answer_stream_is_active():
                        plain = print_mode or isinstance(port, PlainTtyDisplayPort)
                        ui.begin_answer_stream(plain=plain)
                    ui.write_answer_delta(c_piece)
                    content_echoed_to_tty = True
                elif isinstance(port, PlainTtyDisplayPort) and not repl_active:
                    if wait_open:
                        wait.finish()
                        wait_open = False
                    visible_stream_started = True
                    sys.stdout.write(c_piece)
                    sys.stdout.flush()
                    content_echoed_to_tty = True
                streamed_any = True
            if t_piece:
                thinking_running = merge_stream_content(thinking_running, t_piece)
                streamed_any = True
                if (
                    live_thinking
                    and not tool_round
                    and not _turn_has_tool_results(messages)
                    and not port.has_committed_thinking()
                ):
                    previous = thinking_displayed_prefix
                    thinking_displayed_prefix = ui.sync_thinking_display_stream(
                        thinking_running,
                        displayed_prefix=thinking_displayed_prefix,
                    )
                    if (
                        thinking_displayed_prefix
                        and thinking_displayed_prefix != previous
                    ):
                        visible_stream_started = True
                        if wait_open:
                            wait.finish()
                            wait_open = False

        if tool_round:
            discard_thinking_display()
        elif (
            think
            and thinking_running.strip()
            and not port.has_committed_thinking()
            and ui.thinking_stream_is_active()
        ):
            commit_thinking_display(
                thinking_running,
                think=think,
                print_mode=print_mode,
            )
        if ui.answer_stream_is_active():
            defer_finalize = live_stream and use_rich and not print_mode
            if not defer_finalize:
                ui.end_answer_stream()
        elif content_echoed_to_tty and not live_stream:
            sys.stdout.write('\n')
            sys.stdout.flush()

        if final is None:
            return {'message': {'role': 'assistant', 'content': ''}}, False, False

        assembled_content = content_running
        assembled_thinking = thinking_running

        if isinstance(final, dict):
            out: dict[str, Any] = dict(final)
            inner = msg_to_dict(out.get('message') or {})
        else:
            out = msg_to_dict(final)
            inner = msg_to_dict(out.get('message') or out)

        if assembled_content and len(assembled_content) >= len((inner.get('content') or '')):
            inner['content'] = assembled_content
        if assembled_thinking and len(assembled_thinking) >= len((inner.get('thinking') or '')):
            inner['thinking'] = assembled_thinking

        if last_tool_calls is not None:
            inner['tool_calls'] = last_tool_calls

        out['message'] = inner

        if isinstance(final, dict):
            prompt_tokens = final.get('prompt_eval_count') or 0
        else:
            prompt_tokens = getattr(final, 'prompt_eval_count', 0) or 0
        if prompt_tokens:
            ctx.last_prompt_tokens = int(prompt_tokens)
            refresh = ctx.repl_ui_refresh
            if refresh is not None:
                refresh()

        return out, streamed_any, content_echoed_to_tty
    except Exception:
        if ui.answer_stream_is_active():
            ui.cancel_answer_stream()
        if ui.thinking_stream_is_active():
            ui.cancel_thinking_stream()
        raise
    finally:
        wait.finish()
