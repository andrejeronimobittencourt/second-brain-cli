"""Interactive REPL: history, Ollama chat+tools, slash commands."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

from brain import ui
from brain.context import get_context
from brain.defaults import default_system_prompt
from brain.messages import msg_to_dict
from brain.sanitize import sanitize_assistant_text
from brain.tools import TOOL_LIST, TOOL_MAP, ToolResult, active_vision_model, search_notes

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


_SEARCH_PREFIX = '/search '

_SLASH_EXIT_COMMANDS = frozenset({'/exit', '/quit', '/bye'})

_HELP_TEXT = """
Available slash commands:
  /help          — show this message
  /clear         — wipe conversation history and start fresh
  /history       — show the number of messages currently in context
  /search <term> — quick vault-wide search (bypasses the LLM)
  /exit          — save session and exit (aliases: /quit, /bye)

CLI: --config <file>  (or env SECOND_BRAIN_USER_CONFIG), --think, --resume,
     --vision-model, --host, --model (override user JSON for one run).

User JSON: ``system_prompt`` replaces the built-in system text when set.
           ``vault_instructions`` is always appended after that base for
           vault- or app-specific hints (e.g. editor plugins, link conventions).
"""


def _effective_system_prompt() -> str:
    """
    Build the system message for Ollama.

    ``UserConfig.system_prompt`` — when non-empty, replaces the built-in default
    entirely. ``UserConfig.vault_instructions`` — when non-empty, is always
    appended after that base (default or custom), for vault- or editor-specific
    rules without discarding the rest of the instructions.
    """
    ctx = get_context()
    custom = (ctx.user.system_prompt or '').strip()
    add = (ctx.user.vault_instructions or '').strip()
    base = custom if custom else default_system_prompt()
    if not add:
        return base
    return f'{base}\n\n{add}'


def _split_truncate_pair_aware(
    messages: list[dict[str, Any]],
    cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Partition *messages* into ``(dropped_prefix, kept_suffix)``.

    The suffix keeps at most *cap* messages; if the slice begins with a stray
    ``tool`` message (no preceding assistant in the window), leading tools are
    removed from the suffix and included in the dropped side logically.
    """
    if len(messages) <= cap:
        return [], messages
    trimmed = messages[-cap:]
    while trimmed and trimmed[0].get('role') == 'tool':
        trimmed.pop(0)
    n_kept = len(trimmed)
    dropped = messages[: len(messages) - n_kept]
    if dropped:
        logger.debug(
            'Context trimmed: dropped %d oldest messages (cap=%d)',
            len(dropped),
            cap,
        )
    return dropped, trimmed


def _truncate_pair_aware(
    messages: list[dict[str, Any]],
    cap: int,
) -> list[dict[str, Any]]:
    """Keep the last *cap* messages, dropping orphaned leading tool results."""
    _, kept = _split_truncate_pair_aware(messages, cap)
    return kept


def _tool_calls_brief(tool_calls: Any) -> str:
    """Short JSON-ish preview of tool calls for summarizer input."""
    if not tool_calls:
        return ''
    try:
        return json.dumps(tool_calls, ensure_ascii=False)[:500]
    except (TypeError, ValueError):
        return str(tool_calls)[:500]


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
    return '\n\n---\n\n'.join(blocks)


def _summarize_dropped_turns(dropped: list[dict[str, Any]]) -> str | None:
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
        'blocks). Be conservative; omit anything unclear.\n\n'
        f'{transcript}'
    )
    resp = ctx.ollama_client.chat(
        model=ctx.user.ollama_model,
        messages=[
            {'role': 'system', 'content': cc.summarizer_system_prompt},
            {'role': 'user', 'content': user_turn},
        ],
        stream=False,
    )
    raw_msg = resp.get('message') if isinstance(resp, dict) else getattr(resp, 'message', resp)
    parsed = msg_to_dict(raw_msg)
    body = (parsed.get('content') or '').strip()
    return body or None


def _load_history() -> list[dict[str, Any]]:
    ctx = get_context()
    path = ctx.history_path
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        ui.print_error(
            f"Session file '{path}' is corrupted (invalid JSON) and will be ignored.",
        )
        return []
    except OSError as exc:
        ui.print_error(f"Could not read session file '{path}': {exc}")
        return []
    if not isinstance(raw, list):
        ui.print_error(
            f"Session file '{path}' has unexpected format and will be ignored.",
        )
        return []
    cap = ctx.defaults.limits.max_history_messages
    return _truncate_pair_aware([m for m in raw if isinstance(m, dict)], cap)


def _save_history(messages: list[dict[str, Any]]) -> None:
    ctx = get_context()
    non_system = [m for m in messages if m.get('role') != 'system']
    cap = ctx.defaults.limits.max_history_messages
    non_system = _truncate_pair_aware(non_system, cap)
    try:
        ctx.history_path.write_text(
            json.dumps(non_system, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    except OSError as exc:
        ui.print_error(f'Session could not be saved: {exc}')


def _trim_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    # --- Determine whether the token-fill threshold has been crossed ---
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

    # --- Always enforce the hard message-count cap ---
    dropped, kept = _split_truncate_pair_aware(rest, cap)

    if not dropped and not token_pressure:
        # Neither signal triggered — nothing to do.
        return system_msgs + kept

    if not cc.enabled or cap < 2:
        # Compression disabled: apply hard cap only.
        return system_msgs + kept

    # --- Token pressure without message-cap overflow: force a split ---
    # Evict the older half so the summary replaces roughly half the context,
    # giving the model headroom before the next compression cycle.
    if not dropped and token_pressure:
        split_cap = max(1, len(rest) // 2)
        dropped, kept = _split_truncate_pair_aware(rest, split_cap)
        if not dropped:
            return system_msgs + kept

    # --- Summarize evicted turns ---
    summary_body: str | None = None
    wait = ui.generation_wait_start(ctx.defaults.ui.cli_compression_wait_message)
    try:
        summary_body = _summarize_dropped_turns(dropped)
    except Exception as exc:
        logger.warning('Rolling context summary failed: %s', exc, exc_info=True)
        ui.print_error(
            f'Could not compress prior context ({exc}); oldest messages were dropped.',
        )
    finally:
        wait.finish()

    if not summary_body:
        return system_msgs + kept

    summary_body = sanitize_assistant_text(summary_body, ctx)
    label = cc.summary_message_label.strip()
    summary_msg: dict[str, Any] = {
        'role': 'assistant',
        'content': f'{label}\n\n{summary_body}' if label else summary_body,
    }

    tail_cap = cap - 1
    kept_tail = _truncate_pair_aware(kept, tail_cap) if tail_cap >= 1 else []
    return system_msgs + [summary_msg] + kept_tail


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


def _agent_chat(
    messages: list[dict[str, Any]],
    think: bool,
) -> tuple[Any, bool, bool]:
    """Call Ollama with streaming.

    Returns ``(response, was_streamed, assistant_content_echoed_to_stdout)``.

    When Rich is active, assistant ``content`` deltas are **not** written to the
    TTY so ``print_agent`` can render the full answer in a panel without
    duplication. The last chunk often clears ``content``, ``thinking``, and
    ``tool_calls``; earlier-chunk deltas are merged into ``response``.

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

    use_rich = ctx.console is not None
    wait_msg = (
        ctx.defaults.ui.cli_generation_wait_message_think
        if think
        else ctx.defaults.ui.cli_generation_wait_message
    )
    wait = ui.generation_wait_start(wait_msg)
    # With Rich, assistant text is not streamed to the TTY (panels render after the
    # full chunk loop). Stopping the wait indicator on the first thinking/tool
    # delta leaves a long blank gap; keep the spinner until the stream ends.
    early_finish_wait = not use_rich
    wait_open = True
    final: Any = None
    streamed_any = False
    content_echoed_to_tty = False
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    last_tool_calls: Any = None
    try:
        try:
            resp = ctx.ollama_client.chat(**kwargs)
        except TypeError:
            if think:
                ui.print_error(
                    '--think was ignored: upgrade the `ollama` Python package so '
                    'Client.chat() accepts think= (needs a recent ollama package).',
                )
                kwargs.pop('think', None)
                resp = ctx.ollama_client.chat(**kwargs)
            else:
                raise

        # Non-streaming fallback: some Ollama versions return a dict directly
        if isinstance(resp, dict):
            return resp, False, False

        for chunk in resp:
            final = chunk
            msg = _chunk_message(chunk)
            c_piece, t_piece = _stream_text_deltas(msg)
            tc = _raw_tool_calls(msg)
            if c_piece:
                if wait_open and early_finish_wait:
                    wait.finish()
                    wait_open = False
                content_parts.append(c_piece)
                if not use_rich:
                    sys.stdout.write(c_piece)
                    sys.stdout.flush()
                    content_echoed_to_tty = True
                streamed_any = True
            if t_piece:
                thinking_parts.append(t_piece)
                streamed_any = True
            if tc:
                last_tool_calls = tc

        if content_echoed_to_tty:
            sys.stdout.write('\n')
            sys.stdout.flush()

        if final is None:
            return {'message': {'role': 'assistant', 'content': ''}}, False, False

        assembled_content = ''.join(content_parts)
        assembled_thinking = ''.join(thinking_parts)

        # Ollama's last streaming chunk often has ``message.content == ''`` while earlier
        # chunks held the deltas. Merge accumulated text for history and UI.
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

        # Same pattern as ``content``: the last chunk may clear ``tool_calls`` even though
        # earlier chunks carried the call — keep the last non-empty stream value.
        if last_tool_calls is not None:
            inner['tool_calls'] = last_tool_calls

        out['message'] = inner

        # Track how many prompt tokens were consumed so _trim_context can compare
        # against ctx.num_ctx for a token-fill-based compression trigger.
        if isinstance(final, dict):
            prompt_tokens = final.get('prompt_eval_count') or 0
        else:
            prompt_tokens = getattr(final, 'prompt_eval_count', 0) or 0
        if prompt_tokens:
            ctx.last_prompt_tokens = int(prompt_tokens)

        return out, streamed_any, content_echoed_to_tty
    finally:
        wait.finish()


def _coerce_single_tool_call(tc: Any) -> dict[str, Any]:
    """Normalise one raw tool call (dict or SDK object) into a plain dict."""
    raw = tc if isinstance(tc, dict) else msg_to_dict(tc)
    fn = raw.get('function') or {}
    name = fn.get('name', '')
    args = fn.get('arguments', {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            logger.warning(
                'Tool call %r had malformed JSON arguments (falling back to {}): %r',
                name,
                args,
            )
            args = {}
    return {'function': {'name': name, 'arguments': args}}


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Coerce SDK tool call objects into dicts with ``function.name`` / ``arguments``."""
    if not raw:
        return []
    return [_coerce_single_tool_call(tc) for tc in raw]


def _run_tool_loop(
    messages: list[dict[str, Any]],
    think: bool,
    empty_fallback: str,
) -> list[dict[str, Any]]:
    """
    Drive the inner agentic loop for one user turn: keep calling the model and
    dispatching tool calls until the model produces a final text response.

    Returns the updated ``messages`` list.
    """
    ctx = get_context()
    max_rounds = ctx.defaults.limits.max_tool_rounds
    rounds = 0
    while True:
        response, was_streamed, content_echoed_to_tty = _agent_chat(messages, think)
        raw_msg = response.get('message') or response
        message = msg_to_dict(raw_msg)
        if message.get('content'):
            message['content'] = sanitize_assistant_text(message['content'], ctx)
        if message.get('thinking'):
            message['thinking'] = sanitize_assistant_text(message['thinking'], ctx)

        tool_calls = _normalize_tool_calls(message.get('tool_calls'))
        message['tool_calls'] = tool_calls if tool_calls else None
        messages.append(message)

        if not tool_calls:
            # Reasoning panel only for the final assistant reply (no tool_calls). Thinking
            # on tool-planning turns stays in ``messages`` for the model but is not
            # duplicated in the terminal.
            if think and message.get('thinking'):
                ui.print_thinking(message['thinking'])
            visible = (message.get('content') or '').strip()
            if not was_streamed:
                ui.print_agent(visible if visible else empty_fallback)
            elif not visible:
                ui.print_agent(empty_fallback)
            elif not content_echoed_to_tty:
                ui.print_agent(visible)
            break

        rounds += 1
        if rounds > max_rounds:
            fallback = (
                '_(Stopped: tool-call limit reached. '
                'Please rephrase or simplify your request.)_'
            )
            messages.append({'role': 'assistant', 'content': fallback})
            ui.print_agent(fallback)
            break

        for tool_call in tool_calls:
            func_name = str(tool_call['function']['name'])
            args = tool_call['function']['arguments']
            if not isinstance(args, dict):
                args = {}

            ui.print_tool(func_name, args)
            logger.debug('Tool call: %s(%s)', func_name, args)

            if func_name not in TOOL_MAP:
                result = f"System Error: Unknown tool '{func_name}'."
            else:
                try:
                    result = TOOL_MAP[func_name](**args)
                except Exception as exc:
                    logger.warning('Tool %s raised: %s', func_name, exc)
                    result = f'Error executing {func_name}: {exc}'

            logger.debug('Tool result: %.200s', str(result))
            messages.append({
                'role': 'tool',
                'content': str(result),
                'name': func_name,
            })

    return messages


def run_agent(
    resume: bool = False,
    think: bool = False,
    vision_model: str = '',
) -> None:
    """
    Run the interactive REPL until the user exits.

    Requires ``bootstrap()`` to have installed ``ApplicationContext``.

    :param resume: If True, prepend messages loaded from ``history_path``.
    :param think: If True, pass ``think=True`` to Ollama when the client supports it.
    :param vision_model: Overrides the vision model for ``read_image`` for this session.
    """
    ctx = get_context()
    ctx.vision_model_override = vision_model.strip()

    vault = ctx.vault_path.resolve()
    ui.print_info(f'Vault : {vault}')
    if ctx.vault_path_defaulted:
        ui.print_error(
            'vault_path is not set in your user JSON — defaulting to the script '
            'directory. Set vault_path to your Markdown vault folder (your notes root).',
        )
    ui.print_info(f'Model : {ctx.user.ollama_model}')
    if ctx.num_ctx > 0:
        threshold_pct = ctx.defaults.context_compression.context_fill_ratio * 100
        ui.print_info(
            f'Context : {ctx.num_ctx:,} tokens '
            f'(compresses at {threshold_pct:.0f}% fill)',
        )
    ui.print_info(f'Ollama : {ctx.user.ollama_host}')
    ui.print_info(f'Config : {ctx.config_source}')
    if think:
        ui.print_info('Thinking: on (when supported by your Ollama server / model)')
    ui.print_info(f'Vision (read_image): {active_vision_model()}')
    if (ctx.user.vault_instructions or '').strip():
        ui.print_info('Vault-specific instructions: loaded from user JSON.')
    ui.print_info("Type '/help' for commands; use /exit (or /quit, /bye) to quit.\n")

    system_msg: dict[str, Any] = {'role': 'system', 'content': _effective_system_prompt()}
    messages: list[dict[str, Any]] = [system_msg]

    if resume:
        prior = _load_history()
        if prior:
            messages.extend(prior)
            ui.print_info(f'Resumed previous session ({len(prior)} messages loaded).\n')
        else:
            ui.print_info('No saved session found — starting fresh.\n')

    empty_fallback = ctx.defaults.model.empty_answer_markdown

    def _slash_help() -> None:
        print(_HELP_TEXT)

    def _slash_clear() -> None:
        nonlocal messages
        messages = [system_msg]
        if ctx.history_path.exists():
            ctx.history_path.unlink()
        ui.print_info('Conversation cleared.')

    def _slash_history() -> None:
        count = sum(1 for m in messages if m.get('role') != 'system')
        lim = ctx.defaults.limits.max_context_messages
        info = f'{count} messages in current context (cap: {lim})'
        if ctx.num_ctx > 0:
            fill_pct = (ctx.last_prompt_tokens / ctx.num_ctx * 100) if ctx.last_prompt_tokens else 0
            threshold_pct = ctx.defaults.context_compression.context_fill_ratio * 100
            info += (
                f'; last prompt {ctx.last_prompt_tokens:,} / {ctx.num_ctx:,} tokens'
                f' ({fill_pct:.0f}% — compresses at {threshold_pct:.0f}%)'
            )
        ui.print_info(info + '.')

    slash_handlers: dict[str, Callable[[], None]] = {
        '/help': _slash_help,
        '/clear': _slash_clear,
        '/history': _slash_history,
    }

    while True:
        try:
            user_input = input('You: ').strip()
        except (KeyboardInterrupt, EOFError):
            _save_history(messages)
            ui.print_info('\nSession saved. Goodbye.')
            break

        if not user_input:
            continue

        if user_input.lower() in _SLASH_EXIT_COMMANDS:
            _save_history(messages)
            ui.print_info('Session saved. Goodbye.')
            break

        handler = slash_handlers.get(user_input)
        if handler is not None:
            handler()
            continue

        if user_input.startswith(_SEARCH_PREFIX):
            query = user_input[len(_SEARCH_PREFIX):].strip()
            if query:
                print(search_notes(query))
            else:
                ui.print_error(f'Usage: {_SEARCH_PREFIX}<term>')
            continue

        messages.append({'role': 'user', 'content': user_input})
        messages = _trim_context(messages)
        messages = _run_tool_loop(messages, think, empty_fallback)
        _save_history(messages)
