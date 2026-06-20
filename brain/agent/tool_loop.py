"""Inner tool-calling loop for one user turn."""

from __future__ import annotations

import json
import logging
from typing import Any

from brain.ui import render as ui
from brain.agent.compression import trim_context
from brain.agent.ollama_chat import (
    agent_chat,
    commit_thinking_display,
    discard_thinking_display,
    finalize_assistant_answer,
    truncate_tool_result,
)
from brain.core.context import get_context
from brain.core.events import EventHandler, emit_agent_event
from brain.core.messages import msg_to_dict, parse_tool_arguments
from brain.core.overflow import is_context_overflow
from brain.core.sanitize import sanitize_assistant_text
from brain.core.tool_validate import validate_tool_call
from brain.vault.tools import TOOL_MAP, ToolResult

logger = logging.getLogger(__name__)


def _coerce_single_tool_call(tc: Any) -> dict[str, Any]:
    """Normalise one raw tool call (dict or SDK object) into a plain dict."""
    raw = tc if isinstance(tc, dict) else msg_to_dict(tc)
    fn = raw.get('function') or {}
    name = fn.get('name', '')
    raw_args = fn.get('arguments', {})
    args = parse_tool_arguments(raw_args)
    if isinstance(raw_args, str) and raw_args.strip() and not args:
        try:
            json.loads(raw_args)
        except json.JSONDecodeError:
            logger.warning(
                'Tool call %r had malformed JSON arguments (falling back to {}): %r',
                name,
                raw_args,
            )
    return {'function': {'name': name, 'arguments': args}}


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Coerce SDK tool call objects into dicts with ``function.name`` / ``arguments``."""
    if not raw:
        return []
    return [_coerce_single_tool_call(tc) for tc in raw]


def run_tool_loop(
    messages: list[dict[str, Any]],
    think: bool,
    empty_fallback: str,
    *,
    on_event: EventHandler | None = None,
    print_mode: bool = False,
    interactive: bool = False,
) -> list[dict[str, Any]]:
    """
    Drive the inner agentic loop for one user turn: keep calling the model and
    dispatching tool calls until the model produces a final text response.

    Returns the updated ``messages`` list.
    """
    ctx = get_context()
    ctx.display.reset_turn()
    ctx.display.cancel_answer()
    ctx.display.cancel_thinking()
    max_rounds = ctx.defaults.limits.max_tool_rounds
    stream_answers = ctx.defaults.repl.stream_answers and interactive
    rounds = 0
    overflow_attempted = False
    emit_agent_event(on_event, 'turn_start')
    while True:
        try:
            response, was_streamed, content_echoed_to_tty = agent_chat(
                messages,
                think,
                stream_answers=stream_answers,
                print_mode=print_mode,
                live_thinking=False,
            )
        except Exception as exc:
            if (
                not overflow_attempted
                and is_context_overflow(
                    error=exc,
                    prompt_tokens=ctx.last_prompt_tokens,
                    num_ctx=ctx.num_ctx,
                )
            ):
                overflow_attempted = True
                ui.print_info(ctx.defaults.ui.cli_overflow_retry_message)
                trimmed = trim_context(messages, force=True, on_event=on_event)
                messages.clear()
                messages.extend(trimmed)
                continue
            emit_agent_event(on_event, 'error', message=str(exc))
            raise

        raw_msg = response.get('message') or response
        message = msg_to_dict(raw_msg)
        assistant_text = str(message.get('content') or '')
        if (
            not overflow_attempted
            and is_context_overflow(
                message_text=assistant_text,
                prompt_tokens=ctx.last_prompt_tokens,
                num_ctx=ctx.num_ctx,
                assistant_content=assistant_text,
            )
        ):
            overflow_attempted = True
            ui.print_info(ctx.defaults.ui.cli_overflow_retry_message)
            trimmed = trim_context(messages, force=True, on_event=on_event)
            messages.clear()
            messages.extend(trimmed)
            continue

        if message.get('content'):
            message['content'] = sanitize_assistant_text(message['content'], ctx)
        if message.get('thinking'):
            message['thinking'] = sanitize_assistant_text(message['thinking'], ctx)

        tool_calls = _normalize_tool_calls(message.get('tool_calls'))
        message['tool_calls'] = tool_calls if tool_calls else None
        messages.append(message)

        if not tool_calls:
            if think and not print_mode and not ctx.thinking_committed_this_turn:
                commit_thinking_display(
                    (message.get('thinking') or '').strip(),
                    think=think,
                    print_mode=print_mode,
                )
            visible = (message.get('content') or '').strip()
            finalize_assistant_answer(
                visible,
                empty_fallback,
                print_mode=print_mode,
                was_streamed=was_streamed,
                content_echoed_to_tty=content_echoed_to_tty,
            )
            emit_agent_event(on_event, 'turn_end')
            break

        if think and not print_mode:
            discard_thinking_display()

        rounds += 1
        if rounds > max_rounds:
            fallback = ctx.defaults.model.tool_round_limit_message
            messages.append({'role': 'assistant', 'content': fallback})
            if not print_mode:
                ui.print_agent(fallback)
            emit_agent_event(on_event, 'turn_end')
            break

        for tool_call in tool_calls:
            func_name = str(tool_call['function']['name'])
            args = parse_tool_arguments(tool_call['function']['arguments'])

            emit_agent_event(on_event, 'tool_start', name=func_name, args=args)
            if not print_mode:
                ui.print_tool(func_name, args)
            logger.debug('Tool call: %s(%s)', func_name, args)

            if func_name not in TOOL_MAP:
                result = f"System Error: Unknown tool '{func_name}'."
            else:
                validation_err = validate_tool_call(func_name, args)
                if validation_err:
                    result = ToolResult(False, validation_err)
                else:
                    try:
                        result = TOOL_MAP[func_name](**args)
                    except Exception as exc:
                        logger.warning('Tool %s raised: %s', func_name, exc)
                        result = f'Error executing {func_name}: {exc}'

            result_text = truncate_tool_result(str(result))
            logger.debug('Tool result: %.200s', result_text)
            messages.append({
                'role': 'tool',
                'content': result_text,
                'name': func_name,
            })
            emit_agent_event(on_event, 'tool_end', name=func_name, success=(
                result.success if isinstance(result, ToolResult) else True
            ))

    return messages
