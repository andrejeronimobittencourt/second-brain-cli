"""Interactive REPL: history, Ollama chat+tools, slash commands."""

from __future__ import annotations

import sys
from typing import Any

from brain.ui import render as ui
from brain.agent.compression import (
    load_agent_history,
    save_agent_history,
    trim_context,
)
from brain.agent.system import (
    effective_system_prompt,
    empty_answer_fallback,
    refresh_system_message,
    wrap_print_mode_user_message,
)
from brain.agent.tool_loop import run_tool_loop
from brain.core.context import get_context
from brain.core.events import EventHandler
from brain.repl.commands import (
    SLASH_EXIT_COMMANDS,
    build_slash_handlers,
    final_assistant_text,
    handle_prefixed_command,
    seed_repl_display,
)
from brain.repl.input import ReplExit, ReplStatus, create_repl_input, refresh_repl_toolbar
from brain.repl.display_port import NullDisplayPort, set_display_port
from brain.vault.tools import active_vision_model

__all__ = ['run_agent']


def run_agent(
    resume: bool = False,
    think: bool = False,
    vision_model: str = '',
    *,
    print_prompt: str = '',
    on_event: EventHandler | None = None,
) -> int:
    """
    Run the interactive REPL until the user exits, or one shot with ``print_prompt``.

    Requires ``bootstrap()`` to have installed ``ApplicationContext``.

    :param resume: If True, prepend messages loaded from ``history_path``.
    :param think: If True, pass ``think=True`` to Ollama when the client supports it.
    :param vision_model: Overrides the vision model for ``read_image`` for this session.
    :param print_prompt: When set, run one turn and print the answer to stdout (exit code).
    :param on_event: Optional callback for agent lifecycle events.
    :returns: Exit code (0 success, 1 on error in print mode).
    """
    ctx = get_context()
    ctx.vision_model_override = vision_model.strip()
    print_mode = bool(print_prompt.strip())
    ctx.print_mode = print_mode
    if print_mode:
        set_display_port(NullDisplayPort())

    vault = ctx.vault_path.resolve()
    repl = create_repl_input(vault) if not print_mode else None
    if repl is not None:
        ctx.repl_line_editor = repl.uses_line_editor

    def _repl_status() -> ReplStatus:
        return ReplStatus(
            vault_path=ctx.vault_path,
            chat_model=ctx.user.ollama_model,
            vision_model=active_vision_model(),
            session_path=ctx.history_path,
        )

    system_msg: dict[str, Any] = {'role': 'system', 'content': effective_system_prompt()}
    messages: list[dict[str, Any]] = [system_msg]

    if resume and not print_mode:
        prior = load_agent_history()
        if prior:
            messages.extend(prior)

    empty_fallback = empty_answer_fallback()

    slash_handlers = build_slash_handlers(
        system_msg=system_msg,
        messages=messages,
        on_event=on_event,
    )

    def _run_user_turn(user_input: str) -> None:
        # Input lock (``repl_awaiting_final_answer``) stays until this function
        # returns — covering tool rounds, compression, and answer streaming.
        refresh_system_message(messages)
        prompt = (
            wrap_print_mode_user_message(user_input)
            if print_mode
            else user_input
        )
        messages.append({'role': 'user', 'content': prompt})
        trimmed = trim_context(messages, on_event=on_event)
        messages.clear()
        messages.extend(trimmed)
        run_tool_loop(
            messages,
            think,
            empty_fallback,
            on_event=on_event,
            print_mode=print_mode,
            interactive=not print_mode,
        )
        if not print_mode:
            save_agent_history(messages)
            refresh_repl_toolbar()

    if print_mode:
        try:
            _run_user_turn(print_prompt.strip())
            answer = final_assistant_text(messages)
            if answer:
                print(answer)
            elif not ctx.last_answer_streamed_to_tty:
                print(empty_fallback)
            return 0
        except Exception as exc:
            print(f'Error: {exc}', file=sys.stderr)
            return 1

    assert repl is not None

    def _exit_on_interrupt() -> None:
        save_agent_history(messages)

    def _process_repl_line(user_input: str) -> None:
        if user_input.lower() in SLASH_EXIT_COMMANDS:
            save_agent_history(messages)
            raise ReplExit

        handler = slash_handlers.get(user_input)
        if handler is not None:
            handler()
            return

        if handle_prefixed_command(user_input):
            return

        try:
            _run_user_turn(user_input)
        except KeyboardInterrupt:
            if ui.answer_stream_is_active():
                ui.cancel_answer_stream()
            raise

    repl.run(
        status_fn=_repl_status,
        process_line=lambda line: _process_repl_line(line.strip()),
        on_exit=_exit_on_interrupt,
        seed_display=lambda sink: seed_repl_display(
            sink,
            repl=repl,
            resume=resume,
            messages=messages,
        ),
        farewell_message=ctx.defaults.repl.farewell_message,
    )

    return 0
