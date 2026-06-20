"""Tests for brain.agent context management."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

from brain.agent.compression import (
    summarize_dropped_turns,
    trim_context,
    vault_activity_from_messages,
)
from brain.agent.ollama_chat import agent_chat, truncate_tool_result
from brain.agent.system import effective_system_prompt, empty_answer_fallback
from brain.agent.tool_loop import run_tool_loop
from brain.repl.commands import format_history, format_stats
from brain.core.context import ApplicationContext, set_context
from brain.core.defaults import APP_DEFAULTS, default_system_prompt
from brain.core.user_config import UserConfig
from brain.repl.display import ReplDisplaySink
from brain.repl.display_port import NullDisplayPort, ReplDisplayPort
from tests.conftest import MockOllamaClient


class TestEffectiveSystemPrompt:
    def test_vault_instructions_appended_to_default(self, ctx):
        u = replace(ctx.user, vault_instructions='Use plugin X for queries.', system_prompt='')
        set_context(replace(ctx, user=u))
        text = effective_system_prompt()
        base = default_system_prompt()
        assert text.startswith(base)
        assert 'Use plugin X for queries.' in text
        assert 'Current local time:' in text
        assert 'Vault root:' in text

    def test_vault_instructions_after_custom_system_prompt(self, ctx):
        u = replace(
            ctx.user,
            system_prompt='CUSTOM ONLY',
            vault_instructions='Also respect folder Y.',
        )
        set_context(replace(ctx, user=u))
        text = effective_system_prompt()
        assert text.startswith('CUSTOM ONLY')
        assert 'Also respect folder Y.' in text
        assert text.rstrip().endswith(ctx.vault_path.resolve().as_posix().replace('\\', '/'))

    def test_print_mode_adds_one_shot_instructions(self, ctx):
        set_context(replace(ctx, print_mode=True))
        text = effective_system_prompt()
        assert text.startswith('## One-shot CLI')
        assert 'follow-up questions' in text
        assert 'autonomous assistant' in text

    def test_empty_answer_fallback_is_plain(self, ctx):
        set_context(replace(ctx, print_mode=False))
        repl = empty_answer_fallback()
        assert '**' not in repl
        assert '/clear' in repl
        set_context(replace(ctx, print_mode=True))
        printed = empty_answer_fallback()
        assert printed == ctx.defaults.model.empty_answer_print_message
        assert '/clear' not in printed


class TestFormatHistory:
    def test_counts_roles(self, ctx):
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'a'},
            {'role': 'assistant', 'content': 'b'},
            {'role': 'tool', 'content': 'c', 'name': 'read_note'},
        ]
        text = format_history(msgs)
        assert '3 messages' in text
        assert '1 user' in text
        assert '1 assistant' in text
        assert '1 tool' in text


class TestTrimContext:
    def test_system_messages_preserved(self, ctx):
        system = {'role': 'system', 'content': 'sys'}
        msgs = [system] + [{'role': 'user', 'content': str(i)} for i in range(50)]
        result = trim_context(msgs)
        assert result[0] == system
        non_sys = [m for m in result if m['role'] != 'system']
        assert len(non_sys) <= ctx.defaults.limits.max_context_messages

    def test_rolling_summary_when_enabled(self, ctx_compress):
        system = {'role': 'system', 'content': 'sys'}
        msgs = [system] + [{'role': 'user', 'content': f'm{i}'} for i in range(10)]
        result = trim_context(msgs)
        assert result[0] == system
        non_sys = [m for m in result if m['role'] != 'system']
        cap = ctx_compress.defaults.limits.max_context_messages
        assert len(non_sys) <= cap
        assert any(
            m.get('role') == 'assistant'
            and '[Prior context' in (m.get('content') or '')
            for m in non_sys
        )
        calls = ctx_compress.ollama_client.chat.call_args_list
        summary_calls = [c for c in calls if c.kwargs.get('stream') is False]
        assert summary_calls, 'expected a non-streaming summarizer chat call'
        sc = summary_calls[-1]
        assert 'tools' not in sc.kwargs
        assert sc.kwargs.get('stream') is False


class TestVaultActivity:
    def test_extracts_read_and_modified(self):
        msgs = [
            {
                'role': 'assistant',
                'tool_calls': [{
                    'function': {
                        'name': 'read_note',
                        'arguments': {'note_title': 'Inbox/Idea'},
                    },
                }],
            },
            {
                'role': 'assistant',
                'tool_calls': [{
                    'function': {
                        'name': 'create_note',
                        'arguments': {
                            'note_title': 'New',
                            'folder': 'Projects',
                            'content': 'body',
                        },
                    },
                }],
            },
        ]
        reads, modified = vault_activity_from_messages(msgs)
        assert 'Inbox/Idea' in reads
        assert 'Projects/New' in modified


class TestForceTrim:
    def test_force_compact_reduces_messages(self, ctx_compress):
        system = {'role': 'system', 'content': 'sys'}
        msgs = [system] + [{'role': 'user', 'content': f'm{i}'} for i in range(10)]
        before = len(msgs)
        result = trim_context(msgs, force=True)
        assert len(result) < before
        assert any(
            '[Prior context' in (m.get('content') or '')
            for m in result
            if m.get('role') == 'assistant'
        )


class TestToolResultTruncation:
    def test_truncates_large_result(self, ctx):
        huge = 'x' * 50_000
        out = truncate_tool_result(huge)
        assert 'truncated for model context' in out
        assert len(out) < len(huge)


class TestRunToolLoop:
    def _make_ctx(self, vault, client: Any, *, compress: bool = False) -> ApplicationContext:
        user = UserConfig(vault_path=str(vault))
        if compress:
            lim = replace(APP_DEFAULTS.limits, max_context_messages=6)
            cc = replace(APP_DEFAULTS.context_compression, enabled=True)
            d = replace(APP_DEFAULTS, limits=lim, context_compression=cc)
        else:
            d = replace(
                APP_DEFAULTS,
                context_compression=replace(APP_DEFAULTS.context_compression, enabled=False),
            )
        leak = re.compile(d.model.channel_leak_regex, re.IGNORECASE)
        context = ApplicationContext(
            user=user,
            defaults=d,
            vault_path=vault,
            history_path=vault / '.agent_history.json',
            ollama_client=client,
            latex_pairs=d.latex_symbol_pairs,
            channel_leak=leak,
            num_ctx=4096,
            display=NullDisplayPort(),
        )
        set_context(context)
        return context

    def test_simple_reply(self, vault):
        client = MockOllamaClient()
        client.queue_response({
            'message': {'role': 'assistant', 'content': 'Hello from vault.'},
            'prompt_eval_count': 100,
        })
        self._make_ctx(vault, client)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
        ]
        result = run_tool_loop(msgs, False, 'empty', print_mode=True)
        assert any(
            m.get('role') == 'assistant' and 'Hello from vault' in (m.get('content') or '')
            for m in result
        )

    def test_mutates_messages_in_place(self, vault):
        client = MockOllamaClient()
        client.queue_response({
            'message': {'role': 'assistant', 'content': 'Hello from vault.'},
            'prompt_eval_count': 100,
        })
        self._make_ctx(vault, client)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
        ]
        original = msgs
        result = run_tool_loop(msgs, False, 'empty', print_mode=True)
        assert result is original
        assert len(msgs) == 3
        assert msgs[-1]['role'] == 'assistant'

    def test_clear_extend_same_list_wipes_history(self):
        """Regression guard: never sync with clear+extend on the same list ref."""
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]
        updated = msgs
        msgs.clear()
        msgs.extend(updated)
        assert msgs == []

    def test_overflow_triggers_compact_retry(self, vault):
        client = MockOllamaClient()
        client.queue_error(RuntimeError('prompt too long'))
        client.queue_response({
            'message': {'role': 'assistant', 'content': 'Summary of prior turns.'},
        })
        client.queue_response({
            'message': {'role': 'assistant', 'content': 'Recovered.'},
            'prompt_eval_count': 50,
        })
        ctx = self._make_ctx(vault, client, compress=True)
        ctx.last_prompt_tokens = 4000
        set_context(ctx)
        msgs = [
            {'role': 'system', 'content': 'sys'},
        ] + [{'role': 'user', 'content': f'm{i}'} for i in range(8)]
        result = run_tool_loop(msgs, False, 'empty', print_mode=True)
        assert any('Recovered' in (m.get('content') or '') for m in result)
        assert client.chat.call_count >= 2

    def test_streams_answer_in_print_mode(self, vault, monkeypatch):
        import brain.ui.render as ui_mod

        captured: list[str] = []

        def _capture(text: str) -> None:
            captured.append(text)

        monkeypatch.setattr(ui_mod, 'write_answer_delta', _capture)

        chunks = [
            {'message': {'role': 'assistant', 'content': 'Hel'}, 'prompt_eval_count': 5},
            {'message': {'role': 'assistant', 'content': 'lo!'}, 'prompt_eval_count': 5},
        ]

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            if kwargs.get('stream') is False:
                return chunks[-1]
            for item in chunks:
                yield item

        client.chat = MagicMock(side_effect=_stream)
        self._make_ctx(vault, client)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
        ]
        out, streamed, echoed = agent_chat(
            msgs,
            False,
            stream_answers=True,
            print_mode=True,
        )
        assert streamed
        assert echoed
        assert captured == ['Hel', 'lo!']
        assert 'Hello!' in (out.get('message', {}).get('content') or '')

    def test_streams_answer_with_rich_console(self, vault, monkeypatch):
        from io import StringIO

        from rich.console import Console

        import brain.ui.render as ui_mod

        captured: list[str] = []

        def _capture(text: str) -> None:
            captured.append(text)

        monkeypatch.setattr(ui_mod, 'write_answer_delta', _capture)

        chunks = [
            {'message': {'role': 'assistant', 'content': 'Hel'}, 'prompt_eval_count': 5},
            {'message': {'role': 'assistant', 'content': 'lo!'}, 'prompt_eval_count': 5},
        ]

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            if kwargs.get('stream') is False:
                return chunks[-1]
            for item in chunks:
                yield item

        client.chat = MagicMock(side_effect=_stream)
        ctx = self._make_ctx(vault, client)
        ctx.console = Console(file=StringIO(), width=120)
        from brain.repl.display_port import RichConsoleDisplayPort

        ctx.display = RichConsoleDisplayPort(ctx.console)
        monkeypatch.setattr(ui_mod, '_panel_outer_width', lambda: 80)
        set_context(ctx)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
        ]
        out, streamed, echoed = agent_chat(
            msgs,
            False,
            stream_answers=True,
            print_mode=False,
        )
        assert streamed
        assert echoed
        assert captured == ['Hel', 'lo!']
        assert 'Hello!' in (out.get('message', {}).get('content') or '')
        ui_mod.cancel_answer_stream()

    def test_streams_filtered_thinking_before_answer(self, vault, monkeypatch):
        from io import StringIO
        from unittest.mock import MagicMock

        from rich.console import Console

        import brain.ui.render as ui_mod
        from brain.core.context import get_context, set_context
        from brain.repl.display import ReplDisplaySink

        thinking_updates: list[str] = []
        answer_chunks: list[str] = []

        def _capture_answer(text: str) -> None:
            answer_chunks.append(text)

        original_end = ReplDisplaySink.end_thinking_stream

        def _track_thinking_commit(
            self,
            final_content: str | None = None,
        ) -> None:
            if final_content and final_content.strip():
                thinking_updates.append(final_content.strip())
            original_end(self, final_content=final_content)

        monkeypatch.setattr(ReplDisplaySink, 'end_thinking_stream', _track_thinking_commit)
        monkeypatch.setattr(ui_mod, 'write_answer_delta', _capture_answer)
        ui_mod.cancel_answer_stream()
        ui_mod.cancel_thinking_stream()

        scaffold = (
            "Here's a thinking process:\n\n"
            '1.  **Analyze User Input:** math\n'
            '2.  **Calculate:** 2+2=4\n\n'
            'The sum is four because addition combines both values.'
        )
        chunks = [
            {'message': {'role': 'assistant', 'thinking': scaffold[:20], 'content': ''}},
            {'message': {'role': 'assistant', 'thinking': scaffold[20:], 'content': ''}},
            {'message': {'role': 'assistant', 'thinking': '', 'content': '4'}},
        ]

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            for item in chunks:
                yield item

        client.chat = MagicMock(side_effect=_stream)
        ctx = self._make_ctx(vault, client)
        ctx.console = Console(file=StringIO(), width=120)
        sink = ReplDisplaySink(invalidate=MagicMock())
        ctx.repl_display = sink
        ctx.display = ReplDisplayPort(sink=sink)
        set_context(ctx)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
        ]
        out, streamed, echoed = agent_chat(
            msgs,
            True,
            stream_answers=True,
            print_mode=False,
        )
        assert streamed
        assert echoed
        assert len(thinking_updates) == 1
        assert 'Analyze User Input' not in thinking_updates[0]
        assert 'addition combines both values' in thinking_updates[0]
        assert answer_chunks
        assert '4' in (out.get('message', {}).get('content') or '')
        ui_mod.cancel_answer_stream()

    def test_tool_round_thinking_not_committed(self, vault, monkeypatch):
        from io import StringIO
        from unittest.mock import MagicMock

        from rich.console import Console

        import brain.ui.render as ui_mod
        from brain.vault.tools import TOOL_MAP

        thinking_commits: list[str] = []
        original_end = ReplDisplaySink.end_thinking_stream

        def _track_thinking_commit(
            self,
            final_content: str | None = None,
        ) -> None:
            if final_content and final_content.strip():
                thinking_commits.append(final_content.strip())
            original_end(self, final_content=final_content)

        monkeypatch.setitem(TOOL_MAP, 'search_notes', lambda **kwargs: 'three notes')
        monkeypatch.setattr(ReplDisplaySink, 'end_thinking_stream', _track_thinking_commit)

        planning_thinking = (
            "Here's a thinking process:\n\n"
            '1.  **Analyze User Input:** list notes\n'
            '2.  **Select Tool:** search_notes\n\n'
            'I should search the vault for matching notes.'
        )
        final_thinking = (
            '1.  **Summarize Results:** three notes found\n\n'
            'The search returned three relevant notes about colors.'
        )
        rounds = [
            {
                'message': {
                    'role': 'assistant',
                    'thinking': planning_thinking,
                    'content': '',
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'search_notes',
                                'arguments': {'query': 'colors'},
                            },
                        },
                    ],
                },
            },
            {
                'message': {
                    'role': 'assistant',
                    'thinking': final_thinking,
                    'content': 'Found three notes.',
                },
            },
        ]
        call_idx = {'n': 0}

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            idx = call_idx['n']
            call_idx['n'] += 1
            item = rounds[idx]
            yield item

        client.chat = MagicMock(side_effect=_stream)
        ctx = self._make_ctx(vault, client)
        ctx.console = Console(file=StringIO(), width=120)
        sink = ReplDisplaySink(invalidate=MagicMock())
        ctx.repl_display = sink
        ctx.display = ReplDisplayPort(sink=sink)
        set_context(ctx)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'find color notes'},
        ]
        run_tool_loop(msgs, think=True, empty_fallback='(empty)', interactive=True)
        assert len(thinking_commits) == 1
        assert 'search_notes' not in thinking_commits[0].lower()
        assert 'Analyze User Input' not in thinking_commits[0]
        assert 'three relevant notes' in thinking_commits[0]
        transcript = str(sink.get_transcript_formatted_text())
        assert 'Analyze User Input' not in transcript
        ui_mod.cancel_answer_stream()


    def test_tool_round_does_not_live_stream_thinking(self, vault, monkeypatch):
        from io import StringIO
        from unittest.mock import MagicMock

        from rich.console import Console

        from brain.repl.display import ReplDisplaySink
        from brain.repl.display_port import ReplDisplayPort

        planning_thinking = (
            "Here's a thinking process:\n\n"
            '1.  **Analyze User Input:** list notes\n'
            '2.  **Select Tool:** search_notes\n\n'
            'I should search the vault.'
        )
        chunks = [
            {'message': {'role': 'assistant', 'thinking': planning_thinking[:30], 'content': ''}},
            {'message': {'role': 'assistant', 'thinking': planning_thinking[30:], 'content': ''}},
            {
                'message': {
                    'role': 'assistant',
                    'thinking': '',
                    'content': '',
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'search_notes',
                                'arguments': {'query': 'notes'},
                            },
                        },
                    ],
                },
            },
        ]

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            for item in chunks:
                yield item

        client.chat = MagicMock(side_effect=_stream)
        ctx = self._make_ctx(vault, client)
        ctx.console = Console(file=StringIO(), width=120)
        sink = ReplDisplaySink(invalidate=MagicMock())
        ctx.repl_display = sink
        ctx.display = ReplDisplayPort(sink=sink)
        set_context(ctx)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'find notes'},
        ]
        agent_chat(
            msgs,
            True,
            stream_answers=True,
            print_mode=False,
            live_thinking=False,
        )

        transcript = str(sink.get_transcript_formatted_text())
        assert 'Analyze User Input' not in transcript
        assert 'search the vault' not in transcript
        assert not sink.has_live_thinking()

    def test_final_reasoning_streams_after_tools(self, vault, monkeypatch):
        from io import StringIO
        from unittest.mock import MagicMock

        from rich.console import Console

        from brain.repl.display import ReplDisplaySink
        from brain.repl.display_port import ReplDisplayPort

        final_thinking = (
            '1.  **Summarize Results:** three notes found\n\n'
            'The search returned three relevant notes about colors.'
        )
        chunks = [
            {'message': {'role': 'assistant', 'thinking': final_thinking[:20], 'content': ''}},
            {'message': {'role': 'assistant', 'thinking': final_thinking[20:50], 'content': ''}},
            {'message': {'role': 'assistant', 'thinking': final_thinking[50:], 'content': ''}},
            {'message': {'role': 'assistant', 'thinking': '', 'content': 'Found three.'}},
        ]
        updates: list[str] = []

        def _track_update(content: str, *, raw: str = '') -> None:
            updates.append(content)

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            for item in chunks:
                yield item

        client.chat = MagicMock(side_effect=_stream)
        ctx = self._make_ctx(vault, client)
        ctx.console = Console(file=StringIO(), width=120)
        sink = ReplDisplaySink(invalidate=MagicMock())
        ctx.repl_display = sink
        ctx.display = ReplDisplayPort(sink=sink)
        set_context(ctx)
        original_update = ReplDisplaySink.update_thinking_stream

        def _track_update(self, content: str, *, raw: str = '') -> None:
            updates.append(content)
            original_update(self, content, raw=raw)

        monkeypatch.setattr(ReplDisplaySink, 'update_thinking_stream', _track_update)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'find notes'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'search_notes', 'arguments': {}}}]},
            {'role': 'tool', 'content': 'three notes', 'name': 'search_notes'},
        ]
        agent_chat(
            msgs,
            True,
            stream_answers=True,
            print_mode=False,
        )
        assert len(updates) >= 2
        assert updates[-1] != updates[0]
        assert 'three relevant notes' in updates[-1]

    def test_multi_tool_hop_keeps_one_reasoning_panel(self, vault, monkeypatch):
        from io import StringIO
        from unittest.mock import MagicMock

        from rich.console import Console

        from brain.repl.display import ReplDisplaySink
        from brain.repl.display_port import ReplDisplayPort

        planning = (
            "Here's a thinking process:\n\n"
            '1.  **Analyze User Input:** colors\n'
            '2.  **Select Tool:** search_notes\n\n'
            'Search the vault for color notes.'
        )
        second_hop = (
            '1.  **Read Results:** notes found\n\n'
            'Need another tool to read the first note.'
        )
        final_thinking = (
            '1.  **Summarize:** done\n\n'
            'The note mentions blue and red.'
        )
        rounds = [
            {
                'message': {
                    'role': 'assistant',
                    'thinking': planning,
                    'content': '',
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'search_notes',
                                'arguments': {'query': 'colors'},
                            },
                        },
                    ],
                },
            },
            {
                'message': {
                    'role': 'assistant',
                    'thinking': second_hop,
                    'content': '',
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'read_note',
                                'arguments': {'path': 'colors.md'},
                            },
                        },
                    ],
                },
            },
            {
                'message': {
                    'role': 'assistant',
                    'thinking': final_thinking,
                    'content': 'Blue and red.',
                },
            },
        ]
        call_idx = {'n': 0}
        thinking_commits: list[str] = []
        original_end = ReplDisplaySink.end_thinking_stream

        def _track_commit(self, final_content: str | None = None) -> None:
            if final_content and final_content.strip():
                thinking_commits.append(final_content.strip())
            original_end(self, final_content=final_content)

        monkeypatch.setattr(ReplDisplaySink, 'end_thinking_stream', _track_commit)
        from brain.vault.tools import TOOL_MAP

        monkeypatch.setitem(TOOL_MAP, 'search_notes', lambda **kwargs: 'two notes')
        monkeypatch.setitem(TOOL_MAP, 'read_note', lambda **kwargs: 'Blue and red text.')

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            idx = call_idx['n']
            call_idx['n'] += 1
            yield rounds[idx]

        client.chat = MagicMock(side_effect=_stream)
        ctx = self._make_ctx(vault, client)
        ctx.console = Console(file=StringIO(), width=120)
        sink = ReplDisplaySink(invalidate=MagicMock())
        ctx.repl_display = sink
        ctx.display = ReplDisplayPort(sink=sink)
        set_context(ctx)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'tell me about colors'},
        ]
        run_tool_loop(msgs, think=True, empty_fallback='(empty)', interactive=True)

        assert len(thinking_commits) == 1
        assert 'Analyze User Input' not in thinking_commits[0]
        assert 'Read Results' not in thinking_commits[0]
        assert 'blue and red' in thinking_commits[0].lower()
        transcript = str(sink.get_transcript_formatted_text())
        assert 'Analyze User Input' not in transcript
        assert 'Read Results' not in transcript
        assert transcript.lower().count('reasoning') <= 1 or 'blue and red' in transcript.lower()


    def test_intermediate_tool_hop_never_shows_reasoning(self, vault, monkeypatch):
        from io import StringIO
        from unittest.mock import MagicMock

        from rich.console import Console

        from brain.repl.display import ReplDisplaySink
        from brain.repl.display_port import ReplDisplayPort

        second_hop_thinking = (
            '1.  **Read Results:** notes found\n\n'
            'I will read the first note next.'
        )
        chunks = [
            {'message': {'role': 'assistant', 'thinking': second_hop_thinking[:25], 'content': ''}},
            {'message': {'role': 'assistant', 'thinking': second_hop_thinking[25:], 'content': ''}},
            {
                'message': {
                    'role': 'assistant',
                    'thinking': '',
                    'content': '',
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'read_note',
                                'arguments': {'path': 'colors.md'},
                            },
                        },
                    ],
                },
            },
        ]
        updates: list[str] = []
        original_update = ReplDisplaySink.update_thinking_stream

        def _track_update(self, content: str, *, raw: str = '') -> None:
            updates.append(content)
            original_update(self, content, raw=raw)

        monkeypatch.setattr(ReplDisplaySink, 'update_thinking_stream', _track_update)

        client = MockOllamaClient()

        def _stream(**kwargs: object):
            for item in chunks:
                yield item

        client.chat = MagicMock(side_effect=_stream)
        ctx = self._make_ctx(vault, client)
        ctx.console = Console(file=StringIO(), width=120)
        sink = ReplDisplaySink(invalidate=MagicMock())
        ctx.repl_display = sink
        ctx.display = ReplDisplayPort(sink=sink)
        set_context(ctx)
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'colors'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'search_notes', 'arguments': {}}}]},
            {'role': 'tool', 'content': 'two notes', 'name': 'search_notes'},
        ]
        agent_chat(msgs, True, stream_answers=True, print_mode=False)

        assert updates == []
        assert not sink.has_live_thinking()
        assert not sink.has_committed_thinking()
        transcript = str(sink.get_transcript_formatted_text())
        assert 'Read Results' not in transcript


class TestFormatStats:
    def test_includes_model_and_session(self, ctx):
        msgs = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'hi'}]
        text = format_stats(msgs)
        assert 'model:' in text
        assert 'vision:' in text
        assert 'vault:' in text
        assert 'ollama:' in text
        assert 'session:' in text

    def test_message_cap_summary_is_single_line(self, ctx):
        ctx.num_ctx = 0
        msgs = [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': 'a'},
            {'role': 'assistant', 'content': 'b'},
        ]
        text = format_stats(msgs)
        assert 'context fill' in text
        assert 'messages in context' not in text
        assert not text.endswith('..')


class TestSummarizerHygiene:
    def test_wraps_transcript_in_tags(self, ctx_compress):
        dropped = [
            {'role': 'user', 'content': 'find notes about python'},
            {'role': 'assistant', 'content': 'Searching…'},
        ]
        summary = summarize_dropped_turns(dropped)
        assert summary is not None
        call_args = ctx_compress.ollama_client.chat.call_args
        user_msg = call_args.kwargs['messages'][1]['content']
        assert '<conversation>' in user_msg
        assert 'python' in user_msg
        assert 'ONLY the summary' in user_msg
