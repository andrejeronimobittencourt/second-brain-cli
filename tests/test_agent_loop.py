"""Tests for brain.agent_loop context management."""

from __future__ import annotations

from dataclasses import replace

from brain.agent_loop import (
    _effective_system_prompt,
    _split_truncate_pair_aware,
    _trim_context,
    _truncate_pair_aware,
)
from brain.context import set_context
from brain.defaults import default_system_prompt


class TestEffectiveSystemPrompt:
    def test_vault_instructions_appended_to_default(self, ctx):
        u = replace(ctx.user, vault_instructions='Use plugin X for queries.', system_prompt='')
        set_context(replace(ctx, user=u))
        text = _effective_system_prompt()
        base = default_system_prompt()
        assert text.startswith(base)
        assert text.endswith('Use plugin X for queries.')

    def test_vault_instructions_after_custom_system_prompt(self, ctx):
        u = replace(
            ctx.user,
            system_prompt='CUSTOM ONLY',
            vault_instructions='Also respect folder Y.',
        )
        set_context(replace(ctx, user=u))
        assert _effective_system_prompt() == 'CUSTOM ONLY\n\nAlso respect folder Y.'


class TestTruncatePairAware:
    def test_under_cap_unchanged(self):
        msgs = [{'role': 'user', 'content': 'hi'}]
        assert _truncate_pair_aware(msgs, 10) == msgs

    def test_trims_to_cap(self):
        msgs = [{'role': 'user', 'content': str(i)} for i in range(20)]
        result = _truncate_pair_aware(msgs, 5)
        assert len(result) == 5

    def test_drops_orphaned_tool_messages(self):
        msgs = [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'x'}}]},
            {'role': 'tool', 'content': 'result', 'name': 'x'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'world'},
        ]
        # Cap=2 keeps the last 2: user + assistant. No orphaned tool.
        result = _truncate_pair_aware(msgs, 2)
        assert len(result) == 2
        assert result[0]['role'] == 'user'

    def test_drops_leading_tool_after_slice(self):
        msgs = [
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'x'}}]},
            {'role': 'tool', 'content': 'r1', 'name': 'x'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2'},
        ]
        # Cap=3 slices to [assistant a1, user q2, assistant a2] — fine, no tool.
        result = _truncate_pair_aware(msgs, 3)
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'a1'

        # Cap=4 slices to [tool r1, assistant a1, user q2, assistant a2]
        # Should drop the orphaned tool message.
        result = _truncate_pair_aware(msgs, 4)
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'a1'
        assert len(result) == 3


class TestSplitTruncatePairAware:
    def test_split_matches_truncate(self):
        msgs = [{'role': 'user', 'content': str(i)} for i in range(20)]
        dropped, kept = _split_truncate_pair_aware(msgs, 5)
        assert dropped + kept == msgs
        assert kept == _truncate_pair_aware(msgs, 5)


class TestTrimContext:
    def test_system_messages_preserved(self, ctx):
        system = {'role': 'system', 'content': 'sys'}
        msgs = [system] + [{'role': 'user', 'content': str(i)} for i in range(50)]
        result = _trim_context(msgs)
        assert result[0] == system
        non_sys = [m for m in result if m['role'] != 'system']
        assert len(non_sys) <= ctx.defaults.limits.max_context_messages

    def test_rolling_summary_when_enabled(self, ctx_compress):
        system = {'role': 'system', 'content': 'sys'}
        msgs = [system] + [{'role': 'user', 'content': f'm{i}'} for i in range(10)]
        result = _trim_context(msgs)
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
