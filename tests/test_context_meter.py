"""Tests for brain.repl.context_meter."""

from __future__ import annotations

from brain.repl.context_meter import non_system_message_count, resolve_context_meter


class TestResolveContextMeter:
    def test_prefers_tokens_when_num_ctx_known(self):
        used, total = resolve_context_meter(
            last_prompt_tokens=2048,
            num_ctx=8192,
            message_count=10,
            message_cap=40,
        )
        assert (used, total) == (2048, 8192)

    def test_falls_back_to_message_cap(self):
        used, total = resolve_context_meter(
            last_prompt_tokens=2048,
            num_ctx=0,
            message_count=10,
            message_cap=40,
        )
        assert (used, total) == (10, 40)


class TestNonSystemMessageCount:
    def test_excludes_system(self):
        messages = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]
        assert non_system_message_count(messages) == 2
