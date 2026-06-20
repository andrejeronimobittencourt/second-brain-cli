"""Tests for brain.core.context_messages."""

from __future__ import annotations

from brain.core.context_messages import (
    split_messages_pair_aware,
    truncate_messages_pair_aware,
)


class TestTruncateMessagesPairAware:
    def test_under_cap_unchanged(self):
        msgs = [{'role': 'user', 'content': 'hi'}]
        assert truncate_messages_pair_aware(msgs, 10) == msgs

    def test_trims_to_cap(self):
        msgs = [{'role': 'user', 'content': str(i)} for i in range(20)]
        result = truncate_messages_pair_aware(msgs, 5)
        assert len(result) == 5

    def test_drops_orphaned_tool_messages(self):
        msgs = [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'x'}}]},
            {'role': 'tool', 'content': 'result', 'name': 'x'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'world'},
        ]
        result = truncate_messages_pair_aware(msgs, 2)
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
        result = truncate_messages_pair_aware(msgs, 3)
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'a1'

        result = truncate_messages_pair_aware(msgs, 4)
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'a1'
        assert len(result) == 3


class TestSplitMessagesPairAware:
    def test_split_matches_truncate(self):
        msgs = [{'role': 'user', 'content': str(i)} for i in range(20)]
        dropped, kept = split_messages_pair_aware(msgs, 5)
        assert dropped + kept == msgs
        assert kept == truncate_messages_pair_aware(msgs, 5)
