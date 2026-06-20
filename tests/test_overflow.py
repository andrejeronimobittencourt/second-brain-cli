"""Tests for brain.core.overflow."""

from __future__ import annotations

from brain.core.overflow import (
    is_context_overflow,
    is_context_overflow_message,
    is_silent_overflow,
)


class TestOverflowMessage:
    def test_ollama_prompt_too_long(self):
        assert is_context_overflow_message('prompt too long; exceeded max context length')

    def test_llama_cpp_context(self):
        assert is_context_overflow_message('exceeds the available context size')

    def test_unrelated_error(self):
        assert not is_context_overflow_message('connection refused')


class TestOverflowDetection:
    def test_exception_text(self):
        assert is_context_overflow(error=RuntimeError('prompt too long'))

    def test_token_fill(self):
        assert is_context_overflow(prompt_tokens=4096, num_ctx=4096)

    def test_silent_overflow(self):
        assert is_silent_overflow(3900, 4096, '')
        assert not is_silent_overflow(3800, 4096, '')
        assert not is_silent_overflow(3900, 4096, 'hello')
