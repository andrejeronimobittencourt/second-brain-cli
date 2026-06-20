"""Tests for brain.core.retry."""

from __future__ import annotations

import pytest

from brain.core.defaults import RetryPolicy
from brain.core.retry import is_retryable_connection_error, retry_call


class TestRetryableErrors:
    def test_connection_error(self):
        assert is_retryable_connection_error(ConnectionError('refused'))

    def test_non_retryable(self):
        assert not is_retryable_connection_error(ValueError('bad arg'))


class TestRetryCall:
    def test_succeeds_first_try(self):
        calls = {'n': 0}

        def fn() -> str:
            calls['n'] += 1
            return 'ok'

        assert retry_call(fn, RetryPolicy(enabled=True, max_attempts=3)) == 'ok'
        assert calls['n'] == 1

    def test_retries_then_succeeds(self):
        calls = {'n': 0}

        def fn() -> str:
            calls['n'] += 1
            if calls['n'] < 2:
                raise ConnectionError('refused')
            return 'ok'

        policy = RetryPolicy(enabled=True, max_attempts=3, base_delay_ms=1, max_delay_ms=2)
        assert retry_call(fn, policy) == 'ok'
        assert calls['n'] == 2

    def test_raises_after_max_attempts(self):
        def fn() -> None:
            raise ConnectionError('refused')

        policy = RetryPolicy(enabled=True, max_attempts=2, base_delay_ms=1, max_delay_ms=2)
        with pytest.raises(ConnectionError):
            retry_call(fn, policy)
