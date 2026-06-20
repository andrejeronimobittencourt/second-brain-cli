"""Tests for brain.core.streaming."""

from __future__ import annotations

from brain.core.streaming import merge_stream_content


def test_merge_stream_content_delta_chunks():
    text = merge_stream_content('Hel', 'lo')
    assert text == 'Hello'


def test_merge_stream_content_cumulative_chunks():
    text = merge_stream_content('Hel', 'Hello')
    assert text == 'Hello'


def test_merge_stream_content_does_not_repeat():
    text = merge_stream_content('Hello', 'Hello')
    assert text == 'Hello'
