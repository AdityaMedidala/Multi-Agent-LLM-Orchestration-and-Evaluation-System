"""Tests for SSE streaming with Redis event buffering."""
from __future__ import annotations

import json

import pytest

from app.streaming import _buffer_key, _channel


class TestStreamingHelpers:
    def test_channel_format(self):
        ch = _channel("job-123")
        assert ch == "job:job-123:stream"

    def test_buffer_key_format(self):
        bk = _buffer_key("job-123")
        assert bk == "job:job-123:events"

    def test_channel_unique_per_job(self):
        assert _channel("job-1") != _channel("job-2")

    def test_buffer_key_unique_per_job(self):
        assert _buffer_key("job-1") != _buffer_key("job-2")
