"""Tests for tool implementations — code_executor safety, registry dispatch, tool contracts."""
from __future__ import annotations

import asyncio

import pytest

from app.tools.code_executor import _ast_is_safe, execute_code
from app.tools.registry import ToolResult, call_tool


# ── AST safety checker ────────────────────────────────────────────────────────

class TestAstSafety:
    def test_safe_code(self):
        safe, reason = _ast_is_safe("print(2 + 2)")
        assert safe is True
        assert reason == "ok"

    def test_import_os_blocked(self):
        safe, reason = _ast_is_safe("import os")
        assert safe is False
        assert "blocked_import" in reason

    def test_import_subprocess_blocked(self):
        safe, reason = _ast_is_safe("import subprocess")
        assert safe is False
        assert "blocked_import" in reason

    def test_from_os_import_blocked(self):
        safe, reason = _ast_is_safe("from os import path")
        assert safe is False
        assert "blocked_from_import" in reason

    def test_dunder_import_blocked(self):
        safe, reason = _ast_is_safe("__import__('os')")
        assert safe is False
        assert "blocked_dunder_import" in reason

    def test_dynamic_dunder_import_blocked(self):
        safe, reason = _ast_is_safe("__import__(user_input)")
        assert safe is False
        assert "blocked_dynamic_dunder_import" in reason

    def test_importlib_blocked(self):
        safe, reason = _ast_is_safe("importlib.import_module('os')")
        assert safe is False
        assert "blocked_importlib" in reason

    def test_syntax_error_caught(self):
        safe, reason = _ast_is_safe("def :")
        assert safe is False
        assert "syntax_error" in reason

    def test_safe_math(self):
        safe, reason = _ast_is_safe("import math\nprint(math.sqrt(16))")
        assert safe is True

    def test_safe_multiline(self):
        code = "x = [1, 2, 3]\nfor i in x:\n    print(i * 2)"
        safe, reason = _ast_is_safe(code)
        assert safe is True

    def test_nested_import_blocked(self):
        safe, reason = _ast_is_safe("import os.path")
        assert safe is False

    def test_from_sys_blocked(self):
        safe, reason = _ast_is_safe("from sys import argv")
        assert safe is False


# ── code_executor tool ────────────────────────────────────────────────────────

class TestCodeExecutor:
    def test_normal_execution(self):
        result = asyncio.run(execute_code("print(2 + 2)"))
        assert result.success is True
        assert result.output["stdout"].strip() == "4"
        assert result.output["exit_code"] == 0

    def test_empty_input_rejected(self):
        result = asyncio.run(execute_code(""))
        assert result.success is False
        assert result.failure_reason == "malformed"

    def test_oversized_input_rejected(self):
        result = asyncio.run(execute_code("x = 1\n" * 6000))
        assert result.success is False
        assert result.failure_reason == "malformed"

    def test_unsafe_code_rejected(self):
        result = asyncio.run(execute_code("import os; os.system('ls')"))
        assert result.success is False
        assert result.failure_reason == "malformed"

    def test_stderr_captured(self):
        result = asyncio.run(execute_code("import sys; sys.stderr.write('err')"))
        # sys is blocked, so this should fail at AST check
        assert result.success is False

    def test_nonzero_exit(self):
        result = asyncio.run(execute_code("raise ValueError('boom')"))
        assert result.success is True  # subprocess ran, just errored
        assert result.output["exit_code"] != 0
        assert "ValueError" in result.output["stderr"]


# ── Tool registry ─────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_unknown_tool_returns_malformed(self):
        result = asyncio.run(call_tool("nonexistent_tool", {}))
        assert result.success is False
        assert result.failure_reason == "malformed"
        assert result.tool_name == "nonexistent_tool"

    def test_tool_result_dataclass(self):
        tr = ToolResult(
            tool_name="test",
            success=True,
            output={"key": "value"},
            latency_ms=42,
        )
        assert tr.tool_name == "test"
        assert tr.failure_reason is None

    def test_tool_result_with_failure(self):
        tr = ToolResult(
            tool_name="test",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="timeout",
        )
        assert tr.failure_reason == "timeout"
