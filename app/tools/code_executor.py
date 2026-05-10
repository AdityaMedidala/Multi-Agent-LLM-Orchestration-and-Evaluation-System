from __future__ import annotations

import ast
import asyncio
import functools
import subprocess

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This block is ONLY read by your IDE and type-checkers.
    # Python ignores it at runtime, which prevents the circular import!
    from app.tools.registry import ToolResult

# Modules that must never be imported inside sandboxed code.
_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "importlib", "shutil",
    "pathlib", "socket", "ctypes", "mmap", "signal",
    "multiprocessing", "threading", "pty", "tty",
    "fcntl", "resource", "gc", "inspect",
})


def _ast_is_safe(code: str) -> tuple[bool, str]:
    """
    Parse `code` with the ast module and walk every node.
    Returns (is_safe, reason).

    Catches:
    - import os / import sys / import subprocess (and any blocked module)
    - from os import path  (and any blocked module)
    - __import__('os')     (Call node with __import__ as the callee)
    - importlib.import_module(...)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax_error: {exc}"

    for node in ast.walk(tree):
        # import X  /  import X as Y
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_MODULES:
                    return False, f"blocked_import: {alias.name}"

        # from X import Y
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in _BLOCKED_MODULES:
                return False, f"blocked_from_import: {module}"

        # __import__('os') or __import__('subprocess')
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "__import__":
                # Try to extract the first arg statically
                if node.args and isinstance(node.args[0], ast.Constant):
                    mod = str(node.args[0].value).split(".")[0]
                    if mod in _BLOCKED_MODULES:
                        return False, f"blocked_dunder_import: {mod}"
                else:
                    # Dynamic __import__ — reject outright
                    return False, "blocked_dynamic_dunder_import"

            # importlib.import_module(...)
            if isinstance(func, ast.Attribute) and func.attr == "import_module":
                return False, "blocked_importlib_import_module"

    return True, "ok"


def _run_sync(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )


async def execute_code(code: str) -> ToolResult:
    from app.tools.registry import ToolResult
    # ── 1. Validate input ─────────────────────────────────────────────────────
    if not code or len(code) > 5000:
        return ToolResult(
            tool_name="code_executor",
            success=False,
            output={"reason": "empty or oversized input"},
            latency_ms=0,
            failure_reason="malformed",
        )

    # ── 2. AST safety check ───────────────────────────────────────────────────
    safe, reason = _ast_is_safe(code)
    if not safe:
        return ToolResult(
            tool_name="code_executor",
            success=False,
            output={"reason": reason},
            latency_ms=0,
            failure_reason="malformed",
        )

    # ── 3. Execute in subprocess ──────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    try:
        proc: subprocess.CompletedProcess = await loop.run_in_executor(
            None, functools.partial(_run_sync, code)
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="code_executor",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="timeout",
        )

    return ToolResult(
        tool_name="code_executor",
        success=True,
        output={
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        },
        latency_ms=0,
    )