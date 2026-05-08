from __future__ import annotations

import asyncio
import functools
import re
import subprocess

from app.tools.registry import ToolResult

# Strip any line that imports os, sys, or subprocess (basic sandbox)
_BLOCKED = re.compile(
    r"^\s*(import\s+(os|sys|subprocess)\b.*|from\s+(os|sys|subprocess)\b\s+import\b.*)",
    re.MULTILINE,
)


def _sanitize(code: str) -> str:
    return _BLOCKED.sub("", code)


def _run_sync(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )


async def execute_code(code: str) -> ToolResult:
    if not code or len(code) > 5000:
        return ToolResult(
            tool_name="code_executor",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="malformed",
        )

    safe_code = _sanitize(code)
    loop = asyncio.get_running_loop()

    try:
        proc: subprocess.CompletedProcess = await loop.run_in_executor(
            None, functools.partial(_run_sync, safe_code)
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
