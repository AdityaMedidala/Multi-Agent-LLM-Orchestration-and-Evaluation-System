from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: dict
    latency_ms: int
    failure_reason: str | None = None  # "timeout" | "empty" | "malformed" | None


# Imports come AFTER ToolResult so circular imports in tool files resolve cleanly
# (tool files import ToolResult from here; Python's module cache already has it)
from app.tools.code_executor import execute_code  # noqa: E402
from app.tools.data_lookup import data_lookup  # noqa: E402
from app.tools.self_reflection import self_reflect  # noqa: E402
from app.tools.web_search import web_search  # noqa: E402

TOOL_REGISTRY: dict[str, object] = {
    "web_search": web_search,
    "code_executor": execute_code,
    "data_lookup": data_lookup,
    "self_reflection": self_reflect,
}


async def call_tool(tool_name: str, kwargs: dict) -> ToolResult:
    """Central dispatcher. Returns a malformed ToolResult if tool not found."""
    if tool_name not in TOOL_REGISTRY:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output={},
            latency_ms=0,
            failure_reason="malformed",
        )
    start = time.monotonic()
    result: ToolResult = await TOOL_REGISTRY[tool_name](**kwargs)  # type: ignore[operator]
    result.latency_ms = int((time.monotonic() - start) * 1000)
    return result
