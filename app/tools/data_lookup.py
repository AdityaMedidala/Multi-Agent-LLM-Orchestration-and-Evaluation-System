from __future__ import annotations

import asyncio
import functools

import psycopg2
import psycopg2.extras
from langchain_anthropic import ChatAnthropic

from app.config import settings
from app.tools.registry import ToolResult

_SYSTEM = """\
You are a SQL assistant. Convert natural language queries to SQL.

Tables: jobs(id,query,status,created_at), \
agent_logs(id,job_id,agent_id,event_type,payload), \
tool_calls(id,job_id,tool_name,attempt_number,input_payload,output_payload,accepted)

Instruction: Return ONLY a valid SQL SELECT statement, nothing else.\
"""


def _run_query(sql: str) -> list[dict]:
    conn = psycopg2.connect(settings.database_url_sync)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


async def data_lookup(natural_language_query: str) -> ToolResult:
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    # ── 1. NL → SQL via LLM ──────────────────────────────────────────────────
    try:
        response = await llm.ainvoke(
            [("system", _SYSTEM), ("human", natural_language_query)]
        )
        sql = response.content.strip()
    except Exception:
        return ToolResult(
            tool_name="data_lookup",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="timeout",
        )

    # ── 2. Validate it's a SELECT ─────────────────────────────────────────────
    if not sql.upper().lstrip().startswith("SELECT"):
        return ToolResult(
            tool_name="data_lookup",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="malformed",
        )

    # ── 3. Execute against Postgres (sync driver in executor) ─────────────────
    loop = asyncio.get_running_loop()
    try:
        rows = await loop.run_in_executor(None, functools.partial(_run_query, sql))
    except Exception:
        return ToolResult(
            tool_name="data_lookup",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="timeout",
        )

    if not rows:
        return ToolResult(
            tool_name="data_lookup",
            success=False,
            output={"rows": [], "sql": sql, "row_count": 0},
            latency_ms=0,
            failure_reason="empty",
        )

    return ToolResult(
        tool_name="data_lookup",
        success=True,
        output={"rows": rows, "sql": sql, "row_count": len(rows)},
        latency_ms=0,
    )
