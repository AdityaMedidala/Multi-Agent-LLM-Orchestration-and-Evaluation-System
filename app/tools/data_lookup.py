from __future__ import annotations

import asyncio
import functools
import logging

import psycopg2.extras
from langchain_google_genai import ChatGoogleGenerativeAI

from app.db.sync_pool import get_conn, put_conn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tools.registry import ToolResult

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a SQL assistant. Convert natural language queries to SQL.

Tables: jobs(id,query,status,created_at), \
agent_logs(id,job_id,agent_id,event_type,payload), \
tool_calls(id,job_id,tool_name,attempt_number,input_payload,output_payload,accepted)

Instruction: Return ONLY a valid SQL SELECT statement, nothing else.\
"""


def _run_query(sql: str) -> list[dict]:
    """Execute sql using a connection from the shared pool (not a fresh connect)."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]
    finally:
        put_conn(conn)


async def data_lookup(natural_language_query: str) -> ToolResult:
    from app.tools.registry import ToolResult
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # ── 1. NL → SQL via LLM ──────────────────────────────────────────────────
    try:
        response = await llm.ainvoke(
            [("system", _SYSTEM), ("human", natural_language_query)]
        )
        sql = response.content.strip()
    except Exception as exc:
        log.warning("data_lookup: LLM call failed: %s", exc)
        return ToolResult(
            tool_name="data_lookup",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="llm_error",
        )

    # ── 1a. Strip markdown code fences ───────────────────────────────────────
    if sql.startswith("```"):
        lines = sql.split("\n")
        inner = [l for l in lines[1:] if l.strip() != "```"]
        sql = "\n".join(inner).strip()

    log.debug("data_lookup: generated SQL: %s", sql)

    # ── 2. Validate it's a SELECT ─────────────────────────────────────────────
    if not sql.upper().lstrip().startswith("SELECT"):
        log.warning("data_lookup: non-SELECT SQL rejected: %.120s", sql)
        return ToolResult(
            tool_name="data_lookup",
            success=False,
            output={"sql": sql},
            latency_ms=0,
            failure_reason="malformed",
        )

    # ── 3. Execute against Postgres via pool ──────────────────────────────────
    loop = asyncio.get_running_loop()
    try:
        rows = await loop.run_in_executor(None, functools.partial(_run_query, sql))
    except Exception as exc:
        log.warning("data_lookup: query execution failed: %s | sql: %s", exc, sql)
        return ToolResult(
            tool_name="data_lookup",
            success=False,
            output={"sql": sql},
            latency_ms=0,
            failure_reason="db_error",
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