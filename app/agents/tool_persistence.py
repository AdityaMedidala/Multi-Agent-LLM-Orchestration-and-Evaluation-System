from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid

import psycopg2

from app.config import settings

log = logging.getLogger(__name__)


def _write_tool_call_sync(
    job_id: str,
    tool_name: str,
    agent_id: str,
    attempt: int,
    input_payload: dict,
    output_payload: dict,
    latency_ms: int,
    accepted: bool | None,
    failure_reason: str | None,
) -> None:
    try:
        try:
            job_uuid_str = str(uuid.UUID(job_id))
        except ValueError:
            job_uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, job_id))
        conn = psycopg2.connect(settings.database_url_sync)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tool_calls
              (id, job_id, tool_name, agent_id, attempt_number,
               input_payload, output_payload, latency_ms, accepted,
               failure_reason)
            VALUES (%s,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                str(uuid.uuid4()), job_uuid_str, tool_name, agent_id,
                attempt,
                json.dumps(input_payload, default=str),
                json.dumps(output_payload, default=str),
                latency_ms, accepted, failure_reason,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        log.warning("tool_call write failed: %s", exc)


async def _persist_tool_call(
    job_id: str,
    tool_name: str,
    agent_id: str,
    attempt: int,
    input_payload: dict,
    output_payload: dict,
    latency_ms: int,
    accepted: bool | None,
    failure_reason: str | None,
) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        functools.partial(
            _write_tool_call_sync,
            job_id, tool_name, agent_id, attempt,
            input_payload, output_payload, latency_ms,
            accepted, failure_reason,
        ),
    )
