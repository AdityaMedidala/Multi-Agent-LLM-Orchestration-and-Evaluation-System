from __future__ import annotations

import asyncio
import json
import uuid

import psycopg2

from app.config import settings
from app.eval.runner import run_eval


def trigger_eval(case_ids: list[str] | None = None) -> str:
    """
    Run the eval suite synchronously, persist the result, and return the
    eval_run_id.  Intended to be called from a Celery task or CLI script.
    """
    summary = asyncio.run(run_eval(case_ids))
    eval_run_id = str(uuid.uuid4())

    conn = psycopg2.connect(settings.database_url_sync)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO eval_runs (id, summary, worst_prompt_id) "
            "VALUES (%s, %s, %s)",
            (
                eval_run_id,
                json.dumps(summary, default=str),
                summary.get("worst_case_id"),
            ),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()

    return eval_run_id
