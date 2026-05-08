from __future__ import annotations

import asyncio
import logging

import psycopg2
from celery import Celery

from app.config import settings

log = logging.getLogger(__name__)

celery_app = Celery(
    "mega_ai",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
)


# ── DB helpers (psycopg2 sync — required by Celery worker process) ────────────

def _db_conn():
    return psycopg2.connect(settings.database_url_sync)


def _update_job_status(job_id: str, status: str) -> None:
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = %s, updated_at = NOW() WHERE id = %s::uuid",
                (status, job_id),
            )
        conn.commit()
    finally:
        conn.close()


# ── Tasks ─────────────────────────────────────────────────────────────────────

@celery_app.task(name="run_pipeline", bind=True)
def run_pipeline(self, job_id: str, query: str) -> dict:  # noqa: ANN001
    # Imports are deferred so the Celery worker doesn't instantiate LLM clients
    # at module load time (no ANTHROPIC_API_KEY needed until the task runs).
    from app.agents.orchestrator import OrchestratorAgent
    from app.schemas.context import SharedContext

    _update_job_status(job_id, "running")

    ctx = SharedContext(job_id=job_id, original_query=query)
    orchestrator = OrchestratorAgent()

    try:
        result = asyncio.run(orchestrator.run(ctx))
    except Exception:
        log.exception("Pipeline failed for job %s", job_id)
        _update_job_status(job_id, "failed")
        raise

    # Temporarily store final_answer in job.query until a dedicated column exists
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = %s, query = %s, updated_at = NOW() WHERE id = %s::uuid",
                ("done", ctx.final_answer or query, job_id),
            )
        conn.commit()
    finally:
        conn.close()

    return {"job_id": job_id, "status": "done"}


@celery_app.task(bind=True, name="rerun_eval")
def rerun_eval(self, prompt_rewrite_id: str) -> dict:  # noqa: ANN001
    import asyncio
    import json
    import uuid

    import psycopg2

    from app.config import settings
    from app.eval.runner import run_eval

    conn = psycopg2.connect(settings.database_url_sync)
    try:
        cur = conn.cursor()
        # Verify the prompt rewrite record exists
        cur.execute(
            "SELECT agent_id, dimension FROM prompt_rewrites WHERE id = %s",
            (prompt_rewrite_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"error": "prompt_rewrite_not_found"}

        # Pull failed case IDs from the latest eval run
        cur.execute(
            "SELECT summary FROM eval_runs ORDER BY triggered_at DESC LIMIT 1"
        )
        eval_row = cur.fetchone()
        if not eval_row:
            return {"error": "no_eval_run_found"}

        summary = eval_row[0]
        # psycopg2 may return JSON columns as str on some driver versions
        if isinstance(summary, str):
            summary = json.loads(summary)

        failed_ids = [
            c["id"]
            for cat in summary.get("by_category", {}).values()
            for c in cat.get("cases", [])
            if not c.get("passed", True)
        ]
        cur.close()
    finally:
        conn.close()

    # Re-run eval on failed cases only (or all if none were tracked as failed)
    new_summary = asyncio.run(run_eval(failed_ids or None))

    # Persist the rerun record
    rerun_id = str(uuid.uuid4())
    conn = psycopg2.connect(settings.database_url_sync)
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO eval_reruns
               (id, prompt_rewrite_id, failed_case_ids, delta_scores)
               VALUES (%s, %s, %s, %s)""",
            (
                rerun_id,
                prompt_rewrite_id,
                json.dumps(failed_ids),
                json.dumps(new_summary, default=str),
            ),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()

    return {"rerun_id": rerun_id, "cases_rerun": len(failed_ids)}
