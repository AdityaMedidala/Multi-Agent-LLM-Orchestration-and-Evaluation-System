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


@celery_app.task(name="rerun_eval")
def rerun_eval(prompt_rewrite_id: str) -> dict:
    log.info("rerun_eval called for %s", prompt_rewrite_id)
    return {"status": "stub", "prompt_rewrite_id": prompt_rewrite_id}
