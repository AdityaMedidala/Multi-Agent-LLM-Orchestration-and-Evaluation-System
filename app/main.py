from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.db import AgentLog, EvalRun, Job, PromptRewrite, ToolCall, get_db
from app.db.database import AsyncSessionLocal
from app.worker import rerun_eval, run_pipeline

app = FastAPI(title="mega-ai")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request bodies ────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    query: str


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]


class EvalRerunRequest(BaseModel):
    prompt_rewrite_id: str


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ── POST /jobs ────────────────────────────────────────────────────────────────

@app.post("/jobs", status_code=201)
async def create_job(
    body: CreateJobRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    job = Job(id=uuid.uuid4(), query=body.query, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    job_id = str(job.id)
    run_pipeline.delay(job_id, body.query)
    return {"job_id": job_id, "status": "queued"}


# ── GET /jobs/{job_id}/stream  (SSE) ─────────────────────────────────────────

# Maps AgentLog.event_type values to SSE event names
_LOG_EVENT_MAP = {
    "start": "agent_start",
    "output": "agent_done",
    "budget_violation": "budget_update",
}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> EventSourceResponse:
    async def _generate() -> AsyncGenerator[dict, None]:
        try:
            job_uuid = uuid.UUID(job_id)
        except ValueError:
            yield {"event": "error", "data": json.dumps({"message": "invalid job_id"})}
            return

        seen_log_ids: set[str] = set()
        seen_tool_ids: set[str] = set()

        while True:
            async with AsyncSessionLocal() as session:
                job = await session.get(Job, job_uuid)
                if job is None:
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": "job not found", "job_id": job_id}),
                    }
                    return

                # Stream any new AgentLog entries
                log_rows = (
                    await session.execute(
                        select(AgentLog)
                        .where(AgentLog.job_id == job_uuid)
                        .order_by(AgentLog.created_at)
                    )
                ).scalars().all()

                for log in log_rows:
                    lid = str(log.id)
                    if lid not in seen_log_ids:
                        seen_log_ids.add(lid)
                        yield {
                            "event": _LOG_EVENT_MAP.get(log.event_type, log.event_type),
                            "data": json.dumps({
                                "agent_id": log.agent_id,
                                "event_type": log.event_type,
                                "token_count": log.token_count,
                                "latency_ms": log.latency_ms,
                                "policy_violation": log.policy_violation,
                            }),
                        }

                # Stream any new ToolCall entries
                tool_rows = (
                    await session.execute(
                        select(ToolCall)
                        .where(ToolCall.job_id == job_uuid)
                        .order_by(ToolCall.created_at)
                    )
                ).scalars().all()

                for tc in tool_rows:
                    tid = str(tc.id)
                    if tid not in seen_tool_ids:
                        seen_tool_ids.add(tid)
                        yield {
                            "event": "tool_call",
                            "data": json.dumps({
                                "tool_name": tc.tool_name,
                                "agent_id": tc.agent_id,
                                "attempt": tc.attempt_number,
                                "accepted": tc.accepted,
                                "latency_ms": tc.latency_ms,
                            }),
                        }

                # Terminal states
                if job.status == "done":
                    yield {
                        "event": "final_answer",
                        "data": json.dumps({"answer": job.query}),
                    }
                    yield {"event": "done", "data": "{}"}
                    return

                if job.status == "failed":
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": "job failed", "job_id": job_id}),
                    }
                    yield {"event": "done", "data": "{}"}
                    return

            await asyncio.sleep(0.5)

    return EventSourceResponse(_generate())


# ── GET /jobs/{job_id}/trace ──────────────────────────────────────────────────

def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@app.get("/jobs/{job_id}/trace")
async def get_trace(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid job_id")

    job = await db.get(Job, job_uuid)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": f"Job {job_id} not found",
                "job_id": job_id,
            },
        )

    agent_logs = (
        await db.execute(
            select(AgentLog)
            .where(AgentLog.job_id == job_uuid)
            .order_by(AgentLog.created_at)
        )
    ).scalars().all()

    tool_calls = (
        await db.execute(
            select(ToolCall)
            .where(ToolCall.job_id == job_uuid)
            .order_by(ToolCall.created_at)
        )
    ).scalars().all()

    return {
        "job": {
            "job_id": str(job.id),
            "query": job.query,
            "status": job.status,
            "created_at": _iso(job.created_at),
            "updated_at": _iso(job.updated_at),
        },
        "agent_logs": [
            {
                "id": str(log.id),
                "agent_id": log.agent_id,
                "event_type": log.event_type,
                "latency_ms": log.latency_ms,
                "token_count": log.token_count,
                "policy_violation": log.policy_violation,
                "created_at": _iso(log.created_at),
            }
            for log in agent_logs
        ],
        "tool_calls": [
            {
                "id": str(tc.id),
                "agent_id": tc.agent_id,
                "tool_name": tc.tool_name,
                "attempt_number": tc.attempt_number,
                "accepted": tc.accepted,
                "failure_reason": tc.failure_reason,
                "latency_ms": tc.latency_ms,
                "created_at": _iso(tc.created_at),
            }
            for tc in tool_calls
        ],
    }


# ── GET /eval/latest ──────────────────────────────────────────────────────────

@app.get("/eval/latest")
async def get_latest_eval(db: AsyncSession = Depends(get_db)) -> dict:
    eval_run = (
        await db.execute(
            select(EvalRun).order_by(EvalRun.triggered_at.desc()).limit(1)
        )
    ).scalar_one_or_none()
    if eval_run is None:
        raise HTTPException(status_code=404, detail="No eval runs found")
    return {
        "eval_run_id": str(eval_run.id),
        "triggered_at": eval_run.triggered_at.isoformat(),
        "summary": eval_run.summary,
    }


# ── POST /eval/rerun ──────────────────────────────────────────────────────────

@app.post("/eval/rerun")
async def trigger_eval_rerun(body: EvalRerunRequest) -> dict:
    rerun_eval.delay(body.prompt_rewrite_id)
    return {"status": "enqueued", "prompt_rewrite_id": body.prompt_rewrite_id}


# ── POST /prompts/{rewrite_id}/review ────────────────────────────────────────

@app.post("/prompts/{rewrite_id}/review")
async def review_prompt(
    rewrite_id: str,
    body: ReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        rewrite_uuid = uuid.UUID(rewrite_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid rewrite_id")

    rewrite = await db.get(PromptRewrite, rewrite_uuid)
    if rewrite is None:
        raise HTTPException(status_code=404, detail=f"Rewrite {rewrite_id} not found")

    rewrite.status = body.decision
    if body.decision == "approved":
        rewrite.approved_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(rewrite)

    if body.decision == "approved":
        # Fetch latest eval summary to feed the meta-agent
        eval_run = (
            await db.execute(
                select(EvalRun).order_by(EvalRun.triggered_at.desc()).limit(1)
            )
        ).scalar_one_or_none()

        if eval_run is not None:
            _eval_run_id  = str(eval_run.id)
            _eval_summary = eval_run.summary  # dict from JSON column

            # Fire-and-forget: call meta-agent and persist the new rewrite proposal
            async def _spawn_meta_rewrite() -> None:
                from app.agents.meta_agent import run_meta_agent
                try:
                    rewrite_data = await run_meta_agent(_eval_summary)
                    async with AsyncSessionLocal() as session:
                        new_rewrite = PromptRewrite(
                            id=uuid.uuid4(),
                            eval_run_id=uuid.UUID(_eval_run_id),
                            agent_id=rewrite_data["agent_id"],
                            dimension=rewrite_data["dimension"],
                            original_prompt=rewrite_data["original_prompt"],
                            proposed_prompt=rewrite_data["proposed_prompt"],
                            diff_justification=rewrite_data["diff_justification"],
                            status="pending",
                        )
                        session.add(new_rewrite)
                        await session.commit()
                except Exception:
                    pass  # non-fatal: meta-agent failure should not break the response

            asyncio.create_task(_spawn_meta_rewrite())

        # Enqueue targeted re-eval via Celery
        rerun_eval.delay(rewrite_id)

    return {"rewrite_id": rewrite_id, "status": rewrite.status}
