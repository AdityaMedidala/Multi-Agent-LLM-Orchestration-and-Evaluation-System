from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Literal

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

@app.get("/jobs/{job_id}/stream")
async def stream_job(
    job_id: str, db: AsyncSession = Depends(get_db)
) -> EventSourceResponse:
    from app.streaming import token_stream

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid job_id")

    result = await db.execute(select(Job).where(Job.id == job_uuid))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": f"Job {job_id} not found",
                "job_id": job_id,
            },
        )

    async def event_generator():
        # If job already done, emit final_answer and close immediately
        if job.status == "done":
            # Read final answer from agent_logs (synthesis output payload)
            import json as _json
            from sqlalchemy import text
            log_result = await db.execute(
                text("""
                    SELECT payload FROM agent_logs
                    WHERE job_id = :job_id
                      AND agent_id = 'synthesis'
                      AND event_type = 'output'
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"job_id": str(job_uuid)},
            )
            row = log_result.fetchone()
            if row:
                payload = _json.loads(row[0]) if isinstance(row[0], str) else row[0]
                answer = payload.get("output", {}).get("final_answer", "")
            else:
                answer = ""
            yield {
                "event": "final_answer",
                "data": json.dumps({"answer": answer}),
            }
            yield {"event": "done", "data": "{}"}
            return

        # Stream live events from Redis pub/sub
        async for event_type, data in token_stream(job_id):
            yield {
                "event": event_type,
                "data": json.dumps(data),
            }

    return EventSourceResponse(event_generator())


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

    # Read final_answer from synthesis agent_log output
    import json as _json
    from sqlalchemy import text as _text
    fa_result = await db.execute(
        _text("""
            SELECT payload FROM agent_logs
            WHERE job_id = :job_id
              AND agent_id = 'synthesis'
              AND event_type = 'output'
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"job_id": str(job_uuid)},
    )
    fa_row = fa_result.fetchone()
    final_answer = ""
    if fa_row:
        fa_payload = _json.loads(fa_row[0]) if isinstance(fa_row[0], str) else fa_row[0]
        final_answer = fa_payload.get("output", {}).get("final_answer", "")

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
            "final_answer": final_answer,
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
    result = await db.execute(
        select(EvalRun).order_by(EvalRun.triggered_at.desc()).limit(1)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="No eval runs found")

    # Per-case data lives in summary["by_category"][cat]["cases"].
    # Query/answer are in summary["reproducibility_snapshots"][case_id].
    raw_summary: dict = run.summary or {}
    snapshots: dict = raw_summary.get("reproducibility_snapshots", {})

    cases = []
    for cat in ("baseline", "ambiguous", "adversarial"):
        for c in raw_summary.get("by_category", {}).get(cat, {}).get("cases", []):
            case_id = c.get("id", "")
            snap = snapshots.get(case_id, {})
            dims: dict = c.get("dimensions", {})
            cases.append({
                "case_id": case_id,
                "category": cat,
                "query": snap.get("original_query", ""),
                "passed": c.get("passed"),
                "weighted_score": c.get("score"),
                "dimension_scores": {k: v.get("score") for k, v in dims.items()},
                "justifications": {k: v.get("justification", "") for k, v in dims.items()},
                "answer_snippet": (snap.get("final_answer") or "")[:150],
            })

    # Recompute aggregates from the flat case list
    by_category: dict = {}
    for cat in ("baseline", "ambiguous", "adversarial"):
        cat_cases = [c for c in cases if c["category"] == cat]
        if cat_cases:
            by_category[cat] = {
                "passed": sum(1 for c in cat_cases if c.get("passed")),
                "count": len(cat_cases),
                "avg_score": round(
                    sum(c.get("weighted_score") or 0 for c in cat_cases)
                    / len(cat_cases), 3
                ),
            }

    total_passed = sum(1 for c in cases if c.get("passed"))
    overall_avg = (
        round(sum(c.get("weighted_score") or 0 for c in cases) / len(cases), 3)
        if cases else 0.0
    )

    return {
        "eval_run_id": str(run.id),
        "triggered_at": _iso(run.triggered_at),
        "summary": {
            "total_cases": len(cases),
            "total_passed": total_passed,
            "overall_avg_score": overall_avg,
            "by_category": by_category,
            "worst_case_id": raw_summary.get("worst_case_id"),
            "worst_case_score": raw_summary.get("worst_case_score"),
        },
        "cases": cases,
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
