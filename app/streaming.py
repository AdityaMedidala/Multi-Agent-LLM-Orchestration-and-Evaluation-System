from __future__ import annotations

import asyncio
import json

import redis.asyncio as aioredis

from app.config import settings


def _channel(job_id: str) -> str:
    return f"job:{job_id}:stream"


async def publish_event(job_id: str, event_type: str, data: dict) -> None:
    """Publish one SSE event to Redis pub/sub.

    Creates a fresh connection per call so that Celery tasks running under
    asyncio.run() (each with its own event loop) never hit the
    'Future attached to a different loop' error that a cached module-level
    ConnectionPool would cause.
    """
    r = aioredis.from_url(settings.redis_url)
    try:
        payload = json.dumps({"event": event_type, "data": data})
        await r.publish(_channel(job_id), payload)
    finally:
        await r.aclose()


async def token_stream(job_id: str, timeout_seconds: int = 120):
    """
    Async generator that yields (event_type, data) tuples from Redis pub/sub.
    Yields until a 'done' or 'error' event is received, or timeout.
    """
    r = aioredis.from_url(settings.redis_url)
    pubsub = r.pubsub()
    await pubsub.subscribe(_channel(job_id))
    try:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        async for message in pubsub.listen():
            if asyncio.get_running_loop().time() > deadline:
                yield "error", {"message": "stream timeout"}
                break
            if message["type"] != "message":
                continue
            payload = json.loads(message["data"])
            event_type = payload["event"]
            data = payload["data"]
            yield event_type, data
            if event_type in ("done", "error"):
                break
    finally:
        await pubsub.unsubscribe(_channel(job_id))
        await r.aclose()


def publish_event_sync(job_id: str, event_type: str, data: dict) -> None:
    """Sync wrapper for use in Celery tasks (no running event loop)."""
    try:
        # Always create a fresh event loop — Celery tasks are sync
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(publish_event(job_id, event_type, data))
        finally:
            loop.close()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "publish_event_sync failed: %s", exc
        )
