from __future__ import annotations

import asyncio
import json

import redis.asyncio as aioredis

from app.config import settings


_redis_pool: aioredis.ConnectionPool | None = None


def _get_pool() -> aioredis.ConnectionPool:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.ConnectionPool.from_url(settings.redis_url)
    return _redis_pool


def _channel(job_id: str) -> str:
    return f"job:{job_id}:stream"


async def publish_event(job_id: str, event_type: str, data: dict) -> None:
    """Publish one SSE event to Redis pub/sub. Fire and forget."""
    r = aioredis.Redis(connection_pool=_get_pool())
    payload = json.dumps({"event": event_type, "data": data})
    await r.publish(_channel(job_id), payload)
    # No close — pool manages connections


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
    """Sync wrapper for use in Celery tasks."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(publish_event(job_id, event_type, data))
    except Exception:
        pass  # non-fatal
