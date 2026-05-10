from __future__ import annotations

import asyncio
import json
import logging

import redis.asyncio as aioredis

from app.config import settings

log = logging.getLogger(__name__)


def _channel(job_id: str) -> str:
    return f"job:{job_id}:stream"


def _buffer_key(job_id: str) -> str:
    """Redis list key that stores all events for replay."""
    return f"job:{job_id}:events"


# ── Per-event-loop Redis pool ─────────────────────────────────────────────────
# Keyed by id(loop) so each asyncio event loop (main app, each Celery asyncio.run()
# call) gets its own connection pool.  Avoids "Future attached to a different loop"
# errors across Celery task invocations while eliminating per-publish TCP handshakes.

_pools: dict[int, aioredis.Redis] = {}


async def _get_redis() -> aioredis.Redis:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if loop_id not in _pools:
        _pools[loop_id] = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
    return _pools[loop_id]


async def publish_event(job_id: str, event_type: str, data: dict) -> None:
    """Publish one SSE event to Redis pub/sub AND buffer it in a list for replay."""
    try:
        r = await _get_redis()
        payload = json.dumps({"event": event_type, "data": data})
        # Buffer event in a list so late-subscribing clients can replay
        await r.rpush(_buffer_key(job_id), payload)
        # Set a TTL so completed job buffers don't accumulate forever
        await r.expire(_buffer_key(job_id), 300)  # 5 minutes
        # Publish for live subscribers
        await r.publish(_channel(job_id), payload)
    except Exception as exc:
        log.warning("publish_event failed for job %s event %s: %s", job_id, event_type, exc)


async def token_stream(job_id: str, timeout_seconds: int = 120):
    """
    Async generator that yields (event_type, data) tuples.

    First replays any buffered events from the Redis list (catching up events
    that were published before the client subscribed), then switches to live
    pub/sub for new events.  Yields until a 'done' or 'error' event is
    received, or timeout.
    """
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        # ── Phase 1: Replay buffered events ───────────────────────────────────
        buffered = await r.lrange(_buffer_key(job_id), 0, -1)
        replay_count = len(buffered)
        done_in_replay = False
        for raw in buffered:
            payload = json.loads(raw)
            event_type = payload["event"]
            data = payload["data"]
            yield event_type, data
            if event_type in ("done", "error"):
                done_in_replay = True
                break

        if done_in_replay:
            return

        # ── Phase 2: Subscribe for live events ────────────────────────────────
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

                # Skip events we already replayed from the buffer
                if replay_count > 0:
                    replay_count -= 1
                    continue

                yield event_type, data
                if event_type in ("done", "error"):
                    break
        finally:
            await pubsub.unsubscribe(_channel(job_id))
    finally:
        await r.aclose()


def publish_event_sync(job_id: str, event_type: str, data: dict) -> None:
    """
    Sync wrapper for use in Celery tasks (no running event loop).
    Creates a fresh event loop per call — Celery tasks are sync and short-lived,
    so the overhead is acceptable here. Do NOT use this inside async code.
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            # Use a fresh connection (not the pool) since this loop is ephemeral
            async def _pub() -> None:
                r = aioredis.from_url(settings.redis_url, decode_responses=True)
                try:
                    payload = json.dumps({"event": event_type, "data": data})
                    await r.rpush(_buffer_key(job_id), payload)
                    await r.expire(_buffer_key(job_id), 300)
                    await r.publish(_channel(job_id), payload)
                finally:
                    await r.aclose()

            loop.run_until_complete(_pub())
        finally:
            loop.close()
    except Exception as exc:
        log.warning("publish_event_sync failed: %s", exc)