from __future__ import annotations

import logging
import time

import psycopg2

from app.config import settings

log = logging.getLogger(__name__)

_prompt_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 60.0  # seconds


def get_active_prompt(agent_id: str, fallback: str) -> str:
    """
    Returns the latest approved prompt for agent_id from prompt_rewrites table.
    Results are cached for 60 seconds to avoid a DB round-trip per agent call.
    Falls back to the hardcoded constant if none exists or DB is unavailable.
    """
    # Check cache first
    cached = _prompt_cache.get(agent_id)
    if cached:
        value, expires_at = cached
        if time.monotonic() < expires_at:
            return value

    # Cache miss — query DB
    try:
        conn = psycopg2.connect(settings.database_url_sync)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT proposed_prompt
            FROM prompt_rewrites
            WHERE agent_id = %s
              AND status = 'approved'
            ORDER BY approved_at DESC
            LIMIT 1
            """,
            (agent_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0]:
            log.info("prompt_registry: loaded approved prompt for %s", agent_id)
            _prompt_cache[agent_id] = (row[0], time.monotonic() + _CACHE_TTL)
            return row[0]
        else:
            # Cache the fallback too so we don't hit DB every call
            _prompt_cache[agent_id] = (fallback, time.monotonic() + _CACHE_TTL)
    except Exception as exc:
        log.warning("prompt_registry: DB lookup failed for %s: %s", agent_id, exc)

    return fallback
