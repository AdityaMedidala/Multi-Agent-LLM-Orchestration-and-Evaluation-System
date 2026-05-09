from __future__ import annotations

import logging

import psycopg2

from app.config import settings

log = logging.getLogger(__name__)


def get_active_prompt(agent_id: str, fallback: str) -> str:
    """
    Returns the latest approved prompt for agent_id from prompt_rewrites table.
    Falls back to the hardcoded constant if none exists or DB is unavailable.
    """
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
            return row[0]
    except Exception as exc:
        log.warning("prompt_registry: DB lookup failed for %s: %s", agent_id, exc)
    return fallback
