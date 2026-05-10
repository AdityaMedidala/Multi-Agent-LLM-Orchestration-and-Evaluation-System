from __future__ import annotations

import logging

import psycopg2.pool

from app.config import settings

log = logging.getLogger(__name__)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=settings.database_url_sync,
        )
        log.info("sync_pool: initialized (minconn=2, maxconn=10)")
    return _pool


def get_conn():
    """Get a connection from the pool. Caller must call put_conn() after."""
    return get_pool().getconn()


def put_conn(conn, close: bool = False) -> None:
    """Return a connection to the pool."""
    try:
        get_pool().putconn(conn, close=close)
    except Exception as exc:
        log.warning("sync_pool: putconn failed: %s", exc)
