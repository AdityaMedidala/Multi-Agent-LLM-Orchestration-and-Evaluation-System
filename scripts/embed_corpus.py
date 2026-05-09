#!/usr/bin/env python3
"""
Add OpenAI embeddings to document_chunks after seeding.
Run after seed_corpus.py:
    docker compose exec api uv run python scripts/embed_corpus.py
Requires OPENAI_API_KEY in environment.
"""
from __future__ import annotations

import os

import psycopg2
from openai import OpenAI

DATABASE_URL_SYNC = os.environ.get(
    "DATABASE_URL_SYNC",
    "postgresql://mega:mega@postgres:5432/megaai",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def main() -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)
    conn = psycopg2.connect(DATABASE_URL_SYNC)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, text FROM document_chunks WHERE embedding IS NULL")
    rows = cur.fetchall()
    print(f"Embedding {len(rows)} chunks...")
    for chunk_id, text in rows:
        emb = client.embeddings.create(
            model="text-embedding-3-small", input=[text]
        ).data[0].embedding
        emb_str = "[" + ",".join(map(str, emb)) + "]"
        cur.execute(
            "UPDATE document_chunks SET embedding = %s::vector WHERE chunk_id = %s",
            (emb_str, chunk_id),
        )
        print(f"  ✓ {chunk_id}")
    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. {len(rows)} embeddings added.")


if __name__ == "__main__":
    main()
