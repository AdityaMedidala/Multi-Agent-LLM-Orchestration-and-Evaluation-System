#!/usr/bin/env python3
"""
Seed the document_chunks table with the demo corpus.
Run after docker compose up:
    docker compose exec api uv run python scripts/seed_corpus.py
"""
from __future__ import annotations

import json
import os

import psycopg2

DATABASE_URL_SYNC = os.environ.get(
    "DATABASE_URL_SYNC",
    "postgresql://mega:mega@postgres:5432/megaai",
)

CHUNKS = [
    ("c1",  "Binary search is an efficient algorithm for finding an item in a sorted list. It works by repeatedly dividing the search interval in half. Time complexity is O(log n).", {"source": "algorithms"}),
    ("c2",  "RISC (Reduced Instruction Set Computer) uses simple instructions that execute in one clock cycle. Examples include ARM and MIPS processors.", {"source": "architecture"}),
    ("c3",  "CISC (Complex Instruction Set Computer) uses complex instructions that can execute multi-step operations. x86 is the dominant CISC architecture.", {"source": "architecture"}),
    ("c4",  "Large language models like GPT-4 require significant energy for inference due to their billions of parameters and matrix multiplication operations.", {"source": "ai"}),
    ("c5",  "Mitochondria are the powerhouse of the cell, producing ATP through oxidative phosphorylation and cellular respiration.", {"source": "biology"}),
    ("c6",  "TCP three-way handshake: client sends SYN, server responds SYN-ACK, client sends ACK. This establishes a reliable connection.", {"source": "networking"}),
    ("c7",  "The time complexity of binary search is O(log n) because it halves the search space on each iteration.", {"source": "algorithms"}),
    ("c8",  "The 1929 stock market crash was primarily caused by excessive speculation on margin, overvalued stocks, and weak banking regulation. Investors borrowed heavily to buy stocks, creating a bubble that collapsed in October 1929.", {"source": "history"}),
    ("c9",  "The 2008 financial crisis differed from 1929 in that it was driven by subprime mortgage lending, mortgage-backed securities, and excessive leverage in the banking system rather than stock speculation.", {"source": "history"}),
    ("c10", "Mitosis produces two identical diploid daughter cells. Meiosis produces four genetically unique haploid cells. In meiosis, homologous chromosomes undergo crossover during prophase I, increasing genetic diversity.", {"source": "biology"}),
    ("c11", "Transformers handle long sequences using self-attention mechanisms. The quadratic complexity of attention O(n²) is addressed through techniques like sparse attention, linear attention, and sliding window attention used in models like Longformer.", {"source": "ml"}),
    ("c12", "The classical tongue map showing distinct taste regions is a debunked myth originating from a mistranslation of D.P. Hänig 1901 paper. Modern research shows taste receptors for sweet, salty, sour, and bitter are distributed across the entire tongue.", {"source": "biology"}),
    ("c13", "The relationship between dietary saturated fat and cardiovascular disease is contested. The American Heart Association maintains a causal link, while meta-analyses by Siri-Tarino 2010 and Chowdhury 2014 found weak or no direct association after adjusting for replacement nutrients.", {"source": "medicine"}),
    ("c14", "Albert Einstein excelled at mathematics from an early age. The myth that he failed mathematics as a schoolboy is false. He mastered calculus by age 15 and consistently received top marks in mathematics and physics throughout his schooling in Switzerland.", {"source": "history"}),
    ("c15", "Mercury is the smallest planet in the solar system and closest to the Sun. It has no atmosphere, extreme temperature variations, and a day longer than its year. The element mercury (Hg) is a liquid metal at room temperature used in thermometers.", {"source": "science"}),
]


def main() -> None:
    conn = psycopg2.connect(DATABASE_URL_SYNC)
    cur = conn.cursor()

    # Create table if not exists (mirrors the raw SQL used in deployment)
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id TEXT PRIMARY KEY,
            text     TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding vector(1536),
            ts       TSVECTOR GENERATED ALWAYS AS
                     (to_tsvector('english', coalesce(text, ''))) STORED
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_ts
            ON document_chunks USING GIN (ts);
    """)

    inserted = 0
    for chunk_id, text, metadata in CHUNKS:
        cur.execute(
            """
            INSERT INTO document_chunks (chunk_id, text, metadata)
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE
                SET text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata
            """,
            (chunk_id, text, json.dumps(metadata)),
        )
        inserted += 1
        print(f"  ✓ {chunk_id}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nSeeded {inserted} chunks. Embeddings not included.")
    print("Run embed_corpus.py to add embeddings for vector search.")
    print("BM25 (tsvector) search works immediately without embeddings.")


if __name__ == "__main__":
    print("Seeding document_chunks...")
    main()
