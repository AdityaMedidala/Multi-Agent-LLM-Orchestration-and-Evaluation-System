from __future__ import annotations

import json
import logging

import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI

from app.schemas.context import SharedContext
from app.streaming import publish_event

log = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

_COMPRESS_SYSTEM = """\
You are a context compression agent. Your job is to shorten verbose agent outputs
to free token budget while preserving all structured data exactly.

Rules:
- NEVER change or omit: chunk_ids, scores, citations, JSON keys, numbers, URLs.
- You MAY shorten: prose reasoning, repeated phrasing, redundant explanation.
- Return a JSON object with the same top-level keys as the input, just more concise values.
- Return ONLY valid JSON, no markdown fences.\
"""


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


async def compress_context_if_needed(
    ctx: SharedContext,
    agent_id: str,
    budget: int,
    threshold: float = 0.75,
) -> None:
    """
    If the serialised agent_outputs exceed `threshold * budget` tokens,
    compress the lossy-safe portions (prose reasoning) via an LLM call
    and write the result back to ctx.metadata["compressed_context"].

    Structured data (RAG citations, critique claims) is never compressed.
    Lossless fields are passed through untouched.
    """
    serialized = json.dumps(ctx.agent_outputs, default=str)
    current_tokens = _token_count(serialized)
    ceiling = int(budget * threshold)

    if current_tokens <= ceiling:
        return  # plenty of room — nothing to do

    log.info(
        "compression: %d tokens > %d ceiling for agent %s — compressing",
        current_tokens, ceiling, agent_id,
    )

    # Split into lossless (preserve verbatim) and lossy (compress) buckets.
    # RAG output carries chunk citations — always lossless.
    lossless_keys = {"rag", "critique"}
    lossy_outputs = {
        k: v for k, v in ctx.agent_outputs.items() if k not in lossless_keys
    }

    if not lossy_outputs:
        log.warning("compression: nothing safe to compress — skipping")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    try:
        response = await llm.ainvoke([
            ("system", _COMPRESS_SYSTEM),
            ("human", json.dumps(lossy_outputs, default=str)),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        compressed: dict = json.loads(raw)

        # Merge compressed lossy outputs back — lossless keys untouched
        for key, value in compressed.items():
            if key in ctx.agent_outputs and key not in lossless_keys:
                ctx.agent_outputs[key] = value

        ctx.metadata["compression_triggered"] = True
        ctx.metadata["compression_tokens_before"] = current_tokens
        ctx.metadata["compression_tokens_after"] = _token_count(
            json.dumps(ctx.agent_outputs, default=str)
        )
        log.info(
            "compression: %d → %d tokens",
            current_tokens,
            ctx.metadata["compression_tokens_after"],
        )
        await publish_event(ctx.job_id, "budget_update", {
            "agent": agent_id,
            "event": "compression_triggered",
            "tokens_before": current_tokens,
            "tokens_after": ctx.metadata["compression_tokens_after"],
        })

    except Exception as exc:
        log.warning("compression: failed (%s) — continuing without compression", exc)