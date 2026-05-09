from __future__ import annotations

import asyncio
import functools
import logging
import re
import time

import cohere
import openai
import psycopg2
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import AgentResult, BaseAgent
from app.config import settings
from app.schemas.context import ProvenanceEntry, SharedContext, ToolCallRecord
log = logging.getLogger(__name__)

# ── Retrieval constants (kept identical to scratch/final_retreval.py) ─────────
RRF_K        = 60
RERANK_TOP_N = 10
RERANK_MODEL = "rerank-v3.5"
EMBED_MODEL  = "text-embedding-3-small"
SEARCH_LIMIT = 20

# ── Prompts ───────────────────────────────────────────────────────────────────
_FOLLOWUP_TMPL = """\
Given this query: {original_query}
And these initial retrieved chunks: {top3_texts}
What follow-up question would retrieve the missing context needed to \
fully answer the original query?
Return ONLY the follow-up question, nothing else.\
"""

_ANSWER_SYSTEM = (
    "Answer using ONLY the provided context chunks. "
    "For each claim you make, cite the chunk_id it came from in [chunk_id] format."
)


class RAGAgent(BaseAgent):
    agent_id = "rag"

    def __init__(self) -> None:
        # Guard client init so import-time tests with dummy API keys don't fail.
        try:
            self._openai = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        except Exception:
            self._openai = None  # type: ignore[assignment]

        try:
            self._cohere = cohere.AsyncClient(api_key=settings.cohere_api_key)
        except Exception:
            self._cohere = None  # type: ignore[assignment]

        self._llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # ── Embedding (async) ─────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        response = await self._openai.embeddings.create(
            model=EMBED_MODEL, input=[text.strip()]
        )
        return response.data[0].embedding

    # ── DB search (psycopg2 sync → run in executor) ───────────────────────────

    @staticmethod
    def _db_search_sync(
        query_emb: list[float], query: str, limit: int
    ) -> tuple[list, list]:
        # pgvector expects the embedding as a bracketed string for the ::vector cast
        emb_str = "[" + ",".join(map(str, query_emb)) + "]"
        conn = psycopg2.connect(settings.database_url_sync)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT chunk_id, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                ORDER BY score DESC
                LIMIT %s
                """,
                (emb_str, limit),
            )
            vector_rows = cur.fetchall()

            cur.execute(
                """
                SELECT chunk_id, text, metadata,
                       ts_rank(ts, plainto_tsquery('english', %s)) AS score
                FROM document_chunks
                WHERE ts @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, limit),
            )
            bm25_rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()
        return vector_rows, bm25_rows

    # ── RRF fusion (identical logic to scratch/final_retreval.py) ─────────────

    @staticmethod
    def _rrf_fuse(
        vector_rows: list, bm25_rows: list, limit: int = SEARCH_LIMIT
    ) -> list[tuple[float, dict]]:
        vector_ranks = {row[0]: i + 1 for i, row in enumerate(vector_rows)}
        bm25_ranks   = {row[0]: i + 1 for i, row in enumerate(bm25_rows)}

        all_chunks = {
            row[0]: {"chunk_id": row[0], "text": row[1], "metadata": row[2]}
            for row in vector_rows + bm25_rows
        }

        fused = sorted(
            [
                (
                    1 / (RRF_K + vector_ranks.get(cid, limit + 1))
                    + 1 / (RRF_K + bm25_ranks.get(cid, limit + 1)),
                    item,
                )
                for cid, item in all_chunks.items()
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        return fused[:limit]

    # ── Rerank (async Cohere) ─────────────────────────────────────────────────

    async def _rerank(
        self, query: str, fused: list[tuple[float, dict]]
    ) -> list[tuple[float, dict]]:
        if not fused:
            return fused
        response = await self._cohere.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[item["text"] for _, item in fused],
            top_n=RERANK_TOP_N,
            return_documents=False,
        )
        return [(r.relevance_score, fused[r.index][1]) for r in response.results]

    # ── Single retrieve + rerank pass ─────────────────────────────────────────

    async def _retrieve_and_rerank(
        self, query: str
    ) -> list[tuple[float, dict]]:
        query_emb = await self._embed(query)

        loop = asyncio.get_running_loop()
        vector_rows, bm25_rows = await loop.run_in_executor(
            None,
            functools.partial(self._db_search_sync, query_emb, query, SEARCH_LIMIT),
        )

        fused = self._rrf_fuse(vector_rows, bm25_rows)
        if not fused:
            return []

        return await self._rerank(query, fused)

    # ── Web search fallback ───────────────────────────────────────────────────

    async def _web_search_fallback(
        self, query: str, ctx: SharedContext
    ) -> list[tuple[float, dict]]:
        from app.tools.web_search import web_search as _web_search  # deferred to avoid circular import
        start = time.monotonic()
        tool_result = await _web_search(query, max_results=5)
        latency = int((time.monotonic() - start) * 1000)
        ctx.tool_call_log.append(ToolCallRecord(
            tool_name="web_search",
            attempt=1,
            input={"query": query},
            output=tool_result.output if tool_result.success else {},
            latency_ms=latency,
            accepted=tool_result.success,
            failure_reason=tool_result.failure_reason,
        ))
        if not tool_result.success:
            return []
        results = tool_result.output.get("results", [])
        fused = []
        for i, r in enumerate(results):
            score = max(1.0 - (i * 0.1), 0.1)
            fused.append((score, {
                "chunk_id": f"web_{i}",
                "text": f"{r.get('title', '')}. {r.get('snippet', '')}",
                "metadata": {"source": r.get("url", "web"), "type": "web"},
            }))
        return fused

    # ── Agent entry point ─────────────────────────────────────────────────────

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Budget check ───────────────────────────────────────────────────
        if not self._check_budget(ctx, 800):
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=0,
                error="budget_exceeded",
            )

        try:
            query = ctx.original_query

            # ── 3. Hop 1 — retrieve & rerank ─────────────────────────────────
            hop1 = await self._retrieve_and_rerank(query)

            # ── Web search fallback if corpus has no relevant results ─────────
            top_score = hop1[0][0] if hop1 else 0.0
            if not hop1 or top_score < 0.15:
                web_results = await self._web_search_fallback(query, ctx)
                if web_results:
                    hop1 = web_results + hop1
                    hop1.sort(key=lambda x: x[0], reverse=True)

            # ── 4. Hop 2 — follow-up query then retrieve again ────────────────
            top3_texts = "\n---\n".join(
                item["text"][:300] for _, item in hop1[:3]
            )
            followup_resp = await self._llm.ainvoke(
                [("human", _FOLLOWUP_TMPL.format(
                    original_query=query, top3_texts=top3_texts
                ))]
            )
            follow_up_query = followup_resp.content.strip()

            hop2 = await self._retrieve_and_rerank(follow_up_query)

            # ── Merge & deduplicate by chunk_id, re-sort by score ─────────────
            seen: set[str] = set()
            merged: list[tuple[float, dict, int]] = []  # (score, item, hop)
            for score, item in hop1:
                if item["chunk_id"] not in seen:
                    seen.add(item["chunk_id"])
                    merged.append((score, item, 1))
            for score, item in hop2:
                if item["chunk_id"] not in seen:
                    seen.add(item["chunk_id"])
                    merged.append((score, item, 2))
            merged.sort(key=lambda x: x[0], reverse=True)

            # ── 5. Build citations (top 5) ────────────────────────────────────
            citations = [
                {
                    "chunk_id": item["chunk_id"],
                    "text_snippet": item["text"][:200],
                    "hop": hop,
                    "relevance_score": float(score),
                }
                for score, item, hop in merged[:5]
            ]

            # ── 6. Generate answer ────────────────────────────────────────────
            formatted_chunks = "\n\n".join(
                f"[{item['chunk_id']}]\n{item['text']}"
                for _, item, _ in merged[:5]
            )
            from app.agents.prompt_registry import get_active_prompt
            answer_system = get_active_prompt("rag", _ANSWER_SYSTEM)
            answer_resp = await self._llm.ainvoke(
                [
                    ("system", answer_system),
                    ("human", f"Query: {query}\n\nContext:\n{formatted_chunks}"),
                ]
            )
            answer: str = answer_resp.content.strip()

            # ── 7. Store in ctx ───────────────────────────────────────────────
            ctx.agent_outputs[self.agent_id] = {
                "answer": answer,
                "citations": citations,
                "hop1_chunks": len(hop1),
                "hop2_chunks": len(hop2),
                "follow_up_query": follow_up_query,
            }

            # ── 8. Provenance map — one entry per sentence ────────────────────
            first_chunk_id = citations[0]["chunk_id"] if citations else None
            for sentence in re.split(r"(?<=[.!?])\s+", answer):
                sentence = sentence.strip()
                if sentence:
                    ctx.provenance_map.append(
                        ProvenanceEntry(
                            sentence=sentence,
                            source_agent="rag",
                            source_chunk_id=first_chunk_id,
                        )
                    )

            # ── 9. Return result ──────────────────────────────────────────────
            fu_usage = followup_resp.usage_metadata or {}
            an_usage = answer_resp.usage_metadata or {}
            tokens_used = (
                fu_usage.get("input_tokens", 0)
                + fu_usage.get("output_tokens", 0)
                + an_usage.get("input_tokens", 0)
                + an_usage.get("output_tokens", 0)
            )

            return AgentResult(
                agent_id=self.agent_id,
                success=True,
                output=ctx.agent_outputs[self.agent_id],
                tokens_used=tokens_used,
                latency_ms=int((time.monotonic() - start) * 1000),
            )

        except Exception as exc:
            log.exception("RAGAgent failed for job %s", ctx.job_id)
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=int((time.monotonic() - start) * 1000),
                error=str(exc),
            )
