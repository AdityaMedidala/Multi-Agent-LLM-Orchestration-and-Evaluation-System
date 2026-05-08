from __future__ import annotations

import json
import logging
import time

from langchain_anthropic import ChatAnthropic

from app.agents.base import AgentResult, BaseAgent
from app.schemas.context import ProvenanceEntry, SharedContext

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a synthesis agent. Your job is to produce a final, coherent answer by \
merging outputs from multiple AI agents. You must:
1. Resolve any contradictions flagged by the critique agent
2. Produce a final answer where every sentence is attributable to a source agent
3. Return a provenance map linking each sentence to its source

Return ONLY valid JSON:
{
  "final_answer": "The complete synthesized answer as prose.",
  "provenance": [
    {
      "sentence": "exact sentence from final_answer",
      "source_agent": "rag|decomposition|synthesis",
      "source_chunk_id": "chunk_id or null",
      "resolution": "kept|modified|rejected",
      "resolution_reason": "why this sentence was kept/modified/rejected"
    }
  ],
  "contradictions_resolved": 0,
  "synthesis_reasoning": "one sentence explaining how you merged the outputs"
}\
"""


class SynthesisAgent(BaseAgent):
    agent_id = "synthesis"

    def __init__(self) -> None:
        self._llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Budget check ───────────────────────────────────────────────────
        if not self._check_budget(ctx, 700):
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=0,
                error="budget_exceeded",
            )

        # ── 2. Collect inputs ─────────────────────────────────────────────────
        rag_output   = ctx.agent_outputs.get("rag", {})
        decomp_output = ctx.agent_outputs.get("decomposition", {})
        disagreements = [c for c in ctx.critique_claims if c.disagreement]
        agreements    = [c for c in ctx.critique_claims if not c.disagreement]

        # ── 3. Contradiction summary ──────────────────────────────────────────
        if disagreements:
            contradiction_block = "\n".join(
                f"- FLAGGED: '{c.claim_text}' (from {c.source_agent}): {c.justification}"
                for c in disagreements
            )
        else:
            contradiction_block = "No contradictions flagged."

        # ── 4. Build user message ─────────────────────────────────────────────
        rag_answer   = rag_output.get("answer", "")
        rag_citations = rag_output.get("citations", [])
        decomp_reasoning = decomp_output.get("reasoning", "")
        subtasks     = decomp_output.get("subtasks", [])

        agreed_block = (
            "\n".join(
                f"- SUPPORTED: '{c.claim_text}' (from {c.source_agent})"
                for c in agreements
            )
            if agreements
            else "No explicitly supported claims."
        )

        citations_block = "\n".join(
            f"  [{cit['chunk_id']}] (hop {cit['hop']}, score {cit['relevance_score']:.3f}): "
            f"{cit['text_snippet']}"
            for cit in rag_citations[:5]
        ) if rag_citations else "No citations available."

        user_msg = (
            f"Original query: {ctx.original_query}\n\n"
            f"=== RAG Agent Answer ===\n{rag_answer}\n\n"
            f"=== RAG Citations ===\n{citations_block}\n\n"
            f"=== Decomposition Reasoning ===\n{decomp_reasoning}\n"
            f"Subtasks: {json.dumps(subtasks, default=str)[:600]}\n\n"
            f"=== Critique: Flagged Contradictions ===\n{contradiction_block}\n\n"
            f"=== Critique: Supported Claims ===\n{agreed_block}\n\n"
            "Synthesize all of the above into a final answer."
        )

        try:
            # ── LLM call ──────────────────────────────────────────────────────
            response = await self._llm.ainvoke(
                [("system", _SYSTEM), ("human", user_msg)]
            )
            raw: str = response.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data: dict = json.loads(raw)

        except json.JSONDecodeError:
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=int((time.monotonic() - start) * 1000),
                error="parse_failed",
            )
        except Exception as exc:
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=int((time.monotonic() - start) * 1000),
                error=str(exc),
            )

        # ── 5. Write final answer to ctx ──────────────────────────────────────
        ctx.final_answer = data.get("final_answer", "")

        # ── 6. Rebuild provenance map from synthesis output ───────────────────
        # Replace any RAG-level entries with the richer synthesis provenance
        ctx.provenance_map.clear()
        for entry in data.get("provenance", []):
            try:
                ctx.provenance_map.append(
                    ProvenanceEntry(
                        sentence=entry.get("sentence", ""),
                        source_agent=entry.get("source_agent", "synthesis"),
                        source_chunk_id=entry.get("source_chunk_id") or None,
                        tool_calls_used=[],
                    )
                )
            except Exception:
                log.warning("synthesis: skipping malformed provenance entry: %s", entry)

        # ── 7. Store in ctx.agent_outputs ─────────────────────────────────────
        ctx.agent_outputs[self.agent_id] = {
            "final_answer": ctx.final_answer,
            "provenance_entries": len(ctx.provenance_map),
            "contradictions_resolved": data.get("contradictions_resolved", 0),
            "synthesis_reasoning": data.get("synthesis_reasoning", ""),
            "agents_merged": [
                aid for aid in ctx.agent_outputs if aid != "synthesis"
            ],
        }

        # ── 8. Return result ──────────────────────────────────────────────────
        usage = response.usage_metadata or {}
        tokens_used = (
            usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        )

        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            output=ctx.agent_outputs[self.agent_id],
            tokens_used=tokens_used,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
