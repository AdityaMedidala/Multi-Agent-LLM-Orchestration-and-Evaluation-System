from __future__ import annotations

import json
import logging
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import AgentResult, BaseAgent, count_tokens
from app.schemas.context import ProvenanceEntry, SharedContext

log = logging.getLogger(__name__)

_SYSTEM = """\
IMPORTANT: When the RAG agent reports that context is unavailable or \
the query is underspecified, do NOT simply say 'I cannot answer'. \
Instead:
- If the query is ambiguous or underspecified: acknowledge what \
  information is missing, explain what would be needed to answer, \
  and use words like 'depends', 'specify', 'which', 'clarify'. \
  Example: 'This depends on which version you are referring to. \
  Please specify the software and version to get a useful answer.'
- If context is genuinely absent: say what you would need to answer \
  and suggest the user clarify or provide more context.
- Never produce a one-line refusal. Always engage with the ambiguity.

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
}

If no agent has provided retrieved context, acknowledge this clearly \
and provide a direct answer from your own knowledge, clearly marking \
it as not from retrieved sources. Never return an empty response.\
"""


class SynthesisAgent(BaseAgent):
    agent_id = "synthesis"

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Budget check (pre-flight estimate) ─────────────────────────────
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
        rag_output    = ctx.agent_outputs.get("rag", {})
        decomp_output = ctx.agent_outputs.get("decomposition", {})
        disagreements = [c for c in ctx.critique_claims if c.disagreement]
        agreements    = [c for c in ctx.critique_claims if not c.disagreement]

        # ── 3. Contradiction summary ──────────────────────────────────────────
        contradiction_block = (
            "\n".join(
                f"- FLAGGED: '{c.claim_text}' (from {c.source_agent}): {c.justification}"
                for c in disagreements
            )
            if disagreements else "No contradictions flagged."
        )

        # ── 4. Build user message ─────────────────────────────────────────────
        rag_answer        = rag_output.get("answer", "")
        rag_citations     = rag_output.get("citations", [])
        decomp_reasoning  = decomp_output.get("reasoning", "")
        subtasks          = decomp_output.get("subtasks", [])

        agreed_block = (
            "\n".join(
                f"- SUPPORTED: '{c.claim_text}' (from {c.source_agent})"
                for c in agreements
            )
            if agreements else "No explicitly supported claims."
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
            # ── LLM call (streaming) ──────────────────────────────────────────
            from app.agents.prompt_registry import get_active_prompt
            from app.streaming import publish_event
            system = get_active_prompt("synthesis", _SYSTEM)
            raw = ""
            await publish_event(ctx.job_id, "agent_start", {
                "agent": self.agent_id, "ts": time.monotonic()
            })
            async for chunk in self._llm.astream(
                [("system", system), ("human", user_msg)]
            ):
                token = chunk.content
                if token:
                    raw += token
                    await publish_event(ctx.job_id, "token", {
                        "agent": self.agent_id, "token": token
                    })
            raw = raw.strip()

            # ── Actual token count (tiktoken) ─────────────────────────────────
            tokens_used = count_tokens(system + user_msg + raw)
            self._update_budget_actual(ctx, tokens_used)

            await publish_event(ctx.job_id, "agent_done", {
                "agent": self.agent_id, "tokens": tokens_used
            })

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

        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            output=ctx.agent_outputs[self.agent_id],
            tokens_used=tokens_used,
            latency_ms=int((time.monotonic() - start) * 1000),
        )