from __future__ import annotations

import json
import logging
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import AgentResult, BaseAgent
from app.schemas.context import CritiqueClaim, SharedContext, ToolCallRecord

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a critique agent. You review outputs from other AI agents and identify \
specific claims that may be incorrect, unsupported, or contradictory. You MUST \
flag specific text spans — not whole outputs.

For each agent output provided, identify up to 3 specific claims.
For each claim assign a confidence score (0.0-1.0) of how confident you are the \
claim is CORRECT. Flag disagreement=true if you believe the claim is wrong or \
unsupported.

Return ONLY valid JSON:
{
  "claims": [
    {
      "claim_text": "exact span of text from the agent output",
      "confidence": 0.85,
      "disagreement": false,
      "source_agent": "rag",
      "justification": "This claim is well-supported because..."
    }
  ],
  "overall_agreement_rate": 0.75
}\
"""


class CritiqueAgent(BaseAgent):
    agent_id = "critique"

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Budget check ───────────────────────────────────────────────────
        if not self._check_budget(ctx, 600):
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=0,
                error="budget_exceeded",
            )

        # ── 2. Collect other agents' outputs ──────────────────────────────────
        outputs_to_review = {
            agent_id: output
            for agent_id, output in ctx.agent_outputs.items()
            if agent_id != "critique"
        }
        if not outputs_to_review:
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=0,
                error="no_outputs_to_review",
            )

        # ── 3. Self-reflection: detect intra-agent contradictions ────────────
        from app.tools.registry import call_tool
        from app.agents.tool_persistence import _persist_tool_call
        contradictions_summary: list[dict] = []
        _ctx_dict = {"agent_outputs": ctx.agent_outputs}
        for _aid in list(outputs_to_review.keys()):
            _sr = await call_tool(
                "self_reflection",
                {"job_id": ctx.job_id, "agent_id": _aid, "ctx_dict": _ctx_dict},
            )
            ctx.tool_call_log.append(
                ToolCallRecord(
                    tool_name="self_reflection",
                    attempt=1,
                    input={"job_id": ctx.job_id, "agent_id": _aid},
                    output=_sr.output,
                    latency_ms=_sr.latency_ms,
                    accepted=_sr.success,
                    failure_reason=_sr.failure_reason,
                )
            )
            await _persist_tool_call(
                job_id=ctx.job_id,
                tool_name="self_reflection",
                agent_id="critique",
                attempt=1,
                input_payload={"job_id": ctx.job_id, "agent_id": _aid},
                output_payload=_sr.output,
                latency_ms=_sr.latency_ms,
                accepted=_sr.success,
                failure_reason=_sr.failure_reason,
            )
            if _sr.success:
                contradictions_summary.extend(_sr.output.get("contradictions", []))

        # ── 4. Single LLM call ────────────────────────────────────────────────
        formatted_outputs = "\n\n---\n".join(
            f"Agent: {aid}\nOutput: {json.dumps(out)[:1000]}"
            for aid, out in outputs_to_review.items()
        )
        if contradictions_summary:
            formatted_outputs += (
                "\n\n---\nSelf-reflection detected these contradictions:\n"
                + json.dumps(contradictions_summary, indent=2)
            )
        try:
            from app.agents.prompt_registry import get_active_prompt
            from app.streaming import publish_event
            system = get_active_prompt("critique", _SYSTEM)
            raw = ""
            await publish_event(ctx.job_id, "agent_start", {
                "agent": self.agent_id, "ts": time.monotonic()
            })
            async for chunk in self._llm.astream(
                [
                    ("system", system),
                    ("human", f"Review these agent outputs:\n{formatted_outputs}"),
                ]
            ):
                token = chunk.content
                if token:
                    raw += token
                    await publish_event(ctx.job_id, "token", {
                        "agent": self.agent_id, "token": token
                    })
            await publish_event(ctx.job_id, "agent_done", {
                "agent": self.agent_id, "tokens": len(raw)
            })
            raw = raw.strip()

            # ── 4. Parse JSON (strip markdown fences if present) ──────────────
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

        # ── 5. Convert to CritiqueClaim objects and append to ctx ─────────────
        for claim_dict in data.get("claims", []):
            try:
                ctx.critique_claims.append(CritiqueClaim(**claim_dict))
            except Exception:
                log.warning("critique: skipping malformed claim: %s", claim_dict)

        # ── 6. Store in ctx.agent_outputs ─────────────────────────────────────
        ctx.agent_outputs[self.agent_id] = {
            "claims_reviewed": len(ctx.critique_claims),
            "disagreement_count": sum(
                1 for c in ctx.critique_claims if c.disagreement
            ),
            "overall_agreement_rate": data.get("overall_agreement_rate", 1.0),
            "agents_reviewed": list(outputs_to_review.keys()),
        }

        # ── 7. Return result ──────────────────────────────────────────────────
        tokens_used = len(raw.split())  # estimate; astream has no usage_metadata

        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            output=ctx.agent_outputs[self.agent_id],
            tokens_used=tokens_used,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
