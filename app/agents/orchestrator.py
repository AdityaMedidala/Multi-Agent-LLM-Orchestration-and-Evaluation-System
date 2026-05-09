from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
import uuid

import psycopg2
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import AgentResult, BaseAgent
from app.agents.critique import CritiqueAgent
from app.agents.decomposition import DecompositionAgent
from app.agents.rag import RAGAgent
from app.agents.synthesis import SynthesisAgent
from app.config import settings
from app.schemas.context import AgentBudget, SharedContext

log = logging.getLogger(__name__)

AGENT_REGISTRY: dict[str, BaseAgent] = {
    "decomposition": DecompositionAgent(),
    "rag": RAGAgent(),
    "critique": CritiqueAgent(),
    "synthesis": SynthesisAgent(),
}

_ROUTING_SYSTEM = """\
You are an orchestration planner for a multi-agent research system.
Given a user query, decide which agents to invoke and in what order.

Available agents:
- decomposition: breaks the query into structured sub-tasks
- rag: retrieves relevant documents and evidence
- critique: fact-checks and flags unsupported claims
- synthesis: assembles a final, coherent answer

Return ONLY valid JSON in this exact shape — no prose, no markdown:
{
  "routing_plan": [
    {"agent": "<agent_name>", "reason": "<one sentence>", "budget_tokens": <int>},
    ...
  ],
  "orchestrator_reasoning": "<one sentence explaining the overall plan>"
}
"""

_ROUTING_USER = """\
Query: {query}

Produce the routing plan JSON now.
"""


# ── Agent-log persistence (sync DB write, run via executor) ──────────────────

def _write_agent_log_sync(
    job_id: str,
    agent_id: str,
    event_type: str,
    input_payload: dict,
    output_payload: dict,
    latency_ms: int,
    token_count: int,
    policy_violation: bool = False,
) -> None:
    """Write one AgentLog row synchronously (called via run_in_executor)."""
    def _hash(d: dict) -> str:
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    # Normalise job_id: the column is UUID — convert non-UUID strings
    # (e.g. "log-test-1") to a deterministic UUID so the INSERT never
    # fails on a cast error.  In production job_id is always a real UUID.
    try:
        job_uuid_str = str(uuid.UUID(job_id))
    except ValueError:
        job_uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, job_id))

    try:
        conn = psycopg2.connect(settings.database_url_sync)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO agent_logs
              (id, job_id, agent_id, event_type,
               input_hash, output_hash, latency_ms,
               token_count, payload, policy_violation)
            VALUES (%s,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                str(uuid.uuid4()), job_uuid_str, agent_id, event_type,
                _hash(input_payload), _hash(output_payload),
                latency_ms, token_count,
                json.dumps(
                    {"input": input_payload, "output": output_payload},
                    default=str,
                ),
                policy_violation,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        # Non-fatal — logging failure must never abort the pipeline
        log.warning("agent_log write failed: %s", exc)


async def _log_agent_event(
    job_id: str,
    agent_id: str,
    event_type: str,
    input_payload: dict,
    output_payload: dict,
    latency_ms: int,
    token_count: int,
    policy_violation: bool = False,
) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        functools.partial(
            _write_agent_log_sync,
            job_id, agent_id, event_type,
            input_payload, output_payload,
            latency_ms, token_count, policy_violation,
        ),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    agent_id = "orchestrator"

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Ask the LLM for a routing plan ────────────────────────────────
        messages = [
            ("system", _ROUTING_SYSTEM),
            ("human", _ROUTING_USER.format(query=ctx.original_query)),
        ]
        routing_plan, overall_reasoning = await self._get_routing_plan(messages)
        ctx.log_routing(f"orchestrator_reasoning: {overall_reasoning}")

        # Log the orchestrator's own routing decision
        await _log_agent_event(
            job_id=ctx.job_id,
            agent_id="orchestrator",
            event_type="routing",
            input_payload={"query": ctx.original_query},
            output_payload={"plan": routing_plan, "reasoning": overall_reasoning},
            latency_ms=0,
            token_count=0,
        )

        # ── 2. Execute agents in the decided order ────────────────────────────
        results: list[AgentResult] = []
        for step in routing_plan:
            agent_name: str = step["agent"]
            reason: str = step.get("reason", "")
            budget = max(step.get("budget_tokens", 4000), 4000)

            ctx.log_routing(f"invoking '{agent_name}': {reason}")

            agent = AGENT_REGISTRY.get(agent_name)
            if agent is None:
                ctx.metadata[f"{agent_name}_error"] = "agent not found in registry"
                ctx.log_routing(f"SKIPPED '{agent_name}': not in registry")
                continue

            # Initialise budget before the agent runs
            ctx.get_budget(agent_name, budget)

            # Log agent start
            await _log_agent_event(
                job_id=ctx.job_id,
                agent_id=agent_name,
                event_type="start",
                input_payload={"query": ctx.original_query, "routing_reason": reason},
                output_payload={},
                latency_ms=0,
                token_count=0,
            )

            try:
                result = await agent.run(ctx)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                ctx.metadata[f"{agent_name}_error"] = error_msg
                ctx.log_routing(f"FAILED '{agent_name}': {error_msg}")

                # Log agent exception
                await _log_agent_event(
                    job_id=ctx.job_id,
                    agent_id=agent_name,
                    event_type="error",
                    input_payload={"query": ctx.original_query},
                    output_payload={"error": str(exc)},
                    latency_ms=0,
                    token_count=0,
                )

                results.append(
                    AgentResult(
                        agent_id=agent_name,
                        success=False,
                        output={},
                        tokens_used=0,
                        latency_ms=0,
                        error=error_msg,
                    )
                )
                continue

            if not result.success:
                ctx.log_routing(f"FAILED '{agent_name}': {result.error}")

            # Log agent output (success or soft failure)
            await _log_agent_event(
                job_id=ctx.job_id,
                agent_id=agent_name,
                event_type="output",
                input_payload={"query": ctx.original_query},
                output_payload=result.output,
                latency_ms=result.latency_ms,
                token_count=result.tokens_used,
                policy_violation=ctx.budgets.get(
                    agent_name,
                    AgentBudget(agent_id=agent_name, max_tokens=0),
                ).violated,
            )

            results.append(result)

        latency_ms = int((time.monotonic() - start) * 1000)
        total_tokens = sum(r.tokens_used for r in results)

        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            output={
                "routing_plan": routing_plan,
                "orchestrator_reasoning": overall_reasoning,
                "agent_results": [r.model_dump() for r in results],
            },
            tokens_used=total_tokens,
            latency_ms=latency_ms,
        )

    async def _get_routing_plan(
        self, messages: list
    ) -> tuple[list[dict], str]:
        """Call the LLM and parse the routing JSON. Falls back to a safe default."""
        try:
            from app.agents.prompt_registry import get_active_prompt
            system = get_active_prompt("orchestrator", _ROUTING_SYSTEM)
            # Rebuild messages with potentially rewritten system prompt
            messages = [
                (role, system if role == "system" else content)
                for role, content in messages
            ]
            response = await self._llm.ainvoke(messages)
            raw = response.content
            # Strip accidental markdown fences
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            plan = data.get("routing_plan", [])
            reasoning = data.get("orchestrator_reasoning", "")
            return plan, reasoning
        except Exception as exc:  # noqa: BLE001
            # Safe fallback: run all agents in default order
            fallback_plan = [
                {"agent": "decomposition", "reason": "fallback", "budget_tokens": 6000},
                {"agent": "rag",           "reason": "fallback", "budget_tokens": 6000},
                {"agent": "critique",      "reason": "fallback", "budget_tokens": 6000},
                {"agent": "synthesis",     "reason": "fallback", "budget_tokens": 6000},
            ]
            return fallback_plan, f"LLM routing failed ({exc}); using default order"
