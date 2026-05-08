from __future__ import annotations

import json
import time

from langchain_anthropic import ChatAnthropic

from app.agents.base import AgentResult, BaseAgent
from app.agents.critique import CritiqueAgent
from app.agents.decomposition import DecompositionAgent
from app.agents.rag import RAGAgent
from app.agents.synthesis import SynthesisAgent
from app.schemas.context import SharedContext

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


class OrchestratorAgent(BaseAgent):
    agent_id = "orchestrator"

    def __init__(self) -> None:
        self._llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Ask the LLM for a routing plan ────────────────────────────────
        messages = [
            ("system", _ROUTING_SYSTEM),
            ("human", _ROUTING_USER.format(query=ctx.original_query)),
        ]
        routing_plan, overall_reasoning = await self._get_routing_plan(messages)
        ctx.log_routing(f"orchestrator_reasoning: {overall_reasoning}")

        # ── 2. Execute agents in the decided order ────────────────────────────
        results: list[AgentResult] = []
        for step in routing_plan:
            agent_name: str = step["agent"]
            reason: str = step.get("reason", "")
            budget_tokens: int = step.get("budget_tokens", 4000)

            ctx.log_routing(f"invoking '{agent_name}': {reason}")

            agent = AGENT_REGISTRY.get(agent_name)
            if agent is None:
                ctx.metadata[f"{agent_name}_error"] = "agent not found in registry"
                ctx.log_routing(f"SKIPPED '{agent_name}': not in registry")
                continue

            # Initialise budget before the agent runs
            ctx.get_budget(agent_name, budget_tokens)

            try:
                result = await agent.run(ctx)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                ctx.metadata[f"{agent_name}_error"] = error_msg
                ctx.log_routing(f"FAILED '{agent_name}': {error_msg}")
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
                {"agent": "decomposition", "reason": "fallback", "budget_tokens": 4000},
                {"agent": "rag",           "reason": "fallback", "budget_tokens": 6000},
                {"agent": "critique",      "reason": "fallback", "budget_tokens": 3000},
                {"agent": "synthesis",     "reason": "fallback", "budget_tokens": 4000},
            ]
            return fallback_plan, f"LLM routing failed ({exc}); using default order"
