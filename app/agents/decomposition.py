from __future__ import annotations

import json
import logging
import time

from langchain_anthropic import ChatAnthropic

from app.agents.base import AgentResult, BaseAgent
from app.schemas.context import SharedContext, SubTask

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a query decomposition agent. Break the user query into 2-4 \
typed sub-tasks. Each sub-task must have:
- a unique task_id (e.g. t1, t2, t3)
- a task_type: one of decompose, retrieve, critique, synthesize
- a description: what exactly needs to be done
- dependencies: list of task_ids that must complete before this one \
  (empty list if no dependencies)

Return ONLY valid JSON, no markdown:
{
  "subtasks": [
    {
      "task_id": "t1",
      "task_type": "retrieve",
      "description": "...",
      "dependencies": []
    },
    {
      "task_id": "t2",
      "task_type": "critique",
      "description": "...",
      "dependencies": ["t1"]
    }
  ],
  "decomposition_reasoning": "one sentence explaining the breakdown"
}\
"""


class DecompositionAgent(BaseAgent):
    agent_id = "decomposition"

    def __init__(self) -> None:
        self._llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Budget check ───────────────────────────────────────────────────
        if not self._check_budget(ctx, 500):
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                output={},
                tokens_used=0,
                latency_ms=0,
                error="budget_exceeded",
            )

        try:
            # ── 2. LLM call ───────────────────────────────────────────────────
            response = await self._llm.ainvoke(
                [("system", _SYSTEM), ("human", ctx.original_query)]
            )
            raw: str = response.content.strip()

            # ── 3. Parse JSON (strip markdown fences if present) ──────────────
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data: dict = json.loads(raw)

            subtask_dicts: list[dict] = data.get("subtasks", [])
            reasoning: str = data.get("decomposition_reasoning", "")

            # ── 4. Validate dependencies ──────────────────────────────────────
            known_ids = {s["task_id"] for s in subtask_dicts}
            for s in subtask_dicts:
                bad = [d for d in s.get("dependencies", []) if d not in known_ids]
                if bad:
                    log.warning(
                        "decomposition: dropping unknown dependencies %s from %s",
                        bad,
                        s["task_id"],
                    )
                    s["dependencies"] = [
                        d for d in s["dependencies"] if d in known_ids
                    ]

            # ── 5. Materialise SubTask objects into ctx ───────────────────────
            ctx.subtasks = [SubTask(**s) for s in subtask_dicts]

            # ── 6. Write agent output ─────────────────────────────────────────
            ctx.agent_outputs[self.agent_id] = {
                "subtasks": [s.model_dump() for s in ctx.subtasks],
                "reasoning": reasoning,
                "subtask_count": len(ctx.subtasks),
            }

            # ── 7. Build AgentResult ──────────────────────────────────────────
            usage = response.usage_metadata or {}
            tokens_used = (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            return AgentResult(
                agent_id=self.agent_id,
                success=True,
                output=ctx.agent_outputs[self.agent_id],
                tokens_used=tokens_used,
                latency_ms=elapsed_ms,
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
