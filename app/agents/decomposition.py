from __future__ import annotations

import json
import logging
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import AgentResult, BaseAgent, count_tokens
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
        self._llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        # ── 1. Budget check (pre-flight estimate) ─────────────────────────────
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
            # ── 2. LLM call (streaming) ───────────────────────────────────────
            from app.agents.prompt_registry import get_active_prompt
            from app.streaming import publish_event
            system = get_active_prompt("decomposition", _SYSTEM)
            raw = ""
            await publish_event(ctx.job_id, "agent_start", {
                "agent": self.agent_id, "ts": time.monotonic()
            })
            async for chunk in self._llm.astream(
                [("system", system), ("human", ctx.original_query)]
            ):
                token = chunk.content
                if token:
                    raw += token
                    await publish_event(ctx.job_id, "token", {
                        "agent": self.agent_id, "token": token
                    })
            raw = raw.strip()

            # ── 3. Actual token count (tiktoken, not word-split) ──────────────
            tokens_used = count_tokens(system + ctx.original_query + raw)
            self._update_budget_actual(ctx, tokens_used)

            await publish_event(ctx.job_id, "agent_done", {
                "agent": self.agent_id, "tokens": tokens_used
            })

            # ── 4. Parse JSON (strip markdown fences if present) ──────────────
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data: dict = json.loads(raw)

            subtask_dicts: list[dict] = data.get("subtasks", [])
            reasoning: str = data.get("decomposition_reasoning", "")

            # ── 5. Validate dependencies ──────────────────────────────────────
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

            # ── 6. Materialise SubTask objects into ctx ───────────────────────
            ctx.subtasks = [SubTask(**s) for s in subtask_dicts]

            # ── 7. Write agent output ─────────────────────────────────────────
            ctx.agent_outputs[self.agent_id] = {
                "subtasks": [s.model_dump() for s in ctx.subtasks],
                "reasoning": reasoning,
                "subtask_count": len(ctx.subtasks),
            }

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