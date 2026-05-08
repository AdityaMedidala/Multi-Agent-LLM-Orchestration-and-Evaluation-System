from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent
from app.schemas.context import SharedContext


class RAGAgent(BaseAgent):
    agent_id = "rag"

    async def run(self, ctx: SharedContext) -> AgentResult:
        self._check_budget(ctx, 100)
        ctx.agent_outputs[self.agent_id] = {"status": "stub"}
        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            output={"status": "stub"},
            tokens_used=100,
            latency_ms=0,
        )
