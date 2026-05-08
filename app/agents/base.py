from __future__ import annotations

from pydantic import BaseModel

from app.schemas.context import SharedContext


class AgentResult(BaseModel):
    agent_id: str
    success: bool
    output: dict          # agent-specific output, always a dict
    tokens_used: int
    latency_ms: int
    error: str | None = None


class BaseAgent:
    agent_id: str         # must be set by subclass

    async def run(self, ctx: SharedContext) -> AgentResult:
        raise NotImplementedError

    def _check_budget(self, ctx: SharedContext, tokens_needed: int) -> bool:
        """Returns False and marks violated if over budget."""
        budget = ctx.budgets.get(self.agent_id)
        if not budget:
            return True  # no budget set = no enforcement
        if not budget.consume(tokens_needed):
            budget.violated = True
            return False
        return True
