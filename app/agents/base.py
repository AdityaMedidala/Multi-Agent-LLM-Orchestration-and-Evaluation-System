from __future__ import annotations

import tiktoken
from pydantic import BaseModel

from app.schemas.context import SharedContext

# Shared encoder — cl100k_base is a good proxy for Gemini token counts.
# (Gemini has no public tiktoken encoding; cl100k_base is within ~5% for English.)
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_enc.encode(text))


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
        """
        Pre-flight budget gate. Charges `tokens_needed` as an estimate.
        Returns False (and marks violated) if the agent is already over budget.
        Always follow with _update_budget_actual() after the LLM call completes.
        """
        budget = ctx.budgets.get(self.agent_id)
        if not budget:
            return True  # no budget set = no enforcement
        if not budget.consume(tokens_needed):
            budget.violated = True
            return False
        return True

    def _update_budget_actual(self, ctx: SharedContext, actual_tokens: int) -> None:
        """
        Post-call correction: replace the pre-flight estimate with real usage.
        If actual usage exceeds max_tokens, marks the budget as violated and logs
        a policy violation (the orchestrator reads budget.violated when writing
        the agent_log row).

        Call this with the tiktoken count of the full LLM response after streaming
        completes, before returning the AgentResult.
        """
        budget = ctx.budgets.get(self.agent_id)
        if not budget:
            return
        # Replace the pre-charged estimate with the actual figure.
        # We take the max so we never accidentally reduce used_tokens if the
        # pre-charge happened to be higher (shouldn't happen, but defensive).
        budget.used_tokens = max(budget.used_tokens, actual_tokens)
        if budget.used_tokens > budget.max_tokens and not budget.violated:
            budget.violated = True