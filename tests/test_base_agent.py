"""Tests for BaseAgent budget checking and token counting."""
from __future__ import annotations

import pytest

from app.agents.base import BaseAgent, AgentResult, count_tokens
from app.schemas.context import SharedContext


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_longer_than_word_split(self):
        """Tiktoken should give a higher count than naive word split for JSON-like text."""
        text = '{"chunk_id": "c1", "score": 0.923, "text": "Binary search..."}'
        token_count = count_tokens(text)
        word_count = len(text.split())
        assert token_count > word_count

    def test_consistent_results(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert count_tokens(text) == count_tokens(text)


class TestBaseAgentBudget:
    def setup_method(self):
        self.agent = BaseAgent()
        self.agent.agent_id = "test_agent"

    def test_check_budget_no_budget_set(self):
        ctx = SharedContext(job_id="j1", original_query="test")
        # No budget = no enforcement = always passes
        assert self.agent._check_budget(ctx, 99999) is True

    def test_check_budget_within_limit(self):
        ctx = SharedContext(job_id="j1", original_query="test")
        ctx.get_budget("test_agent", 4000)
        assert self.agent._check_budget(ctx, 2000) is True
        assert ctx.budgets["test_agent"].used_tokens == 2000

    def test_check_budget_exceeds_limit(self):
        ctx = SharedContext(job_id="j1", original_query="test")
        ctx.get_budget("test_agent", 4000)
        assert self.agent._check_budget(ctx, 5000) is False
        assert ctx.budgets["test_agent"].violated is True

    def test_update_budget_actual_corrects_estimate(self):
        ctx = SharedContext(job_id="j1", original_query="test")
        ctx.get_budget("test_agent", 4000)
        self.agent._check_budget(ctx, 1000)  # pre-flight estimate
        self.agent._update_budget_actual(ctx, 1500)  # actual was higher
        assert ctx.budgets["test_agent"].used_tokens == 1500

    def test_update_budget_actual_marks_violation(self):
        ctx = SharedContext(job_id="j1", original_query="test")
        ctx.get_budget("test_agent", 4000)
        self.agent._check_budget(ctx, 2000)
        self.agent._update_budget_actual(ctx, 5000)  # actual exceeded max
        assert ctx.budgets["test_agent"].violated is True

    def test_update_budget_actual_no_budget(self):
        ctx = SharedContext(job_id="j1", original_query="test")
        # Should not raise
        self.agent._update_budget_actual(ctx, 1000)


class TestAgentResult:
    def test_creation(self):
        r = AgentResult(
            agent_id="rag",
            success=True,
            output={"answer": "test"},
            tokens_used=500,
            latency_ms=1200,
        )
        assert r.agent_id == "rag"
        assert r.error is None

    def test_with_error(self):
        r = AgentResult(
            agent_id="rag",
            success=False,
            output={},
            tokens_used=0,
            latency_ms=0,
            error="LLM call failed",
        )
        assert r.success is False
        assert "LLM" in r.error
