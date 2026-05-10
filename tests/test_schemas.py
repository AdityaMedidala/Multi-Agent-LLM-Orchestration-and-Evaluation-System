"""Tests for SharedContext, AgentBudget, SubTask, and related Pydantic schemas."""
from __future__ import annotations

import json

import pytest

from app.schemas.context import AgentBudget, SharedContext, SubTask


# ── AgentBudget ───────────────────────────────────────────────────────────────

class TestAgentBudget:
    def test_remaining_fresh(self):
        b = AgentBudget(agent_id="rag", max_tokens=4000)
        assert b.remaining() == 4000
        assert b.used_tokens == 0
        assert b.violated is False

    def test_consume_within_budget(self):
        b = AgentBudget(agent_id="rag", max_tokens=4000)
        assert b.consume(1000) is True
        assert b.used_tokens == 1000
        assert b.remaining() == 3000
        assert b.violated is False

    def test_consume_exact_budget(self):
        b = AgentBudget(agent_id="rag", max_tokens=4000)
        assert b.consume(4000) is True
        assert b.remaining() == 0
        assert b.violated is False

    def test_consume_over_budget(self):
        b = AgentBudget(agent_id="rag", max_tokens=4000)
        assert b.consume(4001) is False
        assert b.violated is True
        # used_tokens should NOT be incremented on violation
        assert b.used_tokens == 0

    def test_consume_multiple_calls(self):
        b = AgentBudget(agent_id="rag", max_tokens=4000)
        assert b.consume(2000) is True
        assert b.consume(1500) is True
        assert b.consume(600) is False  # 2000 + 1500 + 600 = 4100 > 4000
        assert b.violated is True
        assert b.used_tokens == 3500  # only the first two were applied

    def test_remaining_never_negative(self):
        b = AgentBudget(agent_id="rag", max_tokens=100, used_tokens=200)
        assert b.remaining() == 0


# ── SharedContext ─────────────────────────────────────────────────────────────

class TestSharedContext:
    def test_creation(self, ctx):
        assert ctx.job_id == "test-job-001"
        assert ctx.original_query == "What is binary search?"
        assert ctx.subtasks == []
        assert ctx.tool_call_log == []
        assert ctx.agent_outputs == {}
        assert ctx.critique_claims == []
        assert ctx.provenance_map == []
        assert ctx.budgets == {}
        assert ctx.final_answer is None

    def test_get_budget_creates_new(self, ctx):
        b = ctx.get_budget("rag", 4000)
        assert b.agent_id == "rag"
        assert b.max_tokens == 4000
        assert "rag" in ctx.budgets

    def test_get_budget_returns_existing(self, ctx):
        b1 = ctx.get_budget("rag", 4000)
        b1.consume(1000)
        b2 = ctx.get_budget("rag", 8000)  # max_tokens ignored for existing
        assert b2.max_tokens == 4000  # original budget preserved
        assert b2.used_tokens == 1000
        assert b1 is b2

    def test_log_routing(self, ctx):
        ctx.log_routing("step 1: decomposition")
        ctx.log_routing("step 2: rag")
        assert len(ctx.orchestrator_reasoning) == 2
        assert "decomposition" in ctx.orchestrator_reasoning[0]

    def test_model_dump_serializable(self, ctx):
        ctx.log_routing("test reasoning")
        ctx.get_budget("rag", 4000)
        dump = ctx.model_dump()
        # Must be JSON-serializable for DB storage
        serialized = json.dumps(dump, default=str)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_model_dump_with_agents(self, ctx_with_agents):
        dump = ctx_with_agents.model_dump()
        assert "rag" in dump["agent_outputs"]
        assert "synthesis" in dump["agent_outputs"]
        serialized = json.dumps(dump, default=str)
        assert "Binary search" in serialized


# ── SubTask ───────────────────────────────────────────────────────────────────

class TestSubTask:
    def test_creation(self):
        st = SubTask(
            task_id="t1",
            task_type="retrieve",
            description="Find binary search info",
            dependencies=[],
        )
        assert st.status == "pending"
        assert st.result is None

    def test_with_dependencies(self):
        st = SubTask(
            task_id="t3",
            task_type="synthesize",
            description="Merge results",
            dependencies=["t1", "t2"],
        )
        assert st.dependencies == ["t1", "t2"]

    def test_status_update(self):
        st = SubTask(
            task_id="t1",
            task_type="retrieve",
            description="Find info",
            dependencies=[],
        )
        st.status = "done"
        st.result = {"summary": "Found relevant chunks"}
        assert st.status == "done"
        assert st.result["summary"] == "Found relevant chunks"

    def test_invalid_task_type_rejected(self):
        with pytest.raises(Exception):
            SubTask(
                task_id="t1",
                task_type="invalid_type",
                description="Bad type",
                dependencies=[],
            )
