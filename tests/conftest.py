"""Shared fixtures for mega-ai tests."""
from __future__ import annotations

import pytest

from app.schemas.context import (
    AgentBudget,
    CritiqueClaim,
    ProvenanceEntry,
    SharedContext,
    SubTask,
    ToolCallRecord,
)


@pytest.fixture
def ctx() -> SharedContext:
    """A fresh SharedContext with a dummy job_id and query."""
    return SharedContext(
        job_id="test-job-001",
        original_query="What is binary search?",
    )


@pytest.fixture
def ctx_with_agents(ctx: SharedContext) -> SharedContext:
    """SharedContext pre-populated with typical agent outputs."""
    ctx.agent_outputs["decomposition"] = {
        "subtasks": [
            {"task_id": "t1", "task_type": "retrieve", "description": "look up binary search"},
        ],
        "reasoning": "Simple factual query needs one retrieval pass",
        "subtask_count": 1,
    }
    ctx.agent_outputs["rag"] = {
        "answer": "Binary search is O(log n) [c1].",
        "citations": [
            {"chunk_id": "c1", "text_snippet": "Binary search...", "hop": 1, "relevance_score": 0.95},
        ],
        "hop1_chunks": 5,
        "hop2_chunks": 3,
    }
    ctx.provenance_map.append(
        ProvenanceEntry(sentence="Binary search is O(log n).", source_agent="rag", source_chunk_id="c1")
    )
    ctx.critique_claims.append(
        CritiqueClaim(
            claim_text="Binary search is O(log n)",
            confidence=0.95,
            disagreement=False,
            source_agent="rag",
            justification="Well-supported by corpus chunk c1",
        )
    )
    ctx.agent_outputs["critique"] = {
        "claims_reviewed": 1,
        "disagreement_count": 0,
        "overall_agreement_rate": 0.95,
        "agents_reviewed": ["decomposition", "rag"],
    }
    ctx.agent_outputs["synthesis"] = {
        "final_answer": "Binary search runs in O(log n) time.",
        "provenance_entries": 1,
        "contradictions_resolved": 0,
        "synthesis_reasoning": "No contradictions; merged RAG output directly.",
    }
    ctx.final_answer = "Binary search runs in O(log n) time."
    return ctx
