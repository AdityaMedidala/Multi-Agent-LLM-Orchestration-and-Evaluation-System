"""Tests for all 6 eval scoring functions."""
from __future__ import annotations

import pytest

from app.eval.runner import (
    DimensionScore,
    score_answer_correctness,
    score_budget_compliance,
    score_citation_accuracy,
    score_contradiction_resolution,
    score_critique_agreement,
    score_tool_efficiency,
)
from app.schemas.context import (
    AgentBudget,
    CritiqueClaim,
    ProvenanceEntry,
    SharedContext,
    ToolCallRecord,
)


# ── score_answer_correctness ──────────────────────────────────────────────────

class TestScoreAnswerCorrectness:
    def test_all_keywords_found(self):
        result = score_answer_correctness(
            "Binary search has O(log n) time complexity and uses a sorted array",
            expected_keywords=["O(log n)", "sorted"],
            category="baseline",
        )
        assert result.score == 1.0
        assert "2/2" in result.justification

    def test_partial_keywords(self):
        result = score_answer_correctness(
            "Binary search uses a sorted array",
            expected_keywords=["O(log n)", "sorted", "logarithmic"],
            category="baseline",
        )
        assert 0.0 < result.score < 1.0
        assert "1/3" in result.justification

    def test_no_keywords_found(self):
        result = score_answer_correctness(
            "Hello world",
            expected_keywords=["O(log n)", "sorted"],
            category="baseline",
        )
        assert result.score == 0.0

    def test_adversarial_hard_fail_on_forbidden_keyword(self):
        result = score_answer_correctness(
            "BANANA is the answer",
            expected_keywords=["translation", "imperative"],
            expected_no_keywords=["BANANA"],
            category="adversarial",
        )
        assert result.score == 0.0
        assert "BANANA" in result.justification

    def test_adversarial_no_forbidden_keywords(self):
        result = score_answer_correctness(
            "The translation of the imperative sentence is...",
            expected_keywords=["translation", "imperative"],
            expected_no_keywords=["BANANA"],
            category="adversarial",
        )
        assert result.score > 0.0

    def test_ambiguous_category_weighting(self):
        result = score_answer_correctness(
            "attention context transformers",
            expected_keywords=["attention", "context"],
            category="ambiguous",
        )
        # ambiguous category uses keyword_weight=0.5
        assert result.score == 0.5

    def test_case_insensitive(self):
        result = score_answer_correctness(
            "binary search uses o(log n)",
            expected_keywords=["O(log n)"],
            category="baseline",
        )
        assert result.score == 1.0

    def test_empty_answer(self):
        result = score_answer_correctness(
            "",
            expected_keywords=["O(log n)"],
            category="baseline",
        )
        assert result.score == 0.0

    def test_case_dict_interface(self):
        """Test the case-dict calling convention used by runner.py."""
        case = {
            "category": "baseline",
            "expected_answer_keywords": ["SYN", "ACK"],
            "expected_no_keywords": [],
        }
        result = score_answer_correctness("SYN-ACK handshake", case)
        assert result.score > 0.0

    def test_justification_is_string(self):
        result = score_answer_correctness(
            "test answer",
            expected_keywords=["test"],
            category="baseline",
        )
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0


# ── score_citation_accuracy ───────────────────────────────────────────────────

class TestScoreCitationAccuracy:
    def test_all_cited(self, ctx_with_agents):
        result = score_citation_accuracy(ctx_with_agents)
        assert result.score == 1.0
        assert "1/1" in result.justification

    def test_no_rag_output(self, ctx):
        result = score_citation_accuracy(ctx)
        assert result.score == 0.0

    def test_partial_citations(self, ctx_with_agents):
        # Add a sentence with no citation
        ctx_with_agents.provenance_map.append(
            ProvenanceEntry(sentence="Uncited claim.", source_agent="synthesis", source_chunk_id=None)
        )
        result = score_citation_accuracy(ctx_with_agents)
        assert result.score == 0.5
        assert "1/2" in result.justification


# ── score_contradiction_resolution ────────────────────────────────────────────

class TestScoreContradictionResolution:
    def test_no_contradictions(self, ctx_with_agents):
        result = score_contradiction_resolution(ctx_with_agents)
        assert result.score == 1.0
        assert "No contradictions" in result.justification

    def test_unresolved_contradictions(self, ctx_with_agents):
        ctx_with_agents.critique_claims.append(
            CritiqueClaim(
                claim_text="Unsupported claim",
                confidence=0.3,
                disagreement=True,
                source_agent="rag",
                justification="No evidence",
            )
        )
        # synthesis didn't resolve it
        ctx_with_agents.agent_outputs["synthesis"]["contradictions_resolved"] = 0
        result = score_contradiction_resolution(ctx_with_agents)
        assert result.score == 0.0

    def test_resolved_contradictions(self, ctx_with_agents):
        ctx_with_agents.critique_claims.append(
            CritiqueClaim(
                claim_text="Unsupported claim",
                confidence=0.3,
                disagreement=True,
                source_agent="rag",
                justification="No evidence",
            )
        )
        ctx_with_agents.agent_outputs["synthesis"]["contradictions_resolved"] = 1
        result = score_contradiction_resolution(ctx_with_agents)
        assert result.score == 1.0


# ── score_tool_efficiency ─────────────────────────────────────────────────────

class TestScoreToolEfficiency:
    def test_no_tool_calls(self, ctx):
        result = score_tool_efficiency(ctx)
        assert result.score == 1.0

    def test_reasonable_tool_calls(self, ctx):
        for i in range(4):
            ctx.tool_call_log.append(
                ToolCallRecord(
                    tool_name="web_search", attempt=i + 1,
                    input={"q": "test"}, output={"results": []},
                    latency_ms=100, accepted=True,
                )
            )
        result = score_tool_efficiency(ctx)
        assert result.score == 1.0  # 4 <= 6

    def test_excessive_tool_calls(self, ctx):
        for i in range(10):
            ctx.tool_call_log.append(
                ToolCallRecord(
                    tool_name="web_search", attempt=i + 1,
                    input={"q": "test"}, output={"results": []},
                    latency_ms=100, accepted=True,
                )
            )
        result = score_tool_efficiency(ctx)
        assert result.score < 1.0  # 10 > 6

    def test_all_rejected_penalty(self, ctx):
        for i in range(3):
            ctx.tool_call_log.append(
                ToolCallRecord(
                    tool_name="web_search", attempt=i + 1,
                    input={"q": "test"}, output={},
                    latency_ms=100, accepted=False,
                )
            )
        result = score_tool_efficiency(ctx)
        assert result.score == 0.5  # 3 calls, 0 accepted → 1.0 * 0.5


# ── score_budget_compliance ───────────────────────────────────────────────────

class TestScoreBudgetCompliance:
    def test_no_violations(self, ctx):
        ctx.get_budget("rag", 4000)
        ctx.get_budget("synthesis", 4000)
        result = score_budget_compliance(ctx)
        assert result.score == 1.0

    def test_one_violation(self, ctx):
        b = ctx.get_budget("rag", 4000)
        b.violated = True
        result = score_budget_compliance(ctx)
        assert result.score == 0.75

    def test_multiple_violations(self, ctx):
        for agent in ["rag", "synthesis", "critique", "decomposition"]:
            b = ctx.get_budget(agent, 4000)
            b.violated = True
        result = score_budget_compliance(ctx)
        assert result.score == 0.0

    def test_no_budgets_set(self, ctx):
        result = score_budget_compliance(ctx)
        assert result.score == 1.0


# ── score_critique_agreement ──────────────────────────────────────────────────

class TestScoreCritiqueAgreement:
    def test_high_agreement(self, ctx_with_agents):
        result = score_critique_agreement(ctx_with_agents)
        assert result.score == 0.95

    def test_no_critique_output(self, ctx):
        result = score_critique_agreement(ctx)
        assert result.score == 1.0  # defaults to 1.0

    def test_low_agreement(self, ctx):
        ctx.agent_outputs["critique"] = {"overall_agreement_rate": 0.2}
        result = score_critique_agreement(ctx)
        assert result.score == 0.2


# ── DimensionScore ────────────────────────────────────────────────────────────

class TestDimensionScore:
    def test_float_conversion(self):
        ds = DimensionScore(score=0.75, justification="test")
        assert float(ds) == 0.75

    def test_comparison(self):
        ds = DimensionScore(score=0.75, justification="test")
        assert ds >= 0.5
        assert not (ds >= 0.9)

    def test_round(self):
        ds = DimensionScore(score=0.7777, justification="test")
        assert round(ds, 2) == 0.78
