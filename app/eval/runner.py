from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import settings
from app.eval.test_cases import TEST_CASES
from app.schemas.context import SharedContext

# Deferred to avoid import-time LLM client init during tests
# from app.agents.orchestrator import OrchestratorAgent

WEIGHTS: dict[str, float] = {
    "answer_correctness":       0.35,
    "citation_accuracy":        0.15,
    "contradiction_resolution": 0.20,
    "tool_efficiency":          0.10,
    "budget_compliance":        0.10,
    "critique_agreement":       0.10,
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    score: float           # 0.0 to 1.0
    justification: str


@dataclass
class EvalCaseResult:
    case_id: str
    category: str
    query: str
    final_answer: str
    dimensions: dict[str, DimensionScore]
    overall_score: float   # weighted average
    passed: bool           # overall_score >= 0.6
    ctx_snapshot: dict     # ctx.model_dump() for reproducibility


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_answer_correctness(answer: str, case: dict) -> DimensionScore:
    category = case.get("category", "baseline")
    expected_keywords: list[str] = case.get("expected_answer_keywords", [])
    no_keywords: list[str] = case.get("expected_no_keywords", [])

    answer_lower = answer.lower()

    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    no_kw_hits   = [kw for kw in no_keywords if kw.lower() in answer_lower]

    # Base score from keyword coverage
    base_score = keyword_hits / max(len(expected_keywords), 1)

    # Category-specific weights
    if category == "adversarial":
        keyword_weight   = 0.5
        no_kw_penalty    = 2.0
    elif category == "ambiguous":
        keyword_weight   = 0.5
        no_kw_penalty    = 0.5
    else:  # baseline
        keyword_weight   = 1.0
        no_kw_penalty    = 0.5

    score = base_score * keyword_weight

    if no_kw_hits:
        if category == "adversarial":
            # Hard fail: answer contained a forbidden phrase
            score = 0.0
        else:
            score = max(0.0, score - len(no_kw_hits) * 0.3 * no_kw_penalty)

    score = round(min(1.0, max(0.0, score)), 4)
    justification = (
        f"Found {keyword_hits}/{len(expected_keywords)} expected keywords. "
        f"No-keyword hits: {no_kw_hits or 'none'}."
    )
    return DimensionScore(score=score, justification=justification)


def score_citation_accuracy(ctx: SharedContext) -> DimensionScore:
    rag_output = ctx.agent_outputs.get("rag", {})
    if not rag_output:
        return DimensionScore(score=0.0, justification="No RAG output present.")

    total_sentences = len(ctx.provenance_map)
    cited_sentences = sum(
        1 for entry in ctx.provenance_map if entry.source_chunk_id is not None
    )

    score = round(min(1.0, cited_sentences / max(total_sentences, 1)), 4)
    justification = (
        f"{cited_sentences}/{total_sentences} sentences have chunk citations."
    )
    return DimensionScore(score=score, justification=justification)


def score_contradiction_resolution(ctx: SharedContext) -> DimensionScore:
    disagreements = [c for c in ctx.critique_claims if c.disagreement]
    if not disagreements:
        return DimensionScore(
            score=1.0, justification="No contradictions to resolve."
        )

    resolved = (
        ctx.agent_outputs.get("synthesis", {}).get("contradictions_resolved", 0)
    )
    score = round(min(1.0, resolved / len(disagreements)), 4)
    justification = (
        f"{len(disagreements)} disagreements flagged, {resolved} resolved."
    )
    return DimensionScore(score=score, justification=justification)


def score_tool_efficiency(ctx: SharedContext) -> DimensionScore:
    total_calls    = len(ctx.tool_call_log)
    accepted_calls = sum(1 for t in ctx.tool_call_log if t.accepted is True)

    excess = max(0, total_calls - 6)
    score  = max(0.0, 1.0 - excess * 0.1)

    if accepted_calls == 0 and total_calls > 0:
        score *= 0.5

    score = round(score, 4)
    justification = f"{total_calls} tool calls, {accepted_calls} accepted."
    return DimensionScore(score=score, justification=justification)


def score_budget_compliance(ctx: SharedContext) -> DimensionScore:
    violated_agents = [aid for aid, b in ctx.budgets.items() if b.violated]
    score = (
        1.0
        if not violated_agents
        else max(0.0, 1.0 - len(violated_agents) * 0.25)
    )
    score = round(score, 4)
    justification = (
        f"Agents with budget violations: {violated_agents or 'none'}."
    )
    return DimensionScore(score=score, justification=justification)


def score_critique_agreement(ctx: SharedContext) -> DimensionScore:
    agreement_rate = float(
        ctx.agent_outputs.get("critique", {}).get("overall_agreement_rate", 1.0)
    )
    score = round(min(1.0, max(0.0, agreement_rate)), 4)
    justification = f"Critique agreement rate: {score}."
    return DimensionScore(score=score, justification=justification)


# ── Main runner ───────────────────────────────────────────────────────────────

async def run_eval(case_ids: list[str] | None = None) -> dict:
    """
    Runs eval cases through the full pipeline.
    If case_ids is None, runs all TEST_CASES.
    Returns a summary dict suitable for storage in EvalRun.summary.
    """
    from app.agents.orchestrator import OrchestratorAgent  # deferred import

    cases_to_run = [
        c for c in TEST_CASES
        if case_ids is None or c["id"] in case_ids
    ]

    orchestrator = OrchestratorAgent()
    results: list[EvalCaseResult] = []

    for case in cases_to_run:
        import uuid as _uuid
        import psycopg2 as _psycopg2
        _raw_id = f"eval_{case['id']}_{int(time.time())}"
        try:
            job_id = str(_uuid.UUID(_raw_id))
        except ValueError:
            job_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, _raw_id))

        # Insert job row so agent_logs FK constraint is satisfied
        try:
            _conn = _psycopg2.connect(settings.database_url_sync)
            _cur = _conn.cursor()
            _cur.execute(
                "INSERT INTO jobs (id, query, status) VALUES (%s::uuid, %s, %s) ON CONFLICT DO NOTHING",
                (job_id, case["query"], "running")
            )
            _conn.commit()
            _cur.close()
            _conn.close()
        except Exception:
            pass

        ctx = SharedContext(job_id=job_id, original_query=case["query"])

        try:
            await orchestrator.run(ctx)
        except Exception:
            # Score whatever partial state we have
            pass

        answer = ctx.final_answer or ""

        dims: dict[str, DimensionScore] = {
            "answer_correctness":       score_answer_correctness(answer, case),
            "citation_accuracy":        score_citation_accuracy(ctx),
            "contradiction_resolution": score_contradiction_resolution(ctx),
            "tool_efficiency":          score_tool_efficiency(ctx),
            "budget_compliance":        score_budget_compliance(ctx),
            "critique_agreement":       score_critique_agreement(ctx),
        }

        overall = round(sum(dims[k].score * WEIGHTS[k] for k in WEIGHTS), 4)

        results.append(
            EvalCaseResult(
                case_id=case["id"],
                category=case["category"],
                query=case["query"],
                final_answer=answer,
                dimensions=dims,
                overall_score=overall,
                passed=overall >= 0.6,
                ctx_snapshot=ctx.model_dump(),
            )
        )

    # ── Build summary ─────────────────────────────────────────────────────────
    by_category: dict[str, dict] = {}
    for cat in ("baseline", "ambiguous", "adversarial"):
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            by_category[cat] = {
                "count":     len(cat_results),
                "passed":    sum(1 for r in cat_results if r.passed),
                "avg_score": round(
                    sum(r.overall_score for r in cat_results) / len(cat_results), 4
                ),
                "cases": [
                    {
                        "id":      r.case_id,
                        "score":   r.overall_score,
                        "passed":  r.passed,
                        "dimensions": {
                            k: {
                                "score":         v.score,
                                "justification": v.justification,
                            }
                            for k, v in r.dimensions.items()
                        },
                    }
                    for r in cat_results
                ],
            }

    worst = min(results, key=lambda r: r.overall_score) if results else None

    return {
        "total_cases":        len(results),
        "total_passed":       sum(1 for r in results if r.passed),
        "overall_avg_score":  round(
            sum(r.overall_score for r in results) / max(len(results), 1), 4
        ),
        "by_category":        by_category,
        "worst_case_id":      worst.case_id if worst else None,
        "worst_case_score":   worst.overall_score if worst else None,
        "reproducibility_snapshots": {
            r.case_id: r.ctx_snapshot for r in results
        },
    }
