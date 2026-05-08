from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic

# Import named prompt constants from each agent module
from app.agents.critique import _SYSTEM as _CRITIQUE_SYSTEM
from app.agents.orchestrator import _ROUTING_SYSTEM
from app.agents.rag import _ANSWER_SYSTEM as _RAG_ANSWER_SYSTEM
from app.agents.synthesis import _SYSTEM as _SYNTHESIS_SYSTEM

# Maps eval dimension names to the agent responsible for that signal
DIMENSION_TO_AGENT: dict[str, str] = {
    "answer_correctness":       "synthesis",
    "citation_accuracy":        "rag",
    "contradiction_resolution": "synthesis",
    "tool_efficiency":          "orchestrator",
    "budget_compliance":        "orchestrator",
    "critique_agreement":       "critique",
}

# The prompt text each agent currently uses
AGENT_PROMPTS: dict[str, str] = {
    "synthesis":    _SYNTHESIS_SYSTEM,
    "rag":          _RAG_ANSWER_SYSTEM,
    "critique":     _CRITIQUE_SYSTEM,
    "orchestrator": _ROUTING_SYSTEM,
}

_META_SYSTEM = (
    "You are a prompt engineering expert. "
    "You analyze AI agent failures and propose targeted prompt improvements."
)

_META_USER_TMPL = """\
The following agent prompt performed poorly on dimension '{dimension}' \
with score {score:.2f} (case: {case_id}).

CURRENT PROMPT:
{current_prompt}

FAILURE CONTEXT:
{failure_context}

Propose a rewritten prompt that specifically addresses this failure.
Return ONLY valid JSON:
{{
  "proposed_prompt": "the full rewritten prompt text",
  "diff_justification": "2-3 sentences explaining what changed and why",
  "expected_improvement": "what dimension score should improve and by how much"
}}\
"""


async def run_meta_agent(eval_summary: dict) -> dict:
    """
    Reads eval failure cases, proposes a prompt rewrite for the
    worst-performing agent+dimension combination.
    Returns a dict ready to INSERT into the prompt_rewrites table.
    """

    # ── 1. Find the worst agent+dimension across all cases ────────────────────
    worst_score:     float = 2.0   # sentinel above any real score
    worst_agent_id:  str   = "synthesis"
    worst_dimension: str   = "answer_correctness"
    worst_case_id:   str   = ""
    worst_case_data: dict  = {}

    for _cat, cat_data in eval_summary.get("by_category", {}).items():
        for case in cat_data.get("cases", []):
            for dim_name, dim_data in case.get("dimensions", {}).items():
                dim_score = dim_data.get("score", 1.0)
                if dim_score < worst_score:
                    worst_score     = dim_score
                    worst_dimension = dim_name
                    worst_agent_id  = DIMENSION_TO_AGENT.get(dim_name, "synthesis")
                    worst_case_id   = case.get("id", "")
                    worst_case_data = case

    # ── 2. Retrieve the current prompt for that agent ─────────────────────────
    current_prompt = AGENT_PROMPTS.get(worst_agent_id, "")

    # ── 3. Call Claude Haiku for a rewrite proposal ───────────────────────────
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    failure_context = json.dumps(worst_case_data, indent=2, default=str)[:1000]

    user_msg = _META_USER_TMPL.format(
        dimension=worst_dimension,
        score=worst_score,
        case_id=worst_case_id,
        current_prompt=current_prompt,
        failure_context=failure_context,
    )

    response = await llm.ainvoke(
        [("system", _META_SYSTEM), ("human", user_msg)]
    )
    raw: str = response.content.strip()

    # ── 4. Parse response (strip markdown fences) ─────────────────────────────
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data: dict = json.loads(raw)

    # ── 5. Return insert-ready dict ───────────────────────────────────────────
    return {
        "agent_id":           worst_agent_id,
        "dimension":          worst_dimension,
        "original_prompt":    current_prompt,
        "proposed_prompt":    data["proposed_prompt"],
        "diff_justification": data["diff_justification"],
        "status":             "pending",
    }
