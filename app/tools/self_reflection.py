from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic

from app.tools.registry import ToolResult

_PROMPT_TEMPLATE = """\
You previously produced this output: {previous_output}
The current context contains these other agent outputs: {other_outputs}
Identify any contradictions or inconsistencies between your output and the others.
Return JSON: {{"contradictions": [{{"claim": "<str>", "conflicts_with": "<str>", "severity": "low"|"medium"|"high"}}]}}\
"""


async def self_reflect(job_id: str, agent_id: str, ctx_dict: dict) -> ToolResult:
    agent_outputs: dict = ctx_dict.get("agent_outputs", {})
    previous_output = agent_outputs.get(agent_id)

    if not previous_output:
        return ToolResult(
            tool_name="self_reflection",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="empty",
        )

    other_outputs = {k: v for k, v in agent_outputs.items() if k != agent_id}

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    prompt = _PROMPT_TEMPLATE.format(
        previous_output=previous_output,
        other_outputs=other_outputs,
    )

    try:
        response = await llm.ainvoke([("human", prompt)])
        raw = response.content.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        contradictions = data.get("contradictions", [])
    except Exception:
        return ToolResult(
            tool_name="self_reflection",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="malformed",
        )

    return ToolResult(
        tool_name="self_reflection",
        success=True,
        output={"contradictions": contradictions, "agent_id": agent_id},
        latency_ms=0,
    )
