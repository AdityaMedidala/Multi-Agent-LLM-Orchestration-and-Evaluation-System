from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import AgentResult, BaseAgent, count_tokens
from app.agents.critique import CritiqueAgent
from app.agents.decomposition import DecompositionAgent
from app.agents.rag import RAGAgent
from app.agents.synthesis import SynthesisAgent
from app.config import settings
from app.schemas.context import AgentBudget, SharedContext

log = logging.getLogger(__name__)


_ROUTING_SYSTEM = """\
You are an orchestration planner for a multi-agent research system.
Given a user query, decide which agents and tools to invoke and in what order.

Available agents:
- decomposition: breaks the query into structured sub-tasks
- rag: retrieves relevant documents and evidence
- critique: fact-checks and flags unsupported claims
- synthesis: assembles a final, coherent answer

Available tools (included in the routing plan exactly like agents):
- data_lookup: executes a structured database query against the system's own
  tables (jobs, agent_logs, tool_calls); use this when the query contains
  counting, listing, or aggregation language such as "how many", "list all",
  "count", "which jobs", "what runs", "show records", or any request for
  statistics or records from the system database
- code_executor: runs a Python snippet in a sandboxed subprocess and returns
  stdout, stderr, and exit code; use this when the query requires a calculation,
  numerical conversion, algorithm demonstration, or any task best answered by
  executing code rather than generating prose

Tool steps execute before agent steps. Place data_lookup or code_executor early
in the plan whenever the query clearly requests computation or database lookups.

Return ONLY valid JSON in this exact shape — no prose, no markdown:
{
  "routing_plan": [
    {"agent": "<agent_name_or_tool_name>", "reason": "<one sentence>", "budget_tokens": <int>},
    ...
  ],
  "orchestrator_reasoning": "<one sentence explaining the overall plan>"
}
"""

_ROUTING_USER = """\
Query: {query}

Produce the routing plan JSON now.
"""


# ── Agent-log persistence (sync DB write, run via executor) ──────────────────

def _write_agent_log_sync(
    job_id: str,
    agent_id: str,
    event_type: str,
    input_payload: dict,
    output_payload: dict,
    latency_ms: int,
    token_count: int,
    policy_violation: bool = False,
) -> None:
    """Write one AgentLog row synchronously (called via run_in_executor)."""
    def _hash(d: dict) -> str:
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    try:
        job_uuid_str = str(uuid.UUID(job_id))
    except ValueError:
        job_uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, job_id))

    try:
        from app.db.sync_pool import get_conn, put_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO agent_logs
                  (id, job_id, agent_id, event_type,
                   input_hash, output_hash, latency_ms,
                   token_count, payload, policy_violation)
                VALUES (%s,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    str(uuid.uuid4()), job_uuid_str, agent_id, event_type,
                    _hash(input_payload), _hash(output_payload),
                    latency_ms, token_count,
                    json.dumps(
                        {"input": input_payload, "output": output_payload},
                        default=str,
                    ),
                    policy_violation,
                ),
            )
            conn.commit()
            cur.close()
        finally:
            put_conn(conn)
    except Exception as exc:
        log.warning("agent_log write failed: %s", exc)


async def _log_agent_event(
    job_id: str,
    agent_id: str,
    event_type: str,
    input_payload: dict,
    output_payload: dict,
    latency_ms: int,
    token_count: int,
    policy_violation: bool = False,
) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        functools.partial(
            _write_agent_log_sync,
            job_id, agent_id, event_type,
            input_payload, output_payload,
            latency_ms, token_count, policy_violation,
        ),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    agent_id = "orchestrator"

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    async def run(self, ctx: SharedContext) -> AgentResult:
        start = time.monotonic()

        from app.streaming import publish_event

        # ── 1. Ask the LLM for a routing plan ────────────────────────────────
        messages = [
            ("system", _ROUTING_SYSTEM),
            ("human", _ROUTING_USER.format(query=ctx.original_query)),
        ]
        await publish_event(ctx.job_id, "agent_start", {
            "agent": self.agent_id, "ts": time.monotonic()
        })
        routing_plan, overall_reasoning = await self._get_routing_plan(messages)
        await publish_event(ctx.job_id, "agent_done", {
            "agent": self.agent_id, "tokens": 0
        })
        ctx.log_routing(f"orchestrator_reasoning: {overall_reasoning}")

        # ── 2a. Execute tool steps before agent steps ─────────────────────────
        from app.tools.registry import call_tool
        from app.schemas.context import ToolCallRecord
        from app.agents.tool_persistence import _persist_tool_call
        _plannable_tools = {"data_lookup", "code_executor"}
        tool_steps = [s for s in routing_plan if s["agent"] in _plannable_tools]
        routing_plan = [s for s in routing_plan if s["agent"] not in _plannable_tools]
        for tool_step in tool_steps:
            tool_name = tool_step["agent"]
            reason = tool_step.get("reason", "")
            ctx.log_routing(f"invoking tool '{tool_name}': {reason}")
            t0 = time.monotonic()

            # Build tool-specific kwargs
            if tool_name == "data_lookup":
                tool_kwargs = {"natural_language_query": ctx.original_query}
            elif tool_name == "code_executor":
                # Extract code from the query or generate a snippet via LLM
                code_snippet = await self._generate_code_snippet(ctx.original_query)
                tool_kwargs = {"code": code_snippet}
            else:
                tool_kwargs = {}

            tool_result = await call_tool(tool_name, tool_kwargs)
            latency = int((time.monotonic() - t0) * 1000)
            ctx.tool_call_log.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    attempt=1,
                    input=tool_kwargs,
                    output=tool_result.output,
                    latency_ms=latency,
                    accepted=tool_result.success,
                    failure_reason=tool_result.failure_reason,
                )
            )
            await _persist_tool_call(
                job_id=ctx.job_id,
                tool_name=tool_name,
                agent_id="orchestrator",
                attempt=1,
                input_payload=tool_kwargs,
                output_payload=tool_result.output,
                latency_ms=latency,
                accepted=tool_result.success,
                failure_reason=tool_result.failure_reason,
            )
            if tool_result.success:
                ctx.metadata[f"{tool_name}_results"] = tool_result.output
                if tool_name == "data_lookup":
                    ctx.log_routing(
                        f"data_lookup returned {tool_result.output.get('row_count', 0)} rows"
                    )
                else:
                    stdout = tool_result.output.get("stdout", "")[:200]
                    ctx.log_routing(f"code_executor returned: {stdout}")
            else:
                ctx.log_routing(f"{tool_name} failed: {tool_result.failure_reason}")
            await _log_agent_event(
                job_id=ctx.job_id,
                agent_id=tool_name,
                event_type="tool_result",
                input_payload=tool_kwargs,
                output_payload=tool_result.output,
                latency_ms=latency,
                token_count=0,
            )

        # Log the orchestrator's own routing decision
        await _log_agent_event(
            job_id=ctx.job_id,
            agent_id="orchestrator",
            event_type="routing",
            input_payload={"query": ctx.original_query},
            output_payload={"plan": routing_plan, "reasoning": overall_reasoning},
            latency_ms=0,
            token_count=0,
        )

        # ── 3. Execute agents in the decided order ────────────────────────────
        results: list[AgentResult] = []
        agent_type_map = {
            "rag": "retrieve",
            "synthesis": "synthesize",
            "critique": "critique",
            "decomposition": "decompose",
        }
        _agent_registry: dict[str, BaseAgent] = {
            "decomposition": DecompositionAgent(),
            "rag": RAGAgent(),
            "critique": CritiqueAgent(),
            "synthesis": SynthesisAgent(),
        }
        plan_queue = list(routing_plan)
        step_idx = 0

        # Lazy import so circular deps resolve at runtime
        from app.agents.compression import compress_context_if_needed

        while step_idx < len(plan_queue):
            step = plan_queue[step_idx]
            step_idx += 1

            agent_name: str = step["agent"]
            reason: str = step.get("reason", "")
            budget = max(step.get("budget_tokens", 4000), 4000)

            ctx.log_routing(f"invoking '{agent_name}': {reason}")

            agent = _agent_registry.get(agent_name)
            if agent is None:
                ctx.metadata[f"{agent_name}_error"] = "agent not found in registry"
                ctx.log_routing(f"SKIPPED '{agent_name}': not in registry")
                continue

            # ── Initialise budget and publish starting budget to SSE ──────────
            b = ctx.get_budget(agent_name, budget)
            await publish_event(ctx.job_id, "budget_update", {
                "agent": agent_name,
                "budget_tokens": budget,
                "used": 0,
                "remaining": budget,
                "violated": False,
            })

            # ── Context compression before each agent if budget is tight ──────
            await compress_context_if_needed(ctx, agent_name, budget)

            # ── Log agent start ───────────────────────────────────────────────
            await _log_agent_event(
                job_id=ctx.job_id,
                agent_id=agent_name,
                event_type="start",
                input_payload={"query": ctx.original_query, "routing_reason": reason},
                output_payload={},
                latency_ms=0,
                token_count=0,
            )

            try:
                result = await agent.run(ctx)
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                ctx.metadata[f"{agent_name}_error"] = error_msg
                ctx.log_routing(f"FAILED '{agent_name}': {error_msg}")

                await _log_agent_event(
                    job_id=ctx.job_id,
                    agent_id=agent_name,
                    event_type="error",
                    input_payload={"query": ctx.original_query},
                    output_payload={"error": str(exc)},
                    latency_ms=0,
                    token_count=0,
                )
                results.append(
                    AgentResult(
                        agent_id=agent_name,
                        success=False,
                        output={},
                        tokens_used=0,
                        latency_ms=0,
                        error=error_msg,
                    )
                )
                continue

            if not result.success:
                ctx.log_routing(f"FAILED '{agent_name}': {result.error}")

            # ── Publish actual budget remaining after agent completes ─────────
            b_after = ctx.budgets.get(agent_name)
            if b_after:
                await publish_event(ctx.job_id, "budget_update", {
                    "agent": agent_name,
                    "budget_tokens": b_after.max_tokens,
                    "used": b_after.used_tokens,
                    "remaining": b_after.remaining(),
                    "violated": b_after.violated,
                })

            # ── Log agent output ──────────────────────────────────────────────
            is_violation = ctx.budgets.get(
                agent_name,
                AgentBudget(agent_id=agent_name, max_tokens=0),
            ).violated
            await _log_agent_event(
                job_id=ctx.job_id,
                agent_id=agent_name,
                event_type="output",
                input_payload={"query": ctx.original_query},
                output_payload=result.output,
                latency_ms=result.latency_ms,
                token_count=result.tokens_used,
                policy_violation=is_violation,
            )

            results.append(result)

            # Mark subtasks complete for this agent
            completed_type = agent_type_map.get(agent_name)
            if completed_type:
                for st in ctx.subtasks:
                    if st.task_type == completed_type and st.status == "pending":
                        st.status = "done"
                        st.result = {"summary": str(result.output)[:200]}
                        break

            # After decomposition populates ctx.subtasks, reorder remaining steps
            if agent_name == "decomposition" and ctx.subtasks:
                remaining = plan_queue[step_idx:]
                reordered = self._build_execution_order(ctx, remaining)
                plan_queue = plan_queue[:step_idx] + reordered

        # Ensure final_answer is always set even if synthesis fails
        if not ctx.final_answer and ctx.agent_outputs:
            for aid in ("synthesis", "rag", "decomposition"):
                fa = (ctx.agent_outputs.get(aid, {}).get("final_answer")
                      or ctx.agent_outputs.get(aid, {}).get("answer"))
                if fa:
                    ctx.final_answer = fa
                    break

        latency_ms = int((time.monotonic() - start) * 1000)
        total_tokens = sum(r.tokens_used for r in results)

        return AgentResult(
            agent_id=self.agent_id,
            success=True,
            output={
                "routing_plan": routing_plan,
                "orchestrator_reasoning": overall_reasoning,
                "agent_results": [r.model_dump() for r in results],
            },
            tokens_used=total_tokens,
            latency_ms=latency_ms,
        )

    def _build_execution_order(
        self, ctx: SharedContext, routing_plan: list[dict]
    ) -> list[dict]:
        """
        Reorders the routing plan to respect subtask dependencies using
        Kahn's topological sort. Falls back to original plan on cycle.
        """
        subtasks = ctx.subtasks
        if not subtasks:
            return routing_plan

        dep_map: dict[str, list[str]] = {
            st.task_id: st.dependencies for st in subtasks
        }
        task_types: dict[str, str] = {
            st.task_id: st.task_type for st in subtasks
        }

        from collections import deque
        in_degree = {tid: 0 for tid in dep_map}
        dependents: dict[str, list[str]] = {tid: [] for tid in dep_map}
        for tid, deps in dep_map.items():
            in_degree[tid] += len(deps)
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(tid)

        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        ordered_task_ids: list[str] = []
        while queue:
            tid = queue.popleft()
            ordered_task_ids.append(tid)
            for dependent in dependents.get(tid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(ordered_task_ids) != len(dep_map):
            log.warning("Cycle in subtask dependency graph, using original plan")
            return routing_plan

        type_to_agent = {
            "retrieve": "rag",
            "synthesize": "synthesis",
            "critique": "critique",
            "decompose": "decomposition",
        }

        reordered: list[dict] = []
        seen_agents: set[str] = set()
        for tid in ordered_task_ids:
            task_type = task_types.get(tid, "")
            agent_name = type_to_agent.get(task_type)
            if agent_name and agent_name not in seen_agents:
                matching = next(
                    (s for s in routing_plan if s["agent"] == agent_name),
                    None,
                )
                if matching:
                    reordered.append(matching)
                    seen_agents.add(agent_name)

        for step in routing_plan:
            if step["agent"] not in seen_agents:
                reordered.append(step)
                seen_agents.add(step["agent"])

        if reordered:
            ctx.log_routing(
                f"dependency_enforcement: reordered {len(routing_plan)} steps "
                f"via {len(ordered_task_ids)} subtask dependencies"
            )
            return reordered

        return routing_plan

    async def _get_routing_plan(
        self, messages: list
    ) -> tuple[list[dict], str]:
        """Call the LLM and parse the routing JSON. Falls back to a safe default."""
        try:
            from app.agents.prompt_registry import get_active_prompt
            system = get_active_prompt("orchestrator", _ROUTING_SYSTEM)
            messages = [
                (role, system if role == "system" else content)
                for role, content in messages
            ]
            response = await self._llm.ainvoke(messages)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            plan = data.get("routing_plan", [])
            reasoning = data.get("orchestrator_reasoning", "")
            return plan, reasoning
        except Exception as exc:
            fallback_plan = [
                {"agent": "decomposition", "reason": "fallback", "budget_tokens": 6000},
                {"agent": "rag",           "reason": "fallback", "budget_tokens": 6000},
                {"agent": "critique",      "reason": "fallback", "budget_tokens": 6000},
                {"agent": "synthesis",     "reason": "fallback", "budget_tokens": 6000},
            ]
            return fallback_plan, f"LLM routing failed ({exc}); using default order"

    async def _generate_code_snippet(self, query: str) -> str:
        """Use the LLM to generate a safe Python snippet from a natural language query."""
        try:
            response = await self._llm.ainvoke([
                ("system",
                 "You are a Python code generator. Given a user query, write a short "
                 "Python snippet that computes the answer and prints it to stdout.\n"
                 "Rules:\n"
                 "- Use only the standard library (math, statistics, datetime, etc.)\n"
                 "- Do NOT import os, sys, subprocess, socket, or any network/file module\n"
                 "- Print the result clearly\n"
                 "- Return ONLY the Python code, no markdown fences, no explanation"),
                ("human", query),
            ])
            code = response.content.strip()
            if code.startswith("```"):
                code = code.split("```")[1]
                if code.startswith("python"):
                    code = code[6:]
                code = code.strip()
            return code
        except Exception as exc:
            log.warning("code snippet generation failed: %s", exc)
            return "print('Code generation failed')"