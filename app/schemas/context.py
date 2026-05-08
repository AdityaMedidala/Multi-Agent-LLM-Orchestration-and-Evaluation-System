from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SubTask(BaseModel):
    task_id: str
    task_type: Literal["decompose", "retrieve", "critique", "synthesize"]
    description: str
    dependencies: list[str]  # task_ids that must complete first
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: dict | None = None


class ToolCallRecord(BaseModel):
    tool_name: str
    attempt: int
    input: dict
    output: dict | None
    latency_ms: int | None
    accepted: bool | None
    failure_reason: str | None = None


class CritiqueClaim(BaseModel):
    claim_text: str       # specific span of text being flagged
    confidence: float     # 0.0 to 1.0
    disagreement: bool    # True if critique agent flags this span
    source_agent: str
    justification: str


class ProvenanceEntry(BaseModel):
    sentence: str
    source_agent: str               # which agent produced this
    source_chunk_id: str | None = None  # for RAG-sourced sentences
    tool_calls_used: list[str] = []     # tool names that contributed


class AgentBudget(BaseModel):
    agent_id: str
    max_tokens: int
    used_tokens: int = 0
    violated: bool = False

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def consume(self, tokens: int) -> bool:
        """Returns False and marks violated if budget would be exceeded."""
        if self.used_tokens + tokens > self.max_tokens:
            self.violated = True
            return False
        self.used_tokens += tokens
        return True


class SharedContext(BaseModel):
    job_id: str
    original_query: str
    subtasks: list[SubTask] = []
    tool_call_log: list[ToolCallRecord] = []
    agent_outputs: dict[str, dict] = {}       # agent_id -> raw output dict
    critique_claims: list[CritiqueClaim] = []
    provenance_map: list[ProvenanceEntry] = []
    budgets: dict[str, AgentBudget] = {}      # agent_id -> budget
    orchestrator_reasoning: list[str] = []    # routing decisions + justifications
    final_answer: str | None = None
    metadata: dict = {}

    def get_budget(self, agent_id: str, max_tokens: int) -> AgentBudget:
        """Return existing budget for agent_id, or create it with max_tokens."""
        if agent_id not in self.budgets:
            self.budgets[agent_id] = AgentBudget(
                agent_id=agent_id, max_tokens=max_tokens
            )
        return self.budgets[agent_id]

    def log_routing(self, reasoning: str) -> None:
        self.orchestrator_reasoning.append(reasoning)
