# AI Usage Attestation

This document records every use of AI assistance in this project.
Every architectural decision, data model, and system design was made by me.
All AI output was reviewed line-by-line before commit.

## Format
Each entry records: what was asked, what was kept, what was changed, and what was caught.

---


## Session 1 — May 8, 2026

### Block 0 — Project setup & config
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Generate app/config.py using pydantic-settings BaseSettings
with 10 fields, and a matching .env.example with comments.
**What was kept:** All fields, Config class, module-level settings instance.
**What was changed:** Nothing — matched spec exactly.
**What was caught:** Deprecation warning on settings.model_fields called on
instance instead of class — cosmetic, not a bug.

---

### Block 1 — Database models & Alembic
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Generate app/db/database.py (async SQLAlchemy engine,
sessionmaker, Base, get_db dependency) and app/db/models.py (6 models:
Job, AgentLog, ToolCall, EvalRun, PromptRewrite, EvalRerun) using
SQLAlchemy 2.x mapped_column style. Wire alembic/env.py to use
settings.database_url_sync and Base.metadata.
**What was kept:** All 6 models, onupdate on Job.updated_at, FK relationships,
JSON payload columns, policy_violation flag on AgentLog.
**What was changed:** Nothing structural — verified table names via import check.
**What was caught:** alembic CLI not on PATH — used `uv run alembic` instead.
sqlalchemy[asyncio] zsh glob issue — fixed by quoting brackets.

---

### Block 2 — Shared context schema
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Generate app/schemas/context.py with 6 Pydantic v2 models:
SubTask, ToolCallRecord, CritiqueClaim, ProvenanceEntry, AgentBudget,
SharedContext. AgentBudget.consume() must enforce budget and mark violated.
SharedContext.get_budget() initializes per-agent budgets lazily.
**What was kept:** All models, consume() mutation logic, log_routing() helper.
**What was changed:** Nothing — passed verification check (remaining: 7500).
**What was caught:** Claude Code noted that AgentBudget must not be cached
across Redis serialize/deserialize round-trips — noted for worker design.
`from __future__ import annotations` required for dict[str, AgentBudget]
under Python 3.13 with Pydantic v2.

### Block 3 — Agent skeleton & orchestrator
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** BaseAgent/AgentResult base classes, OrchestratorAgent
with LLM-driven JSON routing plan via Claude Haiku, 4 stub agents
(decomposition, rag, critique, synthesis).
**What was kept:** Fallback routing plan on LLM parse failure, per-agent
exception handling continuing pipeline, ctx.get_budget() called before
each agent run.
**What was changed:** Nothing structural.
**What was caught:** Empty __init__.py confused Claude Code's write preview
— used touch instead.

### Block 4 — FastAPI endpoints + Celery worker
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** app/main.py with 6 endpoints (SSE stream, trace,
eval, prompt review) and app/worker.py with run_pipeline Celery task.
**What was kept:** Fresh session per SSE poll cycle to avoid stale reads,
lazy import of OrchestratorAgent inside task to avoid startup cost,
asyncio.run() per task for clean event loop.
**What was changed:** Nothing structural.
**What was caught:** Final answer stored temporarily in job.query column —
noted for cleanup when proper column added.


### Block 5 — Docker Compose + Alembic migrations
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** docker-compose.yml with 4 services + healthchecks,
Dockerfile with uv, Alembic migration.
**What was kept:** Port conflict detection logic (Claude Code found 5432
and 5433 both taken by OrbStack + other projects, auto-selected 5434).
**What was changed:** .env credentials were still on placeholder values
from .env.example — fixed manually with sed.
**What was caught:** Two separate issues: wrong port (5432→5434) and
wrong credentials (user:password→mega:mega). Both were .env issues,
not code issues.

### Block 6 — 4 tools with failure contracts
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** ToolResult dataclass, 4 async tools each with
explicit failure_reason contracts (timeout/empty/malformed), tenacity
retry on web_search, subprocess sandbox on code_executor, NL→SQL on
data_lookup, contradiction detection on self_reflection.
**What was kept:** Circular import safety pattern (registry.py defines
ToolResult before any tool imports it), thread executor for psycopg2
blocking calls, markdown-fence strip on LLM JSON responses.
**What was changed:** Nothing structural.
**What was caught:** code_executor sandbox is best-effort only —
__import__("os") bypass not blocked. Noted as known limitation for README.

### Block 7 — DecompositionAgent real implementation
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Replace stub with real LLM-driven decomposition.
JSON parse with fence stripping, dependency validation, SubTask
materialisation into ctx.subtasks.
**What was kept:** Prune-not-reject on invalid dependencies (safer
degradation), .get(...,0) fallback on usage_metadata for test safety.
**What was changed:** Nothing structural.
**What was caught:** Nothing — clean on first run.

### Block 8 — RAGAgent (ported from NOVA project)
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Replace stub with real 2-hop RAG agent. Port
hybrid BM25+vector+RRF+Cohere pipeline from scratch/final_retreval.py
and scratch/retrieval_core.py. Add follow-up query generation for
hop 2, citation building, provenance map.
**What was kept:** Identical RRF_K=60, RERANK_TOP_N=10, RRF fusion
logic. pgvector bracketed-string embedding format. Hop-1 priority
in dedup pass. Sentence boundary split for provenance.
**What was changed:** Sync psycopg2 pool → direct connect in
run_in_executor. Sync OpenAI client → openai.AsyncOpenAI.
Sync Cohere → cohere.AsyncClient.
**What was caught:** Constructor guards needed on OpenAI/Cohere clients
for test environments with dummy API keys.
**Source reuse:** scratch/final_retreval.py (RRF logic, rerank call),
scratch/retrieval_core.py (embed pattern, context building).

### Block 9 — CritiqueAgent
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Real CritiqueAgent reviewing all other agent
outputs with span-level claim flagging, per-claim confidence scores,
disagreement detection.
**What was kept:** Append-not-replace on ctx.critique_claims,
two-tier exception handling (JSONDecodeError vs generic),
per-claim skip on malformed claims.
**What was changed:** Nothing structural.
**What was caught:** Nothing — clean first run.

### Block 10 — SynthesisAgent
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Real SynthesisAgent merging all agent outputs,
resolving critique disagreements, producing provenance map per sentence
with resolution status (kept/modified/rejected).
**What was kept:** ctx.provenance_map.clear() before rebuild to avoid
duplicates from RAG-level entries. Agreements block passed alongside
disagreements to preserve well-supported content. agents_merged built
dynamically from ctx.agent_outputs keys.
**What was changed:** Nothing structural.
**What was caught:** Nothing — clean first run.


### Block 11 — Evaluation test cases
**Tool:** Claude Opus 4.6 (claude.ai) — switched for adversarial design
**What was asked:** 15 test cases across 3 categories with adversarial
cases requiring genuine sophistication: injection hidden in translation
task (tc_11), forged retrieval header (tc_12), confident wrong-premise
Einstein myth (tc_13), operationalized tongue-map myth (tc_14),
contradiction-forcing saturated fat query (tc_15).
**What was kept:** All cases verbatim. DESIGN_NOTES threat model.
Scoring weight guidance (adversarial no_keywords weight=2.0).
**What was changed:** Nothing.
**What was caught:** Opus flagged that tc_15 requires critique agent
to see retrieval distribution, not just synthesis output — architectural
note for runner.py implementation.
**Why Opus:** Sonnet produces generic adversarial cases. Opus generated
genuinely subtle injections with plausible cover stories and picked
real scientific controversies for the contradiction case.

### Block 12 — Eval runner + trigger
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** app/eval/runner.py with 6 scoring dimensions,
adversarial hard-fail on no_keywords, weighted overall score.
app/eval/trigger.py storing results in eval_runs table.
**What was kept:** Deferred OrchestratorAgent import inside run_eval()
to avoid import-time LLM client init. json default=str on DB insert.
Adversarial no_kw_penalty=2.0 with hard 0.0 on hit.
**What was changed:** Nothing structural.
**What was caught:** test_cases.py already existed from Opus output —
Claude Code read it instead of overwriting. Clean detection.

### Block 13 — Meta-agent
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** run_meta_agent() that finds worst agent+dimension,
retrieves current prompt constant, proposes rewrite with structured
diff via Haiku.
**What was kept:** Worst-score sentinel=2.0 (not 1.0) to avoid false
first-iteration replacement. Import aliases for _SYSTEM name collision
between synthesis and critique.
**What was changed:** Nothing structural.
**What was caught:** _ROUTING_SYSTEM already existed in orchestrator.py
as a named constant — no changes needed there.

### Block 14 — Wire meta-agent + rerun_eval
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Implement rerun_eval Celery task with real DB
logic, wire meta-agent into approval endpoint as fire-and-forget.
**What was kept:** Defensive JSON parsing on psycopg2 (str vs dict),
failed_ids or None fallback to full re-run, closure capture of
_eval_summary as local variable to avoid DetachedInstanceError.
**What was changed:** Nothing structural.
**What was caught:** run_meta_agent deferred import inside coroutine
to avoid ANTHROPIC_API_KEY requirement at import time during tests.

### Block 15 — HTML demo client
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Single-file terminal-style SSE client, no npm.
Named SSE listeners per event type, auto-scroll, clear button.
**What was kept:** Named addEventListener per event type (not onmessage),
onerror CLOSED state guard, esc() XSS helper, Ctrl+Enter shortcut.
**What was changed:** Nothing.
**What was caught:** Nothing — clean first run.

### Block 15 — HTML demo client
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Single-file terminal-style SSE client, no npm.
Named SSE listeners per event type, auto-scroll, clear button.
**What was kept:** Named addEventListener per event type (not onmessage),
onerror CLOSED state guard, esc() XSS helper, Ctrl+Enter shortcut.
**What was changed:** Nothing.
**What was caught:** Nothing — clean first run.

---

### Block 16 — Wire rerun_eval + approval endpoint
**Tool:** Claude Sonnet 4.6 (claude.ai)
**What was asked:** Implement rerun_eval Celery task with real DB
logic, wire meta-agent into approval endpoint as fire-and-forget.
**What was kept:** Defensive JSON parsing on psycopg2 (str vs dict),
failed_ids or None fallback to full re-run, closure capture of
_eval_summary as local variable to avoid DetachedInstanceError.
**What was changed:** Nothing structural.
**What was caught:** run_meta_agent deferred import inside coroutine
to avoid ANTHROPIC_API_KEY requirement at import time during tests.

---

### Block 17 — README
**Tool:** Claude Opus 4.6 (claude.ai) — switched for documentation quality
**What was asked:** Full README with architecture diagram, agent table,
tool table, eval harness docs, known limitations, what to build next.
**What was kept:** All sections verbatim. 7 known limitations including
honest assessment of sandbox security, meta-agent attribution being
correlational not causal, 15 cases being too few for statistical significance.
**What was changed:** Nothing.
**Why Opus:** README is a primary evaluation artifact. A senior engineer
reads it first. Sonnet produces adequate READMEs; Opus produces ones
that demonstrate systems thinking in the limitations section.