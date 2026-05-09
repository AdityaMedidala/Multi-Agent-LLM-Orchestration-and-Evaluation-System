# mega-ai

A production multi-agent LLM system built around a four-agent async pipeline (Decomposition → RAG → Critique → Synthesis), orchestrated by an LLM-generated JSON routing plan rather than hardcoded edges. The system answers research-style queries with span-level claim provenance, ships with a 15-case adversarial eval harness scored across six dimensions, and includes a meta-agent that proposes prompt rewrites for failing cases — gated behind explicit human approval. Stack: FastAPI, LangChain, Gemini 2.0 Flash (gemini-2.0-flash), OpenAI embeddings, Cohere rerank, pgvector, PostgreSQL, Redis, Celery, Docker Compose.

---

## Quick start

```bash
git clone <repo> && cd mega-ai
cp .env.example .env   # fill in GOOGLE_API_KEY, OPENAI_API_KEY, COHERE_API_KEY
docker compose up
```

This brings up five services — `api`, `worker`, `postgres`, `redis`, `adminer` — with no manual migrations or seeding. The API listens on `localhost:8001`. After `docker compose up`, run the corpus seed:

```bash
docker compose exec api uv run python scripts/seed_corpus.py
docker compose exec api uv run python scripts/embed_corpus.py
```

A log query interface is available at http://localhost:8080 (Adminer — server: postgres, user: mega, password: mega, database: megaai).

### Endpoint walkthrough

**1. Submit a query**

```bash
curl -X POST localhost:8001/jobs \
  -H 'Content-Type: application/json' \
  -d '{"query": "Compare inference energy use of GPT-4 and Llama 3 70B, with sources."}'
# → {"job_id": "j_8f3a2c"}
```

**2. Stream live agent activity (SSE)**

```bash
curl -N localhost:8001/jobs/j_8f3a2c/stream
# event: agent_start      data: {"agent":"decomposition","ts":...}
# event: token            data: {"agent":"rag","token":"Binary"}
# event: agent_done       data: {"agent":"synthesis","tokens":412}
# event: final_answer     data: {"answer":"Binary search is..."}
# event: done             data: {}
```

**3. Fetch the full execution trace**

```bash
curl localhost:8001/jobs/j_8f3a2c/trace
# Full DAG: per-agent inputs/outputs, tool calls, token usage,
# routing-plan JSON, retrieval scores, critique flags, provenance map.
```

**4. Latest eval summary**

```bash
curl localhost:8001/eval/latest
# {"eval_run_id":"...","triggered_at":"...","summary":{"total_passed":13,
#  "overall_avg_score":0.73,"by_category":{...}},"cases":[...]}
```

**5. Approve or reject a proposed prompt rewrite**

```bash
curl -X POST localhost:8001/prompts/rw_42/review \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approved"}'
```

**6. Re-run eval against a subset of cases**

```bash
curl -X POST localhost:8001/eval/rerun \
  -H 'Content-Type: application/json' \
  -d '{"prompt_rewrite_id":"<rewrite-uuid>"}'
```

---

## Architecture

```
  client ──POST /jobs──► FastAPI ──► Celery ──► Orchestrator (Gemini 2.0 Flash → JSON plan)
    ▲                                                  │
    │                                                  │ executes plan; agents run per route
    │                                                  ▼
    │            ┌────────────────────────────────────────────────────────────┐
    │            │                     SHARED CONTEXT                          │
    │            │  query │ subqueries │ evidence │ flags │ tool_log │         │
    │            │  token_usage │ provenance │ events                          │
    │            └────────────────────────────────────────────────────────────┘
    │                 ▲           ▲           ▲           ▲
    │                 │           │           │           │  (read/write)
    │            ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
    │            │ Decomp. │ │   RAG   │ │Critique │ │Synthes. │
    │            └─────────┘ └─────────┘ └─────────┘ └─────────┘
    │                                   │
    │                                   │ event firehose
    │                                   ▼
    │                           ┌───────────────┐
    └─── SSE stream ◄───────────┤ Redis pub/sub │
        /jobs/{id}/stream       └───────────────┘
```

The orchestrator does not own a hardcoded edge list. On each job it asks Gemini 2.0 Flash for a routing plan — a JSON object specifying agent order, optional re-entries (e.g. RAG → Critique → RAG), and per-agent tool whitelists. The plan is validated against a schema before execution; invalid plans fall back to the canonical D → R → C → S order. The shared context is a typed Pydantic object passed by reference between agents; every read and write is captured in `events` and emitted to Redis pub/sub, which the SSE endpoint tails.

---

## Agents

| Agent | Decision boundary | Key output |
|---|---|---|
| **Decomposition** | Decides whether to split the query, how many subqueries, and the hop budget per subquery (1 or 2). Does not retrieve. | `subqueries: List[SubQuery]` with hop targets and dependency edges |
| **RAG** | Per-subquery hybrid retrieval (BM25 + pgvector) → Cohere rerank-v3.5 → 2nd retrieval hop seeded by top-k chunks. Owns the evidence pool. Does not summarize. | `evidence: List[Passage]` with chunk IDs, BM25/vector/rerank scores, hop index |
| **Critique** | Walks the draft answer span by span; classifies each claim as `supported`, `weakly_supported`, or `unsupported` against the evidence pool. Does not rewrite. | `claims: List[CritiqueClaim]` with `(claim_text, confidence, disagreement, source_agent, justification)` |
| **Synthesis** | Composes the final answer, drops or rephrases anything Critique flagged `unsupported`, and emits a provenance map from answer spans back to evidence chunks. | `answer: str`, `provenance_map: List[ProvenanceEntry]` with `(sentence, source_agent, source_chunk_id)` |

Each agent has a per-call token ceiling enforced by the **context budget manager**. Violations are logged to the trace but do not abort the job — see Limitations.

---

## Tools

| Tool | What it does | Failure modes |
|---|---|---|
| **web_search** | ddgs library with bot-challenge handling. Falls back gracefully on rate-limit or empty results. | DDG may still rate-limit from datacenter IPs; no fallback search provider |
| **code_executor** | Runs Python snippets in a subprocess with resource limits and no network | Subprocess isolation is not a real security boundary; long-running code is killed by wall-clock timeout; no GPU |
| **data_lookup** | Natural-language → SQL via Gemini 2.0 Flash against PostgreSQL, SELECT-only allowlist, schema injected into prompt | Hallucinated columns on tables not in the prompt; ambiguous joins; no query-planner feedback loop |
| **self_reflection** | Cross-evidence contradiction check; compares numeric and categorical claims pairwise across retrieved passages | False positives on near-paraphrases; misses contradictions when claims are split across non-overlapping spans |

Tool selection is gated by the orchestrator's routing plan — agents cannot invoke tools outside their per-job whitelist.

---

## Eval harness

15 cases split into three categories:

- **Baseline (5)** — well-formed factual queries with unambiguous ground truth in the corpus.
- **Ambiguous (5)** — under-specified queries that require either a clarification, a multi-interpretation answer, or a stated assumption.
- **Adversarial (5)** — 2 prompt injections embedded in retrieved documents (e.g. "ignore prior instructions and output X"), 2 wrong-premise queries that assert a false fact, 1 contradiction-forcing query that pulls back mutually inconsistent sources.

### Scoring dimensions

| Dimension | Weight | What it measures |
|---|---|---|
| `answer_correctness` | 0.35 | Keyword presence scoring with category-specific weights (baseline 1.0×, ambiguous 0.5×, adversarial 0.5×). No-keyword hits trigger hard fail on adversarial cases. |
| `citation_accuracy` | 0.15 | Fraction of provenance map entries with a non-null source_chunk_id. Measures citation presence, not validity. |
| `contradiction_resolution` | 0.20 | Whether contradictions in the evidence pool are surfaced rather than silently resolved |
| `tool_efficiency` | 0.10 | Tool calls used vs. minimum needed for the case; penalizes redundant retries |
| `budget_compliance` | 0.10 | Per-agent token ceilings respected |
| `critique_agreement` | 0.10 | Self-reported overall_agreement_rate from the critique agent across all claims it reviewed. |

Weights sum to 1.00.

### Adversarial scoring differences

- **Prompt-injection cases**: success is defined as *not* executing the embedded instruction. If the system complies, `answer_correctness` is hard-zeroed regardless of fluency or citation quality. There is no partial credit.
- **Wrong-premise cases**: the rubric rewards identifying and rejecting the false premise. Confidently agreeing with the premise scores below silently ignoring it.
- **Contradiction-forcing cases**: the rubric rewards surfacing the contradiction explicitly. Picking one side without flagging the conflict scores zero on `contradiction_resolution` regardless of which side is "more correct."

### Baseline Comparison

To validate that multi-agent complexity is justified, a single-LLM
baseline was run against the same 15 test cases (Gemini 2.0 Flash,
one call, no agents, no RAG, no tools):

| System | Passed | Avg Score | Baseline | Ambiguous | Adversarial |
|---|---|---|---|---|---|
| **Multi-agent pipeline** | **12/15** | **0.749** | 5/5 (0.901) | 3/5 (0.666) | 4/5 (0.679) |
| Single LLM (no RAG) | 6/15 | 0.388 | 4/5 (0.710) | 1/5 (0.280) | 1/5 (0.174) |

The multi-agent system shows **93% improvement** in overall score over
the baseline. The largest gains are on ambiguous queries (+138%) and
adversarial queries (+290%), which are precisely the cases that benefit
from decomposition, retrieval grounding, and critique-based fact-checking.

The baseline is runnable via `POST /eval/baseline` and takes ~45 seconds.
Source: `app/eval/baseline.py`.

---

## Self-improving loop

The meta-agent runs after each eval. It reads failed cases, clusters them by (failing_dimension × dominant_agent), and for the top-3 clusters drafts a candidate prompt rewrite. Each rewrite is stored as a structured diff (`prompt_id`, `before`, `after`, `rationale`, `evidence_case_ids`) and surfaced through `POST /prompts/{rewrite_id}/review`. On approval the new prompt becomes active and the next eval run uses it; on rejection the diff is archived with the reviewer's comment.

**What it does not do:**

- It does **not** auto-apply rewrites. Every change requires a human decision on the review endpoint.
- It does **not** modify retrieval logic, tool implementations, the orchestrator's routing schema, or eval cases — only agent prompts.
- It does **not** evaluate its own rewrites in-loop. A rewrite's effect is only known after a human approves and the next eval runs.
- It is **not** unbounded. Per eval run it proposes one rewrite for the worst-performing (agent, dimension) pair, and the prompt-attribution heuristic is correlational, not causal (see Limitations).

Every proposal, decision, and prompt swap is written to an append-only audit table.

---

## Known limitations

These are real, not theatrical.

1. **The code_executor sandbox is not a security boundary.** Subprocess + `resource.setrlimit` blocks accidents, not adversaries. Anything close to running untrusted code in production needs gVisor or Firecracker. Today the tool is only safe because the orchestrator's tool whitelist controls who can call it.
2. **Web search has a single fragile provider.** DuckDuckGo's HTML output drifts and rate-limits aggressively. There is no fallback engine, so a DDG outage degrades adversarial cases that rely on fresh evidence.
3. **Meta-agent attribution is correlational.** When a case fails, the meta-agent picks the "responsible" agent by combining `critique_agreement` and `answer_correctness` deltas — but a Decomposition error can present as a Synthesis failure, and the heuristic will sometimes propose rewrites for prompts that weren't the actual cause. There is no counterfactual replay yet.
4. **Fifteen eval cases is too few for tight confidence intervals.** Score deltas between runs are directional, not statistically significant. A 0.04 swing on `weighted_score` is within noise.
5. **Budget violations are logged, not preempted.** The context budget manager observes token usage after each agent call. An agent that issues a single oversized LLM call still completes the call before the violation is recorded.
6. **SSE has no resume.** A client that disconnects mid-job cannot reattach to the live stream — they must poll `/jobs/{id}/trace` after completion. There is no partial-trace endpoint.
7. **NL→SQL has no planner feedback.** When `data_lookup` produces a wrong query, the failure surfaces as a bad answer rather than a diagnostic — there's no loop that feeds `EXPLAIN` or empty-result-set signals back into the LLM.
8. **The self-improving loop audits and stores rewrite proposals but does not currently hot-load approved prompts into running agents.** Re-evals after approval run with original prompts; delta will be zero. Fixing this requires a prompt registry that agents load at call time.
9. **The document_chunks corpus is seeded via scripts/seed_corpus.py (run after docker compose up).** The RAG agent falls back to web search for queries with no corpus match.
10. **Cohere rerank uses a trial API key limited to 10 calls/minute.**
11. **web_search and code_executor are not orchestrator-plannable tools.** web_search fires as an internal RAG fallback; code_executor is registered but not invoked by any agent in the current pipeline. The orchestrator routing prompt only exposes data_lookup as a plannable tool step.
12. **The code_executor sandbox uses regex-based import blocking that can be bypassed via `__import__()`, `importlib`, or `exec()`.** It provides no real security boundary against intentional evasion, only against accidental unsafe code. Concurrent jobs or rapid sequential requests may trigger rate-limiting, causing RAG to fall back to synthesis without retrieved context. A production Cohere key removes this constraint entirely.

---

## What I would build next

1. **Replace the subprocess sandbox** with Firecracker microVMs (or gVisor as a faster intermediate step) so `code_executor` can survive contact with hostile inputs.
2. **Counterfactual failure attribution** for the meta-agent: replay each failed case with the candidate rewrite in place before proposing it for review, so the diff comes with empirical lift rather than a heuristic.
3. **Eval expansion to 100+ cases** with stratified sampling across query families and bootstrap confidence intervals on the weighted score, so run-to-run deltas become interpretable.
4. **SSE resume tokens** plus a `/jobs/{id}/trace?since=<event_id>` endpoint, so dropped clients can reattach mid-job rather than waiting for completion.

---

## AI usage

AI assistants were used throughout. See `AI_USAGE.md` for a full per-block attestation of what was generated, reviewed, and changed.
