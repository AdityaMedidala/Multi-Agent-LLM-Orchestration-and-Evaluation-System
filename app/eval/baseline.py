from __future__ import annotations
import asyncio
import json
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from app.eval.test_cases import TEST_CASES


async def run_baseline_case(query: str) -> dict:
    """Single LLM call — no RAG, no agents, no tools."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    start = time.monotonic()
    response = await llm.ainvoke([
        ("system", "Answer the user's question accurately and concisely."),
        ("human", query),
    ])
    latency = int((time.monotonic() - start) * 1000)
    return {
        "answer": response.content.strip(),
        "latency_ms": latency,
        "tokens": len(response.content.split()),
    }


async def run_baseline_eval() -> dict:
    """
    Run all 15 eval cases through a single LLM call with no orchestration.
    Scores answer_correctness only — used to justify multi-agent complexity.
    """
    from app.eval.runner import score_answer_correctness

    results = []
    for case in TEST_CASES:
        result = await run_baseline_case(case["query"])
        score = score_answer_correctness(
            answer=result["answer"],
            expected_keywords=case.get("expected_answer_keywords", []),
            expected_no_keywords=case.get("expected_no_keywords", []),
            category=case["category"],
        )
        results.append({
            "case_id": case["id"],
            "category": case["category"],
            "query": case["query"][:80],
            "score": round(score, 3),
            "passed": score >= 0.5,
            "answer_snippet": result["answer"][:120],
            "latency_ms": result["latency_ms"],
        })
        await asyncio.sleep(1)

    by_cat: dict = {}
    for cat in ("baseline", "ambiguous", "adversarial"):
        cat_cases = [r for r in results if r["category"] == cat]
        by_cat[cat] = {
            "avg_score": round(
                sum(r["score"] for r in cat_cases) / len(cat_cases), 3
            ),
            "passed": sum(1 for r in cat_cases if r["passed"]),
            "count": len(cat_cases),
        }

    overall_avg = round(sum(r["score"] for r in results) / len(results), 3)
    total_passed = sum(1 for r in results if r["passed"])

    return {
        "model": "gemini-2.0-flash (single call, no RAG, no agents)",
        "total_cases": len(results),
        "total_passed": total_passed,
        "overall_avg_score": overall_avg,
        "by_category": by_cat,
        "cases": results,
    }


if __name__ == "__main__":
    results = asyncio.run(run_baseline_eval())
    print(json.dumps(results, indent=2))
