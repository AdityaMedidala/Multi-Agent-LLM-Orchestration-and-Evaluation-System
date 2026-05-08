from __future__ import annotations


async def web_search(query: str, max_results: int = 5) -> "ToolResult":
    from app.tools.registry import ToolResult
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return ToolResult(
            tool_name="web_search",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="malformed",
        )

    if not raw:
        return ToolResult(
            tool_name="web_search",
            success=False,
            output={"results": []},
            latency_ms=0,
            failure_reason="empty",
        )

    results = [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
            "relevance_score": round(max(1.0 - i * 0.1, 0.1), 1),
        }
        for i, r in enumerate(raw)
    ]

    return ToolResult(
        tool_name="web_search",
        success=True,
        output={"results": results, "query": query},
        latency_ms=0,
    )
