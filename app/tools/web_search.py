from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_DDG_URL = "https://api.duckduckgo.com/"


@retry(
    stop=stop_after_attempt(3),          # 1 initial + 2 retries
    wait=wait_exponential(min=1, max=4),
    retry=retry_if_exception_type(httpx.TimeoutException),
    reraise=True,
)
async def _fetch(client: httpx.AsyncClient, query: str) -> httpx.Response:
    return await client.get(
        _DDG_URL,
        params={"q": query, "format": "json", "no_html": "1"},
    )


async def web_search(query: str, max_results: int = 5) -> ToolResult:
    from app.tools.registry import ToolResult  # deferred to break circular import with registry.py
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            response = await _fetch(client, query)
        except httpx.TimeoutException:
            return ToolResult(
                tool_name="web_search",
                success=False,
                output={},
                latency_ms=0,
                failure_reason="timeout",
            )
        except Exception:
            return ToolResult(
                tool_name="web_search",
                success=False,
                output={},
                latency_ms=0,
                failure_reason="malformed",
            )

    try:
        data = response.json()
        related: list = data["RelatedTopics"]  # KeyError → malformed
    except (ValueError, KeyError):
        return ToolResult(
            tool_name="web_search",
            success=False,
            output={},
            latency_ms=0,
            failure_reason="malformed",
        )

    # Flatten leaf topics (skip category nodes that have a "Topics" sub-list)
    flat: list[dict] = []
    for item in related:
        if "FirstURL" in item and item["FirstURL"]:
            flat.append(item)
        elif "Topics" in item:
            flat.extend(t for t in item["Topics"] if "FirstURL" in t and t["FirstURL"])

    flat = flat[:max_results]

    if not flat:
        return ToolResult(
            tool_name="web_search",
            success=False,
            output={"results": []},
            latency_ms=0,
            failure_reason="empty",
        )

    results = []
    for i, item in enumerate(flat):
        text: str = item.get("Text", "")
        parts = text.split(" - ", 1)
        title = parts[0].strip() if parts else text
        snippet = parts[1].strip() if len(parts) > 1 else text
        relevance = max(0.1, round(1.0 - i * 0.1, 1))
        results.append(
            {
                "title": title,
                "url": item["FirstURL"],
                "snippet": snippet,
                "relevance_score": relevance,
            }
        )

    return ToolResult(
        tool_name="web_search",
        success=True,
        output={"results": results, "query": query},
        latency_ms=0,
    )
