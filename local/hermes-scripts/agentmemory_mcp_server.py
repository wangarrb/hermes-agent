#!/usr/bin/env python3
"""agentmemory MCP server"""

import os
import json
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agentmemory")

_client = None

def _get_am():
    global _client
    if _client is None:
        import agentmemory
        _client = agentmemory
    return _client


@mcp.tool()
def am_create(category: str, text: str, metadata: Optional[str] = None) -> str:
    """Create a memory in agentmemory. metadata is JSON string or null."""
    am = _get_am()
    meta = json.loads(metadata) if metadata else {}
    mid = am.create_memory(category, text, metadata=meta)
    return mid or "ok"


@mcp.tool()
def am_search(category: str, query: str, n_results: int = 5, include_embeddings: bool = False) -> str:
    """Search memories by semantic query. Returns JSON array of results.
    NOTE: include_embeddings=True has a known bug (returns only 1 result).
    Default False returns all matches correctly."""
    am = _get_am()
    results = am.search_memory(
        category, query,
        n_results=n_results,
        include_embeddings=include_embeddings,
        include_distances=True,
    )
    out = []
    for r in results:
        dist = r.get("distance", 0)
        try:
            dist = round(float(dist), 4)
        except (ValueError, TypeError):
            dist = 0
        item = {"id": r.get("id"), "document": r.get("document", "")[:500], "distance": dist, "metadata": r.get("metadata")}
        out.append(item)
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def am_get(category: str, memory_id: str) -> str:
    """Get a single memory by ID. Returns JSON."""
    am = _get_am()
    results = am.get_memory(category, ids=[memory_id])
    if not results:
        return json.dumps({"error": "not found"})
    r = results[0] if isinstance(results, list) else results
    return json.dumps({"id": r.get("id"), "document": r.get("document", ""), "metadata": r.get("metadata")}, ensure_ascii=False)


@mcp.tool()
def am_update(category: str, memory_id: str, text: Optional[str] = None, metadata: Optional[str] = None) -> str:
    """Update a memory. metadata is JSON string or null."""
    am = _get_am()
    meta = json.loads(metadata) if metadata else None
    am.update_memory(category, memory_id, text=text, metadata=meta)
    return "ok"


@mcp.tool()
def am_delete(category: str, memory_id: str) -> str:
    """Delete a memory by ID."""
    am = _get_am()
    am.delete_memory(category, memory_id)
    return "ok"


@mcp.tool()
def am_list_categories() -> str:
    """List all memory categories (collections)."""
    am = _get_am()
    client = am.get_client()
    collections = client.list_collections()
    names = [c.name if hasattr(c, 'name') else str(c) for c in collections]
    return json.dumps(names)


@mcp.tool()
def am_count(category: str) -> str:
    """Count memories in a category."""
    am = _get_am()
    return str(am.count_memories(category))


@mcp.tool()
def am_search_multi(category: str, queries: str, n_results: int = 3) -> str:
    """Search with multiple queries (JSON array), merge and deduplicate results."""
    am = _get_am()
    query_list = json.loads(queries)
    seen = set()
    all_results = []
    for q in query_list:
        results = am.search_memory(category, q, n_results=n_results, include_embeddings=False, include_distances=True)
        for r in results:
            rid = r.get("id")
            if rid not in seen:
                seen.add(rid)
                dist = r.get("distance", 0)
                try:
                    dist = round(float(dist), 4)
                except (ValueError, TypeError):
                    dist = 0
                all_results.append({"id": rid, "document": r.get("document", "")[:500], "distance": dist, "metadata": r.get("metadata")})
    all_results.sort(key=lambda x: x["distance"])
    return json.dumps(all_results[:n_results * 3], ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
