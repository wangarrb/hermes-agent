#!/usr/bin/env python3
"""Wait for current external 10% import banks and write a compact quality report."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "http://127.0.0.1:8888"
BANKS = [
    "external_chatmemo_10pct_20260518",
    "external_openclaw_10pct_20260518",
]
QUALITY_FILES = {
    "external_chatmemo_10pct_20260518": "/home/wyr/.hermes/hindsight/external_import/manifests/10pct-20260518/chatmemo/20260518-214615-chatmemo-10pct-quality.json",
    "external_openclaw_10pct_20260518": "/home/wyr/.hermes/hindsight/external_import/manifests/10pct-20260518/openclaw/20260518-214615-openclaw-10pct-quality.json",
}
RECALL_QUERIES = {
    "external_chatmemo_10pct_20260518": [
        "AEB 单目测速 CUSUM 回溯",
        "Science 子刊 自动驾驶 世界模型 投稿",
        "驾驶习惯优化系统 AEB 优先级",
    ],
    "external_openclaw_10pct_20260518": [
        "NAS3R WorldTree 自监督3D重建",
        "OpenClaw 代理 使用规则",
        "SparseWorld MambaOcc RK3588",
    ],
}
BAD_MARKERS = [
    "untrusted metadata",
    "sender_id",
    "thinkingSignature",
    "toolCall",
    "toolResult",
    "System (untrusted)",
    "Conversation info (untrusted",
]
OUT = Path("/home/wyr/.hermes/hindsight/external_import/evals/20260518-10pct-final-report.json")


def api_json(method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(API.rstrip("/") + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    return json.loads(raw) if raw.strip() else {}


def bank_path(bank: str, suffix: str) -> str:
    return f"/v1/default/banks/{urllib.parse.quote(bank, safe='')}/{suffix.lstrip('/')}"


def stats(bank: str) -> dict[str, Any]:
    return api_json("GET", bank_path(bank, "stats"), timeout=30)


def ready(bank: str) -> bool:
    s = stats(bank)
    return int(s.get("pending_consolidation") or 0) <= 0 and int(s.get("pending_operations") or 0) <= 0 and int(s.get("failed_consolidation") or 0) <= 0


def wait_all(timeout_s: int = 7200, poll_s: int = 60) -> dict[str, Any]:
    start = time.time()
    last: dict[str, Any] = {}
    while True:
        last = {bank: stats(bank) for bank in BANKS}
        if all(
            int(v.get("pending_consolidation") or 0) <= 0
            and int(v.get("pending_operations") or 0) <= 0
            and int(v.get("failed_consolidation") or 0) <= 0
            for v in last.values()
        ):
            return {"ready": True, "elapsed_s": int(time.time() - start), "stats": last}
        if time.time() - start >= timeout_s:
            return {"ready": False, "timeout": True, "elapsed_s": int(time.time() - start), "stats": last}
        time.sleep(poll_s)


def list_items(bank: str, suffix: str, limit: int = 100) -> list[dict[str, Any]]:
    data = api_json("GET", bank_path(bank, suffix) + f"?limit={limit}", timeout=60)
    return [x for x in data.get("items", []) if isinstance(x, dict)]


def recall(bank: str, query: str) -> dict[str, Any]:
    payload = {"query": query, "types": ["world", "observation", "experience"], "budget": "mid", "max_tokens": 1200, "limit": 5}
    data = api_json("POST", bank_path(bank, "memories/recall"), payload=payload, timeout=120)
    previews: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            text = x.get("text")
            if isinstance(text, str) and text not in previews:
                previews.append(text)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return {"query": query, "preview": previews[:5]}


def marker_scan(bank: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for typ in ["world", "observation", "experience"]:
        for item in list_items(bank, f"memories/list?type={typ}", limit=500):
            text = item.get("text") or ""
            found = [m for m in BAD_MARKERS if m in text]
            if found:
                hits.append({"id": item.get("id"), "type": typ, "markers": found, "text_preview": text[:300]})
    return hits


def main() -> int:
    wait_result = wait_all()
    report: dict[str, Any] = {"wait": wait_result, "banks": {}}
    for bank in BANKS:
        qf = QUALITY_FILES.get(bank)
        quality = json.loads(Path(qf).read_text(encoding="utf-8")) if qf and Path(qf).exists() else None
        s = stats(bank)
        report["banks"][bank] = {
            "stats": s,
            "quality_manifest": quality,
            "sample_documents": list_items(bank, "documents", limit=5),
            "sample_observations": list_items(bank, "memories/list?type=observation", limit=8),
            "sample_world": list_items(bank, "memories/list?type=world", limit=8),
            "bad_marker_hits_in_memory": marker_scan(bank),
            "recall_smoke": [recall(bank, q) for q in RECALL_QUERIES.get(bank, [])],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"external 10pct final report: {OUT}", f"ready={wait_result.get('ready')} elapsed_s={wait_result.get('elapsed_s')}"]
    for bank in BANKS:
        s = report["banks"][bank]["stats"]
        lines.append(
            f"{bank}: docs={s.get('total_documents')} nodes={s.get('total_nodes')} obs={s.get('total_observations')} "
            f"pending={s.get('pending_consolidation')} failed={s.get('failed_operations')}/{s.get('failed_consolidation')} "
            f"bad_markers={len(report['banks'][bank]['bad_marker_hits_in_memory'])}"
        )
    print("\n".join(lines))
    return 0 if wait_result.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
