#!/usr/bin/env python3
"""Evaluation harness for Hindsight recall quality.

Compares:
- direct Hindsight recall
- layered recall helper (local rerank + query variants)

No LLM calls. No Hindsight writes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_BENCH = HERMES_HOME / "hindsight" / "eval" / "benchmark_queries.jsonl"
DEFAULT_OUT = HERMES_HOME / "hindsight" / "eval" / "runs"
DEFAULT_CARDS_ROOT = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_cards"
DEFAULT_API = "http://127.0.0.1:8888"
DEFAULT_BANK = "hermes"
LAYERED_PATH = HERMES_HOME / "scripts" / "hindsight_recall_layered.py"


def load_layered_module():
    spec = importlib.util.spec_from_file_location("hindsight_recall_layered", str(LAYERED_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LAYERED_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            raise SystemExit(f"Invalid JSONL line {i}: {e}")
    return rows


def classify_doc(document_id: str) -> str:
    if document_id.startswith("hermes-offline-canonical::"):
        return "canonical"
    if document_id.startswith("hermes-offline-consolidation::weekly::"):
        return "offline_weekly"
    if document_id.startswith("hermes-offline-consolidation::daily::"):
        return "offline_daily"
    if document_id.startswith("hermes-sqlite::"):
        return "sqlite_import"
    return "other"


def text_contains(text: str, term: str) -> bool:
    if not term:
        return True
    return term.lower() in text.lower()


def evaluate_results(case: dict[str, Any], results: list[dict[str, Any]], k: int) -> dict[str, Any]:
    non_sidecar_top: list[dict[str, Any]] = []
    sidecar = []
    for r in results:
        if r.get("_sidecar"):
            sidecar.append(r)
            continue
        if len(non_sidecar_top) < k:
            non_sidecar_top.append(r)
    top = non_sidecar_top + sidecar
    combined = "\n".join((r.get("text") or "") + "\n" + (r.get("document_id") or "") for r in top)
    expected_terms = case.get("expected_terms") or []
    hit_terms = [t for t in expected_terms if text_contains(combined, str(t))]
    missing_terms = [t for t in expected_terms if t not in hit_terms]
    layers = [r.get("layer") or classify_doc(r.get("document_id") or "") for r in top]
    layer_counts: dict[str, int] = {}
    for layer in layers:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    expected_layers = set(case.get("expected_layers") or [])
    expected_layer_hits = sum(
        1
        for layer in layers
        if layer in expected_layers or (layer == "local_canonical" and "canonical" in expected_layers)
    )
    has_source_trace = any((r.get("metadata") or {}).get("source_count") or (r.get("metadata") or {}).get("output_json") or r.get("source_fact_ids") for r in top)
    term_recall = len(hit_terms) / max(len(expected_terms), 1)
    # Weighted score: term recall is primary; layer/source are secondary.
    score = 100.0 * term_recall
    if expected_layers:
        score += min(20.0, 5.0 * expected_layer_hits)
    if has_source_trace:
        score += 5.0
    return {
        "term_recall": term_recall,
        "hit_terms": hit_terms,
        "missing_terms": missing_terms,
        "layer_counts": layer_counts,
        "expected_layer_hits_top_k": expected_layer_hits,
        "evaluated_non_sidecar_count": len(non_sidecar_top),
        "evaluated_sidecar_count": len(sidecar),
        "evaluated_total_count": len(top),
        "has_source_trace": has_source_trace,
        "score": round(score, 2),
        "top_preview": [
            {
                "rank": i + 1,
                "layer": layers[i] if i < len(layers) else classify_doc(top[i].get("document_id") or ""),
                "type": top[i].get("type"),
                "document_id": top[i].get("document_id"),
                "text": (top[i].get("text") or "").replace("\n", " ")[:220],
            }
            for i in range(len(top))
        ],
    }


def direct_recall(mod: Any, api: str, bank: str, query: str, limit: int, *, include_observations: bool = True) -> list[dict[str, Any]]:
    results = mod.recall(api, bank, query, limit, include_observations=include_observations)
    out = []
    for rank, r in enumerate(results, 1):
        rr = dict(r)
        rr["layer"] = classify_doc(rr.get("document_id") or "")
        rr["score"] = 100 - rank
        out.append(rr)
    return out


def run_eval(cases: list[dict[str, Any]], *, api: str, bank: str, top_k: int, raw_limit: int, limit: int, cards_root: str | None = None, include_observations: bool = True, local_sidecar_limit: int | None = None) -> dict[str, Any]:
    mod = load_layered_module()
    results = []
    aggregate = {
        "direct_score_sum": 0.0,
        "layered_score_sum": 0.0,
        "direct_term_recall_sum": 0.0,
        "layered_term_recall_sum": 0.0,
        "direct_expected_layer_hits": 0,
        "layered_expected_layer_hits": 0,
    }
    for case in cases:
        q = case["query"]
        mode = case.get("mode") or "mixed"
        direct = direct_recall(mod, api, bank, q, raw_limit, include_observations=include_observations)
        layered = mod.layered_recall(api, bank, q, mode, raw_limit, limit, cards_root=cards_root, include_observations=include_observations, local_sidecar_limit=local_sidecar_limit)
        direct_eval = evaluate_results(case, direct, top_k)
        layered_eval = evaluate_results(case, layered, top_k)
        aggregate["direct_score_sum"] += direct_eval["score"]
        aggregate["layered_score_sum"] += layered_eval["score"]
        aggregate["direct_term_recall_sum"] += direct_eval["term_recall"]
        aggregate["layered_term_recall_sum"] += layered_eval["term_recall"]
        aggregate["direct_expected_layer_hits"] += direct_eval["expected_layer_hits_top_k"]
        aggregate["layered_expected_layer_hits"] += layered_eval["expected_layer_hits_top_k"]
        results.append({
            "id": case.get("id"),
            "query": q,
            "mode": mode,
            "direct": direct_eval,
            "layered": layered_eval,
        })
    n = max(len(cases), 1)
    summary = {
        "case_count": len(cases),
        "top_k": top_k,
        "direct_avg_score": round(aggregate["direct_score_sum"] / n, 2),
        "layered_avg_score": round(aggregate["layered_score_sum"] / n, 2),
        "direct_avg_term_recall": round(aggregate["direct_term_recall_sum"] / n, 3),
        "layered_avg_term_recall": round(aggregate["layered_term_recall_sum"] / n, 3),
        "direct_expected_layer_hits": aggregate["direct_expected_layer_hits"],
        "layered_expected_layer_hits": aggregate["layered_expected_layer_hits"],
    }
    return {"generated_at": datetime.now().isoformat(timespec="seconds"), "cards_root": cards_root, "include_hindsight_observations": include_observations, "summary": summary, "results": results}


def render_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# Hindsight Recall Evaluation")
    lines.append("")
    lines.append(f"generated_at: {report['generated_at']}")
    lines.append(f"cards_root: {report.get('cards_root')}")
    lines.append(f"include_hindsight_observations: {report.get('include_hindsight_observations')}")
    lines.append("")
    s = report["summary"]
    lines.append("## Summary")
    lines.append(f"- cases: {s['case_count']}, top_k: {s['top_k']}")
    lines.append(f"- direct_avg_score: {s['direct_avg_score']}")
    lines.append(f"- layered_avg_score: {s['layered_avg_score']}")
    lines.append(f"- direct_avg_term_recall: {s['direct_avg_term_recall']}")
    lines.append(f"- layered_avg_term_recall: {s['layered_avg_term_recall']}")
    lines.append(f"- direct_expected_layer_hits: {s['direct_expected_layer_hits']}")
    lines.append(f"- layered_expected_layer_hits: {s['layered_expected_layer_hits']}")
    lines.append("")
    lines.append("## Cases")
    for r in report["results"]:
        lines.append(f"### {r['id']}")
        lines.append(f"query: {r['query']}")
        lines.append(f"mode: {r['mode']}")
        d = r["direct"]
        l = r["layered"]
        lines.append(f"- direct: score={d['score']} term_recall={d['term_recall']:.2f} layers={d['layer_counts']} missing={d['missing_terms']}")
        lines.append(f"- layered: score={l['score']} term_recall={l['term_recall']:.2f} layers={l['layer_counts']} missing={l['missing_terms']}")
        if l["top_preview"]:
            top = l["top_preview"][0]
            lines.append(f"- layered top1: {top['layer']} / {top['text']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Hindsight recall quality")
    ap.add_argument("--benchmark", default=str(DEFAULT_BENCH))
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--raw-limit", type=int, default=40)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--cards-root", default=str(DEFAULT_CARDS_ROOT), help="Local v2 canonical cards root; used by default when the directory exists")
    ap.add_argument("--use-local-cards", action="store_true", help="Force-enable local v2 canonical cards in layered recall")
    ap.add_argument("--no-local-cards", action="store_true", help="Disable local v2 canonical cards in layered recall")
    ap.add_argument("--no-hindsight-observations", action="store_true", help="Do not request fact_type=observation from Hindsight; useful for raw baseline eval")
    ap.add_argument("--local-sidecar-limit", type=int, default=None, help="Extra local_canonical sidecar results appended after non-local limit")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    bench = Path(args.benchmark).expanduser()
    if not bench.exists():
        raise SystemExit(f"benchmark file not found: {bench}")
    cases = load_benchmark(bench)
    cards_root_path = Path(args.cards_root).expanduser()
    use_cards = not args.no_local_cards and (args.use_local_cards or cards_root_path.exists())
    cards_root = str(cards_root_path) if use_cards else None
    report = run_eval(cases, api=args.api, bank=args.bank, top_k=args.top_k, raw_limit=args.raw_limit, limit=args.limit, cards_root=cards_root, include_observations=not args.no_hindsight_observations, local_sidecar_limit=args.local_sidecar_limit)
    outdir = Path(args.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = outdir / f"hindsight-eval-{ts}.json"
    md_path = outdir / f"hindsight-eval-{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    (outdir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path), **report}, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
        print(f"saved_json: {json_path}")
        print(f"saved_markdown: {md_path}")


if __name__ == "__main__":
    main()
