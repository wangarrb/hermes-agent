#!/usr/bin/env python3
"""Hindsight manual_review sampler: score, sample, and LLM-review stuck sessions.

The session manifest classifier routes ambiguous sessions to ``manual_review``
(bootstrap diagnostic, multi-scope tags, memory-recall bootstrap, ...). Nothing
consumed that queue, so real work sessions were silently never retained (8-day
ingest blackout discovered 2026-09-04). Full manual review of ~2600 stuck
records is unrealistic, so this tool:

1. Scores every pending manual_review record for work-signal importance
   (content volume, semantic tag density, project tags, recency, longest
   single message — long assistant replies indicate real work).
2. Samples the top-N records per run (default 5) deterministically
   (score desc, then document_id for stable ordering).
3. Asks the review LLM (same config as the offline pipeline) to judge each
   sampled record: real work vs noise.
4. Writes two artifacts:
     - a reviewed-curate manifest JSONL (production-actioned records with
       ``reason=llm_review_promoted``) that hindsight_session_retain_runner.py
       can consume directly,
     - a JSON decision log under review_repair/ for auditability.
Promoted records keep their original event_date and tags; nothing is deleted
from the manual_review pool, so repeated runs with different samples are safe.

Usage (typically from hindsight_memory_pipeline.py as a step before retain):
    python3 hindsight_manual_review_sampler.py --manifest <manifest.jsonl> \
        --sample-size 5 --execute --confirm retain-hindsight-session-manifest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_SIZE = 5
DEFAULT_MANIFEST_DIR = "/home/wyr/.hermes/hindsight/session_ingest/manifests"
DEFAULT_REVIEW_DIR = "/home/wyr/.hermes/hindsight/review_repair"
# Domain/task keywords that indicate real work content (any language).
WORK_SIGNAL_RE = re.compile(
    r"(实现|算法|实验|误差|精度|训练|评估|论文|arxiv|修复|部署|baseline|pipeline|"
    r"commit|指标|benchmark|fusion|滤波|检测|跟踪|测距|方案|验证|failure|ablation)",
    re.IGNORECASE,
)
# Bootstrap/noise markers that reduce score.
NOISE_SIGNAL_RE = re.compile(
    r"(^(hi|hello|你好|test|ping)\b|转发成功|esc to|继续|go on)",
    re.IGNORECASE,
)


def load_records(manifest_path: Path) -> list[dict[str, Any]]:
    records = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_reviewed_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"reviewed": {}, "history": []}


def importance_score(record: dict[str, Any]) -> float:
    """Heuristic work-signal score. Higher = more likely real work."""
    score = 0.0
    chars = float(record.get("content_chars") or 0)
    # Content volume (log-scaled: 10K chars ≈ 4, 100K ≈ 5.7)
    import math
    score += min(math.log10(max(chars, 1)) * 10.0, 60.0)
    tags = [t for t in (record.get("tags") or [])]
    semantic = [t for t in tags if t.startswith(("domain:", "project:", "topic:"))]
    # Semantic specificity
    score += min(len(semantic) * 3.0, 15.0)
    # Project-tagged content is more likely real work
    score += 5.0 * len([t for t in tags if t.startswith("project:")])
    # Recency (newer records first; event_date ISO)
    event = str(record.get("event_date") or "")
    try:
        ts = datetime.fromisoformat(event.replace("Z", "+00:00")).timestamp()
        age_days = max((time.time() - ts) / 86400.0, 0.0)
        score += max(0.0, 20.0 - age_days)  # up to 20 points for the last 20 days
    except Exception:
        pass
    # Text signals from the preview/metadata text if available
    meta = record.get("metadata") or {}
    text_blob = " ".join(
        str(x)
        for x in [
            meta.get("text_preview"),
            meta.get("preview"),
            record.get("context"),
        ]
        if x
    )
    score += 4.0 * len(WORK_SIGNAL_RE.findall(text_blob))
    score -= 3.0 * len(NOISE_SIGNAL_RE.findall(text_blob))
    # Deterministic tie-breaker contribution (tiny, based on id hash)
    score += (hash(record.get("document_id", "")) % 100) / 10000.0
    return round(score, 4)


def sample_records(
    records: list[dict[str, Any]], sample_size: int, reviewed: dict[str, Any]
) -> list[dict[str, Any]]:
    pending = [
        r
        for r in records
        if (
            r.get("action") == "manual_review"
            and r.get("document_id") not in reviewed
            # Never sample/promote secret-flagged records: the promote path
            # would submit their full (unredacted) content to retain.
            and r.get("reason") != "secret_or_credential_material"
        )
    ]
    scored = [(importance_score(r), r["document_id"], r) for r in pending]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [r for _, _, r in scored[:sample_size]]


def _rehydrate_preview(record: dict[str, Any], max_chars: int = 1500) -> str:
    """Get real conversation text for LLM review.

    Lean manifests omit content (content_omitted=true); the only text is a
    session-id preview, which made the review LLM judge everything "noise"
    (first sampler run 2026-09-04 promoted 0/3). Rehydrate from the source
    session JSON via the manifest module, mirroring the retain runner.
    """
    try:
        sys.path.insert(0, "/home/wyr/.hermes/scripts")
        import hindsight_session_manifest as session_manifest  # noqa: E402

        meta = record.get("metadata") or {}
        source_path = meta.get("json_path")
        if not source_path:
            return ""
        candidates = session_manifest.records_from_json_file(
            source_path,
            bank_target=record.get("bank_target")
            or meta.get("bank_target")
            or session_manifest.DEFAULT_BANK_TARGET,
            source_profile=str(meta.get("source_profile") or "default"),
        )
        doc_id = record.get("document_id")
        for candidate in candidates:
            if candidate.get("document_id") == doc_id and candidate.get("content"):
                return str(candidate["content"])
        # Split documents (::part-NNN) rehydrate to multiple chunks; fall back
        # to the first candidate's content for preview purposes.
        for candidate in candidates:
            if candidate.get("content"):
                return str(candidate["content"])
    except Exception as exc:
        print(f"  rehydrate failed for {record.get('document_id')}: {exc}", file=sys.stderr)
    return ""


def build_llm_prompt(record: dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    preview = str(
        meta.get("text_preview")
        or meta.get("preview")
        or record.get("context")
        or ""
    )[:1200]
    # Lean manifest: preview is just the session id — rehydrate real text.
    if len(preview) < 200:
        preview = _rehydrate_preview(record, max_chars=1500) or preview
    tags = ", ".join(record.get("tags") or [])
    return (
        "判断以下 Hermes 会话是否包含真实工作内容（技术分析、实验、代码、决策、"
        "论文阅读等），还是只是启动引导/闲聊/环境检查。\n\n"
        f"document_id: {record.get('document_id')}\n"
        f"分类原因: {record.get('reason')}\n"
        f"内容字符数: {record.get('content_chars')}\n"
        f"标签: {tags}\n"
        f"内容预览（截取）:\n{preview}\n\n"
        "只回答 JSON：{\"verdict\": \"production\" | \"noise\", \"confidence\": 0-1, "
        "\"why\": \"一句话理由\"}\n"
        "production=值得进入知识库的真实工作；noise=无信息量。不确定时选 noise。"
    )


def call_review_llm(prompt: str) -> dict[str, Any] | None:
    """Use the same LLM config as hindsight_daily_noagent's research summary."""
    try:
        sys.path.insert(0, "/home/wyr/.hermes/scripts")
        from hindsight_daily_noagent import _call_hindsight_llm  # noqa: E402

        raw = _call_hindsight_llm(
            "你是数据质量审查员。只输出 JSON，不输出其他文字。",
            prompt,
            max_tokens=300,
        )
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception as exc:  # LLM unavailable → fail open, keep manual
        print(f"  LLM review unavailable: {exc}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None, help="manifest JSONL path (default: latest in manifest dir)")
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--execute", action="store_true", help="run LLM review and write curated manifest (default: report only)")
    parser.add_argument("--submit", action="store_true", help="after review, submit promoted records via hindsight_session_retain_runner (waits for completion)")
    parser.add_argument("--bank", default="hermes")
    parser.add_argument("--auto-promote", action="store_true", help="promote production-verdict records without LLM (testing only)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is None:
        candidates = sorted(Path(args.manifest_dir).glob("*-session-manifest.jsonl"))
        if not candidates:
            print("no manifests found", file=sys.stderr)
            return 1
        manifest_path = candidates[-1]

    records = load_records(manifest_path)
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    state_path = review_dir / "manual_review_sampler_state.json"
    state = load_reviewed_state(state_path)
    reviewed: dict[str, Any] = state.get("reviewed", {})

    pending_count = sum(
        1
        for r in records
        if (
            r.get("action") == "manual_review"
            and r.get("document_id") not in reviewed
            # Never sample/promote secret-flagged records: the promote path
            # would submit their full (unredacted) content to retain.
            and r.get("reason") != "secret_or_credential_material"
        )
    )
    sample = sample_records(records, args.sample_size, reviewed)
    if not sample:
        print(f"manual_review pending={pending_count}; nothing new to sample")
        return 0

    print(
        f"manifest={manifest_path.name} pending_manual_review={pending_count} "
        f"sampling={len(sample)}"
    )
    for r in sample:
        print(
            f"  candidate {r.get('document_id')} score={importance_score(r):.1f} "
            f"reason={r.get('reason')} chars={r.get('content_chars')}"
        )

    if not args.execute:
        print("dry-run: rerun with --execute to review and promote")
        return 0

    promoted = []
    decisions = []
    for rec in sample:
        if args.auto_promote:
            verdict = {"verdict": "production", "confidence": 0.0, "why": "auto_promote"}
        else:
            verdict = call_review_llm(build_llm_prompt(rec))
        if not verdict:
            verdict = {"verdict": "noise", "confidence": 0.0, "why": "llm_unavailable"}
        doc_id = rec.get("document_id")
        decisions.append({"document_id": doc_id, **verdict, "score": importance_score(rec)})
        reviewed[doc_id] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict.get("verdict"),
            "why": verdict.get("why"),
        }
        if verdict.get("verdict") == "production":
            promoted_rec = dict(rec)
            promoted_rec["action"] = "production"
            promoted_rec["reason"] = "llm_review_promoted"
            promoted.append(promoted_rec)

    state["reviewed"] = reviewed
    state["history"].append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_path.name,
            "decisions": decisions,
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    if promoted:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = Path(args.output_dir) / f"{stamp}-manual-review-promoted.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in promoted:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"promoted={len(promoted)} curated manifest: {out_path}")
        if args.submit:
            import subprocess
            submit_cmd = [
                sys.executable,
                "/home/wyr/.hermes/scripts/hindsight_session_retain_runner.py",
                "--manifest", str(out_path),
                "--bank", args.bank,
                "--batch-size", "5",
                "--poll-s", "10",
                "--json",
                "--submit-state", "/home/wyr/.hermes/hindsight/session_ingest/submit_state.json",
                "--execute",
                "--confirm", "retain-hindsight-session-manifest",
            ]
            print("  submitting promoted records...", flush=True)
            proc = subprocess.run(submit_cmd, capture_output=True, text=True, timeout=3600)
            tail = (proc.stdout or "").strip().splitlines()[-8:]
            for line in tail:
                print("   ", line)
            if proc.returncode != 0:
                print(f"  submit FAILED rc={proc.returncode}: {(proc.stderr or '')[-300:]}", file=sys.stderr)
                return 3
    else:
        print("promoted=0 (all sampled records judged noise)")

    decisions_path = review_dir / f"sampler-decisions-{datetime.now():%Y%m%d}.jsonl"
    with decisions_path.open("a", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
