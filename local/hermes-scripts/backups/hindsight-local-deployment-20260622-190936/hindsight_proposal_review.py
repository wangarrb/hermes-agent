#!/usr/bin/env python3
"""Review Hindsight repair/retain proposal bundles before any production merge.

This script is intentionally production-safe:
- reads local proposal JSON files only,
- optionally calls an LLM for advisory judgement when explicitly confirmed,
- writes local human-review packets,
- never calls Hindsight retain/merge/delete APIs and never mutates production.

Production merge/retain remains a separate workflow that must have its own
human go/no-go plus rollback/quarantine plan.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_pipeline_common import load_config, path_from_config  # noqa: E402

SCHEMA_VERSION = "hindsight-proposal-review-packet-v1"
CONFIRM_REVIEW_TOKEN = "review-hindsight-proposals"
DEFAULT_LLM_MODEL = os.environ.get("HINDSIGHT_PROPOSAL_REVIEW_LLM_MODEL", "MiniMax-M2.7")
DEFAULT_LLM_BASE_URL = os.environ.get("HINDSIGHT_PROPOSAL_REVIEW_LLM_BASE_URL", "https://api.minimaxi.com/v1")
DEFAULT_LLM_API_KEY_ENV = os.environ.get("HINDSIGHT_PROPOSAL_REVIEW_LLM_API_KEY_ENV", "MINIMAX_API_KEY")

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-.]{12,}\b"),
    re.compile(r"(?i)\b(api[_ -]?key|token|secret|password|passwd)\b\s*[:=]\s*[^\s,;\]}\"']{6,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]
VALID_LLM_DECISIONS = {"merge_ready", "needs_revision", "quarantine", "reject"}
VALID_RISKS = {"low", "medium", "high"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def read_dotenv(path: Path = Path.home() / ".hermes" / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def get_llm_key(api_key_env: str) -> str:
    key = (os.environ.get(api_key_env) or read_dotenv().get(api_key_env, "")).strip()
    if not key or key in {"***", "[REDACTED]"}:
        raise SystemExit(f"{api_key_env} missing; aborting before LLM call")
    return key


def extract_json_object(text: str) -> dict[str, Any] | None:
    s = re.sub(r"<think>.*?</think>", "", (text or "").strip(), flags=re.S | re.I).strip()
    candidates = [m.group(1).strip() for m in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", s, flags=re.S)]
    candidates.append(s)
    decoder = json.JSONDecoder()
    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        for m in re.finditer(r"\{", cand):
            try:
                obj, _ = decoder.raw_decode(cand[m.start():])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def has_secret_like_text(text: str) -> bool:
    return any(p.search(text or "") for p in SECRET_PATTERNS)


def proposal_files_from_args(*, proposal_jsons: list[Path] | None, proposal_dir: Path | None, stem: str | None) -> list[Path]:
    if proposal_jsons:
        return [p.expanduser() for p in proposal_jsons]
    if not proposal_dir:
        return []
    root = proposal_dir.expanduser()
    if stem:
        candidates = [root / f"{stem}-canonical-proposals.json"]
        if not candidates[0].exists():
            candidates = sorted(root.glob(f"{stem}*canonical-proposals.json"))
        return candidates
    return sorted(root.glob("*-canonical-proposals.json"))


def load_proposals(files: list[Path], *, top: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in files:
        data = load_json(path)
        bundle_props = data.get("proposals") or []
        if not isinstance(bundle_props, list):
            bundle_props = []
        for item in bundle_props:
            if not isinstance(item, dict):
                continue
            p = dict(item)
            p["_proposal_file"] = str(path)
            proposals.append(p)
        sources.append({
            "path": str(path),
            "schema_version": data.get("schema_version"),
            "generated_at": data.get("generated_at"),
            "proposal_count": len(bundle_props),
            "quality": data.get("quality"),
        })
    proposals.sort(key=lambda x: (float(x.get("priority_score") or 0), int(x.get("evidence_count") or 0)), reverse=True)
    if top is not None and top > 0:
        proposals = proposals[:top]
    return proposals, sources


def deterministic_findings(proposal: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    text = str(proposal.get("canonical_text") or "")
    if has_secret_like_text(text):
        findings.append({"severity": "block", "code": "secret_like_material", "message": "proposal text contains credential-like material"})
    if proposal.get("production_action") != "proposal_only_no_write":
        findings.append({"severity": "block", "code": "production_action_not_proposal_only", "message": "proposal is not marked proposal_only_no_write"})
    if proposal.get("merge_gate") != "user_approval_required":
        findings.append({"severity": "block", "code": "missing_user_approval_gate", "message": "merge_gate must remain user_approval_required"})
    if int(proposal.get("evidence_count") or 0) < 1:
        findings.append({"severity": "block", "code": "missing_evidence", "message": "proposal has no evidence ids"})
    if int(proposal.get("source_document_count") or 0) < 1:
        findings.append({"severity": "block", "code": "missing_source_documents", "message": "proposal has no source documents"})
    flags = [str(x) for x in (proposal.get("quality_flags") or [])]
    for flag in flags:
        severity = "block" if flag in {"no_evidence_ids", "no_source_documents"} else "warn"
        findings.append({"severity": severity, "code": f"quality_flag:{flag}", "message": f"proposal quality flag: {flag}"})
    return findings


def build_llm_messages(proposal: dict[str, Any]) -> list[dict[str, str]]:
    public = {
        "proposal_id": proposal.get("proposal_id"),
        "topic": proposal.get("topic"),
        "type": proposal.get("type"),
        "tags": proposal.get("tags"),
        "evidence_count": proposal.get("evidence_count"),
        "source_document_count": proposal.get("source_document_count"),
        "quality_flags": proposal.get("quality_flags"),
        "canonical_text": str(proposal.get("canonical_text") or "")[:1400],
        "source_documents": (proposal.get("source_documents") or [])[:12],
        "source_fact_ids": (proposal.get("source_fact_ids") or [])[:20],
    }
    system = (
        "你是 Hindsight production retain/merge proposal 的发布前审查员。"
        "只做证据充分性、可持久性、风险和是否适合进入生产记忆的判断。"
        "不要输出思维链；不要生成新的记忆事实；不要批准任何绕过人工审核的生产写入。"
        "只输出 JSON。"
    )
    user = {
        "task": "judge_hindsight_production_proposal",
        "allowed_decisions": sorted(VALID_LLM_DECISIONS),
        "required_schema": {
            "proposal_id": "same as input",
            "decision": "merge_ready | needs_revision | quarantine | reject",
            "risk": "low | medium | high",
            "reason_brief": "short source-backed reason, no chain-of-thought",
            "required_human_checks": ["concrete checks human must do before go/no-go"],
            "rollback_or_quarantine_notes": ["concrete notes for rollback/quarantine planning"],
        },
        "proposal": public,
        "constraints": [
            "merge_ready only means LLM advisory pass; final decision remains human-only",
            "if evidence is missing, stale, contradictory, secret-like, or too vague: choose needs_revision/quarantine/reject",
            "if quality_flags are non-empty, explain what the human must check",
            "do not invent evidence not present in the proposal metadata",
        ],
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)}]


def normalize_llm_judgement(proposal: dict[str, Any], raw: dict[str, Any] | None, *, status: str) -> dict[str, Any]:
    raw = raw or {}
    decision = str(raw.get("decision") or "needs_revision")
    if decision not in VALID_LLM_DECISIONS:
        decision = "needs_revision"
    risk = str(raw.get("risk") or "medium")
    if risk not in VALID_RISKS:
        risk = "medium"
    return {
        "status": status,
        "proposal_id": proposal.get("proposal_id"),
        "decision": decision,
        "risk": risk,
        "reason_brief": str(raw.get("reason_brief") or "")[:600],
        "required_human_checks": [str(x)[:400] for x in (raw.get("required_human_checks") or []) if str(x).strip()][:20],
        "rollback_or_quarantine_notes": [str(x)[:400] for x in (raw.get("rollback_or_quarantine_notes") or []) if str(x).strip()][:20],
        "advisory_only": True,
    }


def make_openai_llm_fn(*, model: str, base_url: str, api_key_env: str) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    def _call(messages: list[dict[str, str]]) -> dict[str, Any]:
        key = get_llm_key(api_key_env)
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=payload,
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        obj = extract_json_object(raw)
        if obj is None:
            raise RuntimeError("LLM response did not contain a JSON object")
        return obj
    return _call


def standard_rollback_plan(proposal: dict[str, Any], review_root: Path) -> dict[str, Any]:
    pid = str(proposal.get("proposal_id") or "unknown")
    safe_pid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", pid)[:120]
    quarantine_root = review_root / "quarantine" / safe_pid
    return {
        "required_before_any_production_write": True,
        "production_merge_implemented_here": False,
        "quarantine_root": str(quarantine_root),
        "minimum_steps_before_future_merge": [
            "create a production Hindsight snapshot/export before retain/merge",
            "materialize the exact retain payload manifest and source proposal ids",
            "first retain/merge into a quarantine/temp bank, not directly into hermes production",
            "run recall, conflict, lineage, and duplicate audits against the quarantine bank",
            "obtain explicit human go/no-go for the exact payload and rollback plan",
        ],
        "rollback_steps_if_future_merge_is_approved_then_fails": [
            "use the retained payload manifest to identify inserted/updated document ids",
            "quarantine or delete only those ids through a separately confirmed destructive workflow",
            "restore from the pre-merge snapshot if targeted rollback is insufficient",
            "record the superseded/reverted proposal ids in the review packet",
        ],
    }


def compute_go_no_go(
    proposal: dict[str, Any],
    findings: list[dict[str, str]],
    llm: dict[str, Any] | None,
    *,
    require_llm_review: bool,
    require_human_approval: bool,
    review_root: Path,
) -> dict[str, Any]:
    block_codes = [f["code"] for f in findings if f.get("severity") == "block"]
    warn_codes = [f["code"] for f in findings if f.get("severity") == "warn"]
    conditions: list[str] = []
    decision = "conditional_go"
    reason = "deterministic checks passed; human approval is still required"

    if block_codes:
        decision = "no_go"
        reason = "blocked by deterministic proposal checks"
    elif require_llm_review and (not llm or llm.get("status") != "reviewed"):
        decision = "no_go"
        reason = "LLM review is required and still pending"
        conditions.append("run advisory LLM review with --execute-llm --confirm-review review-hindsight-proposals")
    elif llm and llm.get("decision") in {"needs_revision", "quarantine", "reject"}:
        decision = "no_go"
        reason = f"LLM advisory decision is {llm.get('decision')}"
    elif require_human_approval:
        decision = "conditional_go"
        reason = "LLM/deterministic checks allow human review, but final human go/no-go is pending"
        conditions.append("human reviewer must explicitly approve the exact proposal ids")
    else:
        # Kept for config flexibility, but production workflow should keep human approval enabled.
        decision = "conditional_go"
        reason = "human approval requirement disabled in config; production still not executed by this script"

    if warn_codes:
        conditions.append("review warnings before approval: " + ", ".join(warn_codes))
    if llm:
        conditions.extend(str(x) for x in llm.get("required_human_checks") or [])
    conditions.append("prepare and approve rollback/quarantine plan before any future production retain/merge")

    return {
        "go_no_go": decision,
        "reason": reason,
        "deterministic_block_codes": block_codes,
        "deterministic_warning_codes": warn_codes,
        "conditions": conditions,
        "human_final_decision": "pending" if require_human_approval else "not_configured",
        "production_merge_allowed_by_this_packet": False,
        "rollback_plan": standard_rollback_plan(proposal, review_root),
    }


def review_proposals(
    proposals: list[dict[str, Any]],
    *,
    review_root: Path,
    require_llm_review: bool = True,
    require_human_approval: bool = True,
    execute_llm: bool = False,
    confirm_review: str = "",
    max_llm_calls: int = 10,
    llm_fn: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if execute_llm and confirm_review != CONFIRM_REVIEW_TOKEN:
        raise SystemExit(f"Refusing LLM review: pass --confirm-review {CONFIRM_REVIEW_TOKEN}")
    if execute_llm and llm_fn is None:
        raise SystemExit("Refusing LLM review without llm_fn")

    max_llm_calls = max(0, int(max_llm_calls))
    reviews: list[dict[str, Any]] = []
    llm_calls = 0
    for proposal in proposals:
        findings = deterministic_findings(proposal)
        block_codes = [f.get("code") for f in findings if f.get("severity") == "block"]
        llm_judgement: dict[str, Any] | None = None
        if execute_llm and block_codes:
            blocked_decision = "quarantine" if "secret_like_material" in block_codes else "needs_revision"
            llm_judgement = normalize_llm_judgement(
                proposal,
                {
                    "decision": blocked_decision,
                    "risk": "high",
                    "reason_brief": "Skipped advisory LLM review because deterministic checks blocked this proposal before external disclosure.",
                    "required_human_checks": ["inspect deterministic block codes before any future action"],
                    "rollback_or_quarantine_notes": ["keep blocked proposal in local review/quarantine; do not send to external LLM or production retain"],
                },
                status="skipped_deterministic_block",
            )
        elif execute_llm and llm_calls < max_llm_calls:
            try:
                raw = llm_fn(build_llm_messages(proposal)) if llm_fn else None
                llm_judgement = normalize_llm_judgement(proposal, raw, status="reviewed")
            except Exception as exc:
                llm_judgement = normalize_llm_judgement(
                    proposal,
                    {"decision": "needs_revision", "risk": "high", "reason_brief": f"LLM review failed: {type(exc).__name__}: {exc}"},
                    status="failed",
                )
            llm_calls += 1
        elif require_llm_review:
            status = "deferred_by_call_cap" if execute_llm else "pending_not_executed"
            llm_judgement = normalize_llm_judgement(
                proposal,
                {"decision": "needs_revision", "risk": "medium", "reason_brief": "LLM review required but not completed in this run."},
                status=status,
            )

        go_no_go = compute_go_no_go(
            proposal,
            findings,
            llm_judgement,
            require_llm_review=require_llm_review,
            require_human_approval=require_human_approval,
            review_root=review_root,
        )
        reviews.append({
            "proposal_id": proposal.get("proposal_id"),
            "proposal_file": proposal.get("_proposal_file"),
            "priority_score": proposal.get("priority_score"),
            "topic": proposal.get("topic"),
            "type": proposal.get("type"),
            "canonical_text": proposal.get("canonical_text"),
            "evidence_count": proposal.get("evidence_count"),
            "source_document_count": proposal.get("source_document_count"),
            "quality_flags": proposal.get("quality_flags") or [],
            "deterministic_findings": findings,
            "llm_judgement": llm_judgement,
            "go_no_go": go_no_go,
        })

    counts: dict[str, int] = {}
    for r in reviews:
        key = str((r.get("go_no_go") or {}).get("go_no_go") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "proposal_count": len(proposals),
        "review_count": len(reviews),
        "go_no_go_counts": counts,
        "llm_required": require_llm_review,
        "llm_execute": execute_llm,
        "llm_calls_made": llm_calls,
        "max_llm_calls": max_llm_calls,
        "human_approval_required": require_human_approval,
        "production_mutation_allowed": False,
    }
    return reviews, summary


def build_packet(
    *,
    proposals: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    summary: dict[str, Any],
    config_path: str | None,
    notify_enabled: bool,
) -> dict[str, Any]:
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "config_path": config_path,
        "sources": sources,
        "summary": summary,
        "safety": {
            "proposal_only": True,
            "production_mutation_allowed": False,
            "production_merge_or_retain_executed": False,
            "requires_separate_human_go_no_go": True,
            "requires_rollback_quarantine_plan": True,
        },
        "manual_review": {
            "status": "pending",
            "instructions": [
                "Open the markdown review packet and inspect every conditional_go proposal.",
                "If LLM review is pending, run the review script with --execute-llm --confirm-review review-hindsight-proposals before production decisions.",
                "For any future production merge/retain, prepare a separate exact payload, snapshot, quarantine-bank validation, and rollback plan.",
                "Do not treat this packet as approval to mutate production Hindsight.",
            ],
        },
        "reviews": reviews,
    }
    if notify_enabled:
        packet["notification"] = notification_from_packet(packet)
    return packet


def notification_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    summary = packet.get("summary") or {}
    counts = summary.get("go_no_go_counts") or {}
    return {
        "title": "Hindsight proposal review needs human go/no-go",
        "severity": "action_required",
        "message": (
            f"Proposal review packet ready: {summary.get('review_count', 0)} proposals; "
            f"conditional_go={counts.get('conditional_go', 0)}, no_go={counts.get('no_go', 0)}. "
            "Production merge/retain was NOT executed."
        ),
        "human_action_required": True,
        "production_mutation_executed": False,
    }


def render_markdown(packet: dict[str, Any]) -> str:
    summary = packet.get("summary") or {}
    lines = [
        "# Hindsight proposal review packet",
        "",
        f"generated_at: {packet.get('generated_at')}",
        f"schema_version: {packet.get('schema_version')}",
        "",
        "## Safety boundary",
        "",
        "- proposal_only: true",
        "- production_mutation_allowed: false",
        "- production_merge_or_retain_executed: false",
        "- final human go/no-go required: true",
        "- rollback/quarantine plan required before any future production write: true",
        "",
        "## Summary",
        "",
        f"- proposal_count: {summary.get('proposal_count')}",
        f"- review_count: {summary.get('review_count')}",
        f"- go_no_go_counts: {summary.get('go_no_go_counts')}",
        f"- llm_required: {summary.get('llm_required')}",
        f"- llm_execute: {summary.get('llm_execute')}",
        f"- llm_calls_made: {summary.get('llm_calls_made')}",
        "",
        "## Manual review instructions",
        "",
    ]
    for item in (packet.get("manual_review") or {}).get("instructions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Reviews", ""])
    for i, review in enumerate(packet.get("reviews") or [], 1):
        g = review.get("go_no_go") or {}
        llm = review.get("llm_judgement") or {}
        lines.extend([
            f"### {i}. {review.get('proposal_id')}",
            "",
            f"- topic/type: {review.get('topic')} / {review.get('type')}",
            f"- go_no_go: {g.get('go_no_go')} — {g.get('reason')}",
            f"- human_final_decision: {g.get('human_final_decision')}",
            f"- llm: status={llm.get('status')} decision={llm.get('decision')} risk={llm.get('risk')} reason={llm.get('reason_brief')}",
            f"- deterministic_blocks: {g.get('deterministic_block_codes')}",
            f"- deterministic_warnings: {g.get('deterministic_warning_codes')}",
            f"- quality_flags: {review.get('quality_flags')}",
            f"- evidence_count: {review.get('evidence_count')} ; source_document_count: {review.get('source_document_count')}",
            f"- proposal_file: {review.get('proposal_file')}",
            "",
            "Conditions:",
        ])
        for cond in g.get("conditions") or []:
            lines.append(f"- {cond}")
        lines.extend(["", "Canonical text:", "", str(review.get("canonical_text") or ""), ""])
    return "\n".join(lines)


def write_packet(packet: dict[str, Any], review_root: Path, stem: str) -> dict[str, str]:
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "quarantine").mkdir(parents=True, exist_ok=True)
    json_path = review_root / f"{stem}-review-packet.json"
    md_path = review_root / f"{stem}-review-packet.md"
    json_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(packet), encoding="utf-8")
    return {"review_json": str(json_path), "review_md": str(md_path)}


def default_stem(files: list[Path]) -> str:
    if len(files) == 1:
        name = files[0].name
        return name.replace("-canonical-proposals.json", "")
    return "proposal-review-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build local LLM/human review packets for Hindsight proposal bundles")
    ap.add_argument("--config", type=Path, help="Path to pipeline_config.json")
    ap.add_argument("--proposal-json", action="append", type=Path, default=[], help="Specific proposal JSON file. May be repeated.")
    ap.add_argument("--proposal-dir", type=Path, help="Directory containing *-canonical-proposals.json files")
    ap.add_argument("--review-root", type=Path, help="Directory for review packets")
    ap.add_argument("--stem", help="Proposal file stem filter; also used as output stem when --output-stem is omitted")
    ap.add_argument("--output-stem", help="Review packet output stem; does not filter proposal inputs")
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--execute-llm", action="store_true", help="Actually call configured LLM for advisory proposal judgement")
    ap.add_argument("--confirm-review", default="", help=f"Required token for --execute-llm: {CONFIRM_REVIEW_TOKEN}")
    ap.add_argument("--max-llm-calls", type=int, default=None)
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    ap.add_argument("--llm-api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    ap.add_argument("--notify", action="store_true", help="Emit notification block for Hermes/cron delivery")
    ap.add_argument("--no-notify", action="store_true", help="Disable notification even if config enables it")
    ap.add_argument("--json", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    review_cfg = cfg.get("review") or {}
    proposal_review_cfg = review_cfg.get("proposal_review") or {}
    proposal_dir = args.proposal_dir or path_from_config(cfg, "proposal_root")
    review_root = args.review_root or path_from_config(cfg, "review_root")
    files = proposal_files_from_args(proposal_jsons=args.proposal_json, proposal_dir=proposal_dir, stem=args.stem)
    if not files:
        result = {
            "ok": True,
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "no proposal JSON files found",
            "proposal_dir": str(proposal_dir),
            "production_mutation_allowed": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) if args.json else result["reason"])
        return 0

    proposals, sources = load_proposals(files, top=args.top)
    max_llm_calls = args.max_llm_calls
    if max_llm_calls is None:
        max_llm_calls = int(proposal_review_cfg.get("max_llm_calls") or review_cfg.get("max_llm_calls") or 10)
    require_llm_review = bool(review_cfg.get("require_llm_review", True))
    require_human_approval = bool(review_cfg.get("require_human_approval", True))
    notify_cfg = (review_cfg.get("notify") or {}) if isinstance(review_cfg.get("notify"), dict) else {}
    notify_enabled = bool(args.notify or (notify_cfg.get("enabled") and not args.no_notify))

    llm_fn = None
    if args.execute_llm:
        llm_fn = make_openai_llm_fn(model=args.llm_model, base_url=args.llm_base_url, api_key_env=args.llm_api_key_env)
    reviews, summary = review_proposals(
        proposals,
        review_root=review_root,
        require_llm_review=require_llm_review,
        require_human_approval=require_human_approval,
        execute_llm=bool(args.execute_llm),
        confirm_review=args.confirm_review,
        max_llm_calls=max_llm_calls,
        llm_fn=llm_fn,
    )
    stem = args.output_stem or args.stem or default_stem(files)
    packet = build_packet(
        proposals=proposals,
        sources=sources,
        reviews=reviews,
        summary=summary,
        config_path=str(cfg.get("config_path")) if cfg.get("config_path") else None,
        notify_enabled=notify_enabled,
    )
    paths = write_packet(packet, review_root, stem)
    result = {
        "ok": True,
        "status": "review_packet_written",
        "paths": paths,
        "summary": summary,
        "notification": packet.get("notification"),
        "production_mutation_allowed": False,
        "production_merge_or_retain_executed": False,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"wrote {paths['review_json']}")
        print(f"wrote {paths['review_md']}")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if packet.get("notification"):
            print(json.dumps(packet["notification"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
