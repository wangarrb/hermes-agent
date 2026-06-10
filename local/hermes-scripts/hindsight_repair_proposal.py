#!/usr/bin/env python3
"""Generate non-destructive repair proposals for Hindsight conflict cases.

Inputs can be:
- conflict audit JSON with many cases
- one --case-id from that audit
- a manual claim/target pair

Read-only: no DB writes, no Hindsight writes, no LLM calls.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hindsight_conflict_core as core

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_AUDIT_JSON = HERMES_HOME / "hindsight" / "offline_reflect" / "conflict_audit" / "latest.json"
DEFAULT_OUTPUT_DIR = HERMES_HOME / "hindsight" / "offline_reflect" / "repair_proposals"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_cases(report: dict[str, Any], case_id: str | None = None, max_cases: int = 50) -> list[dict[str, Any]]:
    cases = list(report.get("cases") or [])
    if case_id:
        selected = [c for c in cases if str(c.get("case_id")) == case_id]
        if not selected:
            raise SystemExit(f"case-id not found: {case_id}")
        cases = selected
    return cases[:max_cases]


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Hindsight Repair Proposal", ""]
    lines.append(f"generated_at: {report.get('generated_at')}")
    lines.append(f"proposal_count: {len(report.get('proposals') or [])}")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Proposal only: no DB writes, no file deletion, no Hindsight writes, no LLM calls.")
    lines.append("- delete/overwrite/publish/quarantine execution requires explicit user confirmation.")
    lines.append("")
    for p in report.get("proposals") or []:
        lines.append(f"## {p.get('case_id')}")
        target = p.get("target") or {}
        lines.append(f"target: {target.get('id') or target.get('document_id') or target.get('preview')}")
        lines.append(f"repair_class: {p.get('repair_class')}")
        for a in p.get("recommended_actions") or []:
            lines.append(f"- {a.get('action')}: {a.get('detail')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate read-only repair proposals from Hindsight conflict cases")
    ap.add_argument("--audit-json", default=str(DEFAULT_AUDIT_JSON))
    ap.add_argument("--case-id")
    ap.add_argument("--manual-claim")
    ap.add_argument("--target-id")
    ap.add_argument("--target-text")
    ap.add_argument("--allow-untargeted", action="store_true", help="Allow manual proposal without target id/text; otherwise fail closed")
    ap.add_argument("--max-cases", type=int, default=50)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.manual_claim:
        if not args.allow_untargeted and not (args.target_id or args.target_text):
            raise SystemExit("manual proposal requires --target-id or --target-text; use --allow-untargeted only for intake notes")
        cases = [core.manual_conflict_case(claim=args.manual_claim, target_id=args.target_id, target_text=args.target_text, severity="P1")]
        source = "manual"
    else:
        audit_path = Path(args.audit_json).expanduser()
        audit = load_json(audit_path)
        cases = select_cases(audit, args.case_id, args.max_cases)
        source = str(audit_path)
    proposals = [core.repair_proposal_for_case(c) for c in cases]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "case_count": len(cases),
        "proposals": proposals,
        "safety": ["proposal_only", "no_llm_calls", "no_hindsight_writes", "destructive_actions_require_user_confirmation"],
    }
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = args.case_id or ("manual-" + core.sha1_short(args.manual_claim or "manual", 8) if args.manual_claim else "latest")
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", suffix)[:80].strip("-._") or "proposal"
    json_path = output_dir / f"repair-proposal-{suffix}-{ts}.json"
    md_path = output_dir / f"repair-proposal-{suffix}-{ts}.md"
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
        print(f"saved_json: {json_path}")
        print(f"saved_markdown: {md_path}")


if __name__ == "__main__":
    main()
