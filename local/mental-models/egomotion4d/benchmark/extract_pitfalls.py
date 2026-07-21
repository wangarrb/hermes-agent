#!/usr/bin/env python3
"""Extract algorithm-level pitfalls from pitfalls.md into structured pitfall_index.json.

This is a one-time bootstrap script. After initial extraction, the daily
maintenance pipeline will maintain pitfall_index.json going forward.

Classification rules (per design doc §5.2):
- INCLUDE: changes algorithm representation/objective/metric/association/observer/gate/route;
           reveals load-bearing assumption failure/identifiability/evaluation bias/cross-domain failure;
           high recurrence with symptom->root cause->prevention evidence chain.
- EXCLUDE: single function bugs, variable/path errors, CLI mistakes, dependency install,
           temp service/GPU/permission issues, code style, commit-specific implementation details.
"""
import json, re, os, sys
from pathlib import Path
from datetime import datetime

PITFALLS_FILE = Path("/home/wyr/wiki/auto-maintenance/project/egomotion4d/pitfalls.md")
OUTPUT_FILE = Path(os.path.expanduser("~/.hermes/mental-models/egomotion4d/pitfall_index.json"))

# Implementation-level keywords that indicate non-algorithmic pitfalls
IMPL_KEYWORDS = [
    "工具错误", "wrapper 覆盖", "legacy correction-group graph",
    "patch temporal v1工具", "fragmentation 与 chaining 矛盾",
]

def parse_pitfalls(content: str) -> list[dict]:
    """Parse pitfalls.md into structured entries."""
    entries = []
    # Match: ## P<id>: <title> ... ---
    pattern = r'^## (P\d+):\s*(.+?)$(.*?)(?=^## P\d+:|^---$|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    for pid, title, body in matches:
        pid = pid.strip()
        title = title.strip()

        # Extract fields
        pitfall = _extract_field(body, "坑")
        actual = _extract_field(body, "实际")
        lesson = _extract_field(body, "教训")
        similar = _extract_field(body, "同类场景")
        tags = _extract_field(body, "标签")
        date = _extract_field(body, "日期")
        source = _extract_field(body, "来源")

        # Classify: algorithm vs implementation
        is_algorithm = not any(kw.lower() in (title + pitfall + actual).lower() for kw in IMPL_KEYWORDS)

        # Extract trigger aliases from tags
        tag_list = []
        if tags:
            tag_list = [t.strip().lstrip('#') for t in tags.split('#') if t.strip()]

        entry = {
            "p_id": pid,
            "title": title,
            "status": "current" if is_algorithm else "rejected_non_algorithmic",
            "is_algorithm_level": is_algorithm,
            "trigger": pitfall[:200] if pitfall else "",
            "root_cause": actual[:300] if actual else "",
            "lesson": lesson[:300] if lesson else "",
            "tags": tag_list,
            "date": date or "",
            "source": source or "",
            "detail_locator": f"pitfalls.md#{pid}",
            "alias_of": None,
            "superseded_by": None,
        }
        entries.append(entry)

    return entries

def _extract_field(body: str, field_name: str) -> str:
    """Extract **field**: value from body text."""
    pattern = rf'\*\*{re.escape(field_name)}\*\*:\s*(.+?)(?=\n\*\*|\n---|\Z)'
    match = re.search(pattern, body, re.DOTALL)
    return match.group(1).strip() if match else ""

def deduplicate(entries: list[dict]) -> list[dict]:
    """Mark duplicates as alias_of their canonical entry."""
    # Group by similar titles (J scoring variants, USP variants, etc.)
    canonical = []
    aliases = []

    for entry in entries:
        if not entry["is_algorithm_level"]:
            canonical.append(entry)
            continue

        # Check if this is a duplicate of an existing canonical
        found_alias = False
        for can in canonical:
            if _is_duplicate(entry, can):
                entry["status"] = "superseded"
                entry["alias_of"] = can["p_id"]
                aliases.append(entry)
                found_alias = True
                break

        if not found_alias:
            canonical.append(entry)

    return canonical + aliases

def _is_duplicate(a: dict, b: dict) -> bool:
    """Check if entry a is a duplicate of entry b."""
    # J scoring variants: P8-P19 are all about J scoring temporal-over-residual
    j_keywords = ["j scoring", "j_score", "temporal-over-residual", "candidate-only"]
    a_text = (a["title"] + a["trigger"]).lower()
    b_text = (b["title"] + b["trigger"]).lower()

    if any(kw in a_text for kw in j_keywords) and any(kw in b_text for kw in j_keywords):
        return True

    # USP variants: P10/P15/P19/P21 are about USP full-stack valid but no-gain
    usp_keywords = ["usp", "full-stack", "no-gain", "权重失衡"]
    if any(kw in a_text for kw in usp_keywords) and any(kw in b_text for kw in usp_keywords):
        return True

    # Coverage degradation variants
    cov_keywords = ["coverage", "coverage 退化", "coverage断崖"]
    if any(kw in a_text for kw in cov_keywords) and any(kw in b_text for kw in cov_keywords):
        return True

    return False

def main():
    if not PITFALLS_FILE.exists():
        print(f"ERROR: {PITFALLS_FILE} not found")
        return 1

    content = PITFALLS_FILE.read_text(encoding="utf-8")
    entries = parse_pitfalls(content)
    entries = deduplicate(entries)

    algorithm_count = sum(1 for e in entries if e["is_algorithm_level"])
    canonical_count = sum(1 for e in entries if e["status"] == "current" and e["is_algorithm_level"])
    alias_count = sum(1 for e in entries if e["status"] == "superseded")
    rejected_count = sum(1 for e in entries if e["status"] == "rejected_non_algorithmic")

    index = {
        "schema_version": 1,
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(PITFALLS_FILE),
        "total_entries": len(entries),
        "canonical_count": canonical_count,
        "alias_count": alias_count,
        "rejected_count": rejected_count,
        "entries": entries,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(entries)} entries:")
    print(f"  Algorithm-level (canonical): {canonical_count}")
    print(f"  Algorithm-level (alias/superseded): {alias_count}")
    print(f"  Rejected (non-algorithmic): {rejected_count}")
    print(f"  Total algorithm-level: {algorithm_count}")
    print(f"Output: {OUTPUT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
