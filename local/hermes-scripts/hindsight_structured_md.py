#!/usr/bin/env python3
"""Structured markdown processing for Hindsight ingestion.

Transforms raw markdown files into structured context that captures:
- Document hierarchy (section headings + substructure)
- Key decision points and conclusions
- Code/command blocks
- Table summaries
- Cross-references to other project files

This is a processing layer between raw file reads and Hindsight retain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MDStructure:
    """Parsed structure of a markdown document."""
    title: str = ""
    heading_hierarchy: list[tuple[int, str]] = field(default_factory=list)  # (level, text)
    key_decisions: list[str] = field(default_factory=list)  # lines with decisions/conclusions
    code_blocks: list[dict[str, str]] = field(default_factory=list)  # {lang, summary, snippet}
    tables: list[list[list[str]]] = field(default_factory=list)  # parsed tables
    cross_refs: list[str] = field(default_factory=list)  # references to other .md/.py files
    summary_sections: list[str] = field(default_factory=list)  # "Summary", "Conclusion", "Verdict" sections
    gate_conditions: list[str] = field(default_factory=list)  # PASS/FAIL/NO_CLAIM lines
    total_lines: int = 0
    total_chars: int = 0


# ── Heading detection ──────────────────────────────────────────────────────

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?\s*$", re.MULTILINE)

# ── Decision / conclusion detection ─────────────────────────────────────────

DECISION_PATTERNS = [
    re.compile(r"(?:结论|决策|决定|判[断决]|VERDICT|DECISION)[：:]\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:PASS|FAIL|NO_CLAIM|CLAIM_PASS|CODE_PASS|EVIDENCE_FAIL)[：:]?\s*(.*)", re.IGNORECASE),
    re.compile(r"(?:推荐|建议|选择|采用|关闭|放弃|拒绝)[：:]\s*(.+)"),
    re.compile(r"^\s*(?:✅|❌|⚠️|🔴|🟢|🟡)\s*(.+)"),
    re.compile(r"(?:已证伪|已关闭|已 promotion|不可用|永久关闭)[：:]?\s*(.+)", re.IGNORECASE),
]

# ── Code block detection ────────────────────────────────────────────────────

CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# ── Table detection ─────────────────────────────────────────────────────────

TABLE_LINE_RE = re.compile(r"^\|.+\|$")

# ── Cross-reference detection ───────────────────────────────────────────────

CROSS_REF_RE = re.compile(
    r"((?:docs|src|tests|configs|scripts)/[^\s,\)\]>]+\.(?:md|py|yaml|json|toml))",
    re.IGNORECASE,
)

# ── Gate condition detection ────────────────────────────────────────────────

GATE_RE = re.compile(
    r"(?:Gate|验收|gate|acceptance)[：:]\s*(.+)",
    re.IGNORECASE,
)


def extract_md_structure(text: str, *, max_code_snippet: int = 20) -> MDStructure:
    """Parse a markdown document into structured components.

    Args:
        text: Raw markdown content.
        max_code_snippet: Maximum lines to keep from any single code block snippet.

    Returns:
        MDStructure with parsed components.
    """
    structure = MDStructure(total_lines=text.count("\n") + 1, total_chars=len(text))

    lines = text.split("\n")

    # Extract headings for hierarchy
    for match in HEADING_RE.finditer(text):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        structure.heading_hierarchy.append((level, heading_text))
        if level == 1:
            structure.title = heading_text

    # Extract decision/conclusion lines
    for pattern in DECISION_PATTERNS:
        for match in pattern.finditer(text):
            decision = match.group(0).strip()
            if len(decision) > 10:
                structure.key_decisions.append(decision)

    # Extract code blocks with brief summaries
    for match in CODE_BLOCK_RE.finditer(text):
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        snippet_lines = code.split("\n")[:max_code_snippet]
        snippet = "\n".join(snippet_lines)
        if len(code.split("\n")) > max_code_snippet:
            snippet += f"\n... (+{len(code.split(chr(10))) - max_code_snippet} more lines)"
        structure.code_blocks.append({
            "lang": lang,
            "summary": f"{lang} block, {len(code.split(chr(10)))} lines",
            "snippet": snippet,
        })

    # Extract tables (simple pipe-table detection)
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        if TABLE_LINE_RE.match(line):
            if not in_table:
                if table_lines:
                    # Previous table ended
                    structure.tables.append(_parse_table_rows(table_lines))
                    table_lines = []
                in_table = True
            table_lines.append(line)
        else:
            if in_table:
                structure.tables.append(_parse_table_rows(table_lines))
                table_lines = []
                in_table = False
    if table_lines:
        structure.tables.append(_parse_table_rows(table_lines))

    # Extract cross-references
    seen_refs = set()
    for match in CROSS_REF_RE.finditer(text):
        ref = match.group(1)
        if ref not in seen_refs:
            structure.cross_refs.append(ref)
            seen_refs.add(ref)

    # Extract summary/conclusion sections
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^#{1,3}\s*(?:Summary|总结|结论|Conclusion|Verdict|判决|最终判断)", stripped, re.IGNORECASE):
            # Collect this section
            section_lines = [stripped]
            for j in range(i + 1, min(i + 30, len(lines))):
                if HEADING_RE.match(lines[j]):
                    break
                if lines[j].strip():
                    section_lines.append(lines[j].strip())
            structure.summary_sections.append("\n".join(section_lines))

    # Extract gate conditions
    for match in GATE_RE.finditer(text):
        structure.gate_conditions.append(match.group(0).strip()[:200])

    return structure


def _parse_table_rows(lines: list[str]) -> list[list[str]]:
    """Parse pipe-table lines into rows of cells."""
    rows = []
    for line in lines:
        # Skip separator lines like |---|---|
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # Remove leading/trailing empty cells from split
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


def format_structured_for_retain(
    text: str,
    path: Path,
    source_title: str = "Project Markdown Artifact",
) -> str:
    """Format a markdown file as structured context for Hindsight retain.

    Produces a rich text block that captures the document's narrative arc,
    not just raw content.
    """
    structure = extract_md_structure(text)

    parts: list[str] = []

    # Header
    parts.append(f"Title: {structure.title or path.stem}")
    parts.append(f"Source: {source_title}")
    parts.append(f"Path: {path}")
    parts.append(f"Lines: {structure.total_lines} | Characters: {structure.total_chars}")
    parts.append("")

    # Document structure summary
    if structure.heading_hierarchy:
        parts.append("-- Document Structure --")
        for level, heading in structure.heading_hierarchy[:30]:
            indent = "  " * (level - 1)
            parts.append(f"{indent}## {heading}")
        parts.append("")

    # Key decisions / conclusions
    if structure.key_decisions:
        parts.append("-- Key Decisions & Conclusions --")
        for i, d in enumerate(structure.key_decisions[:15]):
            parts.append(f"{i+1}. {d}")
        parts.append("")

    # Gate conditions
    if structure.gate_conditions:
        parts.append("-- Gate / Acceptance Criteria --")
        for g in structure.gate_conditions[:10]:
            parts.append(f"  {g}")
        parts.append("")

    # Code block summaries
    if structure.code_blocks:
        parts.append("-- Code Blocks --")
        for i, cb in enumerate(structure.code_blocks[:10]):
            parts.append(f"  [{i+1}] {cb['summary']}")
        parts.append("")

    # Cross-references
    if structure.cross_refs:
        parts.append("-- Cross-Referenced Files --")
        for ref in structure.cross_refs[:20]:
            parts.append(f"  {ref}")
        parts.append("")

    # Summary sections
    if structure.summary_sections:
        parts.append("-- Summary Sections --")
        for s in structure.summary_sections[:5]:
            parts.append(s)
            parts.append("")
        parts.append("")

    # Table previews
    if structure.tables:
        parts.append("-- Tables --")
        for i, table in enumerate(structure.tables[:5]):
            parts.append(f"  Table {i+1}: {len(table)} rows x {len(table[0]) if table else 0} cols")
            for row in table[:3]:
                parts.append(f"    | {' | '.join(row[:5])} |")
        parts.append("")

    # Full content (trimmed if very large)
    parts.append("-- Full Content --")
    if len(text) > 80000:
        parts.append(text[:40000])
        parts.append(f"\n\n... [TRUNCATED: {len(text)} total chars, showing first 40000] ...")
        parts.append(f"\n\n... [LAST 10000 CHARS] ...")
        parts.append(text[-10000:])
    else:
        parts.append(text)

    return "\n".join(parts)


def summarize_md_for_index(text: str, path: Path) -> dict[str, Any]:
    """Produce a compact index entry for a markdown document.

    Returns a dict suitable for use as document metadata in Hindsight.
    """
    structure = extract_md_structure(text)
    return {
        "title": structure.title or path.stem,
        "path": str(path),
        "heading_count": len(structure.heading_hierarchy),
        "decision_count": len(structure.key_decisions),
        "code_block_count": len(structure.code_blocks),
        "table_count": len(structure.tables),
        "cross_ref_count": len(structure.cross_refs),
        "gate_count": len(structure.gate_conditions),
        "total_lines": structure.total_lines,
        "total_chars": structure.total_chars,
        "headings": [h[1] for h in structure.heading_hierarchy[:10]],
        "key_decisions": structure.key_decisions[:5],
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Structured markdown analysis for Hindsight")
    ap.add_argument("file", type=Path, help="Markdown file to analyze")
    ap.add_argument("--json", action="store_true", help="Output JSON structure")
    args = ap.parse_args()

    text = args.file.read_text(encoding="utf-8", errors="replace")
    if args.json:
        summary = summarize_md_for_index(text, args.file)
        structure = extract_md_structure(text)
        print(json.dumps({"summary": summary, "structure": {
            "heading_hierarchy": structure.heading_hierarchy,
            "key_decisions": structure.key_decisions,
            "code_blocks": [cb["summary"] for cb in structure.code_blocks],
            "cross_refs": structure.cross_refs,
            "gate_conditions": structure.gate_conditions,
        }}, indent=2, ensure_ascii=False))
    else:
        formatted = format_structured_for_retain(text, args.file)
        print(formatted[:5000])
        if len(formatted) > 5000:
            print(f"\n... [total {len(formatted)} chars]")
