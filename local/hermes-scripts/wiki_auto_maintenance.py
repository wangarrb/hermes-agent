#!/usr/bin/env python3
"""Automated wiki maintenance pass that writes isolated candidate reports.

This script intentionally does not edit the main wiki. It audits the wiki,
scans source notes, reads high-level Hindsight offline outputs, and writes a
standalone report under ~/wiki/auto-maintenance/. The user can later decide
what to merge into the curated wiki.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_WIKI = Path(os.environ.get("WIKI_PATH", str(HOME / "wiki"))).expanduser()
DEFAULT_MEMORIES = HERMES_HOME / "memories"
DEFAULT_HINDSIGHT = HERMES_HOME / "hindsight" / "offline_reflect"
DEFAULT_OUTPUT_SUBDIR = "auto-maintenance/reports"
LOCK_PATH = HERMES_HOME / "hindsight" / "wiki_auto_maintenance.lock"
HINDSIGHT_PIPELINE_LOCK = HERMES_HOME / "hindsight" / "offline_pipeline.lock"
HINDSIGHT_WRAPPER = HERMES_HOME / "scripts" / "hindsight_minimax_import.py"

BASE_RELEVANCE_KEYWORDS = [
    "research", "paper", "experiment", "benchmark", "architecture", "pipeline", "memory", "wiki",
    "hindsight", "canonical", "observation", "conflict", "lineage", "quality", "evaluation", "dataset",
    "model", "agent", "deployment", "debugging", "lesson", "decision", "risk", "open question",
    "研究", "论文", "实验", "评测", "架构", "流程", "记忆", "知识库", "冲突", "溯源", "质量",
    "结论", "决策", "风险", "问题", "数据集", "模型", "部署", "排障", "经验", "教训",
]

SKIP_KEYWORDS = [
    "password", "api key", "token", "secret", "credential", "身份证", "银行卡", "手机号",
]

@dataclass
class WikiPage:
    rel: str
    path: Path
    slug: str
    title: str
    links: list[str]
    tags: list[str]
    updated: str | None
    type: str | None
    line_count: int


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def iso_today() -> str:
    return date.today().isoformat()


def read_text(path: Path, limit_chars: int | None = None) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[READ_ERROR {path}: {e}]"
    if limit_chars and len(text) > limit_chars:
        return text[:limit_chars] + f"\n\n[TRUNCATED {len(text)-limit_chars} chars]"
    return text


def split_csv_words(raw: str | None) -> list[str]:
    if not raw:
        return []
    words: list[str] = []
    for part in re.split(r"[,;\n]", raw):
        item = part.strip()
        if item:
            words.append(item)
    return words


def load_domain_keywords(wiki: Path | None = None) -> list[str]:
    """Load generic relevance keywords plus optional domain keywords.

    Keep the script shareable: domain/project terms belong in WIKI_MAINTENANCE_KEYWORDS
    or an optional wiki/SCHEMA.md taxonomy, not in this script.
    """
    keywords = list(BASE_RELEVANCE_KEYWORDS)
    keywords.extend(split_csv_words(os.environ.get("WIKI_MAINTENANCE_KEYWORDS")))
    if wiki:
        schema = wiki / "SCHEMA.md"
        if schema.exists():
            text = read_text(schema, limit_chars=20000)
            # Reuse tags/taxonomy words as soft relevance hints; this stays generic.
            for m in re.finditer(r"(?:tags?|taxonomy|标签|分类)[:：]?\s*\[?([^\n\]]+)", text, re.I):
                keywords.extend(split_csv_words(m.group(1).replace("-", ",")))
    seen: set[str] = set()
    out: list[str] = []
    for kw in keywords:
        key = kw.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(kw.strip())
    return out


def summarize_v2_rebuild(root: Path) -> dict[str, object]:
    """Summarize latest high-level Hindsight canonical state for the wiki report."""
    latest = root / "v2_rebuild" / "latest.json"
    if not latest.exists():
        return {"available": False, "reason": f"missing {latest}"}
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as e:
        return {"available": False, "reason": f"read_error {latest}: {e}"}
    steps = data.get("steps", {}) if isinstance(data.get("steps"), dict) else {}
    conflict = steps.get("conflict_audit", {}) if isinstance(steps.get("conflict_audit"), dict) else {}
    publish = steps.get("publish", {}) if isinstance(steps.get("publish"), dict) else {}
    gate = steps.get("gate", {}) if isinstance(steps.get("gate"), dict) else {}
    return {
        "available": True,
        "path": str(latest),
        "mode": data.get("mode"),
        "decision": data.get("decision"),
        "published": data.get("published"),
        "errors": data.get("errors"),
        "gate_decision": gate.get("decision"),
        "conflict_summary": conflict.get("summary"),
        "inserted_documents": publish.get("inserted_documents"),
        "inserted_observations": publish.get("inserted_observations"),
        "backup_path": publish.get("backup_path"),
    }


def parse_frontmatter(text: str) -> dict[str, object]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    fm = text[4:end]
    out: dict[str, object] = {}
    for line in fm.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            vals = [] if not inner else [x.strip().strip('"\'') for x in inner.split(",")]
            out[k] = vals
        else:
            out[k] = v.strip('"\'')
    return out


def strip_markdown_code(text: str) -> str:
    """Remove fenced and inline code before wikilink linting.

    Documentation files such as SCHEMA.md intentionally contain examples like
    `[[wikilinks]]`. Those literals are not real Obsidian links, so lint should
    not report them as broken links.
    """
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return text


def slug_for_page(path: Path, wiki: Path) -> str:
    rel = path.relative_to(wiki).with_suffix("").as_posix()
    return rel


def scan_wiki(wiki: Path) -> tuple[list[WikiPage], dict[str, list[str]]]:
    pages: list[WikiPage] = []
    issues: dict[str, list[str]] = defaultdict(list)
    for path in sorted(wiki.rglob("*.md")):
        rel = path.relative_to(wiki).as_posix()
        if rel.startswith("raw/") or rel.startswith("auto-maintenance/") or rel.startswith("_archive/"):
            continue
        text = read_text(path)
        fm = parse_frontmatter(text)
        link_text = strip_markdown_code(text)
        links = re.findall(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]", link_text)
        tags = fm.get("tags") if isinstance(fm.get("tags"), list) else []
        title = str(fm.get("title") or (re.search(r"^#\s+(.+)$", text, re.M).group(1) if re.search(r"^#\s+(.+)$", text, re.M) else path.stem))
        page = WikiPage(
            rel=rel,
            path=path,
            slug=slug_for_page(path, wiki),
            title=title,
            links=links,
            tags=[str(x) for x in tags],
            updated=str(fm.get("updated")) if fm.get("updated") else None,
            type=str(fm.get("type")) if fm.get("type") else None,
            line_count=text.count("\n") + 1,
        )
        pages.append(page)
        if rel not in {"SCHEMA.md", "index.md", "log.md"}:
            for required in ["title", "created", "updated", "type", "tags", "sources"]:
                if required not in fm:
                    issues["frontmatter"].append(f"{rel}: missing {required}")
            if page.line_count > 220:
                issues["large_pages"].append(f"{rel}: {page.line_count} lines")
    slugs = {p.slug for p in pages}
    basenames = {Path(p.slug).name: p.slug for p in pages}
    inbound: Counter[str] = Counter()
    for p in pages:
        for link in p.links:
            link = link.strip().strip("/").removesuffix(".md")
            target = link if link in slugs else basenames.get(Path(link).name)
            if target:
                inbound[target] += 1
            else:
                issues["broken_links"].append(f"{p.rel}: [[{link}]]")
    for p in pages:
        if p.rel in {"SCHEMA.md", "index.md", "log.md"}:
            continue
        if inbound[p.slug] == 0:
            issues["orphans"].append(p.rel)
    index_text = read_text(wiki / "index.md") if (wiki / "index.md").exists() else ""
    for p in pages:
        if p.rel in {"SCHEMA.md", "index.md", "log.md"}:
            continue
        if p.slug not in index_text and Path(p.slug).name not in index_text:
            issues["index_missing"].append(p.rel)
    return pages, issues


def newest_files(root: Path, globs: list[str], *, days: int, limit: int = 80) -> list[Path]:
    if not root.exists():
        return []
    cutoff = datetime.now().timestamp() - days * 86400
    files: list[Path] = []
    for g in globs:
        files.extend(root.glob(g))
    files = [p for p in files if p.is_file() and p.stat().st_mtime >= cutoff]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def score_relevance(text: str, path: Path, keywords: list[str] | None = None) -> tuple[int, list[str]]:
    low = (str(path) + "\n" + text[:6000]).lower()
    hits = []
    for kw in (keywords or BASE_RELEVANCE_KEYWORDS):
        if kw.lower() in low:
            hits.append(kw)
    if any(k in low for k in SKIP_KEYWORDS):
        return -100, hits + ["sensitive-skip"]
    # prefer durable notes and high-level Hindsight layers over raw daily logs
    score = len(set(hits))
    s = str(path)
    if "/topics/" in s or "/details/" in s:
        score += 4
    if "/papers/" in s:
        score += 3
    if "/v2_cards/" in s or "/v2_rebuild/" in s or "canonical-retain-proposal" in s:
        score += 5
    if "/weekly/" in s:
        score += 2
    if "/daily/" in s:
        score -= 4
    return score, sorted(set(hits))[:12]


def collect_source_candidates(memories: Path, days: int, limit: int, *, keywords: list[str] | None = None) -> list[dict[str, object]]:
    globs = ["details/*.md", "topics/*.md", "papers/*.md", "papers/**/*.md", "projects/*.md"]
    candidates = []
    for p in newest_files(memories, globs, days=days, limit=limit * 3):
        text = read_text(p, limit_chars=8000)
        score, hits = score_relevance(text, p, keywords)
        if score <= 0:
            continue
        title = next((m.group(1).strip() for m in re.finditer(r"^#\s+(.+)$", text, re.M)), p.stem)
        candidates.append({
            "path": str(p),
            "title": title,
            "score": score,
            "hits": hits,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "excerpt": re.sub(r"\s+", " ", text[:800]).strip(),
        })
    candidates.sort(key=lambda x: (int(x["score"]), str(x["mtime"])), reverse=True)
    return candidates[:limit]


def latest_hindsight_outputs(root: Path, days: int, limit: int) -> list[Path]:
    if not root.exists():
        return []
    cutoff = datetime.now().timestamp() - days * 86400
    # Preferred high-level layers for wiki candidate generation. The published
    # canonical proposal mirrors what is in the main Hindsight DB; v2_cards and
    # weekly outputs are useful fallback/high-level sources.
    files = [p for p in root.glob("v2_rebuild/gate/canonical-retain-proposal.md") if p.is_file()]
    files += [p for p in root.glob("v2_cards/**/*.md") if p.is_file() and p.stat().st_mtime >= cutoff]
    files += [p for p in root.glob("weekly/**/*.md") if p.is_file() and p.stat().st_mtime >= cutoff]
    # fallback include daily if not enough weekly/canonical material
    files += [p for p in root.glob("daily/**/*.md") if p.is_file() and p.stat().st_mtime >= cutoff]
    def priority(p: Path) -> int:
        s = str(p)
        if "/v2_rebuild/gate/" in s:
            return 0
        if "/v2_cards/" in s:
            return 1
        if "/weekly/" in s:
            return 2
        return 3
    files.sort(key=lambda p: (priority(p), -p.stat().st_mtime))
    return files[:limit]


def extract_hindsight_points(files: list[Path], limit_points: int = 30, *, keywords: list[str] | None = None) -> list[dict[str, object]]:
    points = []
    for p in files:
        text = read_text(p, limit_chars=60000)
        for m in re.finditer(r"^- (.+?)(?=\n- |\n## |\Z)", text, re.M | re.S):
            block = m.group(1).strip()
            one = re.sub(r"\s+", " ", block)
            score, hits = score_relevance(one, p, keywords)
            if score <= 0:
                continue
            # skip transient progress-only lines unless paired with a durable pipeline lesson
            if re.search(r"\b(proc_|pid|timestamp|single timestamp|one-time|临时)\b", one.lower()) and not re.search(r"pipeline|配置|bug|risk|lesson|教训|default|默认", one.lower()):
                continue
            points.append({"source": str(p), "score": score, "hits": hits, "text": one[:1000]})
    # de-duplicate by normalized first 120 chars
    seen = set(); uniq = []
    for pt in sorted(points, key=lambda x: int(x["score"]), reverse=True):
        key = re.sub(r"\W+", "", str(pt["text"]).lower())[:160]
        if key in seen:
            continue
        seen.add(key); uniq.append(pt)
        if len(uniq) >= limit_points:
            break
    return uniq


def wait_hindsight_if_requested(timeout: int, poll: int, *, lock_timeout: int = 21600) -> str:
    if timeout < 0:
        return "skipped"
    messages: list[str] = []
    HINDSIGHT_PIPELINE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now().timestamp()
    try:
        with open(HINDSIGHT_PIPELINE_LOCK, "a+", encoding="utf-8") as lock_fh:
            while True:
                try:
                    fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    messages.append(f"offline_pipeline_lock_acquired={HINDSIGHT_PIPELINE_LOCK}")
                    fcntl.flock(lock_fh, fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    elapsed = datetime.now().timestamp() - start
                    if elapsed > lock_timeout:
                        messages.append(f"offline_pipeline_lock_timeout={lock_timeout}s")
                        break
                    messages.append(f"waiting_for_offline_pipeline_lock elapsed={int(elapsed)}s")
                    import time
                    time.sleep(min(poll, 60))
    except Exception as e:
        messages.append(f"offline_pipeline_lock_error={e!r}")
    if HINDSIGHT_WRAPPER.exists():
        cmd = [sys.executable, str(HINDSIGHT_WRAPPER), "wait-queue", "--poll", str(poll), "--timeout", str(timeout)]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=(timeout + 120 if timeout else None))
            messages.append(f"wait_queue_exit={r.returncode}\n" + r.stdout[-3000:])
            return "\n".join(messages)
        except Exception as e:
            messages.append(f"wait_queue_error={e!r}")
            return "\n".join(messages)
    messages.append("missing_wrapper")
    return "\n".join(messages)


def render_report(*, wiki: Path, pages: list[WikiPage], issues: dict[str, list[str]], source_candidates: list[dict[str, object]], hindsight_files: list[Path], hindsight_points: list[dict[str, object]], hindsight_summary: dict[str, object], wait_status: str, args: argparse.Namespace) -> str:
    counts = {k: len(v) for k, v in issues.items()}
    lines = []
    lines.append(f"# Wiki Auto Maintenance Report - {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- mode: isolated report only; main wiki files were not modified")
    lines.append(f"- wiki: {wiki}")
    lines.append(f"- pages scanned: {len(pages)}")
    lines.append(f"- issue counts: {json.dumps(counts, ensure_ascii=False)}")
    lines.append(f"- source candidates: {len(source_candidates)}")
    lines.append(f"- Hindsight files considered: {len(hindsight_files)}")
    lines.append(f"- Hindsight candidate points: {len(hindsight_points)}")
    lines.append("")
    lines.append("## Hindsight Wait/Queue Status")
    lines.append("```text")
    lines.append(wait_status.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Hindsight Canonical State")
    if hindsight_summary.get("available"):
        lines.append(f"- latest: {hindsight_summary.get('path')}")
        lines.append(f"- mode: {hindsight_summary.get('mode')}")
        lines.append(f"- decision: {hindsight_summary.get('decision')}")
        lines.append(f"- published: {hindsight_summary.get('published')}")
        lines.append(f"- inserted_documents: {hindsight_summary.get('inserted_documents')}")
        lines.append(f"- inserted_observations: {hindsight_summary.get('inserted_observations')}")
        lines.append(f"- conflict_summary: {json.dumps(hindsight_summary.get('conflict_summary'), ensure_ascii=False)}")
    else:
        lines.append(f"- unavailable: {hindsight_summary.get('reason')}")
    lines.append("")
    lines.append("## Wiki Health Findings")
    if not issues:
        lines.append("- No issues found by lightweight audit.")
    else:
        for key in ["broken_links", "frontmatter", "index_missing", "orphans", "large_pages"]:
            vals = issues.get(key, [])
            if not vals:
                continue
            lines.append(f"### {key} ({len(vals)})")
            for v in vals[:80]:
                lines.append(f"- {v}")
            if len(vals) > 80:
                lines.append(f"- ... {len(vals)-80} more")
    lines.append("")
    lines.append("## Hindsight High-Level Candidates")
    lines.append("筛选原则：只保留研究/工作相关、可复用、稳定的高层结论；不直接写入主 wiki。")
    if not hindsight_points:
        lines.append("- No Hindsight candidate points found in the selected window.")
    else:
        for i, pt in enumerate(hindsight_points, 1):
            lines.append(f"### H{i}. score={pt['score']} hits={', '.join(pt['hits'])}")
            lines.append(f"- source: {pt['source']}")
            lines.append(f"- candidate: {pt['text']}")
            lines.append("- suggested_action: review; merge into existing wiki page only if still valid")
    lines.append("")
    lines.append("## Source Note Candidates")
    if not source_candidates:
        lines.append("- No source-note candidates found.")
    else:
        for i, c in enumerate(source_candidates, 1):
            lines.append(f"### S{i}. {c['title']}")
            lines.append(f"- path: {c['path']}")
            lines.append(f"- mtime: {c['mtime']}")
            lines.append(f"- score: {c['score']} hits={', '.join(c['hits'])}")
            lines.append(f"- excerpt: {c['excerpt']}")
            lines.append("- suggested_action: compare with index.md; create/update page only after review")
    lines.append("")
    lines.append("## Suggested Merge Targets")
    target_hints = Counter()
    existing_slugs = {p.slug for p in pages}
    existing_names = {Path(p.slug).name: p.slug for p in pages}
    for pt in hindsight_points:
        for hit in pt.get("hits", [])[:5]:
            slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", str(hit).lower()).strip("-")
            if not slug:
                continue
            target = existing_names.get(slug) or next((s for s in existing_slugs if slug in s.lower()), None)
            if target:
                target_hints[f"update {target}"] += 1
            else:
                target_hints[f"candidate topic: {slug}"] += 1
    if target_hints:
        for k, v in target_hints.most_common(12):
            lines.append(f"- {k}: {v} candidate points")
    else:
        lines.append("- No strong merge target inferred.")
    lines.append("")
    lines.append("## Machine-Readable JSON")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wiki": str(wiki),
        "pages_scanned": len(pages),
        "issue_counts": counts,
        "source_candidates": source_candidates,
        "hindsight_files": [str(p) for p in hindsight_files],
        "hindsight_summary": hindsight_summary,
        "hindsight_points": hindsight_points,
        "args": vars(args),
    }
    lines.append("```json")
    lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run isolated wiki auto-maintenance report")
    ap.add_argument("--wiki", default=str(DEFAULT_WIKI))
    ap.add_argument("--memories", default=str(DEFAULT_MEMORIES))
    ap.add_argument("--hindsight-dir", default=str(DEFAULT_HINDSIGHT))
    ap.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    ap.add_argument("--days", type=int, default=14, help="lookback window for source notes and Hindsight outputs")
    ap.add_argument("--source-limit", type=int, default=40)
    ap.add_argument("--hindsight-file-limit", type=int, default=12)
    ap.add_argument("--hindsight-point-limit", type=int, default=30)
    ap.add_argument("--wait-hindsight", action="store_true", help="wait for Hindsight queue before scanning")
    ap.add_argument("--wait-timeout", type=int, default=21600)
    ap.add_argument("--wait-poll", type=int, default=60)
    ap.add_argument("--lock-timeout", type=int, default=21600, help="seconds to wait for Hindsight offline pipeline lock")
    args = ap.parse_args()

    wiki = Path(args.wiki).expanduser()
    memories = Path(args.memories).expanduser()
    hindsight_dir = Path(args.hindsight_dir).expanduser()
    if not wiki.exists():
        raise SystemExit(f"wiki path not found: {wiki}")
    for required in ["SCHEMA.md", "index.md", "log.md"]:
        if not (wiki / required).exists():
            raise SystemExit(f"wiki missing {required}: {wiki}")

    wait_status = wait_hindsight_if_requested(args.wait_timeout, args.wait_poll, lock_timeout=args.lock_timeout) if args.wait_hindsight else "not requested"
    pages, issues = scan_wiki(wiki)
    keywords = load_domain_keywords(wiki)
    source_candidates = collect_source_candidates(memories, args.days, args.source_limit, keywords=keywords)
    hfiles = latest_hindsight_outputs(hindsight_dir, args.days, args.hindsight_file_limit)
    hsummary = summarize_v2_rebuild(hindsight_dir)
    hpoints = extract_hindsight_points(hfiles, args.hindsight_point_limit, keywords=keywords)

    out_dir = wiki / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wiki-maintenance-{now_stamp()}.md"
    report = render_report(
        wiki=wiki,
        pages=pages,
        issues=issues,
        source_candidates=source_candidates,
        hindsight_files=hfiles,
        hindsight_points=hpoints,
        hindsight_summary=hsummary,
        wait_status=wait_status,
        args=args,
    )
    out_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "report": str(out_path),
        "pages_scanned": len(pages),
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "source_candidates": len(source_candidates),
        "hindsight_files": len(hfiles),
        "hindsight_points": len(hpoints),
        "hindsight_summary": hsummary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
