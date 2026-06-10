#!/usr/bin/env python3
"""Manual external conversation -> Hindsight manifest builder.

This script is deliberately standalone and manual-only. It is not imported by the
regular daily/weekly Hindsight pipeline. Supported sources:

- chat-memo txt exports with Title/URL/Platform/Created headers
- OpenClaw lcm.db conversations, using strict deterministic filters
- Markdown artifacts produced by Hermes/OpenClaw workspaces, parsed by heading/list structure

It writes reviewed manifest JSONL files only; no Hindsight API writes happen here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hindsight_session_manifest as session_manifest  # noqa: E402

DEFAULT_HERMES_HOME = Path.home() / ".hermes"
DEFAULT_OUTPUT_DIR = DEFAULT_HERMES_HOME / "hindsight" / "external_import" / "manifests"
DEFAULT_CHATMEMO_BANK = "hermes"
DEFAULT_OPENCLAW_BANK = "hermes"
DEFAULT_MIXED_BANK = "hermes"
DEFAULT_OPENCLAW_DB = Path.home() / ".openclaw" / "lcm.db"
DEFAULT_MARKDOWN_SCAN_PATHS = [
    Path.home() / ".openclaw" / "workspace" / "memory",
    Path.home() / ".openclaw" / "workspace" / "docs",
    Path.home() / ".openclaw" / "workspace" / ".learnings",
    Path.home() / ".hermes" / "plans",
]
# Backward-compatible alias. The default markdown-artifact CLI path no longer
# scans these roots blindly; it discovers conversation-produced .md files first.
DEFAULT_MARKDOWN_PATHS = DEFAULT_MARKDOWN_SCAN_PATHS
DEFAULT_HERMES_SESSION_ROOTS = [DEFAULT_HERMES_HOME / "sessions"]
DEFAULT_HERMES_PROFILE_ROOT = DEFAULT_HERMES_HOME / "profiles"
DEFAULT_OPENCLAW_SESSION_ROOTS = [Path.home() / ".openclaw" / "agents" / "main" / "sessions"]
OPENCLAW_ROOT_CONTROL_MD_NAMES = {
    "AGENTS.md", "CLAUDE.md", "DREAMS.md", "HEARTBEAT.md", "IDENTITY.md",
    "MEMORY.md", "SOUL.md", "TOOLS.md", "USER.md",
}
DEFAULT_MIN_AGE_SECONDS = 900
DEFAULT_MAX_SEGMENT_CHARS = 80_000
DEFAULT_MAX_SEGMENT_TURNS = 60
DEFAULT_GAP_SPLIT_HOURS = 6.0
DEFAULT_RETAIN_CHUNK_SIZE = 8000
SCHEMA_VERSION = "external-retain-v1"
CLEANING_VERSION = "external-clean-v2"
TAG_RULE_VERSION = "external-tag-rules-v7"
CHATMEMO_ADAPTER_VERSION = "chat-memo-txt-v1"
OPENCLAW_ADAPTER_VERSION = "openclaw-lcm-v1"
MARKDOWN_ADAPTER_VERSION = "markdown-artifact-v1"
MIN_CONTENT_CHARS = 30
TAG_TEXT_HEAD_CHARS = 12_000
SEMANTIC_TAG_PREFIXES = ("domain:", "project:", "topic:")

MESSAGE_MARKER_RE = re.compile(r"^(User|AI|Assistant):\s*\[([^\]]+)\]\s*(.*)$", re.IGNORECASE)
HEADER_RE = re.compile(r"^(Title|URL|Platform|Created|Messages):\s*(.*)$", re.IGNORECASE)
CHATGPT_CONV_RE = re.compile(r"chatgpt\.com/(?:c|share)/([^/?#]+)", re.IGNORECASE)
DOUBAO_CONV_RE = re.compile(r"doubao\.com/chat/([^/?#]+)", re.IGNORECASE)
URL_LAST_SEGMENT_RE = re.compile(r"/([^/?#]+)/?(?:[?#].*)?$")
LEADING_UI_TIMESTAMP_RE = re.compile(r"^\[[A-Z][a-z]{2}\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+GMT[+\-]\d+\]\s*", re.IGNORECASE)

OPENCLAW_DROP_PREFIXES = [
    "System (untrusted):",
    "Sender (untrusted metadata):",
    "[cron:",
    "Read HEARTBEAT.md",
    "LCM compaction",
    "Command still running",
    "Process still running",
    "Process exited",
    "The user received a system notification",
    "(no new output)",
]
OPENCLAW_DROP_EXACT = {"HEARTBEAT_OK", "HEARTBEAT OK"}
OPENCLAW_STATUS_RE = re.compile(
    r"^(脚本正在运行|进程还在运行|继续等待|等待完成|备份完成|检查备份目录详情|命令正在运行|任务正在运行)[。.!！\s]*$",
    re.IGNORECASE,
)
OPENCLAW_UNTRUSTED_BLOCK_RE = re.compile(
    r"(?:Conversation info|Sender|System)\s*\(untrusted[^)]*\):\s*```(?:json)?\s*.*?```\s*",
    re.IGNORECASE | re.DOTALL,
)
SECRET_RE = session_manifest.SECRET_MATERIAL_RE


@dataclass
class Message:
    role: str
    timestamp: str | None
    content: str
    message_id: int | None = None
    seq: int | None = None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_file_metadata(path: Path, raw_bytes: bytes | None = None) -> dict[str, Any]:
    stat = path.stat()
    if raw_bytes is None:
        raw_bytes = path.read_bytes()
    return {
        "source_mtime_ns": stat.st_mtime_ns,
        "source_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "source_size_bytes": stat.st_size,
        "source_file_sha256": sha256_bytes(raw_bytes),
    }


def slugify(value: str | None, default: str = "unknown") -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "chatgpt": "chatgpt",
        "openai": "chatgpt",
        "gemini": "gemini",
        "google gemini": "gemini",
        "豆包": "doubao",
        "doubao": "doubao",
        "openclaw": "openclaw",
    }
    if raw in mapping:
        return mapping[raw]
    out: list[str] = []
    prev_dash = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    return "".join(out).strip("-") or default


def platform_slug(platform: str | None) -> str:
    return slugify(platform, default="unknown")


def conversation_id_from_url_or_path(url: str | None, path: Path, raw_bytes: bytes) -> str:
    url = (url or "").strip()
    for rx in [CHATGPT_CONV_RE, DOUBAO_CONV_RE]:
        m = rx.search(url)
        if m:
            return slugify(m.group(1), default=sha256_bytes(raw_bytes)[:16])
    if url:
        m = URL_LAST_SEGMENT_RE.search(url)
        if m and m.group(1):
            return slugify(m.group(1), default=sha256_bytes(raw_bytes)[:16])
    return slugify(path.stem, default=sha256_bytes(raw_bytes)[:16])


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"]:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def hours_between(a: str | None, b: str | None) -> float | None:
    da = parse_dt(a)
    db = parse_dt(b)
    if da is None or db is None:
        return None
    try:
        return abs((db - da).total_seconds()) / 3600.0
    except TypeError:
        # naive/aware mismatch: normalize by dropping tz for gap purposes.
        return abs((db.replace(tzinfo=None) - da.replace(tzinfo=None)).total_seconds()) / 3600.0


def is_recent_timestamp(value: str | None, min_age_seconds: int) -> bool:
    if not min_age_seconds or min_age_seconds <= 0:
        return False
    dt = parse_dt(value)
    if dt is None:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() < min_age_seconds


def clean_external_text(text: Any) -> str:
    return session_manifest.clean_text(text)


def _bounded_tag_haystack(text: str, title: str | None = None) -> str:
    """Return a short, high-signal haystack for external tag inference.

    External exports can be very long and often contain examples, quoted prior
    answers, or generic words like paper/recall/claim deep in the transcript.
    Tagging over the entire body caused over-broad tags, so only use title plus
    the beginning of cleaned content.
    """
    title_part = str(title or "")
    if re.match(r"^openclaw conversation \d+$", title_part.strip(), re.IGNORECASE):
        title_part = ""
    body_part = clean_external_text(text or "")
    body_part = re.sub(r"(?im)^(Title|Platform|URL|Created|Source):\s*.*$", "", body_part)
    body_part = body_part[:TAG_TEXT_HEAD_CHARS]
    return f"{title_part}\n{body_part}".lower()


def _contains_any(hay: str, needles: Iterable[str]) -> bool:
    return any(str(n).lower() in hay for n in needles)


def _word_count(hay: str, words: Iterable[str]) -> int:
    total = 0
    for word in words:
        pattern = r"(?<![a-z0-9_])" + re.escape(str(word).lower()) + r"(?![a-z0-9_])"
        if re.search(pattern, hay, re.IGNORECASE):
            total += 1
    return total


def semantic_tags_for_text(text: str, title: str | None = None, extra: str | None = None) -> list[str]:
    """Infer conservative semantic tags for third-party conversation imports.

    Do not reuse the generic Hermes session tagger directly: it is intentionally
    broad for first-party sessions and creates false positives on long external
    transcripts (e.g. any occurrence of "paper", "patent", or "recall").
    """
    hay = _bounded_tag_haystack(text, title)
    title_l = str(title or "").lower()
    tags: set[str] = set()

    if _contains_any(hay, ["egomotion4d", "trackingworld", "vggt4d", "dggt", "dgegt", "dage", "ate_metric", "ate metric", "joint ba", "roma2"]):
        tags.update(["project:egomotion4d", "domain:autodrive"])
    if _contains_any(hay, ["vggt-long", "vggt long", "loop closure", "salad", "dinov2"]):
        tags.update(["project:vggt-long", "domain:autodrive"])
    if _contains_any(hay, ["openclaw", "clawhub", "gateway probe"]):
        tags.add("project:openclaw")

    autodrive_strong = ["aeb", "adas", "自动驾驶", "智驾", "车道线", "单目测速", "单目3d", "l2级", "l2 ", "bev", "occupancy", "occflow", "pdms", "collision rate", "world model", "vla"]
    if _contains_any(hay, autodrive_strong):
        tags.add("domain:autodrive")

    hindsight_strong = [
        "hindsight",
        "hidesight",  # common typo in historical exports
        "memory provider",
        "memory bank",
        "memory_units",
        "long-term memory",
        "长期记忆",
        "记忆库",
        "记忆系统",
        "hermes 内置的长期记忆",
        "hermes内置的长期记忆",
    ]
    if _contains_any(hay, hindsight_strong) or ("hermes" in hay and _contains_any(hay, ["召回", "记忆", "retain", "reflect", "consolidation", "observations"])):
        tags.update(["domain:hindsight", "topic:memory-management"])
    if "domain:hindsight" in tags and _contains_any(hay, ["native consolidation", "consolidation", "observations", "observation_scopes", "enable_observations", "pending_consolidation"]):
        tags.add("topic:native-consolidation")
    if "domain:hindsight" in tags and _contains_any(hay, ["recall cache", "auto_recall", "conditional recall", "recall_cache"]):
        tags.add("topic:recall-cache")

    patent_title_signal = _contains_any(title_l, ["专利", "审查意见", "权利要求", "office action", "oa1"])
    patent_body_signal = (
        _contains_any(hay, ["审查意见", "权利要求", "office action", "oa1"])
        or hay.count("专利") >= 2
        or (_word_count(hay, ["patent", "claim", "claims"]) >= 2)
    )
    if patent_title_signal or patent_body_signal:
        tags.add("domain:patent")

    paper_title_signal = _contains_any(title_l, ["论文", "文章投", "science", "子刊", "投稿", "arxiv", "paper", "citation", "abstract"])
    paper_body_signal = _contains_any(hay, ["arxiv", "citation", "abstract", "投science", "投 science", "论文", "投稿"]) or _word_count(hay, ["paper", "papers"]) >= 2
    if paper_title_signal or paper_body_signal:
        tags.add("domain:paper")

    return sorted(tags)


def external_action_for_content(text: str, tags: list[str]) -> tuple[str, str]:
    clean = clean_external_text(text)
    if len(clean.strip()) < MIN_CONTENT_CHARS:
        return "skip", "empty_or_too_short"
    if session_manifest.is_low_signal_conversation(clean):
        return "skip", "low_signal_short_or_chitchat"
    if SECRET_RE.search(clean):
        return "manual_review", "secret_or_credential_material"
    semantic = [t for t in tags if t.startswith(SEMANTIC_TAG_PREFIXES)]
    if not semantic:
        return "manual_review", "no_semantic_tags"
    project_tags = [t for t in semantic if t.startswith("project:")]
    if len(project_tags) > 1 or len(semantic) > 5:
        return "manual_review", "multi_scope_or_overbroad_tags"
    return "production", "semantic_tags_detected"


def observation_scopes_for_tags(tags: Iterable[str], include: bool = True) -> list[list[str]]:
    if not include:
        return []
    return session_manifest.observation_scopes_for_tags(tags)


def render_conversation_content(*, title: str | None, platform: str, url: str | None, created_at: str | None, source: str, messages: list[Message]) -> str:
    header = [
        f"Title: {title or ''}".rstrip(),
        f"Platform: {platform}",
    ]
    if url:
        header.append(f"URL: {url}")
    if created_at:
        header.append(f"Created: {created_at}")
    header.append(f"Source: {source}")
    parts = ["\n".join(header).strip()]
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        ts = f" [{msg.timestamp}]" if msg.timestamp else ""
        parts.append(f"{role}:{ts}\n{msg.content}".strip())
    return clean_external_text("\n\n".join(p for p in parts if p))


def split_text(text: str, max_chars: int) -> list[str]:
    return session_manifest.split_text(text, max_chars)


# ---------------------------------------------------------------------------
# Markdown artifact adapter


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*+]\s+|\d+[.)、]\s+|[（(]?\d+[）)]\s+|[一二三四五六七八九十百]+[、.．]\s+)(.*)$"
)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
REPORT_DATE_RE = re.compile(r"(20\d{2}[-_年.]\d{1,2}[-_月.]\d{1,2})")
SKIP_MD_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
PRODUCED_MD_HINT_RE = re.compile(
    r"(Successfully\s+wrote|bytes_written|File\s+['\"][^'\"]+\.md['\"]\s+written|"
    r"\b(wrote|written|created|saved|generated)\b|已写入|写入|保存到|生成(?:了)?|创建(?:了)?)",
    re.IGNORECASE,
)
NEGATED_MD_PRODUCTION_RE = re.compile(
    r"(不是新写入|不是写入|没有写入|未写入|无需写入|直接看已有|读取|read(?:ing)?\s+(?:the\s+)?(?:existing\s+)?file|not\s+(?:newly\s+)?written)",
    re.IGNORECASE,
)
SUCCESSFULLY_WROTE_MD_RE = re.compile(r"Successfully\s+wrote\s+\d+\s+bytes\s+to\s+(?P<path>[^\r\n]+?\.md)", re.IGNORECASE)
ABS_MD_PATH_RE = re.compile(r"(?P<path>(?:~|/[^\x00\r\n`\"'<>]+?\.md))")
REL_MD_PATH_RE = re.compile(
    r"(?P<path>(?:\.learnings|memory|docs|plans|wiki|references|raw|auto-maintenance|reports|outputs)/"
    r"[^\x00\r\n`\"'<>]+?\.md)"
)
BACKTICK_MD_PATH_RE = re.compile(r"`(?P<path>[^`]+?\.md)`")
TRAILING_PATH_PUNCT = " \t\r\n,，.。;；:：)）]】}>'\""


def safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def iter_text_values(obj: Any, *, max_items: int = 200) -> Iterable[str]:
    """Yield text leaves from small JSON-like payloads for path discovery."""
    stack = [obj]
    seen = 0
    while stack and seen < max_items:
        cur = stack.pop()
        seen += 1
        if isinstance(cur, str):
            yield cur
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def extract_md_path_candidates(text: Any) -> list[str]:
    if not isinstance(text, str) or ".md" not in text:
        return []
    candidates: list[str] = []
    wrote_matches = list(SUCCESSFULLY_WROTE_MD_RE.finditer(text))
    if wrote_matches:
        for m in wrote_matches:
            raw = (m.group("path") or "").strip().strip(TRAILING_PATH_PUNCT)
            if raw and raw.lower().endswith(".md"):
                candidates.append(raw)
        return list(dict.fromkeys(candidates))
    for rx in [BACKTICK_MD_PATH_RE, ABS_MD_PATH_RE, REL_MD_PATH_RE]:
        for m in rx.finditer(text):
            raw = (m.group("path") or "").strip().strip(TRAILING_PATH_PUNCT)
            if raw and raw.lower().endswith(".md"):
                candidates.append(raw)
    return list(dict.fromkeys(candidates))


def has_positive_md_production_hint(text: Any) -> bool:
    if not isinstance(text, str) or ".md" not in text:
        return False
    return bool(PRODUCED_MD_HINT_RE.search(text) and not NEGATED_MD_PRODUCTION_RE.search(text))


def resolve_md_candidate_path(raw: str, *, origin_hint: str | None = None) -> Path | None:
    value = str(raw or "").strip().strip(TRAILING_PATH_PUNCT)
    if not value or not value.lower().endswith(".md"):
        return None
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        roots: list[Path] = []
        if origin_hint == "openclaw":
            roots.append(Path.home() / ".openclaw" / "workspace")
        elif origin_hint == "hermes":
            roots.append(DEFAULT_HERMES_HOME)
        roots.extend([
            Path.home() / ".openclaw" / "workspace",
            DEFAULT_HERMES_HOME,
            Path.home(),
        ])
        for root in roots:
            candidates.append(root / value)
    for cand in candidates:
        try:
            resolved = cand.expanduser().resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".md":
                return resolved
        except OSError:
            continue
    return None


def is_recent_file(path: Path, min_file_age_seconds: int) -> bool:
    if not min_file_age_seconds or min_file_age_seconds <= 0:
        return False
    cutoff_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    try:
        return path.stat().st_mtime_ns > cutoff_ns
    except OSError:
        return True


def add_discovered_md_path(
    discovered: dict[str, dict[str, Any]],
    missing: Counter,
    raw_path: str,
    *,
    origin_hint: str,
    producer: str,
    evidence: dict[str, Any],
    min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    allowed_roots: Iterable[str | Path] | None = None,
) -> None:
    resolved = resolve_md_candidate_path(raw_path, origin_hint=origin_hint)
    if resolved is None:
        missing[str(raw_path)] += 1
        return
    if allowed_roots is not None:
        allowed = False
        for root in allowed_roots:
            try:
                resolved.relative_to(Path(root).expanduser().resolve())
                allowed = True
                break
            except Exception:
                pass
        if not allowed:
            missing[f"outside_allowed_roots::{resolved}"] += 1
            return
    if any(part in SKIP_MD_DIRS for part in resolved.parts):
        missing[str(raw_path)] += 1
        return
    if origin_hint == "openclaw" and str(resolved).startswith(str((Path.home() / ".openclaw" / "workspace").resolve())) and not is_openclaw_allowed_artifact_path(resolved):
        missing[f"openclaw_control_file::{resolved}"] += 1
        return
    if is_recent_file(resolved, min_file_age_seconds):
        missing[f"too_recent::{resolved}"] += 1
        return
    key = str(resolved)
    item = discovered.setdefault(key, {"path": key, "origin_hint": origin_hint, "producer": producer, "evidence": []})
    ev = {k: v for k, v in evidence.items() if v not in (None, "", [], {})}
    if len(item["evidence"]) < 5:
        item["evidence"].append(ev)


def is_openclaw_allowed_artifact_path(path: Path) -> bool:
    try:
        p = path.expanduser().resolve()
        workspace = (Path.home() / ".openclaw" / "workspace").resolve()
        rel = p.relative_to(workspace)
    except Exception:
        return False
    if len(rel.parts) == 1 and rel.name in OPENCLAW_ROOT_CONTROL_MD_NAMES:
        return False
    return True


def infer_artifact_origin(path: Path) -> str:
    text = str(path.expanduser())
    home = str(Path.home())
    if text.startswith(f"{home}/.openclaw/"):
        return "openclaw"
    if text.startswith(f"{home}/.hermes/"):
        return "hermes"
    return "external"


def infer_markdown_artifact_type(path: Path, title: str | None, text: str) -> str:
    name = path.name.lower()
    hay = f"{path.name}\n{title or ''}\n{text[:2000]}".lower()
    if "周报" in hay or "weekly report" in hay or name.startswith("weekly"):
        return "weekly_report"
    if "日报" in hay or "daily report" in hay:
        return "daily_report"
    if name in {"agents.md", "soul.md", "tools.md", "user.md", "memory.md"}:
        return "agent_memory_or_rules"
    if "plan" in name or "计划" in hay:
        return "plan"
    if "learn" in name or "经验" in hay or "错误" in hay:
        return "learning"
    return "markdown_document"


def report_date_from_markdown(path: Path, title: str | None, text: str) -> str | None:
    for value in [path.name, title or "", text[:1000]]:
        m = REPORT_DATE_RE.search(value)
        if m:
            raw = m.group(1)
            raw = raw.replace("年", "-").replace("月", "-").replace("日", "")
            raw = raw.replace("_", "-").replace(".", "-")
            parts = [p for p in raw.split("-") if p]
            if len(parts) >= 3:
                y, mo, d = parts[:3]
                return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return None


def relative_artifact_path(path: Path) -> str:
    p = path.expanduser().resolve()
    for root in [Path.home() / ".openclaw", Path.home() / ".hermes", Path.home()]:
        try:
            return str(p.relative_to(root.expanduser().resolve()))
        except Exception:
            pass
    return str(p)


def markdown_artifact_id(path: Path) -> str:
    rel = relative_artifact_path(path)
    slug = slugify(rel, default="markdown")[:80]
    return f"{slug}-{sha256_text(str(path.expanduser().resolve()))[:12]}"


def default_hermes_session_roots(include_profiles: bool = True) -> list[Path]:
    roots = list(DEFAULT_HERMES_SESSION_ROOTS)
    if include_profiles and DEFAULT_HERMES_PROFILE_ROOT.exists():
        roots.extend(sorted(p / "sessions" for p in DEFAULT_HERMES_PROFILE_ROOT.iterdir() if (p / "sessions").is_dir()))
    return roots


def iter_hermes_session_files(roots: Iterable[str | Path] | None = None, *, limit_sessions: int | None = None) -> Iterable[Path]:
    count = 0
    for root in roots or default_hermes_session_roots(include_profiles=True):
        r = Path(root).expanduser()
        if not r.exists():
            continue
        files = [r] if r.is_file() else sorted(r.glob("session_*.json"), key=lambda p: p.stat().st_mtime_ns if p.exists() else 0)
        for path in files:
            if not path.is_file():
                continue
            yield path
            count += 1
            if limit_sessions is not None and count >= limit_sessions:
                return


def is_hermes_cron_session(session_path: Path, data: dict[str, Any]) -> bool:
    session_id = str(data.get("session_id") or "")
    platform = str(data.get("platform") or "")
    return bool(platform == "cron" or session_id.startswith("cron_") or session_path.name.startswith("session_cron_"))


def _tool_call_function_name(tool_call: dict[str, Any]) -> str:
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    if isinstance(func, dict):
        return str(func.get("name") or "")
    return str(tool_call.get("name") or "") if isinstance(tool_call, dict) else ""


def _tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    raw = func.get("arguments") if isinstance(func, dict) else tool_call.get("arguments") if isinstance(tool_call, dict) else None
    if isinstance(raw, dict):
        return raw
    parsed = safe_json_loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def discover_markdown_paths_from_hermes_sessions(
    roots: Iterable[str | Path] | None = None,
    *,
    include_cron: bool = False,
    min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    limit_sessions: int | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    missing: Counter = Counter()
    diagnostics: dict[str, Any] = {
        "producer": "hermes_sessions",
        "session_roots": [str(p) for p in (roots or default_hermes_session_roots(include_profiles=True))],
        "sessions_seen": 0,
        "sessions_skipped_cron": 0,
        "candidate_mentions": 0,
    }
    for session_path in iter_hermes_session_files(roots, limit_sessions=limit_sessions):
        try:
            data = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        diagnostics["sessions_seen"] += 1
        if not include_cron and is_hermes_cron_session(session_path, data):
            diagnostics["sessions_skipped_cron"] += 1
            continue
        session_id = str(data.get("session_id") or session_path.stem)
        for msg_idx, msg in enumerate(data.get("messages") or []):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            # Strongest evidence: Hermes write_file tool calls contain an explicit target path.
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fname = _tool_call_function_name(tc)
                    args = _tool_call_arguments(tc)
                    if fname == "write_file" and str(args.get("path") or "").lower().endswith(".md"):
                        diagnostics["candidate_mentions"] += 1
                        add_discovered_md_path(
                            discovered,
                            missing,
                            str(args.get("path")),
                            origin_hint="hermes",
                            producer="hermes_write_file_tool",
                            min_file_age_seconds=min_file_age_seconds,
                            evidence={"session": str(session_path), "session_id": session_id, "message_index": msg_idx, "tool": fname},
                        )
                content = str(msg.get("content") or "")
                if has_positive_md_production_hint(content):
                    for raw in extract_md_path_candidates(content):
                        diagnostics["candidate_mentions"] += 1
                        add_discovered_md_path(
                            discovered,
                            missing,
                            raw,
                            origin_hint="hermes",
                            producer="hermes_assistant_text",
                            min_file_age_seconds=min_file_age_seconds,
                            evidence={"session": str(session_path), "session_id": session_id, "message_index": msg_idx, "role": role},
                        )
            elif role == "tool" and str(msg.get("name") or "") in {"terminal", "execute_code"}:
                parsed = safe_json_loads(msg.get("content"))
                texts = []
                if isinstance(parsed, dict):
                    for key in ["output", "stdout", "content", "result"]:
                        if isinstance(parsed.get(key), str):
                            texts.append(parsed[key])
                elif isinstance(msg.get("content"), str):
                    texts.append(str(msg.get("content")))
                for text in texts:
                    if has_positive_md_production_hint(text):
                        for raw in extract_md_path_candidates(text):
                            diagnostics["candidate_mentions"] += 1
                            add_discovered_md_path(
                                discovered,
                                missing,
                                raw,
                                origin_hint="hermes",
                                producer=f"hermes_tool_output:{msg.get('name')}",
                                min_file_age_seconds=min_file_age_seconds,
                                evidence={"session": str(session_path), "session_id": session_id, "message_index": msg_idx, "tool": msg.get("name")},
                            )
    paths = [Path(item["path"]) for item in discovered.values()]
    diagnostics["paths_found"] = len(paths)
    diagnostics["missing_or_skipped_candidates"] = dict(sorted(missing.items())[:100])
    diagnostics["evidence_sample"] = list(discovered.values())[:20]
    return sorted(paths), diagnostics


def iter_openclaw_session_files(roots: Iterable[str | Path] | None = None, *, limit_sessions: int | None = None) -> Iterable[Path]:
    count = 0
    for root in roots or DEFAULT_OPENCLAW_SESSION_ROOTS:
        r = Path(root).expanduser()
        if not r.exists():
            continue
        files = [r] if r.is_file() else sorted(r.glob("*.jsonl"), key=lambda p: p.stat().st_mtime_ns if p.exists() else 0)
        for path in files:
            if not path.is_file():
                continue
            yield path
            count += 1
            if limit_sessions is not None and count >= limit_sessions:
                return


def is_openclaw_session_cron(session_path: Path, event: dict[str, Any]) -> bool:
    session_key = str(event.get("sessionKey") or event.get("session_key") or "")
    session_id = str(event.get("sessionId") or event.get("session_id") or "")
    path_text = str(session_path)
    return bool("cron" in session_key or session_key.startswith("agent:main:cron:") or session_id.startswith("cron_") or "/cron/" in path_text or session_path.name.startswith("session_cron_"))


def discover_markdown_paths_from_openclaw_sessions(
    roots: Iterable[str | Path] | None = None,
    *,
    include_cron: bool = False,
    min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    limit_sessions: int | None = None,
    allowed_roots: Iterable[str | Path] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    missing: Counter = Counter()
    diagnostics: dict[str, Any] = {
        "producer": "openclaw_sessions",
        "session_roots": [str(p) for p in (roots or DEFAULT_OPENCLAW_SESSION_ROOTS)],
        "sessions_seen": 0,
        "sessions_skipped_cron": 0,
        "candidate_mentions": 0,
    }
    for session_path in iter_openclaw_session_files(roots, limit_sessions=limit_sessions):
        try:
            raw_text = session_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if is_recent_file(session_path, min_file_age_seconds):
            continue
        diagnostics["sessions_seen"] += 1
        file_skipped_cron = False
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not include_cron and is_openclaw_session_cron(session_path, event):
                file_skipped_cron = True
                break
            texts = list(iter_text_values(event))
            for text in texts:
                if not has_positive_md_production_hint(text):
                    continue
                for raw in extract_md_path_candidates(text):
                    diagnostics["candidate_mentions"] += 1
                    add_discovered_md_path(
                        discovered,
                        missing,
                        raw,
                        origin_hint="openclaw",
                        producer="openclaw_session_jsonl",
                        min_file_age_seconds=min_file_age_seconds,
                        allowed_roots=list(allowed_roots) if allowed_roots is not None else [Path.home() / ".openclaw" / "workspace"],
                        evidence={
                            "session": str(session_path),
                            "line": line_no,
                            "session_id": str(event.get("sessionId") or event.get("session_id") or ""),
                            "session_key": str(event.get("sessionKey") or event.get("session_key") or ""),
                            "event_type": str(event.get("type") or ""),
                        },
                    )
        if file_skipped_cron:
            diagnostics["sessions_skipped_cron"] += 1
    paths = [Path(item["path"]) for item in discovered.values()]
    diagnostics["paths_found"] = len(paths)
    diagnostics["missing_or_skipped_candidates"] = dict(sorted(missing.items())[:100])
    diagnostics["evidence_sample"] = list(discovered.values())[:20]
    return sorted(paths), diagnostics


def _openclaw_structured_write_paths(content: Any) -> list[str]:
    parsed = safe_json_loads(content)
    if parsed is None:
        return []
    out: list[str] = []
    def walk(obj: Any) -> None:
        if isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            name = str(obj.get("name") or "").lower()
            typ = str(obj.get("type") or "").replace("-", "_").lower()
            if typ in {"toolcall", "tool_call"} and name in {"write", "write_file"}:
                args = obj.get("arguments")
                if isinstance(args, str):
                    args = safe_json_loads(args)
                if isinstance(args, dict) and str(args.get("path") or "").lower().endswith(".md"):
                    out.append(str(args.get("path")))
            else:
                for value in obj.values():
                    walk(value)
    walk(parsed)
    return out


def discover_markdown_paths_from_openclaw_lcm(
    db_path: str | Path = DEFAULT_OPENCLAW_DB,
    *,
    include_dingtalk: bool = True,
    include_history_aggregate: bool = False,
    min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    limit_messages: int | None = None,
    allowed_roots: Iterable[str | Path] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    db_path = Path(db_path).expanduser()
    discovered: dict[str, dict[str, Any]] = {}
    missing: Counter = Counter()
    diagnostics: dict[str, Any] = {
        "producer": "openclaw_lcm",
        "db": str(db_path),
        "messages_seen": 0,
        "conversations_skipped": Counter(),
        "candidate_mentions": 0,
    }
    if not db_path.exists():
        diagnostics["missing_db"] = True
        return [], diagnostics
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT m.message_id, m.conversation_id, m.seq, m.role, m.content, m.created_at,
                   c.session_key, c.title, c.created_at AS conversation_created_at, c.updated_at AS conversation_updated_at
            FROM messages m
            JOIN conversations c ON c.conversation_id = m.conversation_id
            WHERE m.content LIKE '%.md%'
              AND m.role IN ('assistant', 'tool')
            ORDER BY m.message_id
        """
        rows = conn.execute(query).fetchall()
        if limit_messages is not None:
            rows = rows[:limit_messages]
        for row in rows:
            diagnostics["messages_seen"] += 1
            reason = openclaw_session_key_reason(row["session_key"], include_dingtalk=include_dingtalk)
            if reason:
                diagnostics["conversations_skipped"][reason] += 1
                continue
            if is_history_aggregate_title(row["title"]) and not include_history_aggregate:
                diagnostics["conversations_skipped"]["history_aggregate"] += 1
                continue
            raw_paths: list[str] = []
            role = str(row["role"] or "")
            content = str(row["content"] or "")
            if role == "assistant":
                raw_paths.extend(_openclaw_structured_write_paths(content))
            elif role == "tool" and SUCCESSFULLY_WROTE_MD_RE.search(content):
                raw_paths.extend(extract_md_path_candidates(content))
            for raw in list(dict.fromkeys(raw_paths)):
                diagnostics["candidate_mentions"] += 1
                add_discovered_md_path(
                    discovered,
                    missing,
                    raw,
                    origin_hint="openclaw",
                    producer=f"openclaw_lcm:{role}",
                    min_file_age_seconds=min_file_age_seconds,
                    allowed_roots=list(allowed_roots) if allowed_roots is not None else [Path.home() / ".openclaw" / "workspace"],
                    evidence={
                        "db": str(db_path),
                        "conversation_id": str(row["conversation_id"]),
                        "message_id": str(row["message_id"]),
                        "seq": str(row["seq"]),
                        "role": role,
                    },
                )
    finally:
        conn.close()
    paths = [Path(item["path"]) for item in discovered.values()]
    diagnostics["paths_found"] = len(paths)
    diagnostics["missing_or_skipped_candidates"] = dict(sorted(missing.items())[:100])
    diagnostics["evidence_sample"] = list(discovered.values())[:20]
    diagnostics["conversations_skipped"] = dict(diagnostics["conversations_skipped"])
    return sorted(paths), diagnostics


def discover_conversation_markdown_artifact_paths(
    *,
    hermes_roots: Iterable[str | Path] | None = None,
    openclaw_db: str | Path = DEFAULT_OPENCLAW_DB,
    openclaw_session_roots: Iterable[str | Path] | None = None,
    include_hermes: bool = False,
    include_openclaw: bool = True,
    include_openclaw_sessions: bool = True,
    include_cron: bool = False,
    include_dingtalk: bool = True,
    min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
) -> tuple[list[Path], dict[str, Any]]:
    all_paths: dict[str, Path] = {}
    diagnostics: dict[str, Any] = {"source_kind": "markdown_artifact_md", "discovery_mode": "conversation_produced", "sources": []}
    if include_hermes:
        paths, diag = discover_markdown_paths_from_hermes_sessions(
            hermes_roots,
            include_cron=include_cron,
            min_file_age_seconds=min_file_age_seconds,
        )
        diagnostics["sources"].append(diag)
        for p in paths:
            all_paths[str(p)] = p
    if include_openclaw:
        paths, diag = discover_markdown_paths_from_openclaw_lcm(
            openclaw_db,
            include_dingtalk=include_dingtalk,
            min_file_age_seconds=min_file_age_seconds,
        )
        diagnostics["sources"].append(diag)
        for p in paths:
            all_paths[str(p)] = p
    if include_openclaw_sessions:
        paths, diag = discover_markdown_paths_from_openclaw_sessions(
            openclaw_session_roots,
            include_cron=include_cron,
            min_file_age_seconds=min_file_age_seconds,
        )
        diagnostics["sources"].append(diag)
        for p in paths:
            all_paths[str(p)] = p
    diagnostics["paths_found"] = len(all_paths)
    diagnostics["paths"] = sorted(all_paths)
    return [all_paths[k] for k in sorted(all_paths)], diagnostics


def parse_markdown_blocks(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse markdown into structural blocks without assuming section names."""
    lines = text.splitlines()
    heading_stack: list[tuple[int, str]] = []
    title: str | None = None
    blocks: list[dict[str, Any]] = []
    para: list[str] = []
    current_list: list[str] | None = None
    in_fence = False

    def section_path() -> list[str]:
        return [t for _lvl, t in heading_stack]

    def flush_para() -> None:
        nonlocal para
        content = clean_external_text("\n".join(para))
        if content:
            blocks.append({
                "block_type": "paragraph",
                "content": content,
                "section_path": section_path(),
                "heading_level": heading_stack[-1][0] if heading_stack else 0,
                "section_title": heading_stack[-1][1] if heading_stack else None,
            })
        para = []

    def flush_list() -> None:
        nonlocal current_list
        if current_list is None:
            return
        content = clean_external_text("\n".join(current_list))
        if content:
            blocks.append({
                "block_type": "list_item",
                "content": content,
                "section_path": section_path(),
                "heading_level": heading_stack[-1][0] if heading_stack else 0,
                "section_title": heading_stack[-1][1] if heading_stack else None,
            })
        current_list = None

    for raw in lines:
        line = raw.rstrip("\n")
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if current_list is not None:
                current_list.append(line)
            else:
                para.append(line)
            continue
        if not in_fence:
            hm = HEADING_RE.match(line)
            if hm:
                flush_list()
                flush_para()
                level = len(hm.group(1))
                heading = clean_external_text(hm.group(2))
                if heading and title is None:
                    title = heading
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
                blocks.append({
                    "block_type": "heading",
                    "content": heading,
                    "section_path": section_path(),
                    "heading_level": level,
                    "section_title": heading,
                })
                continue
            lm = LIST_ITEM_RE.match(line)
            if lm:
                flush_para()
                flush_list()
                current_list = [lm.group(1).strip() or line.strip()]
                continue
            if not line.strip():
                flush_list()
                flush_para()
                continue
        if current_list is not None:
            current_list.append(line)
        else:
            para.append(line)
    flush_list()
    flush_para()
    return title, blocks


def render_markdown_artifact_content(
    *,
    path: Path,
    title: str | None,
    artifact_origin: str,
    artifact_type: str,
    report_date: str | None,
    record_kind: str,
    section_path: list[str] | None,
    item_index: int | None,
    body: str,
) -> str:
    header = [
        "Markdown Artifact",
        f"Title: {title or path.stem}",
        f"Origin: {artifact_origin}",
        f"Artifact-Type: {artifact_type}",
        f"Source: {path}",
        f"Record-Kind: {record_kind}",
    ]
    if report_date:
        header.append(f"Report-Date: {report_date}")
    if section_path:
        header.append("Section-Path: " + " > ".join(section_path))
    if item_index is not None:
        header.append(f"Item-Index: {item_index}")
    return clean_external_text("\n".join(header) + "\n\n" + (body or ""))


def markdown_artifact_action(text: str) -> tuple[str, str]:
    clean = clean_external_text(text)
    if len(clean.strip()) < MIN_CONTENT_CHARS:
        return "skip", "empty_or_too_short"
    if SECRET_RE.search(clean):
        return "manual_review", "secret_or_credential_material"
    return "production", "markdown_artifact_structured"


def records_from_markdown_file(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_MIXED_BANK,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    include_observation_scopes: bool = True,
    max_section_chars: int = 16_000,
    record_granularity: str = "items",
) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    full_text = clean_external_text(raw_text)
    title, blocks = parse_markdown_blocks(full_text)
    artifact_origin = infer_artifact_origin(path)
    artifact_type = infer_markdown_artifact_type(path, title, full_text)
    report_date = report_date_from_markdown(path, title, full_text)
    source_meta = source_file_metadata(path, raw_bytes)
    artifact_id = markdown_artifact_id(path)
    base_doc_id = f"external-md-artifact::{artifact_origin}::{artifact_id}"
    base_tags = {"source:external-markdown-artifact", f"origin:{artifact_origin}", f"artifact:{artifact_type}", "topic:markdown-artifact"}
    if artifact_type == "weekly_report":
        base_tags.add("topic:weekly-report")
    semantic_tags = set(semantic_tags_for_text(full_text, title, artifact_origin))
    if artifact_origin == "openclaw":
        semantic_tags.add("project:openclaw")
    elif artifact_origin == "hermes":
        semantic_tags.add("project:hermes")
    tags = sorted(base_tags | semantic_tags)
    records: list[dict[str, Any]] = []
    granularity = str(record_granularity or "items").lower()
    include_doc = granularity in {"full", "all", "sections", "items"}
    include_items = granularity in {"all", "items"}
    include_sections = granularity in {"all", "sections"}

    def add_record(
        *,
        suffix: str,
        record_kind: str,
        body: str,
        section_path: list[str] | None = None,
        section_title: str | None = None,
        heading_level: int | None = None,
        item_index: int | None = None,
        block_type: str | None = None,
    ) -> None:
        content = render_markdown_artifact_content(
            path=path,
            title=title,
            artifact_origin=artifact_origin,
            artifact_type=artifact_type,
            report_date=report_date,
            record_kind=record_kind,
            section_path=section_path,
            item_index=item_index,
            body=body,
        )
        action, reason = markdown_artifact_action(content)
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "adapter_version": MARKDOWN_ADAPTER_VERSION,
            "cleaning_version": CLEANING_VERSION,
            "tag_rule_version": TAG_RULE_VERSION,
            "source_kind": "markdown_artifact_md",
            "source_label": "markdown-artifact",
            "source_path": str(path),
            "artifact_origin": artifact_origin,
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "title": title or path.stem,
            "report_date": report_date,
            "record_kind": record_kind,
            "section_title": section_title,
            "section_path": section_path or [],
            "heading_level": heading_level,
            "item_index": item_index,
            "block_type": block_type,
            "content_sha256": sha256_text(content),
            "full_content_sha256": sha256_text(full_text),
            "bank_target": bank_target,
            **source_meta,
        }
        records.append({
            "document_id": f"{base_doc_id}::{suffix}",
            "bank_target": bank_target,
            "action": action,
            "reason": reason,
            "content": content,
            "content_chars": len(content),
            "estimated_retain_chunks": max(1, math.ceil(len(content) / max(1, retain_chunk_size))) if content else 0,
            "event_date": report_date or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "tags": tags,
            "observation_scopes": observation_scopes_for_tags(tags, include_observation_scopes),
            "metadata": metadata,
            "context": "external_markdown_artifact",
            "update_mode": "replace",
        })

    outline = []
    for block in blocks:
        if block.get("block_type") == "heading":
            lvl = int(block.get("heading_level") or 1)
            outline.append("  " * max(0, lvl - 1) + "- " + str(block.get("content") or ""))
    if include_doc:
        add_record(
            suffix="doc",
            record_kind="document_outline",
            body="Outline:\n" + "\n".join(outline[:200]) if outline else full_text[:max_section_chars],
            section_path=[],
            block_type="document",
        )

    section_chunks: dict[tuple[str, ...], list[str]] = defaultdict(list)
    section_meta: dict[tuple[str, ...], dict[str, Any]] = {}
    item_index = 0
    section_index_map: dict[tuple[str, ...], int] = {}
    for block in blocks:
        btype = str(block.get("block_type") or "")
        if btype == "heading":
            key = tuple(block.get("section_path") or [])
            if key not in section_index_map:
                section_index_map[key] = len(section_index_map) + 1
                section_meta[key] = block
            continue
        body = str(block.get("content") or "").strip()
        if not body:
            continue
        key = tuple(block.get("section_path") or [])
        section_chunks[key].append(body)
        if include_items:
            item_index += 1
            add_record(
                suffix=f"item-{item_index:04d}",
                record_kind="item",
                body=body,
                section_path=list(key),
                section_title=block.get("section_title"),
                heading_level=block.get("heading_level"),
                item_index=item_index,
                block_type=btype,
            )
    if include_sections:
        for key, parts in sorted(section_chunks.items(), key=lambda kv: section_index_map.get(kv[0], 10_000)):
            body = clean_external_text("\n\n".join(parts))
            if not body:
                continue
            meta = section_meta.get(key, {})
            for idx, chunk in enumerate(split_text(body, max_section_chars) or [body], start=1):
                sec_idx = section_index_map.get(key, 0)
                add_record(
                    suffix=f"sec-{sec_idx:03d}-part-{idx:03d}",
                    record_kind="section",
                    body=chunk,
                    section_path=list(key),
                    section_title=meta.get("section_title") or (key[-1] if key else None),
                    heading_level=meta.get("heading_level"),
                    item_index=None,
                    block_type="section",
                )
    return records


def iter_markdown_files(paths: Iterable[Path], *, min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS, limit: int | None = None) -> Iterable[Path]:
    cutoff_ns = None
    if min_file_age_seconds and min_file_age_seconds > 0:
        cutoff_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    count = 0
    seen: set[Path] = set()
    for root in paths:
        root = Path(root).expanduser()
        candidates = [root] if root.is_file() else sorted(root.rglob("*.md")) if root.exists() else []
        for file_path in candidates:
            try:
                if not file_path.is_file() or file_path.suffix.lower() != ".md":
                    continue
                if any(part in SKIP_MD_DIRS for part in file_path.parts):
                    continue
                resolved = file_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if cutoff_ns is not None and file_path.stat().st_mtime_ns > cutoff_ns:
                    continue
                yield file_path
                count += 1
                if limit is not None and count >= limit:
                    return
            except OSError:
                continue


def records_from_markdown_artifacts(
    paths: Iterable[str | Path] | None = None,
    *,
    bank_target: str = DEFAULT_MIXED_BANK,
    min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    include_observation_scopes: bool = True,
    limit: int | None = None,
    record_granularity: str = "items",
    include_hermes_discovery: bool = False,
    include_openclaw_sessions: bool = True,
    openclaw_db: str | Path = DEFAULT_OPENCLAW_DB,
    openclaw_session_roots: Iterable[str | Path] | None = None,
    include_dingtalk: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discovery_diagnostics: dict[str, Any] | None = None
    if paths is None:
        discovered_paths, discovery_diagnostics = discover_conversation_markdown_artifact_paths(
            openclaw_db=openclaw_db,
            openclaw_session_roots=openclaw_session_roots,
            include_hermes=include_hermes_discovery,
            include_openclaw=True,
            include_openclaw_sessions=include_openclaw_sessions,
            include_dingtalk=include_dingtalk,
            min_file_age_seconds=min_file_age_seconds,
        )
        roots = list(discovered_paths)
    else:
        roots = [Path(p) for p in paths]
    diagnostics: dict[str, Any] = {
        "source_kind": "markdown_artifact_md",
        "paths": [str(p) for p in roots],
        "discovery_mode": "conversation_produced" if discovery_diagnostics is not None else "explicit_paths",
        "files_seen": 0,
        "read_errors": [],
    }
    if discovery_diagnostics is not None:
        diagnostics["discovery"] = discovery_diagnostics
    records: list[dict[str, Any]] = []
    for file_path in iter_markdown_files(roots, min_file_age_seconds=min_file_age_seconds, limit=limit):
        diagnostics["files_seen"] += 1
        try:
            records.extend(records_from_markdown_file(
                file_path,
                bank_target=bank_target,
                retain_chunk_size=retain_chunk_size,
                include_observation_scopes=include_observation_scopes,
                record_granularity=record_granularity,
            ))
        except Exception as exc:
            diagnostics["read_errors"].append({"path": str(file_path), "error": f"{type(exc).__name__}: {exc}"})
    diagnostics["records"] = len(records)
    return records, diagnostics


# ---------------------------------------------------------------------------
# chat-memo txt adapter


def parse_chat_memo_txt(path: Path) -> tuple[dict[str, str], list[Message], bytes]:
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    headers: dict[str, str] = {}
    messages: list[Message] = []
    current: Message | None = None
    body: list[str] = []

    def flush_current() -> None:
        nonlocal current, body
        if current is not None:
            content = clean_external_text("\n".join(body))
            if content:
                current.content = content
                messages.append(current)
        current = None
        body = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        hm = HEADER_RE.match(line)
        if hm and current is None and not messages:
            headers[hm.group(1).lower()] = hm.group(2).strip()
            continue
        mm = MESSAGE_MARKER_RE.match(line)
        if mm:
            flush_current()
            role_raw = mm.group(1).lower()
            role = "assistant" if role_raw in {"ai", "assistant"} else "user"
            current = Message(role=role, timestamp=mm.group(2).strip(), content="")
            tail = mm.group(3).strip()
            body = [tail] if tail else []
        else:
            body.append(line)
    flush_current()
    return headers, messages, raw_bytes


def records_from_chat_memo_file(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_CHATMEMO_BANK,
    max_document_chars: int = session_manifest.DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    include_observation_scopes: bool = True,
) -> list[dict[str, Any]]:
    path = Path(path)
    headers, messages, raw_bytes = parse_chat_memo_txt(path)
    platform = headers.get("platform") or "unknown"
    pslug = platform_slug(platform)
    url = headers.get("url")
    conv_id = conversation_id_from_url_or_path(url, path, raw_bytes)
    created = headers.get("created") or (messages[0].timestamp if messages else None)
    title = headers.get("title") or path.stem
    clean_messages = [m for m in messages if clean_external_text(m.content)]
    full_content = render_conversation_content(
        title=title,
        platform=platform,
        url=url,
        created_at=created,
        source=str(path),
        messages=clean_messages,
    )
    chunks = split_text(full_content, max_document_chars)
    base_doc_id = f"external-chatmemo::{pslug}::{conv_id}"
    source_meta = source_file_metadata(path, raw_bytes)
    records: list[dict[str, Any]] = []
    semantic_tags = semantic_tags_for_text(full_content, title, platform)
    source_tags = ["source:external-chatmemo", f"platform:{pslug}"]
    tags = sorted(set(semantic_tags + source_tags))
    action, reason = external_action_for_content(full_content, semantic_tags)
    if len([m for m in clean_messages if m.role == "user"]) < 1 or len([m for m in clean_messages if m.role == "assistant"]) < 1:
        action, reason = "skip", "missing_user_or_assistant"
    for idx, chunk in enumerate(chunks or [""]):
        part_index = idx + 1
        part_count = len(chunks) if chunks else 1
        document_id = base_doc_id if part_count == 1 else f"{base_doc_id}::part-{part_index:03d}"
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "adapter_version": CHATMEMO_ADAPTER_VERSION,
            "cleaning_version": CLEANING_VERSION,
            "tag_rule_version": TAG_RULE_VERSION,
            "source_kind": "chat_memo_txt",
            "source_label": "chat-memo",
            "source_path": str(path),
            "url": url,
            "platform": platform,
            "external_conversation_id": conv_id,
            "title": title,
            "created_at": created,
            "message_count": len(clean_messages),
            "content_sha256": sha256_text(chunk),
            "full_content_sha256": sha256_text(full_content),
            "part_index": part_index,
            "part_count": part_count,
            "bank_target": bank_target,
            **source_meta,
        }
        records.append({
            "document_id": document_id,
            "bank_target": bank_target,
            "action": action,
            "reason": reason,
            "content": chunk,
            "content_chars": len(chunk),
            "estimated_retain_chunks": max(1, math.ceil(len(chunk) / max(1, retain_chunk_size))) if chunk else 0,
            "event_date": created,
            "tags": tags,
            "observation_scopes": observation_scopes_for_tags(tags, include_observation_scopes),
            "metadata": metadata,
            "context": "external_conversation",
            "update_mode": "replace",
        })
    return records


def iter_chat_memo_files(path: Path, *, min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS, limit: int | None = None) -> Iterable[Path]:
    cutoff_ns = None
    if min_file_age_seconds and min_file_age_seconds > 0:
        cutoff_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    count = 0
    for file_path in sorted(path.glob("*.txt")):
        if cutoff_ns is not None and file_path.stat().st_mtime_ns > cutoff_ns:
            continue
        yield file_path
        count += 1
        if limit is not None and count >= limit:
            return


def dedupe_chat_memo_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    doc_counts = Counter(str(r.get("document_id") or "") for r in records if r.get("document_id"))
    duplicate_document_ids = {k: v for k, v in sorted(doc_counts.items()) if v > 1}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for rec in records:
        meta = rec.get("metadata") or {}
        if meta.get("source_kind") == "chat_memo_txt" and meta.get("external_conversation_id"):
            key = (
                str(meta.get("source_kind") or ""),
                platform_slug(str(meta.get("platform") or "")),
                str(meta.get("external_conversation_id") or ""),
            )
            groups[key].append(rec)
        else:
            passthrough.append(rec)
    deduped: list[dict[str, Any]] = list(passthrough)
    for group_records in groups.values():
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in group_records:
            by_source[str((rec.get("metadata") or {}).get("source_path") or "")].append(rec)
        # Prefer the newest source export when the same conversation appears in
        # multiple exported files. Keep all parts from that source.
        def source_sort_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int, str]:
            source_path, recs = item
            meta = recs[0].get("metadata") or {}
            return (
                int(meta.get("source_mtime_ns") or 0),
                int(meta.get("source_size_bytes") or 0),
                source_path,
            )
        selected_source, selected_records = max(by_source.items(), key=source_sort_key)
        deduped.extend(selected_records)
    deduped.sort(key=lambda r: str((r.get("metadata") or {}).get("source_path") or r.get("document_id") or ""))
    return deduped, duplicate_document_ids


def records_from_chat_memo_dir(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_CHATMEMO_BANK,
    min_file_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    max_document_chars: int = session_manifest.DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    include_observation_scopes: bool = True,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(path)
    diagnostics: dict[str, Any] = {
        "source_kind": "chat_memo_txt",
        "path": str(root),
        "files_seen": 0,
        "read_errors": [],
    }
    records: list[dict[str, Any]] = []
    for file_path in iter_chat_memo_files(root, min_file_age_seconds=min_file_age_seconds, limit=limit):
        diagnostics["files_seen"] += 1
        try:
            records.extend(records_from_chat_memo_file(
                file_path,
                bank_target=bank_target,
                max_document_chars=max_document_chars,
                retain_chunk_size=retain_chunk_size,
                include_observation_scopes=include_observation_scopes,
            ))
        except Exception as e:
            diagnostics["read_errors"].append({"path": str(file_path), "error": f"{type(e).__name__}: {e}"})
    records, duplicate_document_ids = dedupe_chat_memo_records(records)
    diagnostics["duplicate_document_ids"] = duplicate_document_ids
    diagnostics["records"] = len(records)
    return records, diagnostics


# ---------------------------------------------------------------------------
# OpenClaw lcm.db adapter


def openclaw_session_key_reason(session_key: str | None, *, include_dingtalk: bool = True) -> str | None:
    key = str(session_key or "").strip()
    if not key:
        return "session_key_excluded"
    if ":cron:" in key or key.startswith("agent:cron:"):
        return "session_key_excluded"
    if ":subagent:" in key:
        return "session_key_excluded"
    if ":acp:" in key:
        return "session_key_excluded"
    if key == "agent:main:main" or key.startswith("agent:main:tui-"):
        return None
    if include_dingtalk and key.startswith("agent:main:dingtalk:direct:"):
        return None
    return "session_key_excluded"


def is_history_aggregate_title(title: str | None) -> bool:
    return str(title or "").strip().startswith("历史:")


def extract_openclaw_structured_text(value: Any) -> str:
    """Keep human-visible text from OpenClaw structured assistant payloads.

    OpenClaw can persist assistant messages as JSON blocks containing thinking,
    toolCall, and toolResult items. Those are tool traces, not conversation
    memory. Keep only explicit text/content fields from non-tool items.
    """
    skip_types = {"thinking", "toolcall", "tool_call", "toolresult", "tool_result", "function_call", "functioncall"}

    def walk(obj: Any) -> list[str]:
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, list):
            out: list[str] = []
            for item in obj:
                out.extend(walk(item))
            return out
        if isinstance(obj, dict):
            typ = str(obj.get("type") or "").replace("-", "_").lower()
            if typ in skip_types:
                return []
            out: list[str] = []
            for key in ["text", "message"]:
                if isinstance(obj.get(key), str):
                    out.append(obj[key])
            content = obj.get("content")
            if isinstance(content, str):
                out.append(content)
            elif isinstance(content, list):
                out.extend(walk(content))
            return out
        return []

    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped.startswith(("[", "{")):
        return value
    try:
        parsed = json.loads(stripped)
    except Exception:
        return value
    parts = [clean_external_text(x) for x in walk(parsed)]
    return clean_external_text("\n".join(p for p in parts if p))


def clean_openclaw_message_content(text: Any) -> tuple[str, str | None]:
    cleaned = clean_external_text(text)
    cleaned = OPENCLAW_UNTRUSTED_BLOCK_RE.sub("", cleaned)
    cleaned = extract_openclaw_structured_text(cleaned)
    cleaned = LEADING_UI_TIMESTAMP_RE.sub("", cleaned).strip()
    if not cleaned:
        return "", "empty"
    stripped = cleaned.strip()
    if stripped in OPENCLAW_DROP_EXACT:
        return "", "openclaw_system_noise"
    for prefix in OPENCLAW_DROP_PREFIXES:
        if stripped.startswith(prefix):
            return "", "openclaw_system_noise"
    if OPENCLAW_STATUS_RE.search(stripped):
        return "", "operational_status"
    if session_manifest.is_low_signal_message_body(stripped):
        return "", "low_signal"
    return stripped, None


def fetch_openclaw_conversations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT conversation_id, session_id, session_key, title, created_at, updated_at, active
        FROM conversations
        ORDER BY conversation_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_openclaw_messages(conn: sqlite3.Connection, conversation_id: int) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT message_id, conversation_id, seq, role, content, created_at
        FROM messages
        WHERE conversation_id = ?
          AND role IN ('user', 'assistant')
        ORDER BY seq, message_id
        """,
        (conversation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def split_openclaw_messages(
    messages: list[Message],
    *,
    max_segment_turns: int = DEFAULT_MAX_SEGMENT_TURNS,
    max_segment_chars: int = DEFAULT_MAX_SEGMENT_CHARS,
    gap_split_hours: float = DEFAULT_GAP_SPLIT_HOURS,
) -> list[list[Message]]:
    segments: list[list[Message]] = []
    cur: list[Message] = []
    cur_chars = 0
    cur_user_turns = 0
    prev_ts: str | None = None
    for msg in messages:
        msg_chars = len(msg.content) + 32
        gap = hours_between(prev_ts, msg.timestamp) if prev_ts else None
        would_exceed_chars = bool(cur and cur_chars + msg_chars > max_segment_chars)
        would_exceed_turns = bool(cur and msg.role == "user" and cur_user_turns >= max_segment_turns)
        would_gap_split = bool(cur and gap is not None and gap > gap_split_hours)
        if would_exceed_chars or would_exceed_turns or would_gap_split:
            segments.append(cur)
            cur = []
            cur_chars = 0
            cur_user_turns = 0
        cur.append(msg)
        cur_chars += msg_chars
        if msg.role == "user":
            cur_user_turns += 1
        prev_ts = msg.timestamp or prev_ts
    if cur:
        segments.append(cur)
    return segments


def valid_segment(messages: list[Message]) -> tuple[bool, str | None]:
    if not messages:
        return False, "empty_after_filtering"
    if not any(m.role == "user" for m in messages) or not any(m.role == "assistant" for m in messages):
        return False, "missing_user_or_assistant"
    body_chars = sum(len(m.content) for m in messages)
    if body_chars < MIN_CONTENT_CHARS:
        return False, "empty_or_too_short"
    rendered = "\n\n".join(f"{m.role}: {m.content}" for m in messages)
    if session_manifest.is_low_signal_conversation(rendered):
        return False, "low_signal_short_or_chitchat"
    return True, None


def first_timestamp_for_role(messages: list[Message], role: str) -> str | None:
    for msg in messages:
        if msg.role == role and msg.timestamp:
            return msg.timestamp
    return None


def make_openclaw_record(
    *,
    db_path: Path,
    conversation: dict[str, Any],
    segment: list[Message],
    segment_index: int,
    segment_count: int,
    bank_target: str,
    retain_chunk_size: int,
    include_observation_scopes: bool,
) -> dict[str, Any]:
    conversation_id = str(conversation.get("conversation_id"))
    title = conversation.get("title") or f"OpenClaw conversation {conversation_id}"
    content = render_conversation_content(
        title=title,
        platform="OpenClaw",
        url=None,
        created_at=first_timestamp_for_role(segment, "user") or (segment[0].timestamp if segment else conversation.get("created_at")),
        source=str(db_path),
        messages=segment,
    )
    semantic_tags = semantic_tags_for_text(content, str(title), "openclaw")
    source_tags = ["source:external-openclaw", "platform:openclaw"]
    session_key = str(conversation.get("session_key") or "")
    if ":dingtalk:direct:" in session_key:
        source_tags.append("channel:dingtalk")
    tags = sorted(set(semantic_tags + source_tags))
    action, reason = external_action_for_content(content, semantic_tags)
    start_msg = segment[0]
    end_msg = segment[-1]
    doc_id = f"external-openclaw::{conversation_id}::seg-{segment_index:03d}"
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "adapter_version": OPENCLAW_ADAPTER_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "tag_rule_version": TAG_RULE_VERSION,
        "source_kind": "openclaw_lcm",
        "source_label": "openclaw-lcm",
        "source_path": str(db_path),
        "conversation_id": conversation_id,
        "session_id": conversation.get("session_id"),
        "session_key": session_key,
        "title": conversation.get("title"),
        "conversation_created_at": conversation.get("created_at"),
        "conversation_updated_at": conversation.get("updated_at"),
        "conversation_active": conversation.get("active"),
        "segment_index": segment_index,
        "segment_count": segment_count,
        "segment_started_at": start_msg.timestamp,
        "segment_ended_at": end_msg.timestamp,
        "message_id_start": str(start_msg.message_id) if start_msg.message_id is not None else None,
        "message_id_end": str(end_msg.message_id) if end_msg.message_id is not None else None,
        "seq_start": str(start_msg.seq) if start_msg.seq is not None else None,
        "seq_end": str(end_msg.seq) if end_msg.seq is not None else None,
        "message_count": len(segment),
        "content_sha256": sha256_text(content),
        "full_content_sha256": sha256_text(content),
        "bank_target": bank_target,
    }
    return {
        "document_id": doc_id,
        "bank_target": bank_target,
        "action": action,
        "reason": reason,
        "content": content,
        "content_chars": len(content),
        "estimated_retain_chunks": max(1, math.ceil(len(content) / max(1, retain_chunk_size))) if content else 0,
        "event_date": first_timestamp_for_role(segment, "user") or start_msg.timestamp,
        "tags": tags,
        "observation_scopes": observation_scopes_for_tags(tags, include_observation_scopes),
        "metadata": metadata,
        "context": "external_conversation",
        "update_mode": "replace",
    }


def records_from_openclaw_lcm(
    db_path: str | Path = DEFAULT_OPENCLAW_DB,
    *,
    bank_target: str = DEFAULT_OPENCLAW_BANK,
    include_dingtalk: bool = True,
    include_history_aggregate: bool = False,
    min_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
    max_segment_turns: int = DEFAULT_MAX_SEGMENT_TURNS,
    max_segment_chars: int = DEFAULT_MAX_SEGMENT_CHARS,
    gap_split_hours: float = DEFAULT_GAP_SPLIT_HOURS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    include_observation_scopes: bool = True,
    limit_conversations: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    db_path = Path(db_path).expanduser()
    diagnostics: dict[str, Any] = {
        "source_kind": "openclaw_lcm",
        "path": str(db_path),
        "conversations_seen": 0,
        "included_conversations": 0,
        "excluded_conversations_by_reason": Counter(),
        "dropped_messages_by_reason": Counter(),
        "skipped_segments_by_reason": Counter(),
    }
    records: list[dict[str, Any]] = []
    conn = open_sqlite_readonly(db_path)
    try:
        conversations = fetch_openclaw_conversations(conn)
        used = 0
        for conv in conversations:
            diagnostics["conversations_seen"] += 1
            reason = openclaw_session_key_reason(conv.get("session_key"), include_dingtalk=include_dingtalk)
            if reason:
                diagnostics["excluded_conversations_by_reason"][reason] += 1
                continue
            if is_history_aggregate_title(conv.get("title")) and not include_history_aggregate:
                diagnostics["excluded_conversations_by_reason"]["history_aggregate"] += 1
                continue
            if is_recent_timestamp(conv.get("updated_at") or conv.get("created_at"), min_age_seconds):
                diagnostics["excluded_conversations_by_reason"]["too_recent"] += 1
                continue
            raw_messages = fetch_openclaw_messages(conn, int(conv["conversation_id"]))
            clean_messages: list[Message] = []
            for row in raw_messages:
                content, drop_reason = clean_openclaw_message_content(row.get("content"))
                if drop_reason:
                    diagnostics["dropped_messages_by_reason"][drop_reason] += 1
                    continue
                clean_messages.append(Message(
                    role=str(row.get("role")),
                    timestamp=row.get("created_at"),
                    content=content,
                    message_id=row.get("message_id"),
                    seq=row.get("seq"),
                ))
            if not clean_messages:
                diagnostics["excluded_conversations_by_reason"]["empty_after_filtering"] += 1
                continue
            segments = split_openclaw_messages(
                clean_messages,
                max_segment_turns=max_segment_turns,
                max_segment_chars=max_segment_chars,
                gap_split_hours=gap_split_hours,
            )
            valid_segments: list[list[Message]] = []
            for seg in segments:
                ok, seg_reason = valid_segment(seg)
                if ok:
                    valid_segments.append(seg)
                else:
                    diagnostics["skipped_segments_by_reason"][seg_reason or "invalid"] += 1
            if not valid_segments:
                diagnostics["excluded_conversations_by_reason"]["no_valid_segments"] += 1
                continue
            diagnostics["included_conversations"] += 1
            for idx, seg in enumerate(valid_segments, start=1):
                records.append(make_openclaw_record(
                    db_path=db_path,
                    conversation=conv,
                    segment=seg,
                    segment_index=idx,
                    segment_count=len(valid_segments),
                    bank_target=bank_target,
                    retain_chunk_size=retain_chunk_size,
                    include_observation_scopes=include_observation_scopes,
                ))
            used += 1
            if limit_conversations is not None and used >= limit_conversations:
                break
    finally:
        conn.close()
    # Convert Counters for JSON serialization and deterministic tests.
    diagnostics["excluded_conversations_by_reason"] = dict(diagnostics["excluded_conversations_by_reason"])
    diagnostics["dropped_messages_by_reason"] = dict(diagnostics["dropped_messages_by_reason"])
    diagnostics["skipped_segments_by_reason"] = dict(diagnostics["skipped_segments_by_reason"])
    diagnostics["records"] = len(records)
    return records, diagnostics


# ---------------------------------------------------------------------------
# Sampling helpers


def _record_semantic_tags(record: dict[str, Any]) -> list[str]:
    return sorted(str(t) for t in (record.get("tags") or []) if str(t).startswith(SEMANTIC_TAG_PREFIXES))


def _record_title_fingerprint(record: dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    raw = str(meta.get("title") or record.get("document_id") or "")
    raw = re.sub(r"\s+", " ", raw).strip().lower()
    # Keep CJK/ASCII alnum and separators deterministic; this groups repeated
    # exports with near-identical titles while not relying on source filenames.
    raw = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "-", raw).strip("-")
    return raw[:48] or "untitled"


def semantic_bucket_key(record: dict[str, Any]) -> tuple[str, str]:
    semantic = _record_semantic_tags(record)
    project = [t for t in semantic if t.startswith("project:")]
    domain = [t for t in semantic if t.startswith("domain:")]
    topic = [t for t in semantic if t.startswith("topic:")]
    primary = (project or domain or topic or [f"{record.get('action') or 'unknown'}:{record.get('reason') or 'unknown'}"])[0]
    return primary, _record_title_fingerprint(record)


def _sample_score(record: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(record.get("estimated_retain_chunks") or 0),
        int(record.get("content_chars") or 0),
        str(record.get("document_id") or ""),
    )


def diverse_sample_records(records: list[dict[str, Any]], *, limit: int | None, action: str = "production") -> list[dict[str, Any]]:
    """Return a deterministic, small, topic-diverse sample for smoke tests.

    The selector first collapses repeated exports with the same primary semantic
    tag and title fingerprint, then round-robins across primary semantic tags.
    Within each bucket it prefers smaller documents so a full smoke flow remains
    cheap while still covering multiple topics.
    """
    if limit is None or limit <= 0:
        return list(records)
    candidates = [r for r in records if not action or r.get("action") == action]
    bucket_best: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in candidates:
        key = semantic_bucket_key(rec)
        prev = bucket_best.get(key)
        if prev is None or _sample_score(rec) < _sample_score(prev):
            bucket_best[key] = rec
    by_primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, rec in sorted(bucket_best.items()):
        primary, _title_fp = key
        by_primary[primary].append(rec)
    for group in by_primary.values():
        group.sort(key=_sample_score)
    selected: list[dict[str, Any]] = []
    primaries = sorted(by_primary)
    while len(selected) < limit and any(by_primary.values()):
        for primary in primaries:
            group = by_primary.get(primary) or []
            if not group:
                continue
            selected.append(group.pop(0))
            if len(selected) >= limit:
                break
    return selected


# ---------------------------------------------------------------------------
# Manifest writer / CLI


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: Counter = Counter()
    by_reason: Counter = Counter()
    by_source: Counter = Counter()
    tag_counts: Counter = Counter()
    total_chars = 0
    total_chunks = 0
    records_with_observation_scopes = 0
    observation_scope_count = 0
    for rec in records:
        action = str(rec.get("action") or "unknown")
        reason = str(rec.get("reason") or "unknown")
        meta = rec.get("metadata") or {}
        by_action[action] += 1
        by_reason[f"{action}:{reason}"] += 1
        by_source[str(meta.get("source_kind") or "unknown")] += 1
        total_chars += int(rec.get("content_chars") or 0)
        total_chunks += int(rec.get("estimated_retain_chunks") or 0)
        scopes = rec.get("observation_scopes") or []
        if scopes:
            records_with_observation_scopes += 1
            observation_scope_count += len(scopes)
        for tag in rec.get("tags") or []:
            tag_counts[str(tag)] += 1
    return {
        "generated_at": iso_now(),
        "records": len(records),
        "by_action": dict(sorted(by_action.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_source": dict(sorted(by_source.items())),
        "total_content_chars": total_chars,
        "estimated_retain_chunks": total_chunks,
        "records_with_observation_scopes": records_with_observation_scopes,
        "observation_scope_count": observation_scope_count,
        "top_tags": sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30],
        "tag_rule_version": TAG_RULE_VERSION,
        "manual_only": True,
        "daily_pipeline_integrated": False,
    }


def manifest_record_for_write(rec: dict[str, Any], *, include_content: bool) -> dict[str, Any]:
    out = dict(rec)
    if not include_content and "content" in out:
        out.pop("content", None)
        out["content_omitted"] = True
    return out


def _unique_manifest_paths(output_dir: Path, stamp: str) -> tuple[Path, Path]:
    for idx in range(1000):
        suffix = "" if idx == 0 else f"-{idx:02d}"
        manifest_path = output_dir / f"{stamp}{suffix}-external-manifest.jsonl"
        summary_path = output_dir / f"{stamp}{suffix}-external-manifest-summary.json"
        if not manifest_path.exists() and not summary_path.exists():
            return manifest_path, summary_path
    raise RuntimeError(f"could not allocate unique manifest path under {output_dir}")


def write_manifest(records: list[dict[str, Any]], output_dir: Path, *, include_content: bool = False) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path, summary_path = _unique_manifest_paths(output_dir, stamp)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(manifest_record_for_write(rec, include_content=include_content), ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_records(records)
    summary["include_content"] = include_content
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_path = output_dir / "latest.json"
    latest_path.write_text(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path), "summary_data": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "summary": summary_path, "latest": latest_path}


def infer_default_bank(sources: list[str], bank_target: str | None) -> str:
    if bank_target:
        return bank_target
    unique = set(sources)
    if unique == {"chat-memo"}:
        return DEFAULT_CHATMEMO_BANK
    if unique == {"openclaw-lcm"}:
        return DEFAULT_OPENCLAW_BANK
    if unique == {"markdown-artifact"}:
        return DEFAULT_MIXED_BANK
    return DEFAULT_MIXED_BANK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build manual-only external Hindsight manifest; no Hindsight writes.")
    ap.add_argument("--source", action="append", choices=["chat-memo", "openclaw-lcm", "markdown-artifact"], required=True)
    ap.add_argument("--path", type=Path, help="For --source chat-memo: directory containing exported .txt files; for markdown-artifact: file or directory root")
    ap.add_argument("--chat-memo-dir", type=Path, help="Directory containing chat-memo .txt files")
    ap.add_argument("--markdown-path", action="append", type=Path, help="Markdown artifact file or directory. Can be passed multiple times. If omitted, discovers OpenClaw conversation-produced .md files and skips missing files; add --markdown-include-hermes-produced to include Hermes outputs too.")
    ap.add_argument("--markdown-scan-default-roots", action="store_true", help="Legacy/debug mode: scan broad default markdown roots instead of conversation-produced discovery. Not recommended for production imports.")
    ap.add_argument("--markdown-include-hermes-produced", action="store_true", help="Also include Hermes-produced .md files discovered from Hermes sessions. Default is false because Hermes conversations are already handled by the native session pipeline.")
    ap.add_argument("--markdown-include-openclaw-sessions", dest="markdown_include_openclaw_sessions", action="store_true", default=True, help="Also include OpenClaw session JSONL evidence for markdown artifacts (default: enabled).")
    ap.add_argument("--no-markdown-include-openclaw-sessions", dest="markdown_include_openclaw_sessions", action="store_false", help="Disable OpenClaw session JSONL discovery for markdown artifacts.")
    ap.add_argument("--openclaw-session-roots", action="append", type=Path, default=None, help="Override OpenClaw session roots for markdown-artifact discovery.")
    ap.add_argument("--db", type=Path, default=DEFAULT_OPENCLAW_DB, help="For --source openclaw-lcm: path to lcm.db")
    ap.add_argument("--bank-target", default=None)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--include-content", action="store_true", help="Write cleaned content into manifest JSONL. Default omits content and keeps source pointers/hashes.")
    ap.add_argument("--include-observation-scopes", dest="include_observation_scopes", action="store_true", default=True, help="Include semantic observation scopes in manifest (default: enabled).")
    ap.add_argument("--no-observation-scopes", dest="include_observation_scopes", action="store_false", help="Disable observation scopes for raw/manual low-cost imports.")
    ap.add_argument("--min-file-age-seconds", type=int, default=DEFAULT_MIN_AGE_SECONDS)
    ap.add_argument("--max-document-chars", type=int, default=session_manifest.DEFAULT_MAX_DOCUMENT_CHARS)
    ap.add_argument("--retain-chunk-size", type=int, default=DEFAULT_RETAIN_CHUNK_SIZE)
    ap.add_argument("--markdown-record-granularity", choices=["items", "sections", "all", "full"], default="items", help="For markdown-artifact: items=outline+item records (default), sections=outline+section records, all=outline+items+sections, full=outline only.")
    ap.add_argument("--limit", type=int, default=None, help="Limit chat-memo files, OpenClaw conversations, or markdown files for smoke runs")
    ap.add_argument("--sample-records", type=int, default=None, help="After full source filtering, write only a deterministic diverse sample of this many records. Default writes all records.")
    ap.add_argument("--sample-action", default="production", help="Action to sample when --sample-records is set. Default: production.")
    ap.add_argument("--exclude-dingtalk", action="store_true", help="For OpenClaw: exclude agent:main:dingtalk:direct:* conversations")
    ap.add_argument("--include-history-aggregate", action="store_true", help="For OpenClaw: include title='历史:*' aggregate conversations. Default skip.")
    ap.add_argument("--max-segment-turns", type=int, default=DEFAULT_MAX_SEGMENT_TURNS)
    ap.add_argument("--max-segment-chars", type=int, default=DEFAULT_MAX_SEGMENT_CHARS)
    ap.add_argument("--gap-split-hours", type=float, default=DEFAULT_GAP_SPLIT_HOURS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    sources = list(args.source or [])
    bank = infer_default_bank(sources, args.bank_target)
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for src in sources:
        if src == "chat-memo":
            root = args.chat_memo_dir or args.path
            if not root:
                raise SystemExit("--source chat-memo requires --path or --chat-memo-dir")
            recs, diag = records_from_chat_memo_dir(
                root,
                bank_target=bank,
                min_file_age_seconds=args.min_file_age_seconds,
                max_document_chars=args.max_document_chars,
                retain_chunk_size=args.retain_chunk_size,
                include_observation_scopes=args.include_observation_scopes,
                limit=args.limit,
            )
            records.extend(recs)
            diagnostics.append(diag)
        elif src == "openclaw-lcm":
            recs, diag = records_from_openclaw_lcm(
                args.db,
                bank_target=bank,
                include_dingtalk=not args.exclude_dingtalk,
                include_history_aggregate=args.include_history_aggregate,
                min_age_seconds=args.min_file_age_seconds,
                max_segment_turns=args.max_segment_turns,
                max_segment_chars=args.max_segment_chars,
                gap_split_hours=args.gap_split_hours,
                retain_chunk_size=args.retain_chunk_size,
                include_observation_scopes=args.include_observation_scopes,
                limit_conversations=args.limit,
            )
            records.extend(recs)
            diagnostics.append(diag)
        elif src == "markdown-artifact":
            markdown_roots = args.markdown_path or ([args.path] if args.path else None)
            if args.markdown_scan_default_roots and markdown_roots is None:
                markdown_roots = DEFAULT_MARKDOWN_SCAN_PATHS
            recs, diag = records_from_markdown_artifacts(
                markdown_roots,
                bank_target=bank,
                min_file_age_seconds=args.min_file_age_seconds,
                retain_chunk_size=args.retain_chunk_size,
                include_observation_scopes=args.include_observation_scopes,
                limit=args.limit,
                record_granularity=args.markdown_record_granularity,
                include_hermes_discovery=args.markdown_include_hermes_produced,
                include_openclaw_sessions=args.markdown_include_openclaw_sessions,
                openclaw_db=args.db,
                openclaw_session_roots=args.openclaw_session_roots,
                include_dingtalk=not args.exclude_dingtalk,
            )
            records.extend(recs)
            diagnostics.append(diag)

    full_summary = summarize_records(records)
    sample_info: dict[str, Any] | None = None
    if args.sample_records is not None:
        original_count = len(records)
        original_action_counts = full_summary.get("by_action", {})
        records = diverse_sample_records(records, limit=args.sample_records, action=args.sample_action)
        sample_info = {
            "enabled": True,
            "sample_records": args.sample_records,
            "sample_action": args.sample_action,
            "source_records_before_sampling": original_count,
            "source_by_action_before_sampling": original_action_counts,
            "selected_records": len(records),
            "selected_buckets": ["::".join(semantic_bucket_key(r)) for r in records],
        }

    paths = write_manifest(records, args.output_dir, include_content=args.include_content)
    summary = summarize_records(records)
    if sample_info is not None:
        summary["sampling"] = sample_info
    summary["bank_target"] = bank
    summary["paths"] = {k: str(v) for k, v in paths.items()}
    summary["diagnostics"] = diagnostics
    summary["filters"] = {
        "sources": sources,
        "markdown_paths": [str(p) for p in (args.markdown_path or ([args.path] if args.path else []))],
        "min_file_age_seconds": args.min_file_age_seconds,
        "include_dingtalk": not args.exclude_dingtalk,
        "include_history_aggregate": args.include_history_aggregate,
        "max_segment_turns": args.max_segment_turns,
        "max_segment_chars": args.max_segment_chars,
        "gap_split_hours": args.gap_split_hours,
        "include_observation_scopes": args.include_observation_scopes,
        "tag_rule_version": TAG_RULE_VERSION,
        "sample_records": args.sample_records,
        "sample_action": args.sample_action,
        "markdown_record_granularity": args.markdown_record_granularity,
        "markdown_scan_default_roots": args.markdown_scan_default_roots,
        "markdown_include_hermes_produced": args.markdown_include_hermes_produced,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"records={summary['records']} actions={summary['by_action']} sources={summary['by_source']} chars={summary['total_content_chars']} chunks={summary['estimated_retain_chunks']}")
        print(f"manifest={paths['manifest']}")
        print(f"summary={paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
