#!/usr/bin/env python3
"""Dry-run Hermes session -> Hindsight manifest builder.

This script is deliberately non-mutating: it reads JSON session files / optional
SQLite metadata, cleans deterministic noise, proposes document_id/tags/scopes,
and writes a manifest for review before any Hindsight retain call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_HERMES_HOME = Path.home() / ".hermes"
DEFAULT_SESSIONS_DIR = DEFAULT_HERMES_HOME / "sessions"
DEFAULT_STATE_DB = DEFAULT_HERMES_HOME / "state.db"
DEFAULT_PROFILE_ROOT = DEFAULT_HERMES_HOME / "profiles"
DEFAULT_CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
DEFAULT_DEEPSEEK_SESSIONS_DIR = Path.home() / ".deepseek" / "sessions"
DEFAULT_KANBAN_WORKSPACE_ROOTS = [Path.home() / "code"]
DEFAULT_OUTPUT_DIR = DEFAULT_HERMES_HOME / "hindsight" / "session_ingest" / "manifests"
DEFAULT_SCAN_STATE_PATH = DEFAULT_HERMES_HOME / "hindsight" / "session_ingest" / "manifest_scan_state.json"
DEFAULT_BANK_TARGET = "hermes_v3"
DEFAULT_MAX_DOCUMENT_CHARS = 120_000
DEFAULT_RETAIN_CHUNK_SIZE = 8000
DEFAULT_MIN_FILE_AGE_SECONDS = int(os.environ.get("HINDSIGHT_SESSION_MANIFEST_MIN_FILE_AGE_SECONDS", "900"))
MIN_CONTENT_CHARS = 30
SCHEMA_VERSION = "session-retain-v3"
CLEANING_VERSION = "deterministic-clean-v2"
CANDIDATE_FILTER_VERSION = "lightweight-candidate-filter-v3"
SCAN_STATE_VERSION = "session-manifest-scan-state-v1"

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
MEMORY_CONTEXT_RE = re.compile(r"<memory-context>.*?</memory-context>", re.IGNORECASE | re.DOTALL)
MEDIA_RE = re.compile(r"MEDIA:[^\s]+")
MODEL_SWITCH_NOTE_RE = re.compile(
    r"\[Note:\s*model was just switched from .*? Adjust your self-identification accordingly\.\]",
    re.IGNORECASE | re.DOTALL,
)
THINK_BLOCK_RE = re.compile(r"<\s*(?:think|thinking)\b[^>]*>.*?<\s*/\s*(?:think|thinking)\s*>", re.IGNORECASE | re.DOTALL)
MULTI_BLANK_RE = re.compile(r"\n{3,}")

NOISE_PATTERNS = [
    re.compile(r"^\s*\[CONTEXT COMPACTION\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*\[Your active task list was preserved across context compression\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*<tool_result\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*\{\s*\"tool_calls_made\"", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*Review the conversation above and consider saving or updating a skill if appropriate\.", re.IGNORECASE | re.DOTALL),
]
TAG_RULES: list[tuple[str, list[str], list[str]]] = [
    ("project:egomotion4d", ["egomotion4d", "trackingworld", "vggt4d", "dggt", "dgegt", "dage", "ate_metric", "ate metric", "joint ba", "roma2"], ["domain:autodrive"]),
    ("project:vggt-long", ["vggt-long", "vggt long", "loop closure", "salad", "dinov2"], ["domain:autodrive"]),
    ("project:openclaw", ["openclaw", "clawhub", "approval", "gateway probe"], []),
    ("domain:hindsight", ["hindsight", "memory provider", "memory bank", "observations", "consolidation", "recall", "reflect", "retain", "memory_units", "记忆", "召回"], []),
    ("topic:memory-management", ["hindsight", "memory", "记忆", "recall", "retain", "consolidation", "observations", "reflect", "bank", "discard", "quarantine"], []),
    ("topic:native-consolidation", ["native consolidation", "observations", "consolidation", "enable_observations"], ["domain:hindsight"]),
    ("topic:recall-cache", ["recall cache", "auto_recall", "conditional recall", "recall_cache"], ["domain:hindsight"]),
    ("domain:patent", ["patent", "oa1", "office action", "专利", "审查意见", "权利要求"], []),
    ("domain:paper", ["paper", "论文", "arxiv", "citation", "abstract"], []),
    ("domain:autodrive", ["aeb", "adas", "自动驾驶", "智驾", "车道线", "单目测速"], []),
]

SYSTEM_LABELS = {"hermes", "sqlite", "incremental", "daily", "canonical", "source:tmp"}

BOOTSTRAP_DIAGNOSTIC_RE = re.compile(
    r"(who are u|who are you|你是谁|你是.*助手|我是臭臭|检查下你的环境|检查环境|hermes claw migrate|claw migrate|上下文上限|context.*上限|重启了.*上下文|lossless-claw|lcm\.db|model_metadata|glm-5.*上下文)",
    re.IGNORECASE,
)

BROAD_AGGREGATE_RE = re.compile(
    r"(阅读所有(专题|详细笔记|项目记忆)文件|所有专题文件|所有详细笔记文件|所有项目记忆文件|~/?\.hermes/memories/(details|projects)?|提取核心技术知识|提取核心知识点|项目记忆文件结构化总结|专题知识总结)",
    re.IGNORECASE,
)

MEMORY_RECALL_BOOTSTRAP_RE = re.compile(
    r"(?:^|\n\n)\s*User:\s*(回忆|回顾|帮我回忆|查.*项目记忆|查.*知识图谱|recall\b|remember\b|go on\b)",
    re.IGNORECASE | re.DOTALL,
)
MEMORY_RECALL_RESPONSE_RE = re.compile(
    r"(项目记忆|知识图谱|帮你回忆|查询.*记忆|先查.*卡片|先把.*记忆|memory|recall)",
    re.IGNORECASE,
)
CONTEXT_RESUME_BOOTSTRAP_RE = re.compile(
    r"(?:^|\n\n)\s*User:\s*(刚才聊了什么|刚聊到哪[儿了]*|刚聊到哪了|上次聊到哪|重启会话后怎么继续干活|怎么继续干活)",
    re.IGNORECASE | re.DOTALL,
)
CONTEXT_RESUME_RESPONSE_RE = re.compile(
    r"(当前窗口|长期记忆|之前|上下文|刚才|会话|context|memory)",
    re.IGNORECASE,
)

SECRET_MATERIAL_RE = re.compile(
    r"(?i)(\bsk-[A-Za-z0-9_.-]{12,}\b|\bsk-[A-Za-z0-9_.-]{2,}\.\.\.[A-Za-z0-9_.-]{3,}\b|(?:api[_ -]?key|secret|token|password)\s*[:=]\s*[^\s,;|]+)",
)

CODEX_APPLY_PATCH_MD_RE = re.compile(r"^\*\*\* (?:Add|Update) File:\s*(?P<path>.+?\.md)\s*$", re.IGNORECASE | re.MULTILINE)
CODEX_BACKTICK_MD_RE = re.compile(r"`(?P<path>[^`]+?\.md)`")
CODEX_ABS_MD_RE = re.compile(r"(?P<path>(?:~|/[^\x00\r\n`\"'<>]+?\.md))")
CODEX_REL_MD_RE = re.compile(r"(?P<path>(?:\.{1,2}/)?[A-Za-z0-9_.@+~ -][^\x00\r\n`\"'<>]*?\.md)")
CODEX_MD_POSITIVE_HINT_RE = re.compile(
    r"(Successfully\s+wrote|Success\.\s+Updated\s+the\s+following\s+files|"
    r"\b(wrote|written|created|saved|generated|updated)\b|已写入|写入|保存到|生成(?:了)?|创建(?:了)?|更新(?:了)?)",
    re.IGNORECASE,
)
CODEX_MD_NEGATIVE_HINT_RE = re.compile(r"(读取|参考|read(?:ing)?|existing|已有|不是新写入|未写入|没有写入)", re.IGNORECASE)
TRAILING_PATH_PUNCT = " \t\r\n,，.。;；:：)）]】}>'\""

LOW_SIGNAL_EXACT_PHRASES = {
    "hi",
    "hello",
    "hey",
    "yo",
    "ok",
    "okay",
    "k",
    "yes",
    "no",
    "thanks",
    "thankyou",
    "thx",
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "好",
    "好的",
    "好嘞",
    "行",
    "可以",
    "收到",
    "明白",
    "嗯",
    "嗯嗯",
    "对",
    "是",
    "继续",
    "继续吧",
    "接着",
    "接着说",
    "继续说",
    "谢谢",
    "辛苦",
    "你是谁",
    "whoareyou",
    "whoareu",
    "whoaruu",
    "whoru",
}
LOW_SIGNAL_IDENTITY_ASSISTANT_RE = re.compile(
    r"^(我是|我是臭臭|这里是|我是.*ai助手|我是.*助手|臭臭[，,]?).{0,40}(助手|ai|自动驾驶|领域)",
    re.IGNORECASE,
)
LOW_SIGNAL_PUNCT_RE = re.compile(r"[\s\.,!?:;\-_/\\'\"`~@#$%^&*+=|()[\]{}<>，。！？：；、（）【】《》“”‘’…·]+")
LOW_SIGNAL_MAX_BODY_CHARS = 120


def split_clean_message_bodies(text: str) -> list[str]:
    bodies: list[str] = []
    for block in re.split(r"\n{2,}", text or ""):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"^(User|Assistant):\s*", "", block, flags=re.IGNORECASE)
        if block:
            bodies.append(block)
    return bodies


def normalize_low_signal_phrase(text: str) -> str:
    return LOW_SIGNAL_PUNCT_RE.sub("", (text or "").strip().lower())


def is_low_signal_message_body(text: str) -> bool:
    body = clean_text(text)
    if not body:
        return True
    norm = normalize_low_signal_phrase(body)
    if not norm:
        return True
    if norm in LOW_SIGNAL_EXACT_PHRASES:
        return True
    # Common assistant boilerplate after a greeting / ack / identity check. Keep
    # this narrow so short technical prompts are not dropped just because they
    # contain “继续”.
    if len(body) <= 80 and LOW_SIGNAL_IDENTITY_ASSISTANT_RE.search(body):
        return True
    if len(body) <= 40 and re.search(r"^(在|我在).{0,12}(有什么|需要).{0,18}(处理|帮|做)", body):
        return True
    if len(body) <= 30 and re.search(r"^(好|好的|收到|明白).{0,8}(继续|处理|接着)", body):
        return True
    return False


def is_low_signal_conversation(text: str) -> bool:
    bodies = split_clean_message_bodies(text)
    if not bodies:
        return True
    body_chars = sum(len(clean_text(b)) for b in bodies)
    if body_chars > LOW_SIGNAL_MAX_BODY_CHARS:
        return False
    return all(is_low_signal_message_body(b) for b in bodies)


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "")
    text = CONTROL_RE.sub("", text)
    text = MEMORY_CONTEXT_RE.sub("", text)
    text = MODEL_SWITCH_NOTE_RE.sub("", text)
    text = THINK_BLOCK_RE.sub("", text)
    text = MEDIA_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def extract_message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return clean_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                part = clean_text(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if item_type in {"thinking", "reasoning", "reasoning_content", "thought", "chain_of_thought"}:
                    part = ""
                else:
                    part = clean_text(item.get("text") or item.get("content") or "")
            else:
                part = ""
            if part:
                parts.append(part)
        return clean_text("\n".join(parts))
    if isinstance(content, dict):
        item_type = str(content.get("type") or "").lower()
        if item_type in {"thinking", "reasoning", "reasoning_content", "thought", "chain_of_thought"}:
            return ""
        return clean_text(content.get("text") or content.get("content") or "")
    return ""


def is_noise_message(text: str) -> bool:
    if not text:
        return True
    return any(p.search(text) for p in NOISE_PATTERNS)


def extract_clean_conversation(session_data: dict[str, Any]) -> tuple[str, dict[str, int]]:
    kept: list[str] = []
    stats = {"kept_messages": 0, "dropped_messages": 0, "dropped_noise_messages": 0}
    for msg in session_data.get("messages", []) or []:
        if not isinstance(msg, dict):
            stats["dropped_messages"] += 1
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            stats["dropped_messages"] += 1
            continue
        text = extract_message_text(msg)
        if is_noise_message(text):
            stats["dropped_noise_messages"] += 1
            continue
        prefix = "User" if role == "user" else "Assistant"
        kept.append(f"{prefix}: {text}")
        stats["kept_messages"] += 1
    return clean_text("\n\n".join(kept)), stats


def stable_session_id(path: Path, session_data: dict[str, Any]) -> str:
    stem = path.stem
    if stem.startswith("session_"):
        # Hermes session JSON filenames are the durable unique identity. Some
        # files can carry a stale embedded session_id copied from an earlier
        # session, which would create duplicate Hindsight document_ids in one
        # retain batch. Prefer the filename for session_*.json inputs.
        return stem[len("session_"):]
    sid = str(session_data.get("session_id") or session_data.get("id") or "").strip()
    if sid:
        return sid
    return stem


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5_text(text: str) -> str:
    """Return an md5 digest for compatibility with legacy markdown dedupe state.

    md5 is used only as a non-security content identity for duplicate detection;
    sha256 remains the authoritative integrity hash in metadata.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def md5_bytes(data: bytes) -> str:
    """Return an md5 digest for non-security file duplicate detection."""
    return hashlib.md5(data).hexdigest()


def profile_segment(profile: str | None) -> str:
    """Return a stable, URL/tag-safe profile segment."""
    raw = str(profile or "default").strip() or "default"
    out: list[str] = []
    prev_dash = False
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    return "".join(out).strip("-_") or "default"


def session_document_base_id(session_id: str, source_profile: str = "default") -> str:
    segment = profile_segment(source_profile)
    if segment == "default":
        # Preserve legacy default-profile document IDs so incremental submit
        # state does not re-retain the entire historical default session set.
        return f"hermes-session::{session_id}"
    return f"hermes-session::{segment}::{session_id}"


def profile_tags(source_profile: str = "default") -> list[str]:
    segment = profile_segment(source_profile)
    tags = [f"profile:{segment}"]
    tags.append("source:hermes-session" if segment == "default" else "source:kanban-profile")
    return tags


def profile_source_label(source_profile: str = "default") -> str:
    segment = profile_segment(source_profile)
    return "hermes" if segment == "default" else f"hermes-profile:{segment}"


def source_file_metadata(path: Path, raw_bytes: bytes | None = None) -> dict[str, Any]:
    stat = path.stat()
    if raw_bytes is None:
        raw_bytes = path.read_bytes()
    return {
        "source_mtime_ns": stat.st_mtime_ns,
        "source_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "source_size_bytes": stat.st_size,
        "source_file_sha256": sha256_bytes(raw_bytes),
        "source_file_md5": md5_bytes(raw_bytes),
    }


def is_path_under(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
        return True
    except Exception:
        return False


def split_text(text: str, max_chars: int = DEFAULT_MAX_DOCUMENT_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for paragraph in paragraphs:
        add_len = len(paragraph) + (2 if cur else 0)
        if cur and cur_len + add_len > max_chars:
            chunks.append("\n\n".join(cur))
            cur = [paragraph]
            cur_len = len(paragraph)
        elif len(paragraph) > max_chars:
            if cur:
                chunks.append("\n\n".join(cur))
                cur = []
                cur_len = 0
            for i in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[i:i + max_chars])
        else:
            cur.append(paragraph)
            cur_len += add_len
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c]


def propose_tags(text: str, session_data: dict[str, Any] | None = None) -> list[str]:
    hay = "\n".join([
        text or "",
        str((session_data or {}).get("title") or ""),
        str((session_data or {}).get("model") or ""),
    ]).lower()
    tags: set[str] = set()
    for tag, keywords, implied in TAG_RULES:
        if any(k.lower() in hay for k in keywords):
            tags.add(tag)
            tags.update(implied)
    return sorted(t for t in tags if t not in SYSTEM_LABELS)


def is_bootstrap_or_environment_diagnostic(text: str, tags: list[str]) -> bool:
    hay = (text or "").lower()
    if not BOOTSTRAP_DIAGNOSTIC_RE.search(hay):
        return False
    # If the only semantic signal is generic assistant intro/autodrive profile,
    # do not route the session to production. These startup/debug sessions are
    # useful evidence but poor canonical memory seeds.
    strong_project_tags = [t for t in tags if t.startswith("project:") and t != "project:openclaw"]
    if strong_project_tags:
        return False
    return True


def is_broad_aggregate_summary(text: str) -> bool:
    return bool(BROAD_AGGREGATE_RE.search((text or "").lower()))


def is_memory_recall_or_context_bootstrap(text: str) -> bool:
    hay = clean_text(text or "")
    head = hay[:1200]
    matches = list(MEMORY_RECALL_BOOTSTRAP_RE.finditer(head))
    if not matches:
        return False
    if any((m.group(1) or "").lower() != "go on" for m in matches):
        return True
    return bool(MEMORY_RECALL_RESPONSE_RE.search(head))


def is_context_resume_or_handoff(text: str) -> bool:
    hay = clean_text(text or "")
    head = hay[:1200]
    return bool(CONTEXT_RESUME_BOOTSTRAP_RE.search(head) and CONTEXT_RESUME_RESPONSE_RE.search(head))


def contains_secret_material(text: str) -> bool:
    return bool(SECRET_MATERIAL_RE.search(clean_text(text or "")))


def is_automated_cron_session(session_data: dict[str, Any]) -> bool:
    """Return True for Hermes scheduled-job sessions.

    Cron sessions contain system prompts, tool schemas, skill injection text, and
    generated operational reports rather than direct user/Hermes dialogue. Keeping
    them out of production retain avoids self-ingestion and prompt/schema pollution.
    """
    platform = str(session_data.get("platform") or session_data.get("source") or "").lower()
    session_id = str(session_data.get("session_id") or session_data.get("id") or "")
    return platform == "cron" or session_id.startswith("cron_")


def action_for_content(text: str, tags: list[str]) -> tuple[str, str]:
    if len(text.strip()) < MIN_CONTENT_CHARS:
        return "skip", "empty_or_too_short"
    if is_low_signal_conversation(text):
        return "skip", "low_signal_short_or_chitchat"
    if contains_secret_material(text):
        return "manual_review", "secret_or_credential_material"
    if is_bootstrap_or_environment_diagnostic(text, tags):
        return "manual_review", "bootstrap_or_environment_diagnostic"
    if is_memory_recall_or_context_bootstrap(text):
        return "manual_review", "memory_recall_or_context_bootstrap"
    if is_context_resume_or_handoff(text):
        return "manual_review", "context_resume_or_handoff"
    if is_broad_aggregate_summary(text):
        return "manual_review", "broad_aggregate_summary"
    if not tags:
        return "manual_review", "no_semantic_tags"
    project_tags = [t for t in tags if t.startswith("project:")]
    semantic_tags = [t for t in tags if t.startswith(("domain:", "project:", "topic:"))]
    if len(project_tags) > 1 or len(semantic_tags) > 4:
        return "manual_review", "multi_scope_or_overbroad_tags"
    return "production", "semantic_tags_detected"


def session_event_date(session_data: dict[str, Any]) -> str | None:
    """Return the conversation occurrence time for Hindsight top-level event_date.

    Hermes JSON sessions normally store session-level timestamps but not
    per-message timestamps. Use the actual session start when available and only
    fall back to last_updated/ended_at for older or partial records. This must be
    sent as a top-level Hindsight item field, not as LLM-visible metadata, so
    retain does not default extracted facts to the import/runtime date.
    """
    for key in ["session_start", "started_at", "created_at", "last_updated", "ended_at"]:
        value = session_data.get(key)
        if value:
            return str(value)
    return None


def record_for_chunk(*, session_path: Path, session_data: dict[str, Any], clean_full_text: str, chunk_text: str, session_id: str, bank_target: str, part_index: int, part_count: int, retain_chunk_size: int, source_meta: dict[str, Any] | None = None, source_profile: str = "default") -> dict[str, Any]:
    tags = propose_tags(clean_full_text, session_data)
    for tag in profile_tags(source_profile):
        if tag not in tags:
            tags.append(tag)
    tags = sorted(tags)
    # For split sessions, action is decided on the full cleaned session rather than
    # an individual tail chunk; otherwise a short final assistant reply can be
    # incorrectly skipped even though it is part of a valid session document.
    if is_automated_cron_session(session_data):
        action, reason = "skip", "automated_cron_session"
    else:
        action, reason = action_for_content(clean_full_text, tags)
    base_doc_id = session_document_base_id(session_id, source_profile)
    document_id = base_doc_id if part_count == 1 else f"{base_doc_id}::part-{part_index:03d}"
    source_profile_segment = profile_segment(source_profile)
    metadata = {
        "source_kind": "hermes_json",
        "source_label": profile_source_label(source_profile),
        "source_profile": source_profile_segment,
        "json_path": str(session_path),
        "session_id": session_id,
        "model": session_data.get("model"),
        "platform": session_data.get("platform") or session_data.get("source"),
        "started_at": session_data.get("session_start") or session_data.get("started_at"),
        "last_updated": session_data.get("last_updated") or session_data.get("ended_at"),
        "session_last_updated": session_data.get("last_updated") or session_data.get("ended_at"),
        "message_count": len(session_data.get("messages", []) or []),
        "content_sha256": sha256_text(chunk_text),
        "full_content_sha256": sha256_text(clean_full_text),
        "schema_version": SCHEMA_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "candidate_filter_version": CANDIDATE_FILTER_VERSION,
        "part_index": part_index,
        "part_count": part_count,
        "bank_target": bank_target,
    }
    metadata.update(source_meta or source_file_metadata(session_path))
    return {
        "document_id": document_id,
        "bank_target": bank_target,
        "action": action,
        "reason": reason,
        "content": chunk_text,
        "content_chars": len(chunk_text),
        "estimated_retain_chunks": max(1, math.ceil(len(chunk_text) / max(1, retain_chunk_size))),
        "event_date": session_event_date(session_data),
        "tags": tags,
        "metadata": metadata,
        "context": "hermes_session",
        "update_mode": "replace",
    }


def records_from_json_file(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    source_profile: str = "default",
    include_markdown_artifacts: bool = True,
    min_markdown_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
) -> list[dict[str, Any]]:
    path = Path(path)
    raw_bytes = path.read_bytes()
    source_meta = source_file_metadata(path, raw_bytes)
    session_data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    if not isinstance(session_data, dict):
        return []
    clean_full_text, stats = extract_clean_conversation(session_data)
    if not clean_full_text:
        session_id = stable_session_id(path, session_data)
        source_profile_segment = profile_segment(source_profile)
        return [{
            "document_id": session_document_base_id(session_id, source_profile),
            "bank_target": bank_target,
            "action": "skip",
            "reason": "empty_after_cleaning",
            "content": "",
            "content_chars": 0,
            "estimated_retain_chunks": 0,
            "event_date": session_event_date(session_data),
            "tags": profile_tags(source_profile),
            "metadata": {
                "source_kind": "hermes_json",
                "source_label": profile_source_label(source_profile),
                "source_profile": source_profile_segment,
                "json_path": str(path),
                "session_id": session_id,
                "session_last_updated": session_data.get("last_updated") or session_data.get("ended_at"),
                "schema_version": SCHEMA_VERSION,
                "cleaning_version": CLEANING_VERSION,
                "candidate_filter_version": CANDIDATE_FILTER_VERSION,
                "cleaning_stats": stats,
                "bank_target": bank_target,
                **source_meta,
            },
            "context": "hermes_session",
            "update_mode": "replace",
        }]
    session_id = stable_session_id(path, session_data)
    chunks = split_text(clean_full_text, max_document_chars)
    records = []
    for idx, chunk in enumerate(chunks):
        rec = record_for_chunk(
            session_path=path,
            session_data=session_data,
            clean_full_text=clean_full_text,
            chunk_text=chunk,
            session_id=session_id,
            bank_target=bank_target,
            part_index=idx,
            part_count=len(chunks),
            retain_chunk_size=retain_chunk_size,
            source_meta=source_meta,
            source_profile=source_profile,
        )
        rec["metadata"]["cleaning_stats"] = stats
        records.append(rec)
    if include_markdown_artifacts and not is_automated_cron_session(session_data):
        for artifact_path, producer in discover_hermes_markdown_artifacts_from_session(
            session_data,
            path,
            min_file_age_seconds=min_markdown_file_age_seconds,
        ):
            try:
                records.extend(record_from_hermes_markdown_artifact(
                    artifact_path,
                    bank_target=bank_target,
                    retain_chunk_size=retain_chunk_size,
                    producer=producer,
                    source_session_path=str(path),
                    session_id=session_id,
                    source_profile=source_profile,
                ))
            except Exception:
                continue
    return records


def codex_session_id_from_meta(path: Path, meta: dict[str, Any]) -> str:
    sid = str(meta.get("id") or meta.get("session_id") or "").strip()
    if sid:
        return sid
    m = re.search(r"rollout-[^-]+(?:-[^-]+)*-(?P<id>019[a-f0-9-]+)\.jsonl$", path.name)
    if m:
        return m.group("id")
    return path.stem


def extract_codex_content_text(content: Any) -> str:
    if isinstance(content, str):
        return clean_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                part = clean_text(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if item_type in {"input_text", "output_text", "text"}:
                    part = clean_text(item.get("text") or item.get("content") or "")
                else:
                    part = ""
            else:
                part = ""
            if part:
                parts.append(part)
        return clean_text("\n".join(parts))
    if isinstance(content, dict):
        item_type = str(content.get("type") or "").lower()
        if item_type in {"input_text", "output_text", "text"}:
            return clean_text(content.get("text") or content.get("content") or "")
    return ""


def is_codex_context_injection(text: str) -> bool:
    stripped = clean_text(text)
    if not stripped:
        return True
    if stripped.startswith("# AGENTS.md instructions"):
        return True
    if stripped.startswith("<environment_context>") or "<environment_context>" in stripped[:800]:
        return True
    return False


def codex_apply_patch_succeeded(output: Any) -> bool:
    text = ""
    parsed = None
    if isinstance(output, str):
        text = output
        try:
            parsed = json.loads(output)
        except Exception:
            parsed = None
    elif isinstance(output, dict):
        parsed = output
        text = json.dumps(output, ensure_ascii=False)
    if isinstance(parsed, dict):
        if parsed.get("metadata", {}).get("exit_code") not in (None, 0):
            return False
        text = "\n".join(str(v) for v in parsed.values())
    return "Success." in text or "Updated the following files" in text


def extract_codex_md_path_candidates(text: Any, *, patch_only: bool = False) -> list[str]:
    if not isinstance(text, str) or ".md" not in text:
        return []
    candidates: list[str] = []
    if patch_only:
        for m in CODEX_APPLY_PATCH_MD_RE.finditer(text):
            candidates.append((m.group("path") or "").strip().strip(TRAILING_PATH_PUNCT))
        return list(dict.fromkeys(c for c in candidates if c.lower().endswith(".md")))
    for rx in [CODEX_BACKTICK_MD_RE, CODEX_ABS_MD_RE, CODEX_REL_MD_RE]:
        for m in rx.finditer(text):
            raw = (m.group("path") or "").strip().strip(TRAILING_PATH_PUNCT)
            if raw and raw.lower().endswith(".md"):
                candidates.append(raw)
    return list(dict.fromkeys(candidates))


def has_codex_positive_md_artifact_hint(text: Any) -> bool:
    if not isinstance(text, str) or ".md" not in text:
        return False
    return bool(CODEX_MD_POSITIVE_HINT_RE.search(text) and not CODEX_MD_NEGATIVE_HINT_RE.search(text))


def resolve_codex_md_path(raw: str, *, cwd: str | None = None) -> Path | None:
    value = str(raw or "").strip().strip(TRAILING_PATH_PUNCT)
    if not value or not value.lower().endswith(".md"):
        return None
    path = Path(value).expanduser()
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        roots = []
        if cwd:
            roots.append(Path(cwd))
        roots.extend([Path.home(), DEFAULT_HERMES_HOME])
        candidates = [root / path for root in roots]
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".md":
                return resolved
        except OSError:
            continue
    return None


def codex_artifact_document_id(path: Path) -> str:
    return f"codex-artifact::{sha256_text(str(path.expanduser().resolve()))[:16]}"


def hermes_artifact_document_id(path: Path, source_profile: str = "default") -> str:
    segment = profile_segment(source_profile)
    return f"hermes-artifact::{segment}::{sha256_text(str(path.expanduser().resolve()))[:16]}"


def is_recent_file(path: Path, min_file_age_seconds: int) -> bool:
    if not min_file_age_seconds or min_file_age_seconds <= 0:
        return False
    cutoff_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    try:
        return path.stat().st_mtime_ns > cutoff_ns
    except OSError:
        return True


def safe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def tool_call_function_name(tool_call: dict[str, Any]) -> str:
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    if isinstance(func, dict):
        return str(func.get("name") or "")
    return str(tool_call.get("name") or "") if isinstance(tool_call, dict) else ""


def tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    raw = func.get("arguments") if isinstance(func, dict) else tool_call.get("arguments") if isinstance(tool_call, dict) else None
    return safe_json_dict(raw)


def discover_hermes_markdown_artifacts_from_session(
    session_data: dict[str, Any],
    session_path: Path,
    *,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
) -> list[tuple[Path, str]]:
    """Find markdown files that a Hermes/deepseek-style session actually wrote.

    Discovery is evidence-based: write_file tool calls are strong evidence, and
    assistant/tool text is accepted only when it contains a production hint.
    """
    cwd = str(session_data.get("cwd") or session_data.get("working_dir") or session_path.parent)
    candidates: dict[Path, str] = {}
    for msg in session_data.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fname = tool_call_function_name(tc)
                args = tool_call_arguments(tc)
                raw_path = str(args.get("path") or "")
                if fname in {"write_file", "write"} and raw_path.lower().endswith(".md"):
                    resolved = resolve_codex_md_path(raw_path, cwd=cwd)
                    if resolved is not None and not is_recent_file(resolved, min_file_age_seconds):
                        candidates.setdefault(resolved, "hermes_write_file_tool")
            text = extract_message_text(msg)
            if has_codex_positive_md_artifact_hint(text):
                for raw_path in extract_codex_md_path_candidates(text):
                    resolved = resolve_codex_md_path(raw_path, cwd=cwd)
                    if resolved is not None and not is_recent_file(resolved, min_file_age_seconds):
                        candidates.setdefault(resolved, "hermes_assistant_text")
        elif role == "tool":
            text = extract_message_text(msg) or clean_text(msg.get("content") or "")
            if has_codex_positive_md_artifact_hint(text):
                producer = f"hermes_tool_output:{msg.get('name') or 'tool'}"
                for raw_path in extract_codex_md_path_candidates(text):
                    resolved = resolve_codex_md_path(raw_path, cwd=cwd)
                    if resolved is not None and not is_recent_file(resolved, min_file_age_seconds):
                        candidates.setdefault(resolved, producer)
    return sorted(candidates.items(), key=lambda item: str(item[0]))


def kanban_prompt_parts(path: Path) -> tuple[str, str, str, str] | None:
    """Return (producer, board, profile, task_id) for a Kanban prompt markdown path."""
    parts = path.parts
    for marker, producer in ((".codex-kanban", "codex-kanban"), (".deepseek-kanban", "deepseek-kanban")):
        if marker not in parts:
            continue
        idx = parts.index(marker)
        tail = parts[idx + 1:]
        if len(tail) < 3:
            return None
        board, profile = tail[0], tail[1]
        task_id = Path(tail[-1]).stem
        if board and profile and task_id:
            return producer, board, profile, task_id
    return None


def kanban_prompt_document_id(*, board: str, task_id: str, part_index: int | None = None) -> str:
    base = f"kanban-markdown::{board}::{task_id}"
    return base if part_index is None else f"{base}::part-{part_index:03d}"


def record_from_kanban_prompt_markdown(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    parsed = kanban_prompt_parts(path)
    if parsed is None:
        return []
    producer, board, profile, task_id = parsed
    raw_bytes = path.read_bytes()
    full_text = clean_text(raw_bytes.decode("utf-8", errors="replace"))
    if not full_text:
        return []
    tags = sorted(set(["source:kanban-markdown", "topic:kanban", "topic:markdown-artifact", *propose_tags(full_text, {})]))
    action, reason = ("manual_review", "secret_or_credential_material") if contains_secret_material(full_text) else ("production", "kanban_prompt_markdown")
    source_meta = source_file_metadata(path, raw_bytes)
    chunks = split_text(full_text, DEFAULT_MAX_DOCUMENT_CHARS)
    records: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks or [full_text]):
        document_id = kanban_prompt_document_id(
            board=board,
            task_id=task_id,
            part_index=None if len(chunks) <= 1 else idx,
        )
        content = clean_text(
            "\n".join([
                f"Title: Kanban prompt {board}/{task_id}",
                "Source: Hermes Kanban generated Markdown prompt",
                f"Board: {board}",
                f"Profile: {profile}",
                f"Task: {task_id}",
                f"Path: {path}",
                "",
                chunk,
            ])
        )
        metadata = {
            "source_kind": "kanban_prompt_markdown",
            "source_label": "kanban",
            "source_path": str(path),
            "producer": producer,
            "board": board,
            "profile": profile,
            "task_id": task_id,
            "content_md5": md5_text(content),
            "full_content_md5": md5_text(full_text),
            "content_sha256": sha256_text(content),
            "full_content_sha256": sha256_text(full_text),
            "schema_version": SCHEMA_VERSION,
            "cleaning_version": CLEANING_VERSION,
            "candidate_filter_version": CANDIDATE_FILTER_VERSION,
            "part_index": idx,
            "part_count": len(chunks),
            "bank_target": bank_target,
            **source_meta,
        }
        records.append({
            "document_id": document_id,
            "bank_target": bank_target,
            "action": action,
            "reason": reason,
            "content": content,
            "content_chars": len(content),
            "estimated_retain_chunks": max(1, math.ceil(len(content) / max(1, retain_chunk_size))) if content else 0,
            "event_date": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "tags": tags,
            "metadata": metadata,
            "context": "kanban_prompt_markdown",
            "update_mode": "replace",
        })
    return records


def _kanban_prompt_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (path.stat().st_mtime_ns, str(path))
    except OSError:
        return (0, str(path))


def iter_kanban_prompt_markdown_files(
    workspace_roots: Iterable[Path],
    *,
    since_mtime_ns: int | None = None,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
) -> Iterable[Path]:
    cutoff_mtime_ns = None
    if min_file_age_seconds and min_file_age_seconds > 0:
        cutoff_mtime_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    seen_paths: set[Path] = set()
    task_winners: dict[tuple[str, str], Path] = {}
    for root in workspace_roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for marker in [".codex-kanban", ".deepseek-kanban"]:
            for base in root.rglob(marker):
                if not base.is_dir():
                    continue
                for path in base.glob("*/*/*.md"):
                    try:
                        resolved = path.resolve()
                        stat = resolved.stat()
                    except OSError:
                        continue
                    if resolved in seen_paths:
                        continue
                    if since_mtime_ns is not None and stat.st_mtime_ns <= since_mtime_ns:
                        continue
                    if cutoff_mtime_ns is not None and stat.st_mtime_ns > cutoff_mtime_ns:
                        continue
                    parsed = kanban_prompt_parts(resolved)
                    if parsed is None:
                        continue
                    _, board, _, task_id = parsed
                    seen_paths.add(resolved)
                    key = (board, task_id)
                    current = task_winners.get(key)
                    if current is None or _kanban_prompt_sort_key(resolved) >= _kanban_prompt_sort_key(current):
                        task_winners[key] = resolved
    for path in sorted(task_winners.values(), key=str):
        yield path


def kanban_comment_document_id(*, board: str, task_id: str, comment_id: int) -> str:
    return f"kanban-comment::{board}::{task_id}::{comment_id}"


def _kanban_comment_record_from_row(
    *,
    board: str,
    db_path: Path,
    row: sqlite3.Row | dict[str, Any],
    bank_target: str = DEFAULT_BANK_TARGET,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
) -> dict[str, Any] | None:
    get = row.__getitem__ if hasattr(row, "__getitem__") else row.get  # type: ignore[assignment]
    comment_id = int(get("comment_id"))
    task_id = str(get("task_id") or "")
    body = clean_text(get("comment_body") or "")
    if not task_id or not body:
        return None
    title = clean_text(get("task_title") or "")
    task_body = clean_text(get("task_body") or "")
    result = clean_text(get("task_result") or "")
    assignee = str(get("task_assignee") or "")
    status = str(get("task_status") or "")
    author = str(get("comment_author") or "")
    created_at = int(get("comment_created_at") or 0)
    full_text = clean_text(
        "\n".join([
            f"Title: Kanban comment {board}/{task_id}#{comment_id}",
            "Source: Hermes Kanban task comment",
            f"Board: {board}",
            f"Task: {task_id}",
            f"Task title: {title}",
            f"Assignee: {assignee}",
            f"Status: {status}",
            f"Author: {author}",
            f"Created at: {datetime.fromtimestamp(created_at, timezone.utc).isoformat() if created_at else ''}",
            "",
            "Task body:",
            task_body,
            "",
            "Task result:",
            result,
            "",
            "Comment:",
            body,
        ])
    )
    tags = sorted(set(["source:kanban-comment", "topic:kanban", *propose_tags(full_text, {})]))
    action, reason = ("manual_review", "secret_or_credential_material") if contains_secret_material(full_text) else ("production", "kanban_task_comment")
    content_sha = sha256_text(full_text)
    metadata = {
        "source_kind": "kanban_task_comment",
        "source_label": "kanban",
        "board": board,
        "board_db_path": str(db_path),
        "task_id": task_id,
        "comment_id": comment_id,
        "author": author,
        "task_title": title,
        "task_assignee": assignee,
        "task_status": status,
        "content_sha256": content_sha,
        "full_content_sha256": content_sha,
        "source_mtime_ns": created_at * 1_000_000_000,
        "source_size_bytes": len(body.encode("utf-8")),
        "source_file_sha256": sha256_text(f"{board}:{task_id}:{comment_id}:{body}"),
        "schema_version": SCHEMA_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "candidate_filter_version": CANDIDATE_FILTER_VERSION,
        "bank_target": bank_target,
    }
    return {
        "document_id": kanban_comment_document_id(board=board, task_id=task_id, comment_id=comment_id),
        "bank_target": bank_target,
        "action": action,
        "reason": reason,
        "content": full_text,
        "content_chars": len(full_text),
        "estimated_retain_chunks": max(1, math.ceil(len(full_text) / max(1, retain_chunk_size))) if full_text else 0,
        "event_date": datetime.fromtimestamp(created_at, timezone.utc).isoformat() if created_at else None,
        "tags": tags,
        "metadata": metadata,
        "context": "kanban_comment",
        "update_mode": "replace",
    }


def records_from_kanban_board_db(
    board: str,
    db_path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    since_mtime_ns: int | None = None,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
) -> list[dict[str, Any]]:
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        return []
    cutoff_created_at = None
    if min_file_age_seconds and min_file_age_seconds > 0:
        cutoff_created_at = int(datetime.now(timezone.utc).timestamp()) - int(min_file_age_seconds)
    since_created_at = int(since_mtime_ns // 1_000_000_000) if since_mtime_ns is not None else None
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        clauses = []
        params: list[Any] = []
        if since_created_at is not None:
            clauses.append("c.created_at > ?")
            params.append(since_created_at)
        if cutoff_created_at is not None:
            clauses.append("c.created_at <= ?")
            params.append(cutoff_created_at)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = con.execute(
            f"""
            SELECT
              c.id AS comment_id,
              c.task_id AS task_id,
              c.author AS comment_author,
              c.body AS comment_body,
              c.created_at AS comment_created_at,
              t.title AS task_title,
              t.body AS task_body,
              t.assignee AS task_assignee,
              t.status AS task_status,
              t.result AS task_result
            FROM task_comments c
            LEFT JOIN tasks t ON t.id = c.task_id
            {where}
            ORDER BY c.created_at ASC, c.id ASC
            """,
            params,
        ).fetchall()
    finally:
        con.close()
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (board, str(row["task_id"] or ""), int(row["comment_id"]))
        if key in seen:
            continue
        seen.add(key)
        rec = _kanban_comment_record_from_row(
            board=board,
            db_path=db_path,
            row=row,
            bank_target=bank_target,
            retain_chunk_size=retain_chunk_size,
        )
        if rec is not None:
            records.append(rec)
    return records


def discover_kanban_board_db_paths(kanban_home: Path = DEFAULT_HERMES_HOME) -> list[tuple[str, Path]]:
    root = Path(kanban_home).expanduser()
    out: list[tuple[str, Path]] = []
    default_db = root / "kanban.db"
    if default_db.exists():
        out.append(("default", default_db))
    boards_root = root / "kanban" / "boards"
    if boards_root.exists():
        for child in sorted(boards_root.iterdir()):
            db = child / "kanban.db"
            if child.is_dir() and db.exists():
                out.append((child.name, db))
    return out


def build_manifest_from_kanban_sources(
    *,
    workspace_roots: Iterable[Path] | None = None,
    board_db_paths: Iterable[tuple[str, Path]] | None = None,
    bank_target: str = DEFAULT_BANK_TARGET,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    since_mtime_ns: int | None = None,
    prompt_since_mtime_ns: int | None = None,
    comment_since_mtime_ns: int | None = None,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    roots = list(workspace_roots if workspace_roots is not None else DEFAULT_KANBAN_WORKSPACE_ROOTS)
    for path in iter_kanban_prompt_markdown_files(
        roots,
        since_mtime_ns=prompt_since_mtime_ns if prompt_since_mtime_ns is not None else since_mtime_ns,
        min_file_age_seconds=min_file_age_seconds,
    ):
        records.extend(record_from_kanban_prompt_markdown(
            path,
            bank_target=bank_target,
            retain_chunk_size=retain_chunk_size,
        ))
    dbs = list(board_db_paths if board_db_paths is not None else discover_kanban_board_db_paths(DEFAULT_HERMES_HOME))
    seen_comment_docs: set[str] = set()
    for board, db_path in dbs:
        for rec in records_from_kanban_board_db(
            board,
            db_path,
            bank_target=bank_target,
            retain_chunk_size=retain_chunk_size,
            since_mtime_ns=comment_since_mtime_ns if comment_since_mtime_ns is not None else since_mtime_ns,
            min_file_age_seconds=min_file_age_seconds,
        ):
            doc_id = str(rec.get("document_id") or "")
            if doc_id in seen_comment_docs:
                continue
            seen_comment_docs.add(doc_id)
            records.append(rec)
    return records


def record_from_codex_markdown_artifact(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    producer: str = "codex",
    source_rollout_path: str | None = None,
    session_id: str | None = None,
    source_kind: str = "codex_markdown_artifact",
    source_label: str = "codex",
    source_tag: str = "source:codex-artifact",
    context: str = "codex_markdown_artifact",
    document_id_base: str | None = None,
    source_title: str = "Codex generated Markdown artifact",
    source_session_path: str | None = None,
    source_profile: str | None = None,
) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    raw_bytes = path.read_bytes()
    full_text = clean_text(raw_bytes.decode("utf-8", errors="replace"))
    tags = sorted(set([source_tag, "topic:markdown-artifact", *propose_tags(full_text, {})]))
    action, reason = ("manual_review", "secret_or_credential_material") if contains_secret_material(full_text) else ("production", source_kind)
    source_meta = source_file_metadata(path, raw_bytes)
    chunks = split_text(full_text, DEFAULT_MAX_DOCUMENT_CHARS)
    records: list[dict[str, Any]] = []
    base_id = document_id_base or codex_artifact_document_id(path)
    for idx, chunk in enumerate(chunks or [full_text]):
        document_id = base_id if len(chunks) <= 1 else f"{base_id}::part-{idx:03d}"
        content = clean_text(
            "\n".join([
                f"Title: {path.stem}",
                f"Source: {source_title}",
                f"Path: {path}",
                "",
                chunk,
            ])
        )
        metadata = {
            "source_kind": source_kind,
            "source_label": source_label,
            "source_path": str(path),
            "source_rollout_path": source_rollout_path,
            "source_session_path": source_session_path,
            "source_profile": profile_segment(source_profile) if source_profile else None,
            "session_id": session_id,
            "producer": producer,
            "content_md5": md5_text(content),
            "full_content_md5": md5_text(full_text),
            "content_sha256": sha256_text(content),
            "full_content_sha256": sha256_text(full_text),
            "schema_version": SCHEMA_VERSION,
            "cleaning_version": CLEANING_VERSION,
            "candidate_filter_version": CANDIDATE_FILTER_VERSION,
            "part_index": idx,
            "part_count": len(chunks),
            "bank_target": bank_target,
            **source_meta,
        }
        records.append({
            "document_id": document_id,
            "bank_target": bank_target,
            "action": action,
            "reason": reason,
            "content": content,
            "content_chars": len(content),
            "estimated_retain_chunks": max(1, math.ceil(len(content) / max(1, retain_chunk_size))) if content else 0,
            "event_date": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "tags": tags,
            "metadata": metadata,
            "context": context,
            "update_mode": "replace",
        })
    return records


def record_from_hermes_markdown_artifact(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    producer: str = "hermes",
    source_session_path: str | None = None,
    session_id: str | None = None,
    source_profile: str = "default",
) -> list[dict[str, Any]]:
    return record_from_codex_markdown_artifact(
        path,
        bank_target=bank_target,
        retain_chunk_size=retain_chunk_size,
        producer=producer,
        source_rollout_path=None,
        source_session_path=source_session_path,
        session_id=session_id,
        source_profile=source_profile,
        source_kind="hermes_markdown_artifact",
        source_label=profile_source_label(source_profile),
        source_tag="source:hermes-markdown-artifact",
        context="hermes_markdown_artifact",
        document_id_base=hermes_artifact_document_id(Path(path), source_profile),
        source_title="Hermes generated Markdown artifact",
    )


def record_markdown_artifact(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    producer: str = "hermes",
    source_session_path: str | None = None,
    session_id: str | None = None,
    source_profile: str | None = None,
    structured: bool = False,
) -> list[dict[str, Any]]:
    """Wrapper around record_from_codex_markdown_artifact with optional structured formatting.

    When structured=True, uses the structured MD formatter to capture document
    hierarchy, key decisions, code blocks, tables, and cross-references.
    """
    if structured:
        # Use the structured formatter for richer context
        try:
            from hindsight_structured_md import format_structured_for_retain as _format_structured
        except ImportError:
            _format_structured = None

        md_path = Path(path).expanduser()
        if _format_structured is not None and md_path.exists():
            try:
                raw_text = md_path.read_text(encoding="utf-8", errors="replace")
                formatted = _format_structured(raw_text, md_path)
                # Use the formatted output via the standard path, with
                # a custom wrapper that replaces the content
                records = record_from_codex_markdown_artifact(
                    path,
                    bank_target=bank_target,
                    retain_chunk_size=retain_chunk_size,
                    producer=producer,
                    source_session_path=source_session_path,
                    session_id=session_id,
                    source_profile=source_profile,
                    source_kind="deepseek_markdown_artifact_structured",
                    source_label=profile_source_label(source_profile) if source_profile else "deepseek-tui",
                    source_tag="source:deepseek-markdown-artifact",
                    context="deepseek_markdown_artifact",
                    document_id_base=codex_artifact_document_id(md_path),
                    source_title=f"DeepSeek-TUI generated Markdown artifact ({md_path.name})",
                )
                # Replace the content with the structured version
                for rec in records:
                    rec["content"] = formatted
                    rec["content_chars"] = len(formatted)
                    rec["estimated_retain_chunks"] = max(1, math.ceil(len(formatted) / max(1, retain_chunk_size)))
                return records
            except Exception:
                pass

    # Fallback to standard processing
    return record_from_codex_markdown_artifact(
        path,
        bank_target=bank_target,
        retain_chunk_size=retain_chunk_size,
        producer=producer,
        source_session_path=source_session_path,
        session_id=session_id,
        source_profile=source_profile,
        source_kind="markdown_artifact",
        source_label=profile_source_label(source_profile) if source_profile else "unknown",
        source_tag="source:markdown-artifact",
        context="markdown_artifact",
    )


def records_from_codex_rollout_file(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    include_markdown_artifacts: bool = True,
) -> list[dict[str, Any]]:
    path = Path(path)
    raw_bytes = path.read_bytes()
    source_meta = source_file_metadata(path, raw_bytes)
    meta: dict[str, Any] = {}
    kept: list[str] = []
    stats = {"kept_messages": 0, "dropped_messages": 0, "dropped_noise_messages": 0}
    apply_patch_inputs: dict[str, str] = {}
    successful_apply_patch_calls: set[str] = set()
    md_candidates: dict[str, str] = {}

    for raw_line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except Exception:
            continue
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            continue
        if event.get("type") == "session_meta":
            meta = payload
            continue
        if event.get("type") != "response_item":
            continue
        item_type = str(payload.get("type") or "")
        if item_type == "message":
            role = str(payload.get("role") or "")
            if role not in {"user", "assistant"}:
                stats["dropped_messages"] += 1
                continue
            text = extract_codex_content_text(payload.get("content"))
            if role == "user" and is_codex_context_injection(text):
                stats["dropped_noise_messages"] += 1
                continue
            if is_noise_message(text):
                stats["dropped_noise_messages"] += 1
                continue
            prefix = "User" if role == "user" else "Assistant"
            kept.append(f"{prefix}: {text}")
            stats["kept_messages"] += 1
            if include_markdown_artifacts and role == "assistant" and has_codex_positive_md_artifact_hint(text):
                for raw_path in extract_codex_md_path_candidates(text):
                    md_candidates[raw_path] = "codex_assistant_text"
        elif item_type == "custom_tool_call" and str(payload.get("name") or "") == "apply_patch":
            call_id = str(payload.get("call_id") or "")
            if call_id:
                apply_patch_inputs[call_id] = str(payload.get("input") or "")
        elif item_type == "custom_tool_call_output":
            call_id = str(payload.get("call_id") or "")
            if call_id and codex_apply_patch_succeeded(payload.get("output")):
                successful_apply_patch_calls.add(call_id)

    if include_markdown_artifacts:
        for call_id in successful_apply_patch_calls:
            for raw_path in extract_codex_md_path_candidates(apply_patch_inputs.get(call_id, ""), patch_only=True):
                md_candidates[raw_path] = "codex_apply_patch"

    session_id = codex_session_id_from_meta(path, meta)
    clean_full_text = clean_text("\n\n".join(kept))
    records: list[dict[str, Any]] = []
    if clean_full_text:
        tags = sorted(set([*propose_tags(clean_full_text, meta), "source:codex-session"]))
        action, reason = action_for_content(clean_full_text, tags)
        chunks = split_text(clean_full_text, max_document_chars)
        for idx, chunk in enumerate(chunks):
            document_id = f"codex-session::{session_id}" if len(chunks) == 1 else f"codex-session::{session_id}::part-{idx:03d}"
            metadata = {
                "source_kind": "codex_rollout_jsonl",
                "source_label": "codex",
                "source_profile": "codex",
                "jsonl_path": str(path),
                "session_id": session_id,
                "cwd": meta.get("cwd"),
                "originator": meta.get("originator"),
                "cli_version": meta.get("cli_version"),
                "model": meta.get("model"),
                "model_provider": meta.get("model_provider"),
                "started_at": meta.get("timestamp"),
                "content_sha256": sha256_text(chunk),
                "full_content_sha256": sha256_text(clean_full_text),
                "schema_version": SCHEMA_VERSION,
                "cleaning_version": CLEANING_VERSION,
                "candidate_filter_version": CANDIDATE_FILTER_VERSION,
                "cleaning_stats": stats,
                "part_index": idx,
                "part_count": len(chunks),
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
                "estimated_retain_chunks": max(1, math.ceil(len(chunk) / max(1, retain_chunk_size))),
                "event_date": meta.get("timestamp"),
                "tags": tags,
                "metadata": metadata,
                "context": "codex_session",
                "update_mode": "replace",
            })

    if include_markdown_artifacts:
        seen_paths: set[Path] = set()
        for raw_path, producer in md_candidates.items():
            resolved = resolve_codex_md_path(raw_path, cwd=str(meta.get("cwd") or ""))
            if resolved is None or resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                records.extend(record_from_codex_markdown_artifact(
                    resolved,
                    bank_target=bank_target,
                    retain_chunk_size=retain_chunk_size,
                    producer=producer,
                    source_rollout_path=str(path),
                    session_id=session_id,
                ))
            except Exception:
                continue
    return records


def iter_codex_rollout_files(codex_sessions_dir: Path = DEFAULT_CODEX_SESSIONS_DIR, limit: int | None = None, since_mtime_ns: int | None = None, min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS) -> Iterable[Path]:
    count = 0
    cutoff_mtime_ns = None
    if min_file_age_seconds and min_file_age_seconds > 0:
        cutoff_mtime_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    root = Path(codex_sessions_dir).expanduser()
    if not root.exists():
        return
    for path in sorted(root.rglob("rollout-*.jsonl")):
        try:
            stat = path.stat()
        except OSError:
            continue
        if since_mtime_ns is not None and stat.st_mtime_ns <= since_mtime_ns:
            continue
        if cutoff_mtime_ns is not None and stat.st_mtime_ns > cutoff_mtime_ns:
            continue
        yield path
        count += 1
        if limit is not None and count >= limit:
            return


def build_manifest_from_codex_dir(
    *,
    codex_sessions_dir: Path = DEFAULT_CODEX_SESSIONS_DIR,
    bank_target: str = DEFAULT_BANK_TARGET,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    limit: int | None = None,
    since_mtime_ns: int | None = None,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
    include_markdown_artifacts: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_codex_rollout_files(codex_sessions_dir, limit=limit, since_mtime_ns=since_mtime_ns, min_file_age_seconds=min_file_age_seconds):
        try:
            records.extend(records_from_codex_rollout_file(
                path,
                bank_target=bank_target,
                max_document_chars=max_document_chars,
                retain_chunk_size=retain_chunk_size,
                include_markdown_artifacts=include_markdown_artifacts,
            ))
        except Exception as e:
            session_id = path.stem
            records.append({
                "document_id": f"codex-session::{session_id}",
                "bank_target": bank_target,
                "action": "manual_review",
                "reason": f"read_error:{type(e).__name__}:{e}",
                "content": "",
                "content_chars": 0,
                "estimated_retain_chunks": 0,
                "tags": ["source:codex-session"],
                "metadata": {"source_kind": "codex_rollout_jsonl", "source_label": "codex", "jsonl_path": str(path), "schema_version": SCHEMA_VERSION, "cleaning_version": CLEANING_VERSION, "candidate_filter_version": CANDIDATE_FILTER_VERSION, "source_profile": "codex"},
                "context": "codex_session",
                "update_mode": "replace",
            })
    return records


# ── DeepSeek-TUI session handling ─────────────────────────────────────────

DEEPSEEK_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def deepseek_session_id_from_file(path: Path, metadata: dict[str, Any]) -> str:
    """Extract a stable session ID from deepseek session file path and metadata."""
    session_id = str(metadata.get("id") or "")
    if DEEPSEEK_SESSION_ID_RE.match(session_id):
        return session_id
    # Fallback: use filename stem if it looks like a UUID
    stem = path.stem
    if DEEPSEEK_SESSION_ID_RE.match(stem):
        return stem
    # Last resort
    return sha256_text(str(path) + str(metadata.get("created_at", "")))[:16]


def _extract_deepseek_message_text(msg: dict[str, Any]) -> str:
    """Extract CLEAN text from a deepseek-tui message.

    DeepSeek-TUI message content is a list of blocks:
      - type=text: user input or assistant response (PRESERVE)
      - type=thinking: model reasoning tokens (STRIP)
      - type=tool_use: tool call name + input (STRIP — exact tool I/O not needed)

    Tool role messages (role='tool') are also filtered out at the caller level.
    Only user and assistant roles are processed; within those, only text blocks
    are kept. This ensures precise capture of human input + model output only.
    """
    raw = msg.get("content")
    if isinstance(raw, list):
        # DeepSeek-TUI uses content block format
        texts: list[str] = []
        for block in raw:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_block = str(block.get("text") or "")
                # Also strip any remaining thinking blocks embedded in text
                text_block = THINK_BLOCK_RE.sub("", text_block).strip()
                if text_block:
                    texts.append(text_block)
        return clean_text("\n\n".join(texts)).strip()
    elif isinstance(raw, str):
        # Legacy/fallback: plain text
        text = THINK_BLOCK_RE.sub("", raw)
        return clean_text(text).strip()
    return ""


def records_from_deepseek_session_file(
    path: str | Path,
    *,
    bank_target: str = DEFAULT_BANK_TARGET,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    include_markdown_artifacts: bool = True,
) -> list[dict[str, Any]]:
    """Parse a single deepseek-tui session JSON file into manifest records."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    source_meta = source_file_metadata(path, raw_bytes)

    try:
        session_data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception:
        return []

    if not isinstance(session_data, dict):
        return []

    metadata = session_data.get("metadata") if isinstance(session_data.get("metadata"), dict) else {}
    messages = session_data.get("messages") if isinstance(session_data.get("messages"), list) else []

    if not messages:
        return []

    session_id = deepseek_session_id_from_file(path, metadata)

    # Extract conversation, similar to extract_clean_conversation but
    # tailored for deepseek-tui format (user/assistant roles, thinking stripped)
    kept: list[str] = []
    stats = {"kept_messages": 0, "dropped_messages": 0, "dropped_noise_messages": 0}
    md_candidates: dict[str, str] = {}
    cwd = str(metadata.get("workspace") or "")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role not in {"user", "assistant"}:
            stats["dropped_messages"] += 1
            continue

        text = _extract_deepseek_message_text(msg)
        if not text:
            stats["dropped_messages"] += 1
            continue

        if is_noise_message(text):
            stats["dropped_noise_messages"] += 1
            continue

        prefix = "User" if role == "user" else "Assistant"
        kept.append(f"{prefix}: {text}")
        stats["kept_messages"] += 1

        # Discover MD artifacts from assistant messages — two sources:
        #  1. Direct tool_use blocks (deepseek-tui native: write_file/apply_patch)
        #  2. Assistant text hints (legacy: "wrote", "written", "saved")
        if include_markdown_artifacts and role == "assistant":
            raw_content = msg.get("content")
            if isinstance(raw_content, list):
                for block in raw_content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tool_name = str(block.get("name") or "").lower()
                        tool_input = block.get("input")
                        if isinstance(tool_input, dict):
                            raw_path = str(tool_input.get("path") or "")
                        elif isinstance(tool_input, str):
                            raw_path = tool_input
                        else:
                            continue
                        if tool_name in {"write_file", "write", "edit_file", "apply_patch"} and raw_path.lower().endswith(".md"):
                            resolved = resolve_codex_md_path(raw_path, cwd=cwd)
                            if resolved is not None:
                                md_candidates[str(resolved)] = f"deepseek_{tool_name}"
            # Also check text content for hints
            if has_codex_positive_md_artifact_hint(text):
                for raw_path in extract_codex_md_path_candidates(text):
                    resolved = resolve_codex_md_path(raw_path, cwd=cwd)
                    if resolved is not None:
                        md_candidates[str(resolved)] = "deepseek_assistant_text"

    clean_full_text = clean_text("\n\n".join(kept))
    records: list[dict[str, Any]] = []

    if clean_full_text:
        tags = sorted(set([*propose_tags(clean_full_text, session_data), "source:deepseek-session"]))
        action, reason = action_for_content(clean_full_text, tags)
        chunks = split_text(clean_full_text, max_document_chars)

        for idx, chunk in enumerate(chunks):
            document_id = f"deepseek-session::{session_id}" if len(chunks) == 1 else f"deepseek-session::{session_id}::part-{idx:03d}"
            rec_metadata = {
                "source_kind": "deepseek_session_json",
                "source_label": "deepseek-tui",
                "source_profile": "deepseek",
                "json_path": str(path),
                "session_id": session_id,
                "workspace": metadata.get("workspace"),
                "model": metadata.get("model"),
                "mode": metadata.get("mode"),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "message_count": metadata.get("message_count"),
                "total_tokens": metadata.get("total_tokens"),
                "content_sha256": sha256_text(chunk),
                "full_content_sha256": sha256_text(clean_full_text),
                "schema_version": SCHEMA_VERSION,
                "cleaning_version": CLEANING_VERSION,
                "candidate_filter_version": CANDIDATE_FILTER_VERSION,
                "cleaning_stats": stats,
                "part_index": idx,
                "part_count": len(chunks),
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
                "estimated_retain_chunks": max(1, math.ceil(len(chunk) / max(1, retain_chunk_size))),
                "event_date": metadata.get("created_at"),
                "tags": tags,
                "metadata": rec_metadata,
                "context": "deepseek_session",
                "update_mode": "replace",
            })

    # Include discovered MD artifacts with structured processing
    if include_markdown_artifacts:
        seen_paths: set[str] = set()
        for raw_path_str, producer in md_candidates.items():
            if raw_path_str in seen_paths:
                continue
            seen_paths.add(raw_path_str)
            try:
                md_path = Path(raw_path_str).expanduser()
                if not md_path.exists():
                    continue
                records.extend(record_markdown_artifact(
                    md_path,
                    bank_target=bank_target,
                    retain_chunk_size=retain_chunk_size,
                    producer=producer,
                    source_session_path=str(path),
                    session_id=session_id,
                    structured=True,
                ))
            except Exception:
                continue

    return records


def iter_deepseek_session_files(
    deepseek_sessions_dir: Path = DEFAULT_DEEPSEEK_SESSIONS_DIR,
    limit: int | None = None,
    since_mtime_ns: int | None = None,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
) -> Iterable[Path]:
    """Iterate deepseek-tui session JSON files."""
    count = 0
    cutoff_mtime_ns = None
    if min_file_age_seconds and min_file_age_seconds > 0:
        cutoff_mtime_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    root = Path(deepseek_sessions_dir).expanduser()
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        if since_mtime_ns is not None and stat.st_mtime_ns <= since_mtime_ns:
            continue
        if cutoff_mtime_ns is not None and stat.st_mtime_ns > cutoff_mtime_ns:
            continue
        yield path
        count += 1
        if limit is not None and count >= limit:
            return


def build_manifest_from_deepseek_dir(
    *,
    deepseek_sessions_dir: Path = DEFAULT_DEEPSEEK_SESSIONS_DIR,
    bank_target: str = DEFAULT_BANK_TARGET,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    limit: int | None = None,
    since_mtime_ns: int | None = None,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
    include_markdown_artifacts: bool = True,
) -> list[dict[str, Any]]:
    """Build manifest records from ~/.deepseek/sessions/ directory."""
    records: list[dict[str, Any]] = []
    for path in iter_deepseek_session_files(
        deepseek_sessions_dir, limit=limit,
        since_mtime_ns=since_mtime_ns,
        min_file_age_seconds=min_file_age_seconds,
    ):
        try:
            records.extend(records_from_deepseek_session_file(
                path,
                bank_target=bank_target,
                max_document_chars=max_document_chars,
                retain_chunk_size=retain_chunk_size,
                include_markdown_artifacts=include_markdown_artifacts,
            ))
        except Exception as e:
            session_id = path.stem
            records.append({
                "document_id": f"deepseek-session::{session_id}",
                "bank_target": bank_target,
                "action": "manual_review",
                "reason": f"read_error:{type(e).__name__}:{e}",
                "content": "",
                "content_chars": 0,
                "estimated_retain_chunks": 0,
                "tags": ["source:deepseek-session"],
                "metadata": {
                    "source_kind": "deepseek_session_json",
                    "source_label": "deepseek-tui",
                    "json_path": str(path),
                    "schema_version": SCHEMA_VERSION,
                    "cleaning_version": CLEANING_VERSION,
                    "candidate_filter_version": CANDIDATE_FILTER_VERSION,
                    "source_profile": "deepseek",
                },
                "context": "deepseek_session",
                "update_mode": "replace",
            })
    return records


def iter_json_session_files(sessions_dir: Path, limit: int | None = None, since_mtime_ns: int | None = None, min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS) -> Iterable[Path]:
    """Yield stable session JSON files.

    min_file_age_seconds skips files modified too recently. This prevents the
    offline daily cron from retaining its own still-being-written session JSON
    and avoids races with active Hermes writers.
    """
    count = 0
    cutoff_mtime_ns = None
    if min_file_age_seconds and min_file_age_seconds > 0:
        cutoff_mtime_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) - int(min_file_age_seconds * 1_000_000_000)
    for path in sorted(sessions_dir.glob("session_*.json")):
        stat = path.stat()
        if since_mtime_ns is not None and stat.st_mtime_ns <= since_mtime_ns:
            continue
        if cutoff_mtime_ns is not None and stat.st_mtime_ns > cutoff_mtime_ns:
            continue
        yield path
        count += 1
        if limit is not None and count >= limit:
            return


def build_manifest_from_json_dir(*, sessions_dir: Path = DEFAULT_SESSIONS_DIR, bank_target: str = DEFAULT_BANK_TARGET, max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS, retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE, limit: int | None = None, since_mtime_ns: int | None = None, min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS, source_profile: str = "default", include_markdown_artifacts: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_json_session_files(sessions_dir, limit=limit, since_mtime_ns=since_mtime_ns, min_file_age_seconds=min_file_age_seconds):
        try:
            records.extend(records_from_json_file(
                path,
                bank_target=bank_target,
                max_document_chars=max_document_chars,
                retain_chunk_size=retain_chunk_size,
                source_profile=source_profile,
                include_markdown_artifacts=include_markdown_artifacts,
                min_markdown_file_age_seconds=min_file_age_seconds,
            ))
        except Exception as e:
            session_id = path.stem[len("session_"):] if path.stem.startswith("session_") else path.stem
            source_profile_segment = profile_segment(source_profile)
            records.append({
                "document_id": session_document_base_id(session_id, source_profile),
                "bank_target": bank_target,
                "action": "manual_review",
                "reason": f"read_error:{type(e).__name__}:{e}",
                "content": "",
                "content_chars": 0,
                "estimated_retain_chunks": 0,
                "tags": profile_tags(source_profile),
                "metadata": {"source_kind": "hermes_json", "source_label": profile_source_label(source_profile), "source_profile": source_profile_segment, "json_path": str(path), "schema_version": SCHEMA_VERSION, "cleaning_version": CLEANING_VERSION, "candidate_filter_version": CANDIDATE_FILTER_VERSION},
                "context": "hermes_session",
                "update_mode": "replace",
            })
    return records


def _read_profile_memory_provider(profile_dir: Path) -> str:
    config_path = profile_dir / "config.yaml"
    if not config_path.exists():
        return ""
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        memory = data.get("memory") or {}
        return str(memory.get("provider") or "").strip().lower()
    except Exception:
        # Tiny fallback parser for the only field we need; avoids making the
        # manifest builder depend on PyYAML in minimal/script-only contexts.
        in_memory = False
        for raw in text.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw.startswith(" ") and stripped.endswith(":"):
                in_memory = stripped[:-1] == "memory"
                continue
            if in_memory and re.match(r"^\s+provider\s*:", line):
                return line.split(":", 1)[1].strip().strip("'\"").lower()
    return ""


def discover_session_sources(
    *,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    state_db: Path = DEFAULT_STATE_DB,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    profile_mode: str = "hindsight",
) -> list[dict[str, Any]]:
    """Discover default + profile session sources.

    profile_mode:
      - none: only the supplied/default sessions_dir and state_db
      - hindsight: add profiles whose config says memory.provider=hindsight
      - all: add every profile with a sessions/ directory
    """
    sources: list[dict[str, Any]] = [
        {"profile": "default", "sessions_dir": sessions_dir, "state_db": state_db, "provider": "default"}
    ]
    mode = (profile_mode or "hindsight").strip().lower()
    if mode in {"none", "off", "false", "0"} or not profile_root.exists():
        return sources
    for profile_dir in sorted(p for p in profile_root.iterdir() if p.is_dir()):
        profile = profile_dir.name
        sessions = profile_dir / "sessions"
        if not sessions.exists():
            continue
        provider = _read_profile_memory_provider(profile_dir)
        if mode == "hindsight" and provider != "hindsight":
            continue
        sources.append(
            {
                "profile": profile,
                "sessions_dir": sessions,
                "state_db": profile_dir / "state.db",
                "provider": provider,
            }
        )
    return sources


def build_manifest_from_session_sources(
    *,
    sources: list[dict[str, Any]],
    bank_target: str = DEFAULT_BANK_TARGET,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE,
    limit: int | None = None,
    since_mtime_ns: int | None = None,
    min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    remaining = limit
    for source in sources:
        if remaining is not None and remaining <= 0:
            break
        source_records = build_manifest_from_json_dir(
            sessions_dir=Path(source["sessions_dir"]),
            bank_target=bank_target,
            max_document_chars=max_document_chars,
            retain_chunk_size=retain_chunk_size,
            limit=remaining,
            since_mtime_ns=since_mtime_ns,
            min_file_age_seconds=min_file_age_seconds,
            source_profile=str(source.get("profile") or "default"),
        )
        records.extend(source_records)
        if remaining is not None:
            # Count input session files, not split chunks, by unique json_path.
            consumed = len({(r.get("metadata") or {}).get("json_path") for r in source_records})
            remaining -= consumed
    return records


def sqlite_summary(db_path: Path = DEFAULT_STATE_DB) -> dict[str, Any]:
    if not db_path.exists():
        return {"exists": False, "path": str(db_path)}
    out: dict[str, Any] = {"exists": True, "path": str(db_path)}
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        for table in ["sessions", "messages"]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                out[f"{table}_count"] = cur.fetchone()[0]
            except Exception as e:
                out[f"{table}_error"] = str(e)
        con.close()
    except Exception as e:
        out["error"] = str(e)
    return out


def empty_manifest_scan_state() -> dict[str, Any]:
    return {"schema_version": SCAN_STATE_VERSION, "sources": {}}


def load_manifest_scan_state(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return empty_manifest_scan_state()
    p = Path(path).expanduser()
    if not p.exists():
        return empty_manifest_scan_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return empty_manifest_scan_state()
    if not isinstance(data, dict):
        return empty_manifest_scan_state()
    data.setdefault("schema_version", SCAN_STATE_VERSION)
    sources = data.get("sources")
    if not isinstance(sources, dict):
        data["sources"] = {}
    return data


def scan_state_source_mtime_ns(scan_state: dict[str, Any], source_kind: str) -> int | None:
    sources = scan_state.get("sources") or {}
    if not isinstance(sources, dict):
        return None
    entry = sources.get(source_kind)
    if not isinstance(entry, dict):
        return None
    value = entry.get("max_source_mtime_ns")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def resolve_source_since_mtime_ns(
    *,
    explicit_source_since: int | None,
    global_since: int | None,
    scan_state: dict[str, Any],
    source_kind: str,
) -> int | None:
    if explicit_source_since is not None:
        return explicit_source_since
    if global_since is not None:
        return global_since
    return scan_state_source_mtime_ns(scan_state, source_kind)


def max_source_mtime_ns_by_kind(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        meta = rec.get("metadata") or {}
        source_kind = str(meta.get("source_kind") or "")
        if not source_kind:
            continue
        value = meta.get("source_mtime_ns")
        try:
            mtime_ns = int(value)
        except Exception:
            continue
        out[source_kind] = max(out.get(source_kind, 0), mtime_ns)
    return out


def save_manifest_scan_state(
    path: str | Path,
    scan_state: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    p = Path(path).expanduser()
    state = dict(scan_state or empty_manifest_scan_state())
    state["schema_version"] = SCAN_STATE_VERSION
    sources = dict(state.get("sources") or {})
    now = datetime.now(timezone.utc).isoformat()
    for source_kind, max_mtime_ns in max_source_mtime_ns_by_kind(records).items():
        prev = sources.get(source_kind) if isinstance(sources.get(source_kind), dict) else {}
        prev_mtime = 0
        try:
            prev_mtime = int(prev.get("max_source_mtime_ns") or 0)
        except Exception:
            prev_mtime = 0
        sources[source_kind] = {
            **prev,
            "max_source_mtime_ns": max(prev_mtime, max_mtime_ns),
            "updated_at": now,
            "last_manifest": str(manifest_path) if manifest_path else prev.get("last_manifest"),
        }
    state["sources"] = sources
    state["updated_at"] = now
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


MARKDOWN_SOURCE_KINDS = {"codex_markdown_artifact", "hermes_markdown_artifact", "kanban_prompt_markdown"}


def is_markdown_artifact_record(record: dict[str, Any]) -> bool:
    meta = record.get("metadata") or {}
    if str(meta.get("source_kind") or "") in MARKDOWN_SOURCE_KINDS:
        return True
    return "topic:markdown-artifact" in set(record.get("tags") or [])


def markdown_record_part_index(record: dict[str, Any]) -> int:
    try:
        return int((record.get("metadata") or {}).get("part_index") or 0)
    except Exception:
        return 0


def markdown_record_md5(record: dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    return str(meta.get("full_content_md5") or meta.get("source_file_md5") or "")


def dedupe_manifest_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dedupe manifest records before daily retain.

    Markdown artifacts may be discovered from several conversations (Codex
    rollout JSONL, Hermes/deepseek session JSON, and Kanban prompt files). Keep a
    single candidate per source file part and per raw markdown-content md5 part.
    Retain submit_state still performs the final content-hash idempotency check.
    """
    markdown_path_occurrences: dict[tuple[str, int], int] = {}
    markdown_md5_occurrences: dict[tuple[str, int], int] = {}
    for rec in records:
        if not is_markdown_artifact_record(rec):
            continue
        meta = rec.get("metadata") or {}
        part_index = markdown_record_part_index(rec)
        source_path = str(meta.get("source_path") or "")
        if source_path:
            key = (source_path, part_index)
            markdown_path_occurrences[key] = markdown_path_occurrences.get(key, 0) + 1
        content_md5 = markdown_record_md5(rec)
        if content_md5:
            key = (content_md5, part_index)
            markdown_md5_occurrences[key] = markdown_md5_occurrences.get(key, 0) + 1

    deduped: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, int]] = set()
    seen_md5: set[tuple[str, int]] = set()
    skipped_by_path = 0
    skipped_by_md5 = 0
    for rec in records:
        if not is_markdown_artifact_record(rec):
            deduped.append(rec)
            continue
        meta = rec.get("metadata") or {}
        part_index = markdown_record_part_index(rec)
        source_path = str(meta.get("source_path") or "")
        path_key = (source_path, part_index) if source_path else None
        content_md5 = markdown_record_md5(rec)
        md5_key = (content_md5, part_index) if content_md5 else None
        if path_key is not None and path_key in seen_paths:
            skipped_by_path += 1
            continue
        if md5_key is not None and md5_key in seen_md5:
            skipped_by_md5 += 1
            continue
        deduped.append(rec)
        if path_key is not None:
            seen_paths.add(path_key)
        if md5_key is not None:
            seen_md5.add(md5_key)

    duplicate_paths: dict[str, int] = {}
    for (path, _part), count in markdown_path_occurrences.items():
        if count > 1:
            duplicate_paths[path] = max(duplicate_paths.get(path, 0), count)
    duplicate_md5: dict[str, int] = {}
    for (digest, _part), count in markdown_md5_occurrences.items():
        if count > 1:
            duplicate_md5[digest] = max(duplicate_md5.get(digest, 0), count)
    diagnostics = {
        "input_records": len(records),
        "output_records": len(deduped),
        "skipped_markdown_duplicate_source_path": skipped_by_path,
        "skipped_markdown_duplicate_content_md5": skipped_by_md5,
        "duplicate_markdown_source_paths": dict(sorted(duplicate_paths.items())),
        "duplicate_markdown_content_md5": dict(sorted(duplicate_md5.items())),
    }
    return deduped, diagnostics


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    profile_counts: dict[str, int] = {}
    total_chars = 0
    total_chunks = 0
    for rec in records:
        action = rec.get("action", "unknown")
        reason = rec.get("reason", "unknown")
        by_action[action] = by_action.get(action, 0) + 1
        by_reason[f"{action}:{reason}"] = by_reason.get(f"{action}:{reason}", 0) + 1
        total_chars += int(rec.get("content_chars") or 0)
        total_chunks += int(rec.get("estimated_retain_chunks") or 0)
        source_profile = str((rec.get("metadata") or {}).get("source_profile") or "default")
        profile_counts[source_profile] = profile_counts.get(source_profile, 0) + 1
        for tag in rec.get("tags") or []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "by_action": by_action,
        "by_reason": dict(sorted(by_reason.items())),
        "total_content_chars": total_chars,
        "estimated_retain_chunks": total_chunks,
        "by_profile": dict(sorted(profile_counts.items())),
        "top_tags": sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30],
    }


def manifest_record_for_write(rec: dict[str, Any], *, include_content: bool) -> dict[str, Any]:
    out = dict(rec)
    if not include_content and "content" in out:
        out.pop("content", None)
        out["content_omitted"] = True
    return out


def write_manifest(records: list[dict[str, Any]], output_dir: Path, *, include_content: bool = False) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path = output_dir / f"{stamp}-session-manifest.jsonl"
    summary_path = output_dir / f"{stamp}-session-manifest-summary.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(manifest_record_for_write(rec, include_content=include_content), ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_records(records)
    summary["include_content"] = include_content
    latest_path = output_dir / "latest.json"
    paths = {"manifest": manifest_path, "summary": summary_path, "latest": latest_path}
    write_manifest_summary(paths, summary)
    return paths


def write_manifest_summary(paths: dict[str, Path], summary: dict[str, Any]) -> None:
    summary_path = Path(paths["summary"])
    latest_path = Path(paths["latest"])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(
        json.dumps({"manifest": str(paths["manifest"]), "summary": str(summary_path), "summary_data": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build dry-run session/json Hindsight manifest; no Hindsight writes.")
    ap.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    ap.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    ap.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    ap.add_argument("--codex-sessions-dir", type=Path, default=DEFAULT_CODEX_SESSIONS_DIR)
    ap.add_argument(
        "--profile-mode",
        choices=["hindsight", "all", "none"],
        default=os.environ.get("HINDSIGHT_SESSION_PROFILE_MODE", "hindsight"),
        help="Profile session scan mode: hindsight=include profiles whose memory.provider is hindsight (default); all=all profiles; none=default only.",
    )
    ap.add_argument("--bank-target", default=DEFAULT_BANK_TARGET)
    ap.add_argument("--max-document-chars", type=int, default=DEFAULT_MAX_DOCUMENT_CHARS)
    ap.add_argument("--retain-chunk-size", type=int, default=DEFAULT_RETAIN_CHUNK_SIZE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--codex-limit", type=int, default=None)
    ap.add_argument("--since-mtime-ns", type=int, default=None, help="Only scan session JSON files with filesystem mtime_ns greater than this value. Candidate-generation accelerator only; successful retain state is tracked by the retain runner.")
    ap.add_argument("--session-since-mtime-ns", type=int, default=None, help="Only scan Hermes session JSON files with filesystem mtime_ns greater than this value. Overrides --since-mtime-ns for Hermes JSON sessions.")
    ap.add_argument("--codex-since-mtime-ns", type=int, default=None, help="Only scan Codex rollout JSONL files with filesystem mtime_ns greater than this value. Overrides --since-mtime-ns for Codex rollout files.")
    ap.add_argument("--deepseek-since-mtime-ns", type=int, default=None, help="Only scan DeepSeek-TUI session JSON files with filesystem mtime_ns greater than this value. Overrides --since-mtime-ns for DeepSeek-TUI files.")
    ap.add_argument("--kanban-prompt-since-mtime-ns", type=int, default=None, help="Only scan Kanban prompt Markdown files with filesystem mtime_ns greater than this value. Overrides --since-mtime-ns for Kanban prompt Markdown.")
    ap.add_argument("--kanban-comment-since-mtime-ns", type=int, default=None, help="Only scan Kanban task comments with created_at greater than this mtime_ns-equivalent value. Overrides --since-mtime-ns for Kanban comments.")
    ap.add_argument("--scan-state", type=Path, default=None, help="Per-source candidate scan watermark state. Known sources are filtered by their own max source mtime; new sources are scanned fully once.")
    ap.add_argument("--write-scan-state", action="store_true", help="Advance --scan-state after this manifest build. Daily pipeline normally leaves this off and updates scan state only after retain succeeds.")
    ap.add_argument("--min-file-age-seconds", type=int, default=DEFAULT_MIN_FILE_AGE_SECONDS, help="Skip session JSON files modified more recently than this many seconds. Prevents retaining active/still-being-written sessions; set 0 to disable.")
    ap.add_argument("--include-codex", dest="include_codex", action="store_true", default=os.environ.get("HINDSIGHT_INCLUDE_CODEX", "1").lower() not in {"0", "false", "no", "off"}, help="Include Codex rollout JSONL sessions as first-party daily manifest records (default on).")
    ap.add_argument("--no-include-codex", dest="include_codex", action="store_false", help="Disable Codex rollout JSONL scanning for this manifest build.")
    ap.add_argument("--include-codex-markdown", dest="include_codex_markdown", action="store_true", default=os.environ.get("HINDSIGHT_INCLUDE_CODEX_MARKDOWN", "1").lower() not in {"0", "false", "no", "off"}, help="Include Markdown artifacts that Codex sessions show as generated/written (default on).")
    ap.add_argument("--no-include-codex-markdown", dest="include_codex_markdown", action="store_false", help="Disable Codex-generated Markdown artifact records.")
    ap.add_argument("--deepseek-sessions-dir", type=Path, default=DEFAULT_DEEPSEEK_SESSIONS_DIR)
    ap.add_argument("--deepseek-limit", type=int, default=None)
    ap.add_argument("--include-deepseek", dest="include_deepseek", action="store_true", default=os.environ.get("HINDSIGHT_INCLUDE_DEEPSEEK", "1").lower() not in {"0", "false", "no", "off"}, help="Include DeepSeek-TUI session JSON files as daily manifest records (default on).")
    ap.add_argument("--no-include-deepseek", dest="include_deepseek", action="store_false", help="Disable DeepSeek-TUI session scanning.")
    ap.add_argument("--include-deepseek-markdown", dest="include_deepseek_markdown", action="store_true", default=os.environ.get("HINDSIGHT_INCLUDE_DEEPSEEK_MARKDOWN", "1").lower() not in {"0", "false", "no", "off"}, help="Include Markdown artifacts that DeepSeek-TUI sessions show as written (default on).")
    ap.add_argument("--no-include-deepseek-markdown", dest="include_deepseek_markdown", action="store_false", help="Disable DeepSeek-TUI Markdown artifact records.")
    ap.add_argument("--include-kanban", dest="include_kanban", action="store_true", default=os.environ.get("HINDSIGHT_INCLUDE_KANBAN", "1").lower() not in {"0", "false", "no", "off"}, help="Include Hermes Kanban prompt markdown and task comments in the daily manifest (default on).")
    ap.add_argument("--no-include-kanban", dest="include_kanban", action="store_false", help="Disable Kanban prompt/comment scanning for this manifest build.")
    ap.add_argument("--kanban-workspace-root", type=Path, action="append", default=None, help="Root to scan for .codex-kanban/.deepseek-kanban prompt markdown. Repeatable; default ~/code.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--include-content", action="store_true", help="Write cleaned full content into manifest JSONL. Default omits content and keeps hashes/pointers only.")
    ap.add_argument("--json", action="store_true", help="Print JSON summary")
    args = ap.parse_args(argv)
    scan_state = load_manifest_scan_state(args.scan_state)
    session_since_mtime_ns = resolve_source_since_mtime_ns(
        explicit_source_since=args.session_since_mtime_ns,
        global_since=args.since_mtime_ns,
        scan_state=scan_state,
        source_kind="hermes_json",
    )
    codex_since_mtime_ns = resolve_source_since_mtime_ns(
        explicit_source_since=args.codex_since_mtime_ns,
        global_since=args.since_mtime_ns,
        scan_state=scan_state,
        source_kind="codex_rollout_jsonl",
    )
    deepseek_since_mtime_ns = resolve_source_since_mtime_ns(
        explicit_source_since=args.deepseek_since_mtime_ns,
        global_since=args.since_mtime_ns,
        scan_state=scan_state,
        source_kind="deepseek_session_json",
    )
    kanban_prompt_since_mtime_ns = resolve_source_since_mtime_ns(
        explicit_source_since=args.kanban_prompt_since_mtime_ns,
        global_since=args.since_mtime_ns,
        scan_state=scan_state,
        source_kind="kanban_prompt_markdown",
    )
    kanban_comment_since_mtime_ns = resolve_source_since_mtime_ns(
        explicit_source_since=args.kanban_comment_since_mtime_ns,
        global_since=args.since_mtime_ns,
        scan_state=scan_state,
        source_kind="kanban_task_comment",
    )

    sources = discover_session_sources(
        sessions_dir=args.sessions_dir,
        state_db=args.state_db,
        profile_root=args.profile_root,
        profile_mode=args.profile_mode,
    )
    records = build_manifest_from_session_sources(
        sources=sources,
        bank_target=args.bank_target,
        max_document_chars=args.max_document_chars,
        retain_chunk_size=args.retain_chunk_size,
        limit=args.limit,
        since_mtime_ns=session_since_mtime_ns,
        min_file_age_seconds=args.min_file_age_seconds,
    )
    if args.include_codex:
        records.extend(build_manifest_from_codex_dir(
            codex_sessions_dir=args.codex_sessions_dir,
            bank_target=args.bank_target,
            max_document_chars=args.max_document_chars,
            retain_chunk_size=args.retain_chunk_size,
            limit=args.codex_limit,
            since_mtime_ns=codex_since_mtime_ns,
            min_file_age_seconds=args.min_file_age_seconds,
            include_markdown_artifacts=args.include_codex_markdown,
        ))
    if getattr(args, "include_deepseek", True):
        records.extend(build_manifest_from_deepseek_dir(
            deepseek_sessions_dir=getattr(args, "deepseek_sessions_dir", DEFAULT_DEEPSEEK_SESSIONS_DIR),
            bank_target=args.bank_target,
            max_document_chars=args.max_document_chars,
            retain_chunk_size=args.retain_chunk_size,
            limit=getattr(args, "deepseek_limit", None),
            since_mtime_ns=deepseek_since_mtime_ns,
            min_file_age_seconds=args.min_file_age_seconds,
            include_markdown_artifacts=getattr(args, "include_deepseek_markdown", True),
        ))
    if args.include_kanban:
        records.extend(build_manifest_from_kanban_sources(
            workspace_roots=args.kanban_workspace_root or DEFAULT_KANBAN_WORKSPACE_ROOTS,
            board_db_paths=None,
            bank_target=args.bank_target,
            retain_chunk_size=args.retain_chunk_size,
            prompt_since_mtime_ns=kanban_prompt_since_mtime_ns,
            comment_since_mtime_ns=kanban_comment_since_mtime_ns,
            min_file_age_seconds=args.min_file_age_seconds,
        ))
    records, dedupe_diagnostics = dedupe_manifest_records(records)
    paths = write_manifest(records, args.output_dir, include_content=args.include_content)
    saved_scan_state = None
    if args.scan_state and args.write_scan_state:
        saved_scan_state = save_manifest_scan_state(args.scan_state, scan_state, records, manifest_path=paths["manifest"])
    summary = summarize_records(records)
    summary["dedupe"] = dedupe_diagnostics
    summary["scan_state"] = {
        "path": str(args.scan_state) if args.scan_state else None,
        "enabled": bool(args.scan_state),
        "write_enabled": bool(args.write_scan_state),
        "effective_since_mtime_ns": {
            "hermes_json": session_since_mtime_ns,
            "codex_rollout_jsonl": codex_since_mtime_ns,
            "deepseek_session_json": deepseek_since_mtime_ns,
            "kanban_prompt_markdown": kanban_prompt_since_mtime_ns,
            "kanban_task_comment": kanban_comment_since_mtime_ns,
        },
        "saved_sources": sorted((saved_scan_state or {}).get("sources", {}).keys()) if saved_scan_state else [],
    }
    summary["filters"] = {
        "min_file_age_seconds": args.min_file_age_seconds,
        "since_mtime_ns": args.since_mtime_ns,
        "session_since_mtime_ns": session_since_mtime_ns,
        "codex_since_mtime_ns": codex_since_mtime_ns,
        "deepseek_since_mtime_ns": deepseek_since_mtime_ns,
        "kanban_prompt_since_mtime_ns": kanban_prompt_since_mtime_ns,
        "kanban_comment_since_mtime_ns": kanban_comment_since_mtime_ns,
        "profile_mode": args.profile_mode,
        "include_codex": args.include_codex,
        "codex_sessions_dir": str(args.codex_sessions_dir),
        "codex_limit": args.codex_limit,
        "include_codex_markdown": args.include_codex_markdown,
        "include_deepseek": getattr(args, "include_deepseek", True),
        "deepseek_sessions_dir": str(getattr(args, "deepseek_sessions_dir", DEFAULT_DEEPSEEK_SESSIONS_DIR)),
        "deepseek_limit": getattr(args, "deepseek_limit", None),
        "include_deepseek_markdown": getattr(args, "include_deepseek_markdown", True),
        "include_kanban": args.include_kanban,
        "kanban_workspace_roots": [str(p) for p in (args.kanban_workspace_root or DEFAULT_KANBAN_WORKSPACE_ROOTS)],
    }
    summary["paths"] = {k: str(v) for k, v in paths.items()}
    summary["sources"] = [
        {
            "profile": str(s.get("profile")),
            "sessions_dir": str(s.get("sessions_dir")),
            "state_db": str(s.get("state_db")),
            "provider": str(s.get("provider") or ""),
        }
        for s in sources
    ]
    if args.include_codex:
        summary["sources"].append({
            "profile": "codex",
            "sessions_dir": str(args.codex_sessions_dir),
            "state_db": "",
            "provider": "codex-rollout-jsonl",
        })
    if getattr(args, "include_deepseek", True):
        summary["sources"].append({
            "profile": "deepseek",
            "sessions_dir": str(getattr(args, "deepseek_sessions_dir", DEFAULT_DEEPSEEK_SESSIONS_DIR)),
            "state_db": "",
            "provider": "deepseek-tui-session-json",
        })
    sqlite_sources = {str(s.get("profile")): sqlite_summary(Path(s["state_db"])) for s in sources}
    summary["sqlite"] = sqlite_sources.get("default", sqlite_summary(args.state_db))
    summary["sqlite_sources"] = sqlite_sources
    write_manifest_summary(paths, summary)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"records={summary['records']} actions={summary['by_action']} profiles={summary.get('by_profile', {})} chars={summary['total_content_chars']} chunks={summary['estimated_retain_chunks']}")
        print(f"manifest={paths['manifest']}")
        print(f"summary={paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
