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
DEFAULT_OUTPUT_DIR = DEFAULT_HERMES_HOME / "hindsight" / "session_ingest" / "manifests"
DEFAULT_BANK_TARGET = "hermes_v3"
DEFAULT_MAX_DOCUMENT_CHARS = 120_000
DEFAULT_RETAIN_CHUNK_SIZE = 8000
DEFAULT_MIN_FILE_AGE_SECONDS = int(os.environ.get("HINDSIGHT_SESSION_MANIFEST_MIN_FILE_AGE_SECONDS", "900"))
MIN_CONTENT_CHARS = 30
SCHEMA_VERSION = "session-retain-v3"
CLEANING_VERSION = "deterministic-clean-v2"
CANDIDATE_FILTER_VERSION = "lightweight-candidate-filter-v3"

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
    ("topic:native-consolidation", ["native consolidation", "observations", "consolidation", "observation_scopes", "enable_observations"], ["domain:hindsight"]),
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
    }


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


def observation_scopes_for_tags(tags: Iterable[str]) -> list[list[str]]:
    scopes: list[list[str]] = []
    for tag in tags:
        if tag.startswith(("domain:", "project:", "topic:")):
            scopes.append([tag])
    return scopes


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
        "observation_scopes": observation_scopes_for_tags(tags),
        "metadata": metadata,
        "context": "hermes_session",
        "update_mode": "replace",
    }


def records_from_json_file(path: str | Path, *, bank_target: str = DEFAULT_BANK_TARGET, max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS, retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE, source_profile: str = "default") -> list[dict[str, Any]]:
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
            "observation_scopes": [],
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


def build_manifest_from_json_dir(*, sessions_dir: Path = DEFAULT_SESSIONS_DIR, bank_target: str = DEFAULT_BANK_TARGET, max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS, retain_chunk_size: int = DEFAULT_RETAIN_CHUNK_SIZE, limit: int | None = None, since_mtime_ns: int | None = None, min_file_age_seconds: int = DEFAULT_MIN_FILE_AGE_SECONDS, source_profile: str = "default") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_json_session_files(sessions_dir, limit=limit, since_mtime_ns=since_mtime_ns, min_file_age_seconds=min_file_age_seconds):
        try:
            records.extend(records_from_json_file(path, bank_target=bank_target, max_document_chars=max_document_chars, retain_chunk_size=retain_chunk_size, source_profile=source_profile))
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
                "observation_scopes": [],
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
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_path = output_dir / "latest.json"
    latest_path.write_text(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path), "summary_data": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "summary": summary_path, "latest": latest_path}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build dry-run session/json Hindsight manifest; no Hindsight writes.")
    ap.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    ap.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    ap.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
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
    ap.add_argument("--since-mtime-ns", type=int, default=None, help="Only scan session JSON files with filesystem mtime_ns greater than this value. Candidate-generation accelerator only; successful retain state is tracked by the retain runner.")
    ap.add_argument("--min-file-age-seconds", type=int, default=DEFAULT_MIN_FILE_AGE_SECONDS, help="Skip session JSON files modified more recently than this many seconds. Prevents retaining active/still-being-written sessions; set 0 to disable.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--include-content", action="store_true", help="Write cleaned full content into manifest JSONL. Default omits content and keeps hashes/pointers only.")
    ap.add_argument("--json", action="store_true", help="Print JSON summary")
    args = ap.parse_args(argv)

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
        since_mtime_ns=args.since_mtime_ns,
        min_file_age_seconds=args.min_file_age_seconds,
    )
    paths = write_manifest(records, args.output_dir, include_content=args.include_content)
    summary = summarize_records(records)
    summary["filters"] = {
        "min_file_age_seconds": args.min_file_age_seconds,
        "since_mtime_ns": args.since_mtime_ns,
        "profile_mode": args.profile_mode,
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
    sqlite_sources = {str(s.get("profile")): sqlite_summary(Path(s["state_db"])) for s in sources}
    summary["sqlite"] = sqlite_sources.get("default", sqlite_summary(args.state_db))
    summary["sqlite_sources"] = sqlite_sources
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"records={summary['records']} actions={summary['by_action']} profiles={summary.get('by_profile', {})} chars={summary['total_content_chars']} chunks={summary['estimated_retain_chunks']}")
        print(f"manifest={paths['manifest']}")
        print(f"summary={paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
