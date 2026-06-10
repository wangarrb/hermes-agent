#!/usr/bin/env python3
"""DEPRECATED: SQLite day-topic import helper.

Production Hindsight ingestion for this environment must use the session/json
manifest route and native Hindsight retain APIs. This script is kept only for
historical audit/debug dry-runs. Submit mode is blocked by default because the
SQLite day-topic route breaks session boundaries and creates bad tag/scope
pollution (`hermes/sqlite/incremental`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

HINDSIGHT_API = "http://127.0.0.1:8888"
DEFAULT_BANK_ID = "hermes"
DEFAULT_DB_PATH = Path.home() / ".hermes" / "state.db"
DEFAULT_PROGRESS_FILE = Path.home() / ".hermes" / "hindsight" / "sqlite_import_progress.json"
DEFAULT_HINDSIGHT_CONFIG_FILE = Path.home() / ".hermes" / "hindsight" / "config.json"
DEPRECATED_SQLITE_IMPORT_MESSAGE = (
    "SQLite day-topic submit is deprecated/blocked. Use session/json native route: "
    "hindsight_session_manifest.py + hindsight_session_retain_runner.py "
    "or hindsight_minimax_import.py session-manifest-retain-llm."
)

MIN_CONTENT_CHARS = 30
REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
BACKOFF_SECONDS = [5, 15, 30, 60]
RATE_LIMIT_BACKOFF_SECONDS = 300
MAX_BUNDLE_CHARS = 120000
DEFAULT_RETAIN_CHUNK_SIZE = 8000
DEFAULT_RETAIN_EXTRACTION_MODE = "concise"

TOPIC_KEYWORDS = {
    "egomotion4d": [
        "egomotion4d", "trackingworld", "any4d", "dage", "pi3x", "roma2",
        "gtsam", "ate", "rpe", "unidepth", "metricdepth", "trajectory",
        "轨迹", "尺度", "位姿", "前端", "bundle adjustment",
    ],
    "hermes": [
        "hermes", "hindsight", "cch", "gateway", "provider", "memory",
        "mycompress", "/mycompress", "minimax", "openrouter", "claude code",
        "codex", "skills", "toolsets",
    ],
    "openclaw": ["openclaw", "clawhub"],
    "paper": ["paper", "arxiv", "论文", "pdf", "semanticscholar", "scholar"],
}

# 非 LLM 初筛：只在本地用确定性规则判断“是否值得送进 Hindsight retain”。
# 目标不是总结，而是减少 full 导入时的低价值/重复内容进入付费 retain。
VALUE_SIGNAL_PATTERNS: dict[str, tuple[int, list[str]]] = {
    "user_pref_or_rule": (
        5,
        [
            r"用户偏好", r"用户要求", r"默认", r"必须", r"不要", r"记住", r"规则", r"偏好", r"教训",
            r"以后", r"以后.*回答", r"先给结论", r"别啰嗦", r"关键风险", r"说清楚",
        ],
    ),
    "decision_or_conclusion": (4, [r"结论", r"决定", r"方案", r"推荐", r"不建议", r"取舍", r"原因", r"根因", r"风险"]),
    "verified_result": (4, [r"已验证", r"验证", r"通过", r"失败", r"failed", r"passed", r"error", r"exception", r"traceback", r"修复", r"bug"]),
    "project_or_system": (3, [r"egomotion4d", r"hindsight", r"hermes", r"minimax", r"ollama", r"sqlite", r"postgresql", r"docker", r"gtsam", r"pi3x", r"dage", r"openclaw"]),
    "code_or_config": (3, [r"```", r"\bpython3?\b", r"\bcurl\b", r"\bdocker\b", r"\bpsql\b", r"select\s+.+\s+from", r"(?:^|[\s`'\"])(?:~?/|/)[\w./~+-]*[A-Za-z_.-][\w./~+-]*", r"\.(py|md|json|ya?ml|toml|sql|txt)\b"]),
    "numeric_evidence": (2, [r"\b\d+(?:\.\d+)?%?\b", r"ate", r"rpe", r"token", r"chunks?", r"calls?", r"调用", r"次数"]),
}

# 双本地模型兜底后，确定性规则可以更严格地先丢掉“只有过程、没有结论”的噪声。
# 注意：这不是硬删技术内容；只在没有 durable signal 时触发。
TRANSIENT_PROGRESS_PATTERNS = [
    r"\[\s*\d+\s*/\s*\d+\s*\]",
    r"\bsubmitting bundles\b",
    r"\bpending\s*=\s*\d+\b",
    r"\bprocessed\s*[:=]\s*\d+\b",
    r"\bfailed\s*[:=]\s*\d+\b",
    r"继续等待", r"当前进度", r"正在处理", r"已处理\s*[:：]\s*\d+",
    r"\bloading\s+\d+\s*/\s*\d+\b",
]

DURABLE_SIGNAL_PATTERNS = [
    r"用户偏好", r"用户要求", r"默认", r"必须", r"不要", r"记住", r"教训", r"以后.*回答",
    r"结论", r"决定", r"方案", r"推荐", r"不建议", r"取舍", r"根因", r"风险",
    r"已验证", r"验证", r"通过", r"失败", r"error", r"exception", r"traceback", r"修复", r"bug",
    r"egomotion4d", r"hindsight", r"hermes", r"openclaw", r"ollama", r"gtsam", r"pi3x", r"dage",
    r"\bpython3?\b", r"\bcurl\b", r"\bdocker\b", r"\bpsql\b", r"(?:^|[\s`'\"])(?:~?/|/)[\w./~+-]*[A-Za-z_.-][\w./~+-]*", r"\.(py|md|json|ya?ml|toml|sql|txt)\b",
]

LOW_VALUE_SHORT_PATTERNS = [
    r"^好[的嘞]?[。！!\s]*$",
    r"^收到[。！!\s]*$",
    r"^谢谢[。！!\s]*$",
    r"^继续[。！!\s]*$",
    r"^可以[。！!\s]*$",
    r"^ok[。！!\s]*$",
]

# 硬噪声：无论 --prefilter 是否开启，都不能送入 Hindsight retain。
# 这类内容是系统循环/工具包装，不是用户长期记忆；如果放行，容易让单个循环会话
# 生成成千上万条 facts 并污染 recall。
HEARTBEAT_NOISE_MARKERS = [
    "read heartbeat.md if it exists",
    "workspace/heartbeat.md",
    "do not read docs/heartbeat.md",
    "heartbeat_ok",
    "# heartbeat.md",
    "this is a heartbeat prompt",
    "the user has sent another heartbeat prompt",
    "multiple heartbeat prompts",
    "nothing needs attention",
    "recent images",
    "inbound directory",
    "心跳由消息触发",
    "检查钉钉最新图片",
]

INJECTED_CONTEXT_NOISE_MARKERS = [
    "# memory.md - 长期记忆",
    "# user.md - 用户画像",
    "## 用户画像",
    "## 技术知识库",
    "this file is yours to evolve",
    "you are a cli ai agent",
]

PREFILTER_DEFAULT_THRESHOLDS = {
    "none": -10**9,
    "safe": 1,       # 只去掉明显无价值短对话/压缩重复块
    "balanced": 7,   # 双模型兜底后的推荐档：项目/决策/错误/配置/偏好信号更明确才保留
    "strict": 12,    # 高强度降调用：只保留高密度技术/偏好/决策内容
}


@dataclass
class PrefilterDecision:
    keep: bool
    score: int
    reason: str


@dataclass
class FilterStats:
    messages_seen: int = 0
    messages_kept: int = 0
    messages_dropped: int = 0
    rule_kept: int = 0
    rule_dropped: int = 0
    local_model_calls: int = 0
    backup_model_calls: int = 0
    model_kept: int = 0
    model_dropped: int = 0
    model_errors: int = 0
    reasons: dict[str, int] | None = None
    kept_samples: list[dict[str, Any]] | None = None
    dropped_samples: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = {}
        if self.kept_samples is None:
            self.kept_samples = []
        if self.dropped_samples is None:
            self.dropped_samples = []

    def add_reason(self, reason: str) -> None:
        key = (reason or "unknown").split(";", 1)[0].split(",", 1)[0][:80]
        self.reasons[key] = self.reasons.get(key, 0) + 1


class LocalFilter:
    def decide(self, text: str, role: str, score: int, reasons: list[str]) -> PrefilterDecision:
        return PrefilterDecision(True, score, "local_filter_disabled")

    def decide_many(self, candidates: list[tuple[str, str, int, list[str]]]) -> list[PrefilterDecision]:
        return [self.decide(text, role, score, reasons) for text, role, score, reasons in candidates]

    def close(self) -> None:
        return None


class OllamaLocalFilter(LocalFilter):
    """串行调用本地 Ollama 模型，只用于复核规则准备丢弃的灰区内容。"""

    SYSTEM_PROMPT = (
        "你是离线记忆导入的严格初筛复核器。只输出严格 JSON，不要 Markdown，不要解释。"
        "字段：keep(boolean), priority('high'|'medium'|'low'|'drop'), confidence(0到1), reason(string<=40字)。"
        "保留：稳定事实、用户偏好、项目决策、验证结果、工具/环境长期经验、可复现实验数值、路径/配置/端口/命令。"
        "丢弃：寒暄、短确认、一次性进度、压缩handoff、天气/临时查询、纯日志噪声、无结论的运行输出。"
        "若只是当前任务进度且无可复用结论，应丢弃；若包含错误根因、验证结论、长期配置或用户偏好，应保留。"
        "原则：宁可保留可疑技术信息，但不要保留纯过程日志。"
    )

    def __init__(
        self,
        model: str,
        backup_model: str | None = None,
        api: str = "http://127.0.0.1:11434",
        drop_policy: str = "consensus",
        max_chars: int = 2400,
        timeout: int = 60,
        max_calls: int = 300,
        stats: FilterStats | None = None,
    ) -> None:
        self.model = model
        self.backup_model = backup_model
        self.api = api.rstrip("/")
        self.drop_policy = drop_policy
        self.max_chars = max_chars
        self.timeout = timeout
        self.max_calls = max_calls
        self.stats = stats

    def _unload(self, model: str | None) -> None:
        if not model:
            return
        try:
            requests.post(
                f"{self.api}/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass

    def _call(self, model: str, text: str, role: str, score: int, reasons: list[str]) -> PrefilterDecision:
        if self.stats:
            self.stats.local_model_calls += 1
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"role={role}\nrule_score={score}\nrule_reasons={','.join(reasons)}\n"
                        "待复核文本：\n" + text[: self.max_chars]
                    ),
                },
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 128, "num_ctx": 4096},
            "keep_alive": "5m",
        }
        try:
            resp = requests.post(f"{self.api}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
        except Exception as e:
            if self.stats:
                self.stats.model_errors += 1
            return PrefilterDecision(True, score, f"model_error_keep:{type(e).__name__}")

        try:
            match = re.search(r"\{.*\}", content, flags=re.S)
            obj = json.loads(match.group(0) if match else content)
            priority = str(obj.get("priority", "")).lower()
            keep_val = obj.get("keep")
            keep = bool(keep_val) if isinstance(keep_val, bool) else priority not in {"drop", "low"}
            conf = float(obj.get("confidence", 0.0) or 0.0)
            reason = str(obj.get("reason", "model_decision"))[:80]
            return PrefilterDecision(keep, score, f"{model}:keep={keep}:priority={priority}:conf={conf:.2f}:{reason}")
        except Exception as e:
            if self.stats:
                self.stats.model_errors += 1
            return PrefilterDecision(True, score, f"model_parse_error_keep:{type(e).__name__}")

    def _cap_reached_decision(self, score: int) -> PrefilterDecision:
        if self.stats:
            self.stats.model_errors += 1
        return PrefilterDecision(True, score, "local_filter_call_cap_keep")

    def _can_call(self) -> bool:
        return not (self.max_calls > 0 and self.stats and self.stats.local_model_calls >= self.max_calls)

    def decide_many(self, candidates: list[tuple[str, str, int, list[str]]]) -> list[PrefilterDecision]:
        """批量复核，避免 primary/backup 在多条候选之间反复加载/卸载。"""
        results: list[PrefilterDecision | None] = [None] * len(candidates)
        primary_drop_indices: list[int] = []

        # 第一遍：primary 连续处理所有候选。只把 primary 判 drop 的项目交给 backup。
        for idx, (text, role, score, reasons) in enumerate(candidates):
            if not self._can_call():
                results[idx] = self._cap_reached_decision(score)
                continue
            primary = self._call(self.model, text, role, score, reasons)
            if primary.keep:
                if self.stats:
                    self.stats.model_kept += 1
                results[idx] = primary
            elif not self.backup_model or self.drop_policy == "single":
                if self.stats:
                    self.stats.model_dropped += 1
                results[idx] = primary
            else:
                primary_drop_indices.append(idx)
                results[idx] = primary

        if self.backup_model and self.drop_policy != "single" and primary_drop_indices:
            # 避免双模型同时占显存：primary 全部跑完后只卸载一次，再连续跑 backup。
            self._unload(self.model)
            for idx in primary_drop_indices:
                text, role, score, reasons = candidates[idx]
                if not self._can_call():
                    results[idx] = self._cap_reached_decision(score)
                    continue
                if self.stats:
                    self.stats.backup_model_calls += 1
                backup = self._call(self.backup_model, text, role, score, reasons)
                primary = results[idx]
                if backup.keep:
                    if self.stats:
                        self.stats.model_kept += 1
                    results[idx] = PrefilterDecision(True, score, f"backup_rescue:{backup.reason}")
                else:
                    if self.stats:
                        self.stats.model_dropped += 1
                    primary_reason = primary.reason if isinstance(primary, PrefilterDecision) else "primary_drop"
                    results[idx] = PrefilterDecision(False, score, f"consensus_drop:{primary_reason}|{backup.reason}")
            self._unload(self.backup_model)

        return [r if isinstance(r, PrefilterDecision) else PrefilterDecision(True, candidates[i][2], "local_filter_internal_keep") for i, r in enumerate(results)]

    def decide(self, text: str, role: str, score: int, reasons: list[str]) -> PrefilterDecision:
        return self.decide_many([(text, role, score, reasons)])[0]

    def close(self) -> None:
        self._unload(self.model)
        self._unload(self.backup_model)


@dataclass
class SessionRecord:
    session_id: str
    started_at: float
    ended_at: float | None
    first_message_at: float  # 本次导入内容中的最早消息时间；增量导入时可晚于 session started_at
    last_message_at: float  # 本次导入内容中的最晚消息时间
    source: str
    model: str
    title: str | None
    message_count: int
    content: str
    day: str
    topic: str


@dataclass
class Bundle:
    index: int
    group_key: str
    topic: str
    start: str
    end: str
    records: list[SessionRecord]
    content: str
    document_id: str


def sanitize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\x00", "")


def extract_structured_content_text(content: Any) -> str:
    """从 Hermes/OpenAI structured content 中只提取真正展示给用户的正文。

    旧 SQLite 里有些 content 是 JSON 字符串，里面混着 thinking/toolCall/toolResult。
    这些不是长期记忆正文，必须在导入 Hindsight 前剔除。
    """
    parts: list[str] = []

    def add_text(value: Any) -> None:
        if isinstance(value, str):
            cleaned = sanitize_text(value).strip()
            if cleaned:
                parts.append(cleaned)
        elif isinstance(value, (list, dict)):
            nested = extract_structured_content_text(value)
            if nested:
                parts.append(nested)

    def walk(value: Any) -> None:
        if isinstance(value, str):
            add_text(value)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if not isinstance(value, dict):
            return

        typ = str(value.get("type") or "").lower().replace("_", "")
        if typ in {
            "thinking", "reasoning", "toolcall", "toolresult", "functioncall", "functionresult",
            "tooluse", "tooloutput", "inputimage", "image", "audio",
        }:
            return

        # 常见正文字段；不要兜底遍历所有字段，否则会把 metadata / tool args 再吃进来。
        for key in ("text", "content", "message"):
            if key in value:
                add_text(value.get(key))

    walk(content)
    return sanitize_text("\n".join(parts)).strip()


def extract_message_text(msg: dict[str, Any]) -> str:
    """提取单条消息的文本内容（只保留 user/assistant 主文本）。"""
    role = msg.get("role")
    if role not in {"user", "assistant"}:
        return ""

    content = msg.get("content", "")
    if isinstance(content, str):
        raw = sanitize_text(content).strip()
        parsed = None
        if raw and raw[0] in "[{":
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
        text = extract_structured_content_text(parsed) if parsed is not None else raw
    elif isinstance(content, (list, dict)):
        text = extract_structured_content_text(content)
    else:
        text = ""

    if not text:
        return ""

    prefix = "User" if role == "user" else "Assistant"
    return f"{prefix}: {text}"


def normalize_for_prefilter(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_compression_handoff_noise(text: str) -> bool:
    low = normalize_for_prefilter(text)
    return (
        "context compaction" in low
        or "earlier turns were compacted" in low
        or "reference only" in low and "active task" in low
    )


def is_tool_wrapper_noise(text: str) -> bool:
    low = normalize_for_prefilter(text)
    if not low.startswith("assistant:"):
        return False
    if "[toolcall]" in low or "[toolresult]" in low:
        return True
    if low in {"assistant: (no output)", "assistant: [thinking]", "assistant: []"}:
        return True
    if low.startswith("assistant: [thinking]") and len(low) < 120:
        return True
    if low.startswith("assistant: [{") and ('"type":"toolcall"' in low or '"type":"thinking"' in low or "'type': 'toolcall'" in low or "'type': 'thinking'" in low):
        return True
    return False


def heartbeat_marker_count(text: str) -> int:
    low = normalize_for_prefilter(text)
    return sum(1 for marker in HEARTBEAT_NOISE_MARKERS if marker in low)


def is_heartbeat_loop_noise(text: str) -> bool:
    """识别 OpenClaw/Hermes 心跳循环提示与 HEARTBEAT_OK 包装。

    只靠一个词“heartbeat”太宽，会误伤关于 Hindsight 的真实排障记录；这里要求
    明确的 HEARTBEAT.md 指令/路径/HEARTBEAT_OK 等组合信号。
    """
    low = normalize_for_prefilter(text)
    if not low:
        return False
    if heartbeat_marker_count(low) >= 2:
        return True
    if "read heartbeat.md if it exists" in low:
        return True
    if "heartbeat_ok" in low:
        return True
    if "heartbeat.md" in low and ("loop" in low or "stuck" in low or "infinite" in low or "hours" in low or "last 5 minutes" in low or "recent images" in low or "new images" in low or "inbound directory" in low or "nothing to process" in low or "follow it strictly" in low):
        return True
    if "this is a heartbeat prompt" in low or "this is another heartbeat prompt" in low:
        return True
    if "heartbeat_ok" in low and ("nothing needs attention" in low or "heartbeat prompt" in low or "reply" in low or "loop" in low or "glm-5" in low or "stuck" in low):
        return True
    if "heartbeat.md" in low and ("heartbeat_ok" in low or "check" in low or "anything that needs attention" in low):
        return True
    if "heartbeat prompt" in low and ("heartbeat_ok" in low or "heartbeat.md" in low):
        return True
    if "heartbeat.md" in low and "inbound directory" in low and "recent images" in low:
        return True
    if re.fullmatch(r"(?:assistant:\s*)?(?:\[thinking\]\s*)?heartbeat_ok[。.!\s]*", low):
        return True
    return False


def is_injected_context_noise(text: str) -> bool:
    """识别被当成 assistant 正文写入 SQLite 的系统上下文/长期记忆快照。"""
    low = normalize_for_prefilter(text)
    if not low:
        return False
    hits = sum(1 for marker in INJECTED_CONTEXT_NOISE_MARKERS if marker in low)
    if hits >= 2 and "user:" not in low[:500]:
        return True
    if low.startswith("assistant: # memory.md - 长期记忆"):
        return True
    return False


def is_hard_import_noise(text: str) -> bool:
    return (
        is_compression_handoff_noise(text)
        or is_tool_wrapper_noise(text)
        or is_heartbeat_loop_noise(text)
        or is_injected_context_noise(text)
    )


def matches_any(patterns: list[str], text: str) -> bool:
    for p in patterns:
        try:
            if re.search(p, text, flags=re.IGNORECASE):
                return True
        except re.error:
            if p.lower() in text:
                return True
    return False


def is_transient_progress_noise(text: str) -> bool:
    low = normalize_for_prefilter(text)
    return matches_any(TRANSIENT_PROGRESS_PATTERNS, low) and not matches_any(DURABLE_SIGNAL_PATTERNS, low)


def value_score(content: str, title: str | None = None) -> tuple[int, list[str]]:
    """本地确定性价值评分。只用于初筛，不替代 LLM retain。"""
    text = normalize_for_prefilter("\n".join(x for x in [title or "", content] if x))
    if not text:
        return 0, ["empty"]
    if is_hard_import_noise(text):
        return -50, ["hard_import_noise"]
    if len(text) < 80 and any(re.search(p, text, flags=re.IGNORECASE) for p in LOW_VALUE_SHORT_PATTERNS):
        return -10, ["low_value_short_ack"]
    if is_transient_progress_noise(text):
        return -8, ["transient_progress_noise"]

    score = 0
    reasons: list[str] = []
    for name, (weight, patterns) in VALUE_SIGNAL_PATTERNS.items():
        hit_count = 0
        for p in patterns:
            try:
                if re.search(p, text, flags=re.IGNORECASE):
                    hit_count += 1
            except re.error:
                if p.lower() in text:
                    hit_count += 1
        if hit_count:
            # 同一类命中多次只给温和加分，避免长文本靠堆词过阈值。
            score += weight + min(hit_count - 1, 2)
            reasons.append(f"{name}:{hit_count}")

    # 长内容本身有一定价值，但不能让纯噪声长日志直接通过太高阈值。
    if len(text) >= 1200:
        score += 1
        reasons.append("long_content")
    if len(text) >= 5000:
        score += 1
        reasons.append("very_long_content")

    return score, reasons or ["no_signal"]


def prefilter_decision(content: str, title: str | None, mode: str, threshold: int | None = None) -> PrefilterDecision:
    if mode == "none":
        return PrefilterDecision(True, 0, "prefilter_disabled")
    score, reasons = value_score(content, title)
    effective_threshold = PREFILTER_DEFAULT_THRESHOLDS.get(mode, 5) if threshold is None else threshold
    keep = score >= effective_threshold
    return PrefilterDecision(keep, score, ",".join(reasons) + f";threshold={effective_threshold}")


def strip_known_noise(text: str) -> str:
    """删除确定无助于长期记忆的包装噪声，保留正文。"""
    cleaned = text
    cleaned = re.sub(
        r"^(User|Assistant):\s*(Sender|Conversation info) \(untrusted metadata\):\s*```json\s*.*?\s*```\s*",
        r"\1: ",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"\[Note: model was just switched[^\]]*\]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def filter_message_text(text: str, role: str, mode: str, threshold: int | None = None) -> str:
    """消息级初筛。safe 只去明确噪声；balanced/strict 会减少送入 retain 的正文。"""
    if mode == "none":
        return text

    cleaned = strip_known_noise(text)
    if not cleaned:
        return ""

    low = normalize_for_prefilter(cleaned)
    if is_hard_import_noise(low):
        return ""
    if is_transient_progress_noise(low):
        return ""
    if len(low) < 80 and any(re.search(p, low, flags=re.IGNORECASE) for p in LOW_VALUE_SHORT_PATTERNS):
        return ""

    if mode == "safe":
        return cleaned

    score, _ = value_score(cleaned)
    effective_threshold = PREFILTER_DEFAULT_THRESHOLDS.get(mode, 5) if threshold is None else threshold

    if mode == "balanced":
        # 用户侧短指令通常没有长期价值；助手侧只保留带项目/决策/验证/配置信号的内容。
        if role == "user":
            return cleaned if len(cleaned) >= 120 or score >= max(2, effective_threshold // 2) else ""
        return cleaned if score >= effective_threshold or (len(cleaned) >= 2000 and score >= 3) else ""

    if mode == "strict":
        return cleaned if score >= effective_threshold else ""

    return cleaned


def extract_session_content(
    db_path: Path,
    session_id: str,
    since_timestamp: float = 0.0,
    prefilter_mode: str = "none",
    prefilter_threshold: int | None = None,
) -> tuple[str, float, float]:
    """从 SQLite 提取单个 session 的对话内容，返回 (内容, 最早消息时间戳, 最晚消息时间戳)。"""
    # hard_import_noise 必须默认启用，不能依赖 --prefilter；否则 full/none 导入会再次
    # 把 HEARTBEAT 循环、压缩 handoff 等系统包装送进 Hindsight。
    effective_mode = prefilter_mode if prefilter_mode != "none" else "safe"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content, timestamp
        FROM messages
        WHERE session_id = ? AND timestamp > ?
        ORDER BY timestamp
        """,
        (session_id, since_timestamp)
    )

    rows = cursor.fetchall()
    conn.close()

    messages = []
    first_timestamp = 0.0
    last_timestamp = 0.0

    for row in rows:
        msg = {
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
        }
        text = extract_message_text(msg)
        if text:
            text = filter_message_text(text, row["role"], effective_mode, prefilter_threshold)
        if text:
            if first_timestamp <= 0:
                first_timestamp = row["timestamp"]
            messages.append(text)
        if text and row["timestamp"] > last_timestamp:
            last_timestamp = row["timestamp"]

    content = sanitize_text("\n\n".join(messages)).strip()
    return content, first_timestamp, last_timestamp


def get_sessions_since(
    db_path: Path,
    since_timestamp: float = 0.0,
    hours: int | None = None,
    include_main: bool = True,
) -> list[dict[str, Any]]:
    """从 SQLite 获取比 since_timestamp 新的 sessions（或最近 N 小时）。"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 计算时间范围
    if hours is not None:
        cutoff = (datetime.now() - timedelta(hours=hours)).timestamp()
        since_timestamp = max(since_timestamp, cutoff)

    # 获取非 main sessions
    cursor.execute(
        """
        SELECT s.id, s.source, s.model, s.started_at, s.ended_at, s.title, s.message_count,
               s.input_tokens, s.output_tokens, MAX(m.timestamp) AS last_message_at
        FROM sessions s
        JOIN messages m ON m.session_id = s.id
        WHERE s.id != 'main'
        GROUP BY s.id
        HAVING last_message_at > ?
        ORDER BY last_message_at DESC
        """,
        (since_timestamp,)
    )
    rows = cursor.fetchall()

    sessions = []
    for row in rows:
        sessions.append({
            "id": row["id"],
            "source": row["source"],
            "model": row["model"] or "unknown",
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "title": row["title"],
            "message_count": row["message_count"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "last_message_at": row["last_message_at"],
        })

    # 获取 main session（当前会话）
    if include_main:
        cursor.execute(
            """
            SELECT s.id, s.source, s.model, s.started_at, s.ended_at, s.title, s.message_count,
                   s.input_tokens, s.output_tokens, MAX(m.timestamp) AS last_message_at
            FROM sessions s
            JOIN messages m ON m.session_id = s.id
            WHERE s.id = 'main'
            GROUP BY s.id
            """
        )
        main_row = cursor.fetchone()
        if main_row and main_row["last_message_at"] > since_timestamp:
            sessions.append({
                "id": main_row["id"],
                "source": main_row["source"],
                "model": main_row["model"] or "unknown",
                "started_at": main_row["started_at"],
                "ended_at": main_row["ended_at"],
                "title": main_row["title"],
                "message_count": main_row["message_count"],
                "input_tokens": main_row["input_tokens"],
                "output_tokens": main_row["output_tokens"],
                "last_message_at": main_row["last_message_at"],
            })

    conn.close()
    return sessions


def day_of(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.date().isoformat()


def classify_topic(text: str) -> str:
    low = text.lower()
    scores = {}
    for topic, keys in TOPIC_KEYWORDS.items():
        scores[topic] = sum(low.count(k.lower()) for k in keys)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def text_excerpt(text: str, n: int = 220) -> str:
    return re.sub(r"\s+", " ", sanitize_text(text)).strip()[:n]


def record_filter_drop(
    stats: FilterStats | None,
    sess: dict[str, Any],
    reason: str,
    content: str = "",
    score: int | None = None,
) -> None:
    if stats is None:
        return
    stats.messages_dropped += 1
    stats.add_reason(reason)
    stats.dropped_samples.append({
        "session_id": sess.get("id"),
        "title": sess.get("title"),
        "source": sess.get("source"),
        "model": sess.get("model"),
        "started_at": datetime.fromtimestamp(sess.get("started_at") or 0).isoformat() if sess.get("started_at") else None,
        "last_message_at": datetime.fromtimestamp(sess.get("last_message_at") or 0).isoformat() if sess.get("last_message_at") else None,
        "reason": reason,
        "score": score,
        "chars": len(content or ""),
        "excerpt": text_excerpt(content),
    })


def record_filter_keep(stats: FilterStats | None, record: SessionRecord, reason: str = "kept") -> None:
    if stats is None:
        return
    stats.messages_kept += 1
    stats.kept_samples.append({
        "session_id": record.session_id,
        "title": record.title,
        "source": record.source,
        "model": record.model,
        "day": record.day,
        "topic": record.topic,
        "reason": reason,
        "chars": len(record.content),
        "excerpt": text_excerpt(record.content),
    })


def print_filter_sample_report(stats: FilterStats, sample_count: int, sample_seed: int) -> None:
    if sample_count <= 0:
        return
    rng = random.Random(sample_seed)
    print("=== Filter Sample Report ===")
    print(f"Kept sessions: {stats.messages_kept}")
    print(f"Dropped sessions: {stats.messages_dropped}")
    if stats.local_model_calls or stats.backup_model_calls or stats.model_errors:
        print(f"Local model calls: {stats.local_model_calls} (backup={stats.backup_model_calls}, kept={stats.model_kept}, dropped={stats.model_dropped}, errors={stats.model_errors})")
    if stats.reasons:
        print("Drop reasons:")
        for reason, count in sorted(stats.reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {reason}: {count}")
    kept_pool = stats.kept_samples or []
    dropped_pool = stats.dropped_samples or []
    kept_samples = rng.sample(kept_pool, min(sample_count, len(kept_pool))) if kept_pool else []
    dropped_samples = rng.sample(dropped_pool, min(sample_count, len(dropped_pool))) if dropped_pool else []
    print("\nKept samples:")
    if not kept_samples:
        print("  (none)")
    for i, s in enumerate(kept_samples, 1):
        print(f"  [{i}] id={s.get('session_id')} topic={s.get('topic')} chars={s.get('chars')} reason={s.get('reason')} title={s.get('title') or ''}")
        print(f"      {s.get('excerpt') or ''}")
    print("\nDropped samples:")
    if not dropped_samples:
        print("  (none)")
    for i, s in enumerate(dropped_samples, 1):
        print(f"  [{i}] id={s.get('session_id')} reason={s.get('reason')} score={s.get('score')} chars={s.get('chars')} title={s.get('title') or ''}")
        if s.get("excerpt"):
            print(f"      {s.get('excerpt')}")
    print()


def build_records(
    sessions: list[dict[str, Any]],
    db_path: Path,
    skip_short: bool = True,
    since_timestamp: float = 0.0,
    prefilter_mode: str = "none",
    prefilter_threshold: int | None = None,
    filter_stats: FilterStats | None = None,
    local_filter: LocalFilter | None = None,
) -> tuple[list[SessionRecord], dict[str, int]]:
    """构建 SessionRecord 列表。"""
    records: list[SessionRecord] = []
    skipped = {"no_content": 0, "too_short": 0, "prefiltered": 0, "db_error": 0}
    # 本地双模型 gate 需要批处理：先让 primary 连续判断，再让 backup 连续复核，避免频繁加载/卸载模型。
    pending_local: list[tuple[dict[str, Any], str, float, float, PrefilterDecision]] = []

    def make_record(sess: dict[str, Any], content: str, first_msg_at: float, last_msg_at: float) -> SessionRecord:
        return SessionRecord(
            session_id=sess["id"],
            started_at=sess["started_at"],
            ended_at=sess["ended_at"],
            first_message_at=first_msg_at or sess.get("started_at", 0.0),
            last_message_at=last_msg_at or sess.get("last_message_at", sess.get("started_at", 0.0)),
            source=sess["source"],
            model=sess["model"],
            title=sess["title"],
            message_count=sess["message_count"],
            content=content,
            day=day_of(first_msg_at or sess["started_at"]),
            topic=classify_topic(content),
        )

    for sess in sessions:
        if filter_stats:
            filter_stats.messages_seen += 1
        try:
            content, first_msg_at, last_msg_at = extract_session_content(
                db_path,
                sess["id"],
                since_timestamp=since_timestamp,
                prefilter_mode=prefilter_mode,
                prefilter_threshold=prefilter_threshold,
            )
        except Exception as e:
            skipped["db_error"] += 1
            record_filter_drop(filter_stats, sess, f"db_error:{type(e).__name__}")
            print(f"  [skip] {sess['id']}: db error: {e}", file=sys.stderr)
            continue

        if not content:
            skipped["no_content"] += 1
            record_filter_drop(filter_stats, sess, "no_content_after_filter")
            continue

        if skip_short and len(content) < MIN_CONTENT_CHARS:
            skipped["too_short"] += 1
            record_filter_drop(filter_stats, sess, "too_short", content)
            continue

        decision = prefilter_decision(content, sess.get("title"), prefilter_mode, prefilter_threshold)
        if not decision.keep:
            if local_filter is not None:
                pending_local.append((sess, content, first_msg_at, last_msg_at, decision))
                continue
            skipped["prefiltered"] += 1
            record_filter_drop(filter_stats, sess, f"prefiltered:{decision.reason}", content, decision.score)
            print(
                f"  [prefilter] skip {sess['id']}: score={decision.score} {decision.reason} chars={len(content)}",
                file=sys.stderr,
            )
            continue

        record = make_record(sess, content, first_msg_at, last_msg_at)
        records.append(record)
        record_filter_keep(filter_stats, record, "rule_keep")

    if local_filter is not None and pending_local:
        local_inputs = [(content, "session", decision.score, [decision.reason]) for _, content, _, _, decision in pending_local]
        local_decisions = local_filter.decide_many(local_inputs)
        for (sess, content, first_msg_at, last_msg_at, rule_decision), local_decision in zip(pending_local, local_decisions):
            if local_decision.keep:
                record = make_record(sess, content, first_msg_at, last_msg_at)
                records.append(record)
                record_filter_keep(filter_stats, record, f"local_rescue:{local_decision.reason}")
            else:
                skipped["prefiltered"] += 1
                record_filter_drop(filter_stats, sess, f"prefiltered:{local_decision.reason}", content, rule_decision.score)
                print(
                    f"  [prefilter] skip {sess['id']}: score={rule_decision.score} {local_decision.reason} chars={len(content)}",
                    file=sys.stderr,
                )

    records.sort(key=lambda r: r.started_at)
    return records, skipped


def render_bundle_header(records: list[SessionRecord], group_key: str, bundle_index: int) -> str:
    """生成 bundle 的 header。"""
    starts = [r.first_message_at for r in records]
    last_msgs = [r.last_message_at for r in records]
    date_start = datetime.fromtimestamp(min(starts)).isoformat()
    date_end = datetime.fromtimestamp(max(last_msgs)).isoformat()
    topics = sorted(set(r.topic for r in records))

    lines = [
        "# Hermes SQLite 对话合并包（增量导入）",
        "",
        "这是从 SQLite state.db 提取的对话记忆。",
        "请抽取稳定、有复用价值的事实、偏好、项目决策、工具经验；",
        "忽略一次性寒暄、临时状态、执行噪声。",
        "",
        f"source: sqlite_state_db",
        f"import_mode: incremental",
        f"target_bank: {DEFAULT_BANK_ID}",
        f"group_key: {group_key}",
        f"bundle_index: {bundle_index}",
        f"date_range: {date_start} .. {date_end}",
        f"topics: {', '.join(topics)}",
        f"session_count: {len(records)}",
        "",
        "source_sessions:",
    ]
    for r in records[:60]:
        started = datetime.fromtimestamp(r.started_at).isoformat()
        first_msg = datetime.fromtimestamp(r.first_message_at).isoformat()
        last_msg = datetime.fromtimestamp(r.last_message_at).isoformat()
        lines.append(f"  - id={r.session_id} session_start={started} first_msg={first_msg} last_msg={last_msg} topic={r.topic} model={r.model} chars={len(r.content)}")
    if len(records) > 60:
        lines.append(f"  - ... {len(records) - 60} more")
    lines.extend(["", "--- conversations ---", ""])
    return "\n".join(lines)


def split_oversized_record(record: SessionRecord, max_bundle_chars: int) -> list[SessionRecord]:
    """把单个超大 session 硬切成多个虚拟 SessionRecord。

    旧逻辑只在多个 session 合并时按 max_bundle_chars flush；如果单个 session
    自身超过上限，会被原样送给 Hindsight，导致 MiniMax 结构化 JSON 输出更容易
    被长对话内容劫持，出现 JSON parse retry / STUCK。这里在 bundle 层做保守硬切，
    优先按 User/Assistant turn 或空行边界切分。
    """
    # 留出 header、SESSION 标题和 split marker 的空间。
    target_chars = max(20_000, max_bundle_chars - 5_000)
    if len(record.content) + 500 <= max_bundle_chars:
        return [record]

    content = record.content
    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = min(len(content), start + target_chars)
        if end < len(content):
            window = content[start:end]
            candidates = [window.rfind("\nUser:"), window.rfind("\nAssistant:"), window.rfind("\n\n")]
            valid = [c for c in candidates if c >= int(target_chars * 0.55)]
            if valid:
                end = start + max(valid)
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    total = len(chunks)
    parts: list[SessionRecord] = []
    for i, chunk in enumerate(chunks, 1):
        marker = (
            f"[SESSION SPLIT]\n"
            f"original_session_id: {record.session_id}\n"
            f"split_part: {i}/{total}\n"
            f"reason: oversized session split before Hindsight retain to avoid JSON/stuck retry loops\n\n"
        )
        parts.append(
            replace(
                record,
                session_id=f"{record.session_id}::part-{i:02d}-of-{total:02d}",
                title=f"{record.title or record.session_id} [split {i}/{total}]",
                message_count=max(1, math.ceil(record.message_count / total)),
                content=marker + chunk,
            )
        )
    print(
        f"  [split] oversized session {record.session_id}: {len(record.content)} chars -> {total} parts "
        f"(target~{target_chars})",
        file=sys.stderr,
    )
    return parts


def build_bundles(records: list[SessionRecord], group_by: str = "day", max_bundle_chars: int = MAX_BUNDLE_CHARS) -> list[Bundle]:
    """按时间分组构建 bundle；单个超大 session 会先硬切，避免大 prompt 卡死。"""
    expanded_records: list[SessionRecord] = []
    for r in records:
        expanded_records.extend(split_oversized_record(r, max_bundle_chars))

    groups: dict[str, list[SessionRecord]] = {}
    for r in expanded_records:
        if group_by == "all":
            key = "all"
        elif group_by == "day":
            key = r.day
        elif group_by == "topic":
            key = r.topic
        elif group_by == "day-topic":
            key = f"{r.day}__{r.topic}"
        else:
            key = r.day
        groups.setdefault(key, []).append(r)

    bundles = []
    bundle_index = 0

    for gkey in sorted(groups):
        group_records = sorted(groups[gkey], key=lambda r: r.started_at)
        current: list[SessionRecord] = []
        current_len = 0

        def flush():
            nonlocal bundle_index, current, current_len
            if not current:
                return

            header = render_bundle_header(current, gkey, bundle_index)
            blocks = []
            for r in current:
                started = datetime.fromtimestamp(r.started_at).isoformat()
                first_msg = datetime.fromtimestamp(r.first_message_at).isoformat()
                last_msg = datetime.fromtimestamp(r.last_message_at).isoformat()
                blocks.append(f"\n===== SESSION id={r.session_id} session_start={started} first_msg={first_msg} last_msg={last_msg} topic={r.topic} model={r.model} =====\n{r.content}")

            content = header + "".join(blocks)
            starts = [r.first_message_at for r in current]
            last_msgs = [r.last_message_at for r in current]
            topics = sorted(set(r.topic for r in current))

            hash_input = f"sqlite|{group_by}|{gkey}|{bundle_index}|" + "|".join(r.session_id for r in current) + f"|{max(last_msgs)}"
            doc_id = f"hermes-sqlite::{group_by}::{gkey}::{bundle_index:04d}::{short_hash(hash_input)}"

            bundles.append(Bundle(
                index=bundle_index,
                group_key=gkey,
                topic=topics[0] if len(topics) == 1 else "mixed",
                start=datetime.fromtimestamp(min(starts)).isoformat(),
                end=datetime.fromtimestamp(max(last_msgs)).isoformat(),
                records=list(current),
                content=content,
                document_id=doc_id,
            ))

            bundle_index += 1
            current = []
            current_len = 0

        for r in group_records:
            block_len = len(r.content) + 500
            if current and current_len + block_len > max_bundle_chars:
                flush()
            current.append(r)
            current_len += block_len
        flush()

    return bundles


def should_retry(error_text: str) -> bool:
    s = (error_text or "").lower()
    return any(x in s for x in [
        "429", "throttling", "concurrency allocated quota exceeded",
        "read timed out", "timeout", "temporarily unavailable",
        "connection reset", "rate limit",
    ])


def get_retry_delay(error_text: str, attempt: int) -> int:
    s = (error_text or "").lower()
    if "429" in s or "throttling" in s:
        return RATE_LIMIT_BACKOFF_SECONDS
    return BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]


def post_bundle(api: str, bank: str, bundle: Bundle) -> tuple[bool, str | None]:
    """提交 bundle 到 Hindsight。"""
    last_msgs = [r.last_message_at for r in bundle.records]
    max_last_msg = max(last_msgs) if last_msgs else 0

    item = {
        "content": bundle.content,
        "document_id": bundle.document_id,
        "context": "hermes_sqlite_incremental",
        "timestamp": bundle.start,
        "metadata": {
            "source": "hermes_sqlite_import",
            "import_mode": "incremental",
            "group_key": bundle.group_key,
            "topic": bundle.topic,
            "date_range_start": bundle.start,
            "date_range_end": bundle.end,
            "last_message_timestamp": str(max_last_msg),
            "last_message_iso": datetime.fromtimestamp(max_last_msg).isoformat() if max_last_msg > 0 else None,
            "session_count": str(len(bundle.records)),
            "bundle_index": str(bundle.index),
        },
        "tags": ["hermes", "sqlite", "incremental", bundle.topic],
    }

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{api}/v1/default/banks/{bank}/memories",
                json={"async": True, "items": [item]},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code in (200, 201, 202):
                return True, None
            err = f"API error: {resp.status_code} {resp.text[:500]}"
            last_err = err
            if attempt < MAX_RETRIES and should_retry(err):
                delay = get_retry_delay(err, attempt)
                print(f"  retry after {delay}s: {err[:100]}", file=sys.stderr)
                time.sleep(delay)
                continue
            return False, err
        except Exception as e:
            err = str(e)
            last_err = err
            if attempt < MAX_RETRIES and should_retry(err):
                delay = get_retry_delay(err, attempt)
                print(f"  retry after {delay}s: {err[:100]}", file=sys.stderr)
                time.sleep(delay)
                continue
            return False, err

    return False, last_err or "unknown error"


def estimate_internal_chunks(bundles: list[Bundle], retain_chunk_size: int) -> int:
    """估算 Hindsight retain 内部按 retain_chunk_size 切出来的 chunk 数。"""
    if retain_chunk_size <= 0:
        return 0
    return sum(max(1, math.ceil(len(b.content) / retain_chunk_size)) for b in bundles)


def print_bundle_size_warnings(bundles: list[Bundle], retain_chunk_size: int) -> None:
    """打印调用次数相关的关键估算，避免误把 bundle 数当成 LLM 调用数。"""
    if not bundles:
        return
    internal_chunks = estimate_internal_chunks(bundles, retain_chunk_size)
    max_chars = max(len(b.content) for b in bundles)
    avg_chars = sum(len(b.content) for b in bundles) / len(bundles)
    print(f"  Estimated retain chunks: {internal_chunks} (ceil(bundle_chars / {retain_chunk_size}))")
    print(f"  Avg bundle chars: {avg_chars:.0f}")
    print(f"  Max bundle chars: {max_chars}")
    if max_chars > max(retain_chunk_size * 30, 200000):
        print(
            "  WARNING: at least one bundle is very large; Hindsight will split it into many retain chunks. "
            "This may be caused by one huge session or by overly coarse grouping. Consider --no-main for historical imports, "
            "--max-bundle-chars for bundle sizing, or a larger --retain-chunk-size only after quality review."
        )


def check_hindsight_health(api: str) -> bool:
    """检查 Hindsight 服务是否可用。"""
    try:
        resp = requests.get(f"{api}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def get_bank_config(api: str, bank: str) -> dict[str, Any]:
    resp = requests.get(f"{api}/v1/default/banks/{bank}/config", timeout=15)
    resp.raise_for_status()
    return resp.json()


def patch_bank_config(api: str, bank: str, updates: dict[str, Any]) -> dict[str, Any]:
    resp = requests.patch(
        f"{api}/v1/default/banks/{bank}/config",
        json={"updates": updates},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def ensure_retain_config(api: str, bank: str, chunk_size: int, extraction_mode: str, enable_observations: bool = False) -> dict[str, Any]:
    """确保 Hindsight retain 配置为指定 chunk size / extraction mode / observations 开关。"""
    cfg = get_bank_config(api, bank)
    current = cfg.get("config", {})
    updates: dict[str, Any] = {}

    if current.get("retain_chunk_size") != chunk_size:
        updates["retain_chunk_size"] = chunk_size
    if current.get("retain_extraction_mode") != extraction_mode:
        updates["retain_extraction_mode"] = extraction_mode
    if current.get("enable_observations") is not enable_observations:
        updates["enable_observations"] = enable_observations

    if updates:
        cfg = patch_bank_config(api, bank, updates)

    final = cfg.get("config", {})
    if final.get("retain_extraction_mode") != extraction_mode:
        raise RuntimeError(
            f"Unexpected retain_extraction_mode: {final.get('retain_extraction_mode')!r}; expected {extraction_mode!r}"
        )
    if final.get("enable_observations") is not enable_observations:
        raise RuntimeError(
            f"Unexpected enable_observations: {final.get('enable_observations')!r}; expected {enable_observations!r}"
        )
    return cfg


def load_hindsight_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_hindsight_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_auto_retain(path: Path, enabled: bool) -> bool | None:
    """设置 Hermes Hindsight auto_retain，返回原值；配置不存在则返回 None。"""
    cfg = load_hindsight_config(path)
    if cfg is None:
        return None
    old = cfg.get("auto_retain")
    cfg["auto_retain"] = bool(enabled)
    save_hindsight_config(path, cfg)
    return bool(old) if isinstance(old, bool) else None


def restore_auto_retain(path: Path, old_value: bool | None) -> None:
    if old_value is None:
        return
    cfg = load_hindsight_config(path)
    if cfg is None:
        return
    cfg["auto_retain"] = old_value
    save_hindsight_config(path, cfg)


def empty_progress() -> dict[str, Any]:
    return {
        "processed": [],
        "last_imported_timestamp": 0.0,
        "last_imported_iso": None,
        "last_run": None,
        "total_sessions_imported": 0,
        "total_bundles_imported": 0,
    }


def load_progress(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return empty_progress()


def progress_for_run(*, full: bool, path: Path = DEFAULT_PROGRESS_FILE) -> dict[str, Any]:
    """Return progress state for this run.

    A full rebuild must ignore the old processed document_id list. Otherwise a
    DB reset followed by --full can silently skip bundles that only exist in
    the historical progress file.
    """
    if full:
        return empty_progress()
    return load_progress(path)


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def update_progress_cutoff(progress: dict[str, Any], bundles: list[Bundle]) -> None:
    """更新 progress 里的截止时间（取所有成功 bundle 中最晚的消息时间）。"""
    if not bundles:
        return

    all_last_msgs = []
    for b in bundles:
        for r in b.records:
            all_last_msgs.append(r.last_message_at)

    if all_last_msgs:
        max_ts = max(all_last_msgs)
        progress["last_imported_timestamp"] = max_ts
        progress["last_imported_iso"] = datetime.fromtimestamp(max_ts).isoformat()


class RawDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """同时保留 epilog 换行并显示参数默认值。"""


def main():
    parser = argparse.ArgumentParser(
        description="DEPRECATED/BLOCKED submit: SQLite day-topic import helper; use session/json manifest route",
        formatter_class=RawDefaultsHelpFormatter,
        epilog=(
            "当前正式路线：\n"
            "  生成 manifest：python3 hindsight_session_manifest.py --bank-target hermes_v3 --json\n"
            "  dry-run retain：python3 hindsight_minimax_import.py session-manifest-retain-llm --manifest <manifest.jsonl> --bank hermes_v3\n"
            "  正式 retain：同上加 --execute --confirm retain-hindsight-session-manifest\n"
            "本脚本只保留历史审计 dry-run；submit 默认阻断。"
        ),
    )
    parser.add_argument("--hours", type=int, help="时间范围（小时）")
    parser.add_argument("--days", type=int, help="时间范围（天），优先于 hours")
    parser.add_argument("--since", type=str, help="增量起点：ISO 时间或 Unix timestamp")
    parser.add_argument("--incremental", action="store_true", default=True, help="增量导入：使用上次记录的 message cutoff")
    parser.add_argument("--full", action="store_true", help="全量导入：忽略截止时间记录，正式提交前必须先 dry-run")
    parser.add_argument("--mode", choices=["dry-run", "submit"], default="dry-run", help="运行模式；dry-run 不调用 Hindsight/LLM")
    parser.add_argument("--group-by", choices=["day", "topic", "day-topic", "all"], default="day-topic", help="bundle 分组方式；day-topic 是当前离线 pipeline 默认")
    parser.add_argument("--max-bundle-chars", type=int, default=MAX_BUNDLE_CHARS, help="脚本层每个 bundle 的目标最大字符数；单个超大会话也会被硬切，避免 MiniMax JSON/STUCK retry loop")
    parser.add_argument("--prefilter", choices=["none", "safe", "balanced", "strict"], default="safe", help="本地确定性初筛。safe 只挡明显低价值/污染内容；balanced/strict 仅用于成本应急；none 仍保留 hard-noise 清理")
    parser.add_argument("--prefilter-threshold", type=int, help="覆盖初筛阈值；默认 safe=1, balanced=7, strict=12")
    parser.add_argument("--local-filter", help="可选本地 Ollama 复核模型，仅复核规则准备丢弃的灰区 session；默认关闭，例如 llama3.1:8b-local")
    parser.add_argument("--backup-filter", help="可选备份 Ollama 模型；drop-policy=consensus 时用于复核 drop，例如 qwen2:7b-instruct")
    parser.add_argument("--drop-policy", choices=["single", "consensus"], default="consensus", help="本地模型 drop 策略：single=主模型可直接 drop；consensus=主备都 drop 才丢弃")
    parser.add_argument("--ollama-api", default="http://127.0.0.1:11434", help="Ollama API 地址")
    parser.add_argument("--local-filter-max-chars", type=int, default=2400, help="送给本地筛选模型的最大字符数")
    parser.add_argument("--local-filter-timeout", type=int, default=60, help="本地筛选模型单次请求超时秒数")
    parser.add_argument("--local-filter-max-calls", type=int, default=300, help="本地筛选模型最大调用次数；超过后保守保留，避免误丢")
    parser.add_argument("--sample-report", type=int, default=8, help="dry-run 输出 kept/dropped 抽样数量；0 关闭")
    parser.add_argument("--sample-seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--include-main", action="store_true", default=True, help="包含 main session")
    parser.add_argument("--no-main", action="store_true", help="排除 main session")
    parser.add_argument("--bank", default=DEFAULT_BANK_ID, help="目标 Hindsight bank")
    parser.add_argument("--api", default=HINDSIGHT_API, help="Hindsight API 地址")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite 数据库路径")
    parser.add_argument("--retain-chunk-size", type=int, default=DEFAULT_RETAIN_CHUNK_SIZE, help="Hindsight retain chunk size，单位是字符，不是 token")
    parser.add_argument("--retain-extraction-mode", default=DEFAULT_RETAIN_EXTRACTION_MODE, choices=["concise"], help="Hindsight retain extraction mode；只允许 concise，禁止 verbose")
    parser.add_argument("--enable-observations", action="store_true", help="允许 Hindsight 内置 observations/consolidation；默认关闭，离线 pipeline 通常用自有 daily/weekly consolidation")
    parser.add_argument("--hindsight-config", default=str(DEFAULT_HINDSIGHT_CONFIG_FILE), help="Hermes Hindsight config.json 路径，用于 submit 时临时关闭 auto_retain")
    parser.add_argument("--allow-deprecated-sqlite-submit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "submit" and not args.allow_deprecated_sqlite_submit and os.environ.get("HINDSIGHT_ALLOW_DEPRECATED_SQLITE_IMPORT") != "1":
        print(f"ERROR: {DEPRECATED_SQLITE_IMPORT_MESSAGE}", file=sys.stderr)
        sys.exit(2)

    include_main = args.include_main and not args.no_main
    db_path = Path(args.db)

    # 加载 progress。全量重建必须忽略旧 processed document_id，
    # 否则 reset DB 后会因为历史 progress 跳过已经不存在的 bundles。
    progress = progress_for_run(full=args.full)

    # 确定时间起点
    since_timestamp = 0.0

    if args.full:
        # 全量导入，忽略截止时间记录
        since_timestamp = 0.0
    elif args.since:
        # 手动指定起点
        try:
            if args.since.isdigit():
                since_timestamp = float(args.since)
            else:
                dt = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
                since_timestamp = dt.timestamp()
        except Exception:
            print(f"ERROR: invalid --since format: {args.since}", file=sys.stderr)
            sys.exit(1)
    elif args.incremental and progress.get("last_imported_timestamp", 0) > 0:
        # 增量导入，使用上次记录的截止时间
        since_timestamp = progress["last_imported_timestamp"]
        last_iso = progress.get("last_imported_iso")
        print(f"Incremental mode: since {last_iso or since_timestamp}")

    # 如果同时指定了 hours/days，取 max
    hours = None
    if args.days:
        hours = args.days * 24
    elif args.hours:
        hours = args.hours

    print("=" * 60)
    print("SQLite → Hindsight Import Tool (Incremental)")
    print("=" * 60)
    print(f"DB: {db_path}")
    if hours:
        print(f"Time range: last {hours} hours")
    if since_timestamp > 0:
        print(f"Since cutoff: {datetime.fromtimestamp(since_timestamp).isoformat()}")
    print(f"Include main: {include_main}")
    print(f"Group by: {args.group_by}")
    print(f"Max bundle chars: {args.max_bundle_chars}")
    print(f"Prefilter: {args.prefilter}" + (f" threshold={args.prefilter_threshold}" if args.prefilter_threshold is not None else ""))
    print(f"Local filter: {args.local_filter or 'disabled'}" + (f" backup={args.backup_filter} policy={args.drop_policy}" if args.local_filter else ""))
    print(f"Sample report: {args.sample_report}")
    print(f"Mode: {args.mode}")
    print(f"Target bank: {args.bank}")
    print(f"Retain chunk size: {args.retain_chunk_size} chars")
    print(f"Retain extraction mode: {args.retain_extraction_mode}")
    print(f"Enable observations: {args.enable_observations}")
    print("Submit mode is serial: one bundle request at a time")
    print()

    if not db_path.exists():
        print(f"ERROR: SQLite database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # 获取 sessions
    print("Fetching sessions from SQLite...")
    sessions = get_sessions_since(
        db_path,
        since_timestamp=since_timestamp,
        hours=hours,
        include_main=include_main,
    )
    print(f"  Found {len(sessions)} sessions")

    if not sessions:
        print("No sessions in time range. Done.")
        sys.exit(0)

    # 构建记录
    filter_stats = FilterStats()
    local_filter: LocalFilter | None = None
    if args.local_filter:
        local_filter = OllamaLocalFilter(
            model=args.local_filter,
            backup_model=args.backup_filter,
            api=args.ollama_api,
            drop_policy=args.drop_policy,
            max_chars=args.local_filter_max_chars,
            timeout=args.local_filter_timeout,
            max_calls=args.local_filter_max_calls,
            stats=filter_stats,
        )

    print("Extracting content...")
    try:
        records, skipped = build_records(
            sessions,
            db_path,
            since_timestamp=(0.0 if args.full else since_timestamp),
            prefilter_mode=args.prefilter,
            prefilter_threshold=args.prefilter_threshold,
            filter_stats=filter_stats,
            local_filter=local_filter,
        )
    finally:
        if local_filter is not None:
            local_filter.close()
    print(f"  Records: {len(records)}")
    print(f"  Skipped: {skipped}")

    if not records:
        print("No content after filtering. Done.")
        sys.exit(0)

    # 构建 bundle
    print("Building bundles...")
    bundles = build_bundles(records, group_by=args.group_by, max_bundle_chars=args.max_bundle_chars)
    print(f"  Bundles: {len(bundles)}")

    # 统计
    total_chars = sum(len(b.content) for b in bundles)
    all_last_msgs = []
    for b in bundles:
        for r in b.records:
            all_last_msgs.append(r.last_message_at)
    max_last_msg = max(all_last_msgs) if all_last_msgs else 0

    print(f"  Total chars: {total_chars}")
    print_bundle_size_warnings(bundles, args.retain_chunk_size)
    print(f"  Max last_message_at: {datetime.fromtimestamp(max_last_msg).isoformat()}")
    print()

    # Dry-run 只输出统计
    if args.mode == "dry-run":
        print("=== Dry-run Summary ===")
        print(f"Since cutoff: {datetime.fromtimestamp(since_timestamp).isoformat() if since_timestamp > 0 else '0 (full)'}")
        print(f"Sessions: {len(sessions)}")
        print(f"Records with content: {len(records)}")
        print(f"Skipped: {skipped}")
        print(f"Bundles: {len(bundles)}")
        print(f"Estimated retain chunks: {estimate_internal_chunks(bundles, args.retain_chunk_size)}")
        print(f"Total content chars: {total_chars}")
        print(f"New cutoff would be: {datetime.fromtimestamp(max_last_msg).isoformat()}")
        print()
        print_filter_sample_report(filter_stats, args.sample_report, args.sample_seed)
        print("Bundle details:")
        for b in bundles:
            print(f"  [{b.index}] {b.group_key}: {len(b.records)} sessions, {len(b.content)} chars, topic={b.topic}, last_msg={b.end}")
        print()
        print("Run with --mode submit to actually import.")
        sys.exit(0)

    # Submit 模式
    print("Checking Hindsight health...")
    if not check_hindsight_health(args.api):
        print(f"ERROR: Hindsight not reachable at {args.api}", file=sys.stderr)
        sys.exit(1)
    print("  OK")

    print("Ensuring Hindsight retain config...")
    try:
        bank_cfg = ensure_retain_config(
            args.api,
            args.bank,
            chunk_size=args.retain_chunk_size,
            extraction_mode=args.retain_extraction_mode,
            enable_observations=args.enable_observations,
        )
        final_cfg = bank_cfg.get("config", {})
        print(f"  retain_chunk_size={final_cfg.get('retain_chunk_size')}")
        print(f"  retain_extraction_mode={final_cfg.get('retain_extraction_mode')}")
        print(f"  enable_observations={final_cfg.get('enable_observations')}")
    except Exception as e:
        print(f"ERROR: failed to ensure Hindsight retain config: {e}", file=sys.stderr)
        sys.exit(1)

    hindsight_config_path = Path(args.hindsight_config)
    old_auto_retain = set_auto_retain(hindsight_config_path, False)
    if old_auto_retain is None:
        print(f"Temporary auto_retain disable: skipped (config not found or value missing): {hindsight_config_path}")
    else:
        print(f"Temporary auto_retain disable: {old_auto_retain} -> False")

    processed_ids = set(progress.get("processed", []))

    print("Submitting bundles...")
    success_count = 0
    fail_count = 0
    success_bundles = []

    try:
        for i, bundle in enumerate(bundles):
            # 跳过已处理的 bundle
            if bundle.document_id in processed_ids:
                print(f"  [{i+1}/{len(bundles)}] skip (already processed): {bundle.document_id}")
                continue

            print(f"  [{i+1}/{len(bundles)}] submitting {bundle.document_id}...")
            ok, err = post_bundle(args.api, args.bank, bundle)

            if ok:
                processed_ids.add(bundle.document_id)
                success_bundles.append(bundle)
                success_count += 1
                print(f"    ✓ success")
            else:
                fail_count += 1
                print(f"    ✗ failed: {err}")

            # 串行提交，保留短间隔，避免无意义并发占用配额。
            if i < len(bundles) - 1:
                time.sleep(0.5)
    finally:
        restore_auto_retain(hindsight_config_path, old_auto_retain)
        if old_auto_retain is not None:
            print(f"Temporary auto_retain restored: {old_auto_retain}")

    # 更新 progress
    if success_bundles:
        update_progress_cutoff(progress, success_bundles)
        progress["processed"] = list(processed_ids)
        progress["last_run"] = datetime.now().isoformat()
        progress["total_sessions_imported"] = progress.get("total_sessions_imported", 0) + sum(len(b.records) for b in success_bundles)
        progress["total_bundles_imported"] = progress.get("total_bundles_imported", 0) + len(success_bundles)
        save_progress(DEFAULT_PROGRESS_FILE, progress)

    print()
    print("=" * 60)
    print("Import complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    if success_bundles:
        print(f"  New cutoff: {progress.get('last_imported_iso', 'N/A')}")
        print(f"  Total sessions imported: {progress.get('total_sessions_imported', 0)}")
        print(f"  Total bundles imported: {progress.get('total_bundles_imported', 0)}")
    print(f"  Progress: {DEFAULT_PROGRESS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()