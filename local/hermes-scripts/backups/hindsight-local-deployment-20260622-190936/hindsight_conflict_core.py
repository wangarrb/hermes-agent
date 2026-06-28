#!/usr/bin/env python3
"""Shared primitives for Hindsight conflict-driven repair workflow.

This module is intentionally deterministic and read-only. It contains no DB writes,
no Hindsight API calls, and no LLM calls.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
OFFLINE_CONSOLIDATION_DOC_RE = re.compile(
    r"^(hermes-offline-consolidation::(?:daily|weekly)::.+::\d{2})::[0-9a-f]{12}$"
)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:m|cm|mm|%|ms|s|tok/s|tokens|chars|次|条|个|帧|scene)?", re.I)

SEVERITY_ORDER = {"OK": 0, "P3": 1, "P2": 2, "P1": 3, "P0": 4}

CONTAMINATION_PATTERNS = [
    ("context_compaction", re.compile(r"(?m)^\s*(?:\[?CONTEXT COMPACTION|---\s*END OF CONTEXT SUMMARY)", re.I), "P1"),
    # Actual leaked heartbeat lines are usually line-leading markers. Merely
    # mentioning HEARTBEAT as a known pollution class is a legitimate technical fact.
    ("heartbeat", re.compile(r"(?m)^\s*(?:\[[^\]]+\]\s*)?HEARTBEAT\s*[:|\-]", re.I), "P1"),
    # Operational-log vocabulary can be legitimate high-level knowledge. Keep it
    # visible as P3 evidence instead of blocking publish by default.
    ("tool_log_mention", re.compile(r"PENDING_BREAKDOWN|WORKER_TASK|STREAMING RETAIN COMPLETE|docker logs|tcsetattr|exit code 130", re.I), "P3"),
    ("json_retry_loop_mention", re.compile(r"JSON parse retry|STUCK|payload_null|rate limited|HTTP 429", re.I), "P3"),
    ("raw_stack_trace", re.compile(r"Traceback \(most recent call last\)|ModuleNotFoundError|PermissionError|FileNotFoundError", re.I), "P2"),
    ("prompt_leak", re.compile(r"<think>|Let me analyze the input material|strict JSON in the specified schema", re.I), "P1"),
]

GENERIC_TOKENS = {
    "用户", "偏好", "默认", "规则", "必须", "不能", "可以", "建议", "当前", "结论", "问题",
    "项目", "配置", "结果", "数据", "工具", "流程", "系统", "记忆", "发布", "本地", "主库",
    "the", "and", "for", "with", "from", "this", "that", "user", "project", "memory", "offline",
}

POLARITY_GROUPS = [
    ("publish", ["发布", "publish", "eligible"], ["不发布", "未发布", "blocked", "block", "keep local", "local only"]),
    ("allow", ["可以", "允许", "enable", "true", "开启"], ["不能", "禁止", "不允许", "disable", "false", "关闭"]),
    ("retain", ["保留", "retain", "keep"], ["删除", "移除", "drop", "delete", "purge"]),
    ("default", ["默认", "default"], ["不默认", "not default", "opt-in", "显式"]),
]

REQUIRED_FLOW = [
    "conflict_intake",
    "provenance_trace",
    "low_level_feature_audit",
    "root_cause_classification",
    "repair_proposal",
    "human_confirmation_for_destructive_actions",
    "execute_repair",
    "re_eval_gate",
    "publish_or_keep_local",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:n]


def severity_value(severity: str) -> int:
    return SEVERITY_ORDER.get(str(severity or "").upper(), 0)


def severity_at_least(severity: str, threshold: str) -> bool:
    return severity_value(severity) >= severity_value(threshold)


def max_severity(items: list[dict[str, Any]], default: str = "OK") -> str:
    if not items:
        return default
    return max((str(i.get("severity") or default).upper() for i in items), key=severity_value)


def redact(text: str, limit: int | None = None) -> str:
    """Redact common secrets from previews/reports."""
    out = text or ""
    out = re.sub(r"(?i)(api[_-]?key|token|password|secret|credential)\s*[:=]\s*[^\s,'\"]+", r"\1=[REDACTED]", out)
    out = re.sub(r"\bsk-[A-Za-z0-9_\-]{16,}\b", "[REDACTED]", out)
    out = re.sub(r"postgresql://[^\s'\"]+", "[REDACTED]", out)
    if limit is not None and len(out) > limit:
        return out[:limit] + "..."
    return out


def detect_contamination(text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    body = text or ""
    for name, pattern, severity in CONTAMINATION_PATTERNS:
        m = pattern.search(body)
        if m:
            snippet = body[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
            hits.append({"name": name, "severity": severity, "snippet": redact(snippet, 220)})
    return sorted(hits, key=lambda h: (-severity_value(h["severity"]), h["name"]))


def numeric_tokens(text: str) -> list[str]:
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        token = re.sub(r"\s+", "", m.group(0))
        if token:
            out.append(token)
    return out[:80]


NUMERIC_CONTEXT_WORDS = {
    "ate", "rpe", "psnr", "ssim", "lpips", "alignment_score", "score", "ratio", "lat_ratio",
    "precision", "recall", "error", "metric", "length", "distance", "trajectory", "scale",
    "误差", "精度", "指标", "比例", "轨迹", "总长", "长度", "距离", "尺度", "排名",
}
NON_CONFLICT_UNITS = {"scene", "帧", "条", "个", "次", "tokens", "chars", "tok/s"}
CONFLICT_UNITS = {"m", "cm", "mm", "%"}


def numeric_claims(text: str) -> list[dict[str, Any]]:
    """Extract comparable numeric claims with a rough metric/unit key.

    The audit should not compare every number in a sentence. Scene IDs, years,
    frame counts, sample counts, dates, and mixed metrics create many false
    positives. This helper keeps only numbers with a measurement unit or nearby
    metric word, and groups them by `metric|unit` before divergence checks.
    """
    claims: list[dict[str, Any]] = []
    body = text or ""
    lower = body.lower()
    for m in NUMBER_RE.finditer(body):
        token = re.sub(r"\s+", "", m.group(0))
        value_m = re.match(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", token, re.I)
        if not value_m:
            continue
        try:
            value = float(value_m.group(0))
        except Exception:
            continue
        raw_unit = token[value_m.end():]
        unit = raw_unit.lower()
        # `10M` usually means million, not meters; avoid treating uppercase M as metre.
        if raw_unit == "M":
            continue
        if 1900 <= value <= 2100 and not raw_unit:
            continue
        if unit in NON_CONFLICT_UNITS:
            continue
        window_start = max(0, m.start() - 36)
        window_end = min(len(lower), m.end() + 36)
        window = lower[window_start:window_end]
        number_center = m.start() - window_start + (m.end() - m.start()) / 2.0
        metric = ""
        metric_distance = 10**9
        metric_center = -1.0
        for w in sorted(NUMERIC_CONTEXT_WORDS, key=len, reverse=True):
            for wm in re.finditer(re.escape(w), window):
                center = (wm.start() + wm.end()) / 2.0
                dist = abs(center - number_center)
                if dist < metric_distance:
                    metric = w
                    metric_distance = dist
                    metric_center = center
        if not metric:
            # Unit-only values are usually ranges/counts/scene measurements, not
            # comparable claims. Require a nearby metric anchor for conflict audit.
            continue
        if not unit:
            if value > 100:
                continue
            # Unitless metric claims should normally be written as METRIC=VALUE
            # or have the metric immediately before the value. If the nearest
            # metric appears after the number, the number is often a parameter,
            # version, scene/window id, or line/count near a later metric mention.
            if metric_center > number_center:
                continue
            between = window[int(min(metric_center, number_center)): int(max(metric_center, number_center))]
            if metric_distance > 14 and not any(sym in between for sym in ["=", "≈", "~", "≤", "≥", "<", ">", ":"]):
                continue
        key = f"{metric}|{unit or 'unitless'}"
        claims.append({"token": token, "value": value, "unit": unit or "unitless", "metric": metric or None, "key": key})
    return claims[:80]


def _parse_key_values(line: str) -> dict[str, str]:
    pairs = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line):
        pairs[key] = value.strip()
    return pairs


def extract_source_sessions(original_text: str) -> list[dict[str, Any]]:
    """Extract source session headers from import bundle original_text.

    Expected line shape:
      - id=<session> session_start=<iso> first_msg=<iso> ... chars=<int>
    """
    sessions: list[dict[str, Any]] = []
    in_block = False
    for raw in (original_text or "").splitlines():
        line = raw.strip()
        if line == "source_sessions:":
            in_block = True
            continue
        if in_block and not line:
            continue
        if in_block and line.startswith("---"):
            break
        if in_block and line.startswith("- "):
            vals = _parse_key_values(line[2:])
            if "id" in vals:
                item: dict[str, Any] = vals
                if "chars" in item:
                    try:
                        item["chars"] = int(str(item["chars"]))
                    except Exception:
                        pass
                sessions.append(item)
    return sessions


def observation_text(obs: dict[str, Any]) -> str:
    parts = [
        str(obs.get("insight") or ""),
        str(obs.get("applicability") or ""),
        " ".join(map(str, obs.get("tags") or [])),
        str(obs.get("type") or ""),
        str(obs.get("topic") or ""),
    ]
    return "\n".join(p for p in parts if p)


def significant_terms(text: str, *, limit: int = 8) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", text or ""):
        t = token.strip().lower()
        if not t or t in GENERIC_TOKENS:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        terms.append(t)
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def source_refs(obs: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    refs.extend(map(str, obs.get("source_documents") or []))
    refs.extend(map(str, obs.get("evidence_ids") or []))
    seen = set()
    out = []
    for r in refs:
        r = r.strip()
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def offline_consolidation_doc_key(ref: str) -> str | None:
    """Return hash-independent key for offline consolidation document IDs."""
    m = OFFLINE_CONSOLIDATION_DOC_RE.match(str(ref or ""))
    return m.group(1) if m else None


def offline_consolidation_doc_alias(ref: str) -> str | None:
    key = offline_consolidation_doc_key(ref)
    return f"offline_doc_alias::{key}" if key else None


def _case_id(kind: str, target: str, detail: str = "") -> str:
    return f"conflict::{kind}::{sha1_short(target + '|' + detail, 14)}"


def _case(kind: str, severity: str, title: str, target: dict[str, Any], evidence: dict[str, Any], *, source: str = "auto", repair_class: str = "investigate") -> dict[str, Any]:
    cid = _case_id(kind, str(target.get("id") or target.get("preview") or ""), title + str(evidence))
    return {
        "case_id": cid,
        "source": source,
        "type": kind,
        "severity": severity,
        "title": title,
        "target": target,
        "evidence": evidence,
        "repair_class": repair_class,
        "required_flow": list(REQUIRED_FLOW),
        "status": "open",
        "created_at": now_iso(),
    }


def manual_conflict_case(*, claim: str, target_id: str | None = None, target_text: str | None = None, severity: str = "P1") -> dict[str, Any]:
    target = {"id": target_id or "", "preview": redact(target_text or claim, 500)}
    return _case(
        "manual_conflict",
        severity,
        "Manual conflict/quality concern specified by user",
        target,
        {"claim": redact(claim, 1000)},
        source="manual",
        repair_class="manual_trace_required",
    )


def _source_ref_known(ref: str, known_document_ids: set[str], known_memory_ids: set[str], known_file_paths: set[str]) -> bool:
    if not ref:
        return False
    if ref in known_document_ids or ref in known_memory_ids or ref in known_file_paths:
        return True
    alias = offline_consolidation_doc_alias(ref)
    if alias and alias in known_document_ids:
        return True
    if ref.startswith("/") or ref.startswith("~"):
        return Path(ref).expanduser().exists()
    # Paths may be stored relative to home/offline root; caller can add known_file_paths.
    return False


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase_l = phrase.lower()
    if re.fullmatch(r"[a-z0-9_\- ]+", phrase_l):
        return re.search(r"(?<![a-z0-9_\-])" + re.escape(phrase_l) + r"(?![a-z0-9_\-])", text) is not None
    return phrase_l in text


def _polarity(text: str) -> dict[str, set[str]]:
    lower = (text or "").lower()
    out = {"positive": set(), "negative": set()}
    for name, positive, negative in POLARITY_GROUPS:
        neg_hit = any(_contains_phrase(lower, n) for n in negative)
        pos_hit = any(_contains_phrase(lower, p) for p in positive)
        # Negated forms such as "not default" / "不允许" should not also count as
        # positive simply because they contain "default" / "允许" as a substring.
        if neg_hit:
            out["negative"].add(name)
        if pos_hit and not neg_hit:
            out["positive"].add(name)
    return out


def build_conflict_cases(
    observations: list[dict[str, Any]],
    *,
    known_document_ids: set[str] | None = None,
    known_memory_ids: set[str] | None = None,
    known_file_paths: set[str] | None = None,
    max_cases: int = 200,
) -> list[dict[str, Any]]:
    known_document_ids = {str(x) for x in (known_document_ids or set())}
    known_memory_ids = {str(x).lower() for x in (known_memory_ids or set())}
    known_file_paths = {str(x) for x in (known_file_paths or set())}
    cases: list[dict[str, Any]] = []

    for obs in observations:
        oid = str(obs.get("id") or sha1_short(observation_text(obs)))
        target = {
            "id": oid,
            "topic": obs.get("topic"),
            "type": obs.get("type"),
            "preview": redact(str(obs.get("insight") or ""), 500),
        }
        text = observation_text(obs)
        contamination = detect_contamination(text)
        if contamination:
            cases.append(_case(
                "contamination",
                max_severity(contamination, "P1"),
                "High-level observation contains raw/tool/log contamination signals",
                target,
                {"hits": contamination},
                repair_class="quarantine_then_trace",
            ))

        refs = source_refs(obs)
        if not refs:
            cases.append(_case(
                "missing_lineage",
                "P1",
                "Observation has no evidence_ids/source_documents and cannot be traced",
                target,
                {"source_documents": obs.get("source_documents") or [], "evidence_ids": obs.get("evidence_ids") or []},
                repair_class="lineage_required",
            ))
        else:
            traceable_source_refs = []
            for ref in refs:
                if UUID_RE.findall(ref):
                    continue
                if _source_ref_known(ref, known_document_ids, known_memory_ids, known_file_paths):
                    traceable_source_refs.append(ref)
            dangling_docs = []
            dangling_memory = []
            for ref in refs:
                uuids = UUID_RE.findall(ref)
                if uuids:
                    if known_memory_ids:
                        for uid in uuids:
                            if uid.lower() not in known_memory_ids:
                                dangling_memory.append(uid)
                    continue
                if (ref.startswith("hermes-") or ref.startswith("/") or ref.startswith("~")) and not _source_ref_known(ref, known_document_ids, known_memory_ids, known_file_paths):
                    dangling_docs.append(ref)
            if dangling_docs:
                if traceable_source_refs:
                    cases.append(_case(
                        "partial_dangling_source_document",
                        "P3",
                        "Observation has some stale source references, but at least one source document/file is traceable",
                        target,
                        {"dangling_source_documents": dangling_docs[:20], "traceable_source_refs": traceable_source_refs[:20]},
                        repair_class="lineage_cleanup_nonblocking",
                    ))
                else:
                    cases.append(_case(
                        "dangling_source_document",
                        "P1",
                        "Observation references source documents/files that were not found",
                        target,
                        {"dangling_source_documents": dangling_docs[:20]},
                        repair_class="trace_or_quarantine",
                    ))
            if dangling_memory:
                if traceable_source_refs:
                    cases.append(_case(
                        "stale_evidence_id",
                        "P3",
                        "Observation has stale memory UUID references, but source document/file lineage is traceable",
                        target,
                        {"dangling_memory_ids": dangling_memory[:20], "traceable_source_refs": traceable_source_refs[:20]},
                        repair_class="lineage_cleanup_nonblocking",
                    ))
                else:
                    cases.append(_case(
                        "dangling_evidence_id",
                        "P2",
                        "Observation references memory UUIDs not found in DB",
                        target,
                        {"dangling_memory_ids": dangling_memory[:20]},
                        repair_class="trace_or_repair_lineage",
                    ))

    # Conservative candidate detectors. These are P2 by default: useful for review,
    # not automatically destructive.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for obs in observations:
        text = observation_text(obs)
        terms = significant_terms(" ".join(map(str, obs.get("tags") or [])) + " " + text, limit=5)
        if not terms:
            continue
        key = (str(obs.get("topic") or "").lower(), str(obs.get("type") or "").lower(), "|".join(terms[:3]))
        grouped.setdefault(key, []).append(obs)

    for key, rows in grouped.items():
        if len(rows) < 2:
            continue
        num_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for obs in rows:
            for claim in numeric_claims(str(obs.get("insight") or "")):
                num_groups.setdefault(str(claim.get("key")), []).append((obs, claim))
        for numeric_key, entries in num_groups.items():
            obs_ids = {str(o.get("id")) for o, _ in entries}
            if len(obs_ids) < 2:
                continue
            values = [float(c["value"]) for _, c in entries]
            lo, hi = min(values), max(values)
            denom = max(abs(lo), abs(hi), 1e-9)
            if (hi - lo) / denom >= 0.5 and abs(hi - lo) >= 0.1:
                cases.append(_case(
                    "numeric_divergence_candidate",
                    "P2",
                    "Same topic/type/terms have divergent comparable numeric claims; scope may be mixed",
                    {"id": "|".join([str(x) for x in key]), "topic": key[0], "preview": "numeric divergence group"},
                    {
                        "numeric_key": numeric_key,
                        "range": [lo, hi],
                        "examples": [
                            {"id": str(o.get("id")), "insight": redact(str(o.get("insight") or ""), 240), "numbers": [str(c.get("token"))], "metric": c.get("metric"), "unit": c.get("unit")}
                            for o, c in entries[:8]
                        ],
                    },
                    repair_class="scope_trace_required",
                ))

        polarities = []
        for obs in rows:
            p = _polarity(str(obs.get("insight") or ""))
            if p["positive"] or p["negative"]:
                polarities.append((obs, p))
        for name, _, _ in POLARITY_GROUPS:
            positives = [o for o, p in polarities if name in p["positive"]]
            negatives = [o for o, p in polarities if name in p["negative"]]
            if positives and negatives:
                cases.append(_case(
                    "polarity_conflict_candidate",
                    "P2",
                    f"Potential polarity conflict for '{name}' within same topic/type/terms",
                    {"id": "|".join([str(x) for x in key]), "topic": key[0], "preview": f"polarity conflict: {name}"},
                    {
                        "positive_examples": [{"id": str(o.get("id")), "insight": redact(str(o.get("insight") or ""), 220)} for o in positives[:5]],
                        "negative_examples": [{"id": str(o.get("id")), "insight": redact(str(o.get("insight") or ""), 220)} for o in negatives[:5]],
                    },
                    repair_class="scope_or_supersede_review",
                ))
                break

    # Deduplicate stable cases while preserving first occurrence.
    seen = set()
    deduped = []
    for case in cases:
        cid = case["case_id"]
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(case)
        if len(deduped) >= max_cases:
            break
    return deduped


def repair_proposal_for_case(case: dict[str, Any]) -> dict[str, Any]:
    kind = str(case.get("type") or "manual_conflict")
    target = case.get("target") or {}
    base_actions = [
        {"action": "trace_lineage", "detail": "Trace target to source facts/documents/raw spans before any mutation."},
        {"action": "low_level_feature_audit", "detail": "Check evidence count, numeric coverage, contamination hits, source scope, and raw cleaning decision."},
        {"action": "classify_root_cause", "detail": "Classify as raw_cleaning / retain / daily / weekly / v2_reduce / recall_rerank / publish_state."},
    ]
    if kind == "contamination":
        actions = base_actions + [
            {"action": "quarantine_candidate", "detail": "If contamination is confirmed, quarantine the observation/document first; do not delete immediately."},
            {"action": "reprocess_impacted_branch", "detail": "Re-retain/rebuild only the impacted document/day/topic branch after filtering the pollutant."},
            {"action": "add_regression_case", "detail": "Add a contamination smoke query or audit rule for the detected pattern."},
        ]
    elif kind in {"missing_lineage", "dangling_source_document", "dangling_evidence_id"}:
        actions = base_actions + [
            {"action": "repair_or_rebuild_lineage", "detail": "Recover source ids from local JSON/DB/raw bundle; if untraceable, keep local-only or quarantine."},
            {"action": "block_publish_until_traceable", "detail": "Do not publish high-level canonical facts without traceable evidence."},
        ]
    elif kind in {"numeric_divergence_candidate", "polarity_conflict_candidate"}:
        actions = base_actions + [
            {"action": "compare_scopes", "detail": "Check whether conflicting claims belong to different configs/dates/scenes; narrow applicability if needed."},
            {"action": "supersede_or_split_claims", "detail": "If one claim is stale, supersede it; if both valid, split by applicability/scope."},
            {"action": "re_eval_targeted_benchmarks", "detail": "Add/refresh benchmark case covering the conflict before publish."},
        ]
    else:
        actions = base_actions + [
            {"action": "manual_review", "detail": "User-specified conflict: produce a concrete quarantine/supersede/re-retain/delete/rebuild plan after tracing."},
            {"action": "execute_only_after_confirmation", "detail": "Any delete, overwrite, publish, or external write requires explicit user confirmation."},
        ]
    return {
        "case_id": case.get("case_id"),
        "case_type": kind,
        "severity": case.get("severity"),
        "title": case.get("title"),
        "status": case.get("status"),
        "target": target,
        "evidence": case.get("evidence") or {},
        "required_flow": list(case.get("required_flow") or REQUIRED_FLOW),
        "repair_class": case.get("repair_class"),
        "recommended_actions": actions,
        "human_confirmation_required": True,
        "destructive_actions_allowed_without_confirmation": False,
        "notes": "This is a proposal only. It performs no DB/file deletion and no Hindsight writes.",
    }
