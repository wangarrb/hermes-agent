#!/usr/bin/env python3
"""Read-only hardening diagnostics for Hindsight session/json production retain.

Phase-A tool for long-term stable session/json governance:
- classify zero-unit documents using generic structural features;
- score recall-smoke output with generic top-k metrics;
- write reproducible JSON/Markdown reports;
- never mutate Hindsight production data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from hindsight_bank_quality_audit import redact_secrets, sql_literal  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone use
    def redact_secrets(value: Any) -> Any:
        return value

    def sql_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

DEFAULT_BANK = "hermes"
DEFAULT_REPORT_DIR = Path.home() / ".hermes" / "hindsight" / "reports"
DEFAULT_PSQL = "/home/wyr/.pg0/installation/18.1.0/bin/psql"

VALUE_CLASS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "user_preference": [
        re.compile(r"用户.{0,20}(偏好|喜欢|不喜欢|希望|要求|倾向|默认|习惯|风格)"),
        re.compile(r"\b(user|client).{0,40}\b(prefers?|preference|wants?|expects?|requires?)\b", re.I),
    ],
    "durable_decision": [
        re.compile(r"(决定|确认|采用|选择|放弃|废弃|默认|最终|固定|结论|策略|方案|路线)"),
        re.compile(r"\b(decision|decided|choose|chosen|default|strategy|policy|final|deprecate|abandon)\b", re.I),
    ],
    "project_state": [
        re.compile(r"(项目|分支|仓库|版本|状态|进度|当前状态|已完成|未完成|下一步)"),
        re.compile(r"\b(project|repo|branch|version|status|progress|next step|todo|completed|pending)\b", re.I),
    ],
    "experiment_result": [
        re.compile(r"(实验|验证|评测|指标|结果|对比|基线|消融|误差|精度|召回|通过|失败)"),
        re.compile(r"\b(experiment|benchmark|metric|result|baseline|ablation|accuracy|recall|precision|ATE|RPE|pass(ed)?|fail(ed)?)\b", re.I),
    ],
    "tool_lesson": [
        re.compile(r"(工具|脚本|命令|配置|容器|服务|provider|模型|接口|API|skill|cron|systemd)"),
        re.compile(r"\b(tool|script|command|config|docker|service|provider|model|endpoint|api|skill|cron|systemd)\b", re.I),
    ],
    "error_root_cause": [
        re.compile(r"(根因|原因|修复|报错|错误|失败|异常|超时|卡住|回退|坑|风险)"),
        re.compile(r"\b(root cause|fix(ed)?|error|exception|traceback|timeout|stuck|retry|rollback|pitfall|risk|bug|429)\b", re.I),
    ],
    "environment_fact": [
        re.compile(r"(路径|端口|环境变量|数据库|容器|服务|版本|安装|部署|配置文件|日志)"),
        re.compile(r"\b(path|port|env|database|container|service|version|install|deploy|config|log|health)\b", re.I),
    ],
    "open_question": [
        re.compile(r"(待确认|待实现|需要确认|问题|疑问|是否|下一步|TODO|未解决|open question)"),
        re.compile(r"\b(open question|unknown|unclear|needs? confirmation|todo|follow[- ]?up)\b", re.I),
    ],
}

NOISE_PATTERNS: dict[str, re.Pattern[str]] = {
    "tool_marker": re.compile(r"\b(RUN|EXIT|STDOUT|STDERR|tool_calls_made|process|session_id|pid=|returncode)\b", re.I),
    "stack_or_error_log": re.compile(r"\b(Traceback|File \".*\", line \d+|Error while|Exception|BrokenPipeError|Permission denied|ConnectionRefused|Invalid control character)\b", re.I),
    "json_or_config_dump": re.compile(r"^[\s{\[\]}\],:\"'0-9A-Za-z_\-./]+$"),
    "path_or_hash": re.compile(r"(/home/|~/.|[A-Fa-f0-9]{32,}|sha256|\.jsonl?|\.py|\.md|\.log)"),
    "compression_or_resume": re.compile(r"(CONTEXT COMPACTION|Active Task|/mycompress|/resume|刚聊到哪|继续|handoff|compression)", re.I),
    "code_fence": re.compile(r"```"),
}

TOKEN_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "api", "json", "session", "user",
    "用户", "查询", "信息", "数据", "相关", "问题", "方案", "系统",
}


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def tags_of(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    return []


def bucket_number(value: float, buckets: list[tuple[float, str]], default: str) -> str:
    for upper, label in buckets:
        if value < upper:
            return label
    return default


def extract_metadata(doc: dict[str, Any], manifest_record: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    retain_params = safe_json_loads(doc.get("retain_params"), {}) or {}
    retain_meta = safe_json_loads(retain_params.get("metadata"), {}) or {}
    doc_meta = safe_json_loads(doc.get("metadata"), {}) or {}
    manifest_meta = safe_json_loads((manifest_record or {}).get("metadata"), {}) or {}
    for source in (manifest_meta, doc_meta, retain_meta):
        if isinstance(source, dict):
            meta.update(source)
    cleaning = safe_json_loads(meta.get("cleaning_stats"), {}) or {}
    if isinstance(cleaning, dict):
        meta["cleaning_stats"] = cleaning
    return meta


def semantic_hits(text: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    for cls, patterns in VALUE_CLASS_PATTERNS.items():
        count = 0
        for pat in patterns:
            count += len(pat.findall(text))
        if count:
            hits[cls] = count
    return hits


def noise_features(text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_count = max(len(lines), 1)
    counts: dict[str, int] = {}
    noisy_lines = 0
    for name, pat in NOISE_PATTERNS.items():
        matches = pat.findall(text)
        counts[name] = len(matches)
    for ln in lines:
        if any(pat.search(ln) for pat in NOISE_PATTERNS.values()):
            noisy_lines += 1
    code_like_lines = sum(1 for ln in lines if ln.startswith(("{", "}", "[", "]", "- ", "* ")) or "```" in ln)
    counts["code_like_lines"] = code_like_lines
    counts["line_count"] = line_count
    counts["noisy_lines"] = noisy_lines
    ratio = min(1.0, noisy_lines / line_count)
    if line_count <= 3 and sum(counts.values()) > 5:
        ratio = max(ratio, 0.5)
    return {"counts": counts, "transcript_noise_ratio": round(ratio, 4)}


def classify_doc(doc: dict[str, Any], manifest_record: dict[str, Any] | None = None) -> dict[str, Any]:
    text = str(doc.get("original_text") or doc.get("text") or "")
    chars = len(text)
    tags = tags_of((manifest_record or {}).get("tags") or doc.get("tags"))
    unit_count = int(doc.get("unit_count") or 0)
    meta = extract_metadata(doc, manifest_record)
    cleaning = safe_json_loads(meta.get("cleaning_stats"), {}) or {}
    message_count = int(meta.get("message_count") or 0)
    kept_messages = int(cleaning.get("kept_messages") or 0)
    dropped_messages = int(cleaning.get("dropped_messages") or 0)
    dropped_noise_messages = int(cleaning.get("dropped_noise_messages") or 0)
    total_cleaning_messages = kept_messages + dropped_messages
    dropped_ratio = round(dropped_messages / total_cleaning_messages, 4) if total_cleaning_messages else None
    noise = noise_features(text)
    sem = semantic_hits(text)
    semantic_score = sum(min(v, 5) for v in sem.values())
    primary_value_classes = [k for k, _ in sorted(sem.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
    multi_scope = len(tags) >= 3 or len(primary_value_classes) >= 3
    high_value = semantic_score >= 4 or any(cls in sem for cls in ("user_preference", "durable_decision", "experiment_result", "error_root_cause"))
    command_log_heavy = noise["transcript_noise_ratio"] >= 0.35 or noise["counts"].get("tool_marker", 0) >= 8 or noise["counts"].get("stack_or_error_log", 0) >= 3
    compression_like = noise["counts"].get("compression_or_resume", 0) >= 2
    overlong = chars >= 12000 or int((manifest_record or {}).get("estimated_retain_chunks") or 0) >= 2
    cleaning_loss_risk = (dropped_ratio is not None and dropped_ratio >= 0.85 and kept_messages <= 8) or (message_count and kept_messages and kept_messages / max(message_count, 1) <= 0.08)

    if unit_count > 0:
        zero_unit_class = "has_units"
        recommended_route = "keep"
    elif not high_value and chars < 2000:
        zero_unit_class = "true_low_signal"
        recommended_route = "skip_or_raw_only"
    elif compression_like and not high_value:
        zero_unit_class = "context_bootstrap_or_resume_noise"
        recommended_route = "manual_review_or_raw_only"
    elif command_log_heavy and high_value:
        zero_unit_class = "noisy_high_value_transcript"
        recommended_route = "production_windowed"
    elif command_log_heavy:
        zero_unit_class = "noisy_transcript"
        recommended_route = "raw_only"
    elif overlong and multi_scope:
        zero_unit_class = "overlong_or_multi_scope"
        recommended_route = "production_windowed"
    elif cleaning_loss_risk and high_value:
        zero_unit_class = "cleaning_lost_context_risk"
        recommended_route = "retry_less_aggressive_cleaning"
    elif high_value:
        zero_unit_class = "extraction_too_strict_candidate"
        recommended_route = "retry_custom_mission"
    else:
        zero_unit_class = "unclassified_zero_unit"
        recommended_route = "manual_review"

    return {
        "document_id": doc.get("id") or doc.get("document_id"),
        "unit_count": unit_count,
        "chars": chars,
        "tags": tags,
        "metadata": {
            "model": meta.get("model"),
            "started_at": meta.get("started_at"),
            "json_path": meta.get("json_path"),
            "message_count": message_count or None,
            "cleaning_stats": cleaning,
        },
        "features": {
            "semantic_hits": sem,
            "semantic_score": semantic_score,
            "primary_value_classes": primary_value_classes,
            "high_value": high_value,
            "noise": noise,
            "dropped_ratio": dropped_ratio,
            "dropped_noise_messages": dropped_noise_messages,
            "multi_scope": multi_scope,
            "overlong": overlong,
            "cleaning_lost_context_risk": cleaning_loss_risk,
        },
        "zero_unit_class": zero_unit_class,
        "recommended_route": recommended_route,
        "preview": text[:280].replace("\n", " "),
    }


def load_manifest(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            did = rec.get("document_id")
            if did:
                out[str(did)] = rec
    return out


def load_bank_documents(bank: str, *, psql: str | None = None, host: str = "/tmp", port: int = 5432, db: str = "hindsight", user: str = "hindsight") -> list[dict[str, Any]]:
    psql = psql or os.environ.get("HINDSIGHT_PSQL") or DEFAULT_PSQL
    bank_lit = sql_literal(bank)
    sql = rf'''
WITH doc_units AS (
  SELECT d.id, d.bank_id, d.original_text, d.tags, d.metadata, d.retain_params, d.created_at, d.updated_at,
         COUNT(m.id)::int AS unit_count
  FROM documents d
  LEFT JOIN memory_units m ON m.bank_id = d.bank_id AND m.document_id = d.id
  WHERE d.bank_id = {bank_lit}
  GROUP BY d.id, d.bank_id, d.original_text, d.tags, d.metadata, d.retain_params, d.created_at, d.updated_at
)
SELECT COALESCE(jsonb_agg(to_jsonb(doc_units) ORDER BY created_at), '[]'::jsonb)::text FROM doc_units;
'''
    cmd = [psql, "-h", host, "-p", str(port), "-U", user, "-d", db, "-q", "-t", "-A", "-c", sql]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    raw = proc.stdout.strip()
    if not raw:
        return []
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def build_manifest_derived_benchmark_candidates(classified: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    """Build reviewable generic recall benchmark candidates from retained docs.

    This does not call recall and does not assume any project-specific vocabulary.
    It turns observed tags + generic value classes into candidate query specs that
    can later be promoted into a benchmark JSONL after review.
    """
    tag_docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in classified:
        for tag in doc.get("tags") or []:
            tag_docs[str(tag)].append(doc)
    candidates: list[dict[str, Any]] = []
    for tag, docs in tag_docs.items():
        if len(docs) < 2:
            continue
        tag_terms = [part for part in re.split(r"[:_\-/]+", tag) if part and part.lower() not in TOKEN_STOPWORDS]
        class_counts = Counter(cls for d in docs for cls in (d.get("features") or {}).get("primary_value_classes") or [])
        class_terms = [cls.replace("_", " ") for cls, _ in class_counts.most_common(2)]
        query_terms = tag_terms + class_terms
        if not query_terms:
            continue
        candidates.append({
            "id": "derived_tag_" + re.sub(r"[^A-Za-z0-9]+", "_", tag).strip("_").lower(),
            "query": " ".join(query_terms),
            "expected_terms": tag_terms,
            "source": "manifest_tag_value_class",
            "tag": tag,
            "support_doc_count": len(docs),
            "unit_doc_count": sum(1 for d in docs if d.get("unit_count", 0) > 0),
            "zero_unit_doc_count": sum(1 for d in docs if d.get("unit_count", 0) == 0),
            "top_value_classes": class_counts.most_common(5),
        })
    candidates.sort(key=lambda c: (c["support_doc_count"], c["unit_doc_count"]), reverse=True)
    return candidates[:limit]


def summarize_zero_units(classified: list[dict[str, Any]], *, max_samples: int = 20) -> dict[str, Any]:
    docs = classified
    zero = [d for d in docs if d.get("unit_count") == 0]
    total_chars = sum(int(d.get("chars") or 0) for d in docs)
    zero_chars = sum(int(d.get("chars") or 0) for d in zero)
    by_class = Counter(d.get("zero_unit_class") for d in zero)
    by_route = Counter(d.get("recommended_route") for d in zero)
    by_model = Counter((d.get("metadata") or {}).get("model") or "<unknown>" for d in zero)
    tag_counts = Counter(t for d in zero for t in d.get("tags") or [])
    length_buckets = Counter(bucket_number(int(d.get("chars") or 0), [(1000, "<1k"), (3000, "1-3k"), (8000, "3-8k"), (20000, "8-20k")], "20k+") for d in zero)
    dropped_buckets = Counter(bucket_number(float((d.get("features") or {}).get("dropped_ratio") or 0), [(0.25, "<25%"), (0.5, "25-50%"), (0.75, "50-75%"), (0.9, "75-90%")], ">=90%") for d in zero)
    retry_candidates = [d for d in zero if d.get("recommended_route") in {"retry_custom_mission", "production_windowed", "retry_less_aggressive_cleaning"}]
    retry_candidates.sort(key=lambda d: ((d.get("features") or {}).get("semantic_score") or 0, d.get("chars") or 0), reverse=True)
    samples = sorted(zero, key=lambda d: ((d.get("features") or {}).get("semantic_score") or 0, d.get("chars") or 0), reverse=True)[:max_samples]
    return {
        "total_documents": len(docs),
        "zero_unit_documents": len(zero),
        "zero_unit_ratio": round(len(zero) / len(docs), 4) if docs else 0,
        "total_chars": total_chars,
        "zero_unit_chars": zero_chars,
        "zero_unit_chars_ratio": round(zero_chars / total_chars, 4) if total_chars else 0,
        "by_zero_unit_class": dict(by_class.most_common()),
        "by_recommended_route": dict(by_route.most_common()),
        "by_model": dict(by_model.most_common()),
        "by_length_bucket": dict(sorted(length_buckets.items())),
        "by_dropped_ratio_bucket": dict(sorted(dropped_buckets.items())),
        "top_tags": tag_counts.most_common(30),
        "high_value_retry_candidate_count": len(retry_candidates),
        "high_value_retry_candidates": [compact_doc(d) for d in retry_candidates[:max_samples]],
        "samples": [compact_doc(d) for d in samples],
    }


def compact_doc(d: dict[str, Any]) -> dict[str, Any]:
    features = d.get("features") or {}
    return {
        "document_id": d.get("document_id"),
        "chars": d.get("chars"),
        "tags": d.get("tags"),
        "zero_unit_class": d.get("zero_unit_class"),
        "recommended_route": d.get("recommended_route"),
        "semantic_score": features.get("semantic_score"),
        "primary_value_classes": features.get("primary_value_classes"),
        "noise_ratio": (features.get("noise") or {}).get("transcript_noise_ratio"),
        "dropped_ratio": features.get("dropped_ratio"),
        "model": (d.get("metadata") or {}).get("model"),
        "preview": d.get("preview"),
    }


def tokenize_query(query: str) -> list[str]:
    terms: list[str] = []
    for raw in re.split(r"\s+", query.strip()):
        token = raw.strip(" ,.;:!?()[]{}<>`'\"")
        if not token:
            continue
        lower = token.lower()
        if len(lower) <= 2 and not re.search(r"[\u4e00-\u9fff]", lower):
            continue
        if lower in TOKEN_STOPWORDS or token in TOKEN_STOPWORDS:
            continue
        terms.append(token)
    if not terms:
        terms = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", query)
        terms = [t for t in terms if t.lower() not in TOKEN_STOPWORDS and t not in TOKEN_STOPWORDS]
    # Keep order while deduplicating case-insensitively.
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def matched_terms(row: dict[str, Any], terms: list[str]) -> list[str]:
    hay = (str(row.get("text") or "") + " " + " ".join(tags_of(row.get("tags")))).lower()
    return [t for t in terms if t.lower() in hay]


def score_recall_rows(query: str, rows: list[dict[str, Any]], *, expected_terms: list[str] | None = None, min_match_terms: int | None = None) -> dict[str, Any]:
    terms = expected_terms or tokenize_query(query)
    if min_match_terms is None:
        min_match_terms = max(1, min(2, math.ceil(len(terms) * 0.25))) if terms else 1
    scored = []
    relevant_flags: list[bool] = []
    tag_counter: Counter[str] = Counter()
    doc_prefix_counter: Counter[str] = Counter()
    for rank, row in enumerate(rows, start=1):
        mt = matched_terms(row, terms)
        relevant = len(mt) >= min_match_terms
        relevant_flags.append(relevant)
        for tag in tags_of(row.get("tags")):
            tag_counter[tag] += 1
        doc_prefix_counter[str(row.get("doc_prefix") or row.get("document_id") or "<unknown>")] += 1
        scored.append({
            "rank": rank,
            "relevant": relevant,
            "matched_terms": mt,
            "type": row.get("type"),
            "doc_prefix": row.get("doc_prefix"),
            "tags": tags_of(row.get("tags"))[:10],
            "text": str(row.get("text") or "")[:220],
        })
    k = len(rows)
    relevant_count = sum(1 for x in relevant_flags if x)
    first_rank = next((i for i, x in enumerate(relevant_flags, start=1) if x), None)
    dominant_tag, dominant_tag_count = tag_counter.most_common(1)[0] if tag_counter else (None, 0)
    dominant_prefix, dominant_prefix_count = doc_prefix_counter.most_common(1)[0] if doc_prefix_counter else (None, 0)
    covered_terms = sorted({term for s in scored for term in s["matched_terms"]}, key=lambda x: x.lower())
    return {
        "query": query,
        "expected_terms": terms,
        "min_match_terms": min_match_terms,
        "k": k,
        "relevant_count": relevant_count,
        "precision_at_k": round(relevant_count / k, 4) if k else 0,
        "off_topic_rate": round(1 - relevant_count / k, 4) if k else 0,
        "mrr": round(1 / first_rank, 4) if first_rank else 0,
        "first_relevant_rank": first_rank,
        "covered_expected_terms": covered_terms,
        "expected_term_coverage": round(len(covered_terms) / len(terms), 4) if terms else 0,
        "dominant_tag": dominant_tag,
        "dominant_tag_ratio": round(dominant_tag_count / k, 4) if k else 0,
        "dominant_doc_prefix": dominant_prefix,
        "dominant_doc_prefix_ratio": round(dominant_prefix_count / k, 4) if k else 0,
        "rows": scored,
    }


def load_benchmark(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("id") or rec.get("qid") or rec.get("query"))
            out[qid] = rec
    return out


def score_recall_audit(audit_json: str | Path | None, *, benchmark_jsonl: str | Path | None = None) -> dict[str, Any]:
    if not audit_json:
        return {"status": "not_run", "reason": "no_audit_json"}
    p = Path(audit_json)
    if not p.exists():
        return {"status": "not_run", "reason": f"missing_audit_json:{p}"}
    data = json.loads(p.read_text(encoding="utf-8"))
    smoke = data.get("recall_smoke") or {}
    bench = load_benchmark(benchmark_jsonl)
    per_query: dict[str, Any] = {}
    for qid, qdata in smoke.items():
        query = str(qdata.get("query") or qid)
        rows = list(qdata.get("rows") or [])
        spec = bench.get(qid) or bench.get(query) or {}
        expected_terms = spec.get("expected_terms") or None
        if isinstance(expected_terms, str):
            expected_terms = tokenize_query(expected_terms)
        min_match_terms = spec.get("min_match_terms")
        try:
            min_match_terms = int(min_match_terms) if min_match_terms is not None else None
        except Exception:
            min_match_terms = None
        per_query[qid] = score_recall_rows(query, rows, expected_terms=expected_terms, min_match_terms=min_match_terms)
    precision_values = [v["precision_at_k"] for v in per_query.values() if v.get("k")]
    dominant_values = [v["dominant_tag_ratio"] for v in per_query.values() if v.get("k")]
    return {
        "status": "ok",
        "query_count": len(per_query),
        "macro_precision_at_k": round(sum(precision_values) / len(precision_values), 4) if precision_values else 0,
        "max_dominant_tag_ratio": max(dominant_values) if dominant_values else 0,
        "queries_below_precision_0_5": [qid for qid, v in per_query.items() if v.get("k") and v.get("precision_at_k", 0) < 0.5],
        "queries_dominant_tag_over_0_75": [qid for qid, v in per_query.items() if v.get("k") and v.get("dominant_tag_ratio", 0) > 0.75],
        "per_query": per_query,
    }


def write_markdown(result: dict[str, Any]) -> str:
    zero = result.get("zero_unit_report") or {}
    recall = result.get("recall_benchmark") or {}
    lines: list[str] = []
    lines.append("# Hindsight session/json production hardening Phase A")
    lines.append("")
    lines.append(f"- generated_at: `{result.get('generated_at')}`")
    lines.append(f"- bank: `{result.get('bank')}`")
    lines.append(f"- manifest: `{result.get('manifest')}`")
    lines.append(f"- audit_json: `{result.get('audit_json')}`")
    lines.append("")
    lines.append("## Zero-unit summary")
    lines.append("")
    for key in ["total_documents", "zero_unit_documents", "zero_unit_ratio", "zero_unit_chars", "zero_unit_chars_ratio", "high_value_retry_candidate_count"]:
        lines.append(f"- {key}: `{zero.get(key)}`")
    lines.append("")
    lines.append("### By zero-unit class")
    lines.append("```json")
    lines.append(json.dumps(zero.get("by_zero_unit_class", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("### By recommended route")
    lines.append("```json")
    lines.append(json.dumps(zero.get("by_recommended_route", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("### High-value retry candidates")
    for item in (zero.get("high_value_retry_candidates") or [])[:10]:
        lines.append(f"- `{item.get('document_id')}` class={item.get('zero_unit_class')} route={item.get('recommended_route')} sem={item.get('semantic_score')} noise={item.get('noise_ratio')} chars={item.get('chars')} tags={item.get('tags')}")
    lines.append("")
    lines.append("## Recall benchmark")
    lines.append("")
    lines.append(f"- status: `{recall.get('status')}`")
    lines.append(f"- query_count: `{recall.get('query_count')}`")
    lines.append(f"- macro_precision_at_k: `{recall.get('macro_precision_at_k')}`")
    lines.append(f"- max_dominant_tag_ratio: `{recall.get('max_dominant_tag_ratio')}`")
    lines.append(f"- queries_below_precision_0_5: `{recall.get('queries_below_precision_0_5')}`")
    lines.append(f"- queries_dominant_tag_over_0_75: `{recall.get('queries_dominant_tag_over_0_75')}`")
    lines.append("")
    derived = result.get("manifest_derived_benchmark_candidates") or []
    lines.append("## Manifest-derived benchmark candidates")
    lines.append("")
    for item in derived[:12]:
        lines.append(f"- `{item.get('id')}` query=`{item.get('query')}` tag={item.get('tag')} support={item.get('support_doc_count')} units={item.get('unit_doc_count')} zero={item.get('zero_unit_doc_count')}")
    lines.append("")
    for qid, q in (recall.get("per_query") or {}).items():
        lines.append(f"### {qid}")
        lines.append(f"- precision_at_k={q.get('precision_at_k')} off_topic_rate={q.get('off_topic_rate')} mrr={q.get('mrr')} dominant_tag={q.get('dominant_tag')} dominant_tag_ratio={q.get('dominant_tag_ratio')}")
    lines.append("")
    return "\n".join(lines)


def run_phase_a(*, bank: str, manifest: str | Path | None, audit_json: str | Path | None, benchmark_jsonl: str | Path | None, max_samples: int) -> dict[str, Any]:
    manifest_by_id = load_manifest(manifest)
    docs = load_bank_documents(bank)
    classified = []
    for doc in docs:
        did = str(doc.get("id") or doc.get("document_id") or "")
        classified.append(classify_doc(doc, manifest_by_id.get(did)))
    zero_report = summarize_zero_units(classified, max_samples=max_samples)
    recall_report = score_recall_audit(audit_json, benchmark_jsonl=benchmark_jsonl)
    derived_candidates = build_manifest_derived_benchmark_candidates(classified)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bank": bank,
        "manifest": str(manifest) if manifest else None,
        "audit_json": str(audit_json) if audit_json else None,
        "benchmark_jsonl": str(benchmark_jsonl) if benchmark_jsonl else None,
        "mode": "read_only_phase_a",
        "zero_unit_report": zero_report,
        "recall_benchmark": recall_report,
        "manifest_derived_benchmark_candidates": derived_candidates,
    }


def write_reports(result: dict[str, Any], output_dir: str | Path, *, stem: str | None = None) -> dict[str, str]:
    result = redact_secrets(result)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        stem = f"{stamp}-hindsight-session-quality-hardening-{result.get('bank','bank')}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(write_markdown(result), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only Hindsight session/json production hardening diagnostics")
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--manifest", help="Session/json manifest JSONL used for the retain run")
    ap.add_argument("--audit-json", help="Quality audit JSON containing recall_smoke rows")
    ap.add_argument("--benchmark-jsonl", help="Optional generic recall benchmark specs with expected_terms")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--stem")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="Print JSON result to stdout")
    args = ap.parse_args(argv)
    result = run_phase_a(bank=args.bank, manifest=args.manifest, audit_json=args.audit_json, benchmark_jsonl=args.benchmark_jsonl, max_samples=args.max_samples)
    paths = write_reports(result, args.output_dir, stem=args.stem)
    result["paths"] = paths
    result = redact_secrets(result)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        z = result.get("zero_unit_report", {})
        r = result.get("recall_benchmark", {})
        print(f"json={paths['json']}")
        print(f"md={paths['md']}")
        print(f"zero_unit_documents={z.get('zero_unit_documents')} ratio={z.get('zero_unit_ratio')} high_value_retry_candidates={z.get('high_value_retry_candidate_count')}")
        print(f"recall_macro_precision={r.get('macro_precision_at_k')} queries_below_precision_0_5={r.get('queries_below_precision_0_5')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
