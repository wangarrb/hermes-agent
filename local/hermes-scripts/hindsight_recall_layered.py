#!/usr/bin/env python3
"""Layered recall helper for Hindsight.

Modes:
- high-level: prefer canonical/offline weekly/daily summaries, fallback to raw facts.
- evidence: prefer raw sqlite facts and numeric/source evidence.
- mixed: balance high-level summaries and raw evidence.

No Hindsight writes. No LLM calls.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import urllib.request
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_API = "http://127.0.0.1:8888"
DEFAULT_BANK = "hermes"
DEFAULT_CARDS_ROOT = Path.home() / ".hermes" / "hindsight" / "offline_reflect" / "v2_cards"
DEFAULT_REPAIR_SIDECAR_ROOT = Path.home() / ".hermes" / "hindsight" / "review_repair" / "approved"
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]+|[\u4e00-\u9fff]{2,}")
ASCII_SIG_RE = re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z][A-Za-z0-9_\-]{1,}(?![A-Za-z0-9_\-])")
CJK_SEQ_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
SIG_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "onto", "via", "using", "uses", "what", "why", "how",
    "current", "latest", "conclusion", "result", "results", "summary", "关键", "当前", "现在", "什么", "是什么", "怎么",
    "结论", "原因", "分别", "处理", "问题", "项目", "用户", "是否", "能不能",
}
# Generic relation words are useful for normal token overlap, but too weak to
# promote local canonical cards by themselves. Without this filter, cards like
# "不能直接包含 Python CustomFactor" can outrank the actual "wiki 不直接写主库"
# rule just because the query also contains "能不能/直接".
GENERIC_SIG_TOKENS = {
    "不能", "能不", "能直", "直接", "接写", "写主", "新内", "容能", "内容",
    "什么", "是什", "则是", "的强", "护的", "强制", "规则", "默认",
    "结论", "是什", "论是", "关键", "原因", "问题",
}
MEASUREMENT_WORDS = {
    "ate", "rpe", "metric", "score", "precision", "recall", "error", "errors", "baseline", "benchmark", "ranking",
    "alignment_score", "指标", "数值", "误差", "精度", "排名", "对比", "退化", "改善", "ablation",
}
DETAIL_QUERY_WORDS = MEASUREMENT_WORDS | {"数字", "关键值", "evidence", "fact", "事实", "证据"}
RANKING_CUES = {"排名", "ranking", "排序", "优于", "最佳", "最优", ">"}


def query_type_hints(query: str) -> set[str]:
    """Infer desired observation types from generic bilingual query words.

    This stays domain-agnostic: it only recognizes memory-task categories such as
    preferences, decisions, risks, tooling lessons, and open questions. It is used
    to keep local canonical cards from crowding out more relevant Hindsight facts
    when a card merely matches a broad topic name.
    """
    q = (query or "").lower()
    hints: set[str] = set()
    if any(x in q for x in ["偏好", "preference", "preferences", "沟通风格", "稳定规则", "反复纠正"]):
        hints.add("user_preference")
    if any(x in q for x in ["决策", "decision", "decisions", "架构选择", "技术取舍", "推荐方案", "方案"]):
        hints.add("project_decision")
    if any(x in q for x in ["风险", "risk", "risks", "blocker", "注意事项", "不要做", "禁忌"]):
        hints.add("risk")
    if any(x in q for x in ["开放问题", "open question", "open questions", "未解决", "question", "questions"]):
        hints.add("open_question")
    if any(x in q for x in ["工具", "tooling", "配置", "config", "命令", "command", "调试", "debugging"]):
        hints.add("tooling_lesson")
    if any(x in q for x in ["技术经验", "technical lesson", "technical lessons", "经验教训", "lesson", "lessons", "根因", "为什么", "证伪", "ablation", "数值", "指标", "benchmark", "ranking", "排名", "关键值"]):
        hints.add("technical_lesson")
    if any(x in q for x in ["规则", "强制", "能不能", "必须", "默认", "policy", "rule", "rules"]):
        hints.update({"system_rule", "project_decision", "user_preference"})
    return hints


def clean_json_text(text: str) -> str:
    return CONTROL_CHARS.sub("", text or "")


def http_json(url: str, payload: dict[str, Any], timeout: int = 45) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(clean_json_text(raw))


def classify_doc(document_id: str) -> str:
    if document_id.startswith("repair-sidecar::"):
        return "approved_repair_sidecar"
    if document_id.startswith("offline-v2-card::") or document_id.startswith("offline-v2-observation::"):
        return "local_canonical"
    if document_id.startswith("hermes-offline-canonical::"):
        return "canonical"
    if document_id.startswith("hermes-offline-consolidation::weekly::"):
        return "offline_weekly"
    if document_id.startswith("hermes-offline-consolidation::daily::"):
        return "offline_daily"
    if document_id.startswith("hermes-sqlite::"):
        return "sqlite_import"
    return "other"


def tokenize(text: str) -> set[str]:
    toks = {m.group(0).lower() for m in TOKEN_RE.finditer(text or "")}
    # Add individual CJK bi-grams lightly for short Chinese queries.
    cjk = "".join(c for c in (text or "") if "\u4e00" <= c <= "\u9fff")
    if len(cjk) >= 2:
        toks.update(cjk[i : i + 2] for i in range(len(cjk) - 1))
    return toks


def significant_tokens(text: str) -> set[str]:
    """Domain-agnostic discriminative tokens for local observation scoring.

    Full CJK bi-gram overlap is useful for broad rerank, but too permissive for
    local canonical observations. This stricter token set emphasizes identifiers,
    short technical acronyms, and non-generic Chinese phrases.
    """
    out: set[str] = set()
    for m in ASCII_SIG_RE.finditer(text or ""):
        tok = m.group(0).lower().strip("-_")
        if len(tok) >= 2 and tok not in SIG_STOPWORDS:
            out.add(tok)
    for m in CJK_SEQ_RE.finditer(text or ""):
        seq = m.group(0)
        if seq not in SIG_STOPWORDS and len(seq) >= 2:
            out.add(seq)
        if len(seq) >= 4:
            for i in range(len(seq) - 1):
                bg = seq[i : i + 2]
                if bg not in SIG_STOPWORDS:
                    out.add(bg)
    return out


def has_measurement_signal(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in MEASUREMENT_WORDS) or "%" in t


def recall(api: str, bank: str, query: str, limit: int, *, include_observations: bool = True) -> list[dict[str, Any]]:
    # V2 default: include observation units explicitly. Hindsight's API default
    # only searches world/experience, so direct canonical observations would be
    # invisible without this types list. Eval baselines may disable this to
    # isolate local-card improvements from already-published canonical docs.
    types = ["world", "experience"]
    if include_observations:
        types.insert(0, "observation")
    payload = {"query": query, "limit": limit, "types": types}
    data = http_json(f"{api}/v1/default/banks/{bank}/memories/recall", payload)
    return data.get("results") or []


def query_variants(query: str, mode: str) -> list[str]:
    variants = [query]
    if mode in ("high-level", "mixed"):
        variants.extend(
            [
                query + " offline consolidation weekly daily executive summary knowledge points project decisions user preferences high-level conclusion",
                query + " canonical observation stable insight applicability evidence_ids summarized lesson decision",
            ]
        )
    if mode in ("evidence", "mixed"):
        variants.extend(
            [
                query + " exact numeric evidence raw sqlite source fact benchmark ATE RPE command path",
                query + " detailed facts source document values parameters errors",
            ]
        )
    # Preserve order and remove duplicates.
    return list(OrderedDict((v, None) for v in variants).keys())


def local_observation_result(obs: dict[str, Any], *, card_path: Path | None = None, card: dict[str, Any] | None = None, idx: int = 0) -> dict[str, Any] | None:
    text = obs.get("insight") or ""
    if not text:
        return None
    topic = obs.get("topic") or (card or {}).get("topic") or "global"
    doc_id = (card or {}).get("card_id") or f"offline-v2-observation::{topic}::{obs.get('id') or idx}"
    # Keep user-facing text concise, but include applicability when it carries
    # stable context. Tags/source docs stay in _score_text for retrieval only.
    visible = text
    if obs.get("applicability") and str(obs.get("applicability")) not in visible:
        visible = f"{visible} | {obs.get('applicability')}"
    score_parts = [
        visible,
        obs.get("type") or "",
        topic,
        " ".join(map(str, obs.get("tags") or [])),
        " ".join(map(str, obs.get("source_documents") or [])),
    ]
    return {
        "id": obs.get("id") or f"{doc_id}::{idx}",
        "document_id": doc_id,
        "type": obs.get("type") or "canonical_observation",
        "text": visible,
        "metadata": {
            "local_card_path": str(card_path) if card_path else None,
            "card_scope": (card or {}).get("scope"),
            "topic": topic,
            "source_count": len(obs.get("evidence_ids") or []),
            "source_documents": obs.get("source_documents") or [],
            "tags": obs.get("tags") or [],
            "applicability": obs.get("applicability"),
        },
        "source_fact_ids": obs.get("evidence_ids") or [],
        "mentioned_at": obs.get("valid_from"),
        "_local_card": True,
        "_score_text": " ".join(str(x) for x in score_parts if x),
    }


def repair_observation_result(obs: dict[str, Any], *, sidecar_path: Path | None = None, idx: int = 0) -> dict[str, Any] | None:
    if str(obs.get("status") or "approved") not in {"approved", "provisional"}:
        return None
    text = obs.get("insight") or obs.get("text") or ""
    if not text:
        return None
    topic = obs.get("topic") or "repair"
    doc_id = obs.get("id") or f"repair-sidecar::{topic}::{idx}"
    source_documents = obs.get("source_documents") or []
    score_parts = [
        text,
        obs.get("type") or "",
        topic,
        " ".join(map(str, obs.get("tags") or [])),
        " ".join(map(str, source_documents)),
    ]
    return {
        "id": obs.get("id") or f"{doc_id}::{idx}",
        "document_id": str(doc_id),
        "type": obs.get("type") or "repair_observation",
        "text": text,
        "metadata": {
            "repair_sidecar_path": str(sidecar_path) if sidecar_path else None,
            "topic": topic,
            "source_count": len(obs.get("evidence_ids") or obs.get("source_fact_ids") or []),
            "source_documents": source_documents,
            "tags": obs.get("tags") or [],
            "status": obs.get("status") or "approved",
            "source_bank": obs.get("source_bank"),
        },
        "source_fact_ids": obs.get("evidence_ids") or obs.get("source_fact_ids") or [],
        "mentioned_at": obs.get("valid_from"),
        "layer": "approved_repair_sidecar",
        "_repair_sidecar": True,
        "_score_text": " ".join(str(x) for x in score_parts if x),
    }


def load_local_cards(cards_root: str | Path | None) -> list[dict[str, Any]]:
    if not cards_root:
        return []
    root = Path(cards_root).expanduser()
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    index_path = root / "observations_index.jsonl"
    if index_path.exists():
        for idx, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
            except Exception:
                continue
            r = local_observation_result(obs, idx=idx)
            if r:
                out.append(r)
        return out
    for f in glob.glob(str(root / "**" / "*.json"), recursive=True):
        p = Path(f)
        if p.name == "manifest.json":
            continue
        try:
            card = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for idx, obs in enumerate(card.get("canonical_observations") or []):
            r = local_observation_result(obs, card_path=p, card=card, idx=idx)
            if r:
                out.append(r)
    return out


def load_repair_sidecar(repair_sidecar_root: str | Path | None) -> list[dict[str, Any]]:
    if not repair_sidecar_root:
        return []
    root = Path(repair_sidecar_root).expanduser()
    if not root.exists():
        return []
    paths: list[Path] = []
    direct = root / "observations_index.jsonl"
    if direct.exists():
        paths.append(direct)
    paths.extend(sorted(root.glob("*-observations_index.jsonl")))
    seen_paths: set[Path] = set()
    out: list[dict[str, Any]] = []
    for path in paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
            except Exception:
                continue
            r = repair_observation_result(obs, sidecar_path=path, idx=idx)
            if r:
                out.append(r)
    return out


def local_card_candidates(cards_root: str | Path | None, query: str, mode: str, limit: int) -> list[dict[str, Any]]:
    cards = load_local_cards(cards_root)
    if not cards:
        return []
    return score_sidecar_candidates(cards, query, mode, limit, base_score=36.0, strict=False)


def repair_sidecar_candidates(repair_sidecar_root: str | Path | None, query: str, mode: str, limit: int) -> list[dict[str, Any]]:
    cards = load_repair_sidecar(repair_sidecar_root)
    if not cards:
        return []
    # Repair-zone sidecars are useful but still provisional; use stricter lexical
    # gating than local canonical cards so broad topic matches do not leak into
    # unrelated recall contexts.
    return score_sidecar_candidates(cards, query, mode, limit, base_score=28.0, strict=True)


def score_sidecar_candidates(cards: list[dict[str, Any]], query: str, mode: str, limit: int, *, base_score: float, strict: bool) -> list[dict[str, Any]]:
    scored = []
    q_tokens = tokenize(query)
    q_sig_tokens = significant_tokens(query)
    q_numbers = set(NUMBER_RE.findall(query))
    type_hints = query_type_hints(query)
    detail_query = any(x in (query or "").lower() for x in DETAIL_QUERY_WORDS)
    ranking_query = any(x in (query or "").lower() for x in RANKING_CUES)
    for rank, r in enumerate(cards, 1):
        rr = dict(r)
        rr["layer"] = rr.get("layer") or classify_doc(str(rr.get("document_id") or "")) or "local_canonical"
        metadata = rr.get("metadata") or {}
        relevance_blob = " ".join(
            [
                rr.get("text") or "",
                rr.get("_score_text") or "",
                rr.get("type") or "",
                metadata.get("topic") or "",
                " ".join(map(str, metadata.get("tags") or [])),
                " ".join(map(str, metadata.get("source_documents") or [])),
            ]
        )
        overlap = q_tokens & tokenize(relevance_blob)
        sig_overlap = q_sig_tokens & significant_tokens(relevance_blob)
        core_sig_overlap = sig_overlap - GENERIC_SIG_TOKENS
        numeric_hits = q_numbers & set(NUMBER_RE.findall(relevance_blob))
        type_hit = (rr.get("type") or "") in type_hints
        # Local cards are secondary evidence until validated. Require concrete
        # textual/numeric/type relevance; broad topic-only matches are not enough.
        # Type alone is not enough for generic rule/default queries: otherwise an
        # unrelated user_preference/project_decision with only "不能/直接" overlap
        # can crowd out exact source evidence.
        if type_hints:
            if not type_hit and len(core_sig_overlap) < 2 and len(overlap) < 4 and not numeric_hits:
                continue
            if type_hit and len(core_sig_overlap) < 1 and len(overlap) < 2 and not numeric_hits:
                continue
        elif len(core_sig_overlap) < 1 and len(overlap) < 3 and not numeric_hits:
            continue
        if strict and len(core_sig_overlap) < 1 and len(overlap) < 2 and not numeric_hits:
            continue
        # Give local observations a canonical boost, but do not let file/order
        # position dominate relevance. All local observations use a neutral rank
        # prior; lexical/type/numeric overlap decides the order.
        visible_text = rr.get("text") or ""
        visible_numbers = set(NUMBER_RE.findall(visible_text))
        blob_numbers = set(NUMBER_RE.findall(relevance_blob))
        decimal_count = sum(1 for n in visible_numbers if "." in n)
        number_count = len(visible_numbers)
        rr["score"] = base_score + 14.0 * len(core_sig_overlap) + 3.0 * len(sig_overlap - core_sig_overlap) + 2.0 * min(6, len(overlap))
        if rr.get("source_fact_ids") or metadata.get("source_count"):
            rr["score"] += 3.0
        if type_hit:
            rr["score"] += 14.0
        if numeric_hits:
            rr["score"] += 10.0 * len(numeric_hits)
        if detail_query and has_measurement_signal(relevance_blob):
            # Prefer cards that carry actual numeric evidence in the visible
            # observation text. Source document IDs contain dates/hashes, which
            # should not make a vague local card outrank precise source facts.
            rr["score"] += 4.0 * min(4, number_count)
            rr["score"] += 8.0 * min(4, decimal_count)
        if detail_query and number_count == 0 and not numeric_hits:
            rr["score"] -= 18.0
        if detail_query and blob_numbers and not has_measurement_signal(relevance_blob):
            rr["score"] -= 6.0
        if ranking_query and any(cue in relevance_blob.lower() for cue in RANKING_CUES):
            rr["score"] += 16.0
        if "alignment_score" in (query or "").lower() and "alignment_score" in relevance_blob.lower():
            rr["score"] += 18.0
        if "ablation" in (query or "").lower() and "ablation" in relevance_blob.lower():
            rr["score"] += 10.0
        if type_hints and not type_hit and (rr.get("type") or "") in {"open_question", "risk"}:
            rr["score"] -= 16.0
        scored.append(rr)
    scored.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    return scored[:limit]


def result_key(r: dict[str, Any]) -> str:
    if r.get("id"):
        return str(r["id"])
    return (r.get("document_id") or "") + "::" + (r.get("text") or "")[:120]


def layer_boost(layer: str, fact_type: str | None, mode: str) -> float:
    # Layer is a secondary signal. Relevance must dominate; otherwise generic
    # offline-consolidation bookkeeping facts outrank useful raw evidence.
    if layer == "approved_repair_sidecar":
        return 12 if mode in ("high-level", "mixed") else -5
    if mode == "high-level":
        boost = {
            "local_canonical": 34,
            "canonical": 45,
            "offline_weekly": 28,
            "offline_daily": 16,
            "sqlite_import": 0,
            "other": 0,
        }.get(layer, 0)
        if fact_type == "observation":
            boost += 30
        return boost
    if mode == "evidence":
        boost = {
            "local_canonical": 0,
            "canonical": 0,
            "offline_weekly": -8,
            "offline_daily": 5,
            "sqlite_import": 28,
            "other": 5,
        }.get(layer, 0)
        if fact_type == "observation":
            boost -= 5
        return boost
    boost = {
        "local_canonical": 26,
        "canonical": 35,
        "offline_weekly": 22,
        "offline_daily": 15,
        "sqlite_import": 12,
        "other": 3,
    }.get(layer, 0)
    if fact_type == "observation":
        boost += 20
    return boost


def consolidation_meta_penalty(text: str) -> float:
    t = (text or "").lower()
    meta_patterns = [
        "performed offline daily consolidation",
        "performed offline daily reflection",
        "completed daily offline reflection",
        "daily consolidation task",
        "daily offline reflection and consolidation process",
        "processing source documents for topic",
        "processing 120 source documents",
        "reviewing 111 source entries",
    ]
    if any(p in t for p in meta_patterns):
        return -55.0
    return 0.0


def score_result(r: dict[str, Any], *, query: str, mode: str, rank: int, variant_index: int) -> float:
    layer = classify_doc(r.get("document_id") or "")
    text = r.get("_score_text") or r.get("text") or r.get("content") or ""
    metadata = r.get("metadata") or {}
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    overlap = len(q_tokens & t_tokens)
    overlap_score = 16.0 * overlap / math.sqrt(max(len(q_tokens), 1))
    q_sig_tokens = significant_tokens(query)
    core_sig_overlap = (q_sig_tokens & significant_tokens(text)) - GENERIC_SIG_TOKENS
    core_sig_score = 5.0 * len(core_sig_overlap)
    q_numbers = set(NUMBER_RE.findall(query))
    text_numbers = set(NUMBER_RE.findall(text))
    numeric_score = 0.0
    if q_numbers:
        numeric_score = 12.0 * len(q_numbers & text_numbers)
    detail_query = any(x in (query or "").lower() for x in DETAIL_QUERY_WORDS)
    ranking_query = any(x in (query or "").lower() for x in RANKING_CUES)
    detail_score = 0.0
    if detail_query and has_measurement_signal(text):
        decimal_count = sum(1 for n in text_numbers if "." in n)
        detail_score += 4.0 * min(5, decimal_count)
    if ranking_query and any(cue in (text or "").lower() for cue in RANKING_CUES):
        detail_score += 10.0
    if "alignment_score" in (query or "").lower():
        detail_score += 24.0 if "alignment_score" in (text or "").lower() else -8.0
    source_trace_score = 0.0
    if r.get("source_fact_ids") or metadata.get("source_count") or metadata.get("output_json"):
        source_trace_score += 4.0
    # Keep rank as a weak prior only. Semantic recall rank matters, but local
    # rerank must be allowed to demote high-ranked generic meta facts.
    base = 35.0 - rank * 0.8 - variant_index * 1.5
    lb = layer_boost(layer, r.get("type"), mode)
    if mode in ("high-level", "mixed"):
        # Do not let layer preference promote unrelated offline summaries.
        # If there is no lexical/numeric overlap with the user query, the result
        # may still be useful but should not outrank directly relevant raw facts.
        relevance_gate = min(1.0, (overlap + len(q_numbers & text_numbers)) / 2.0)
        lb *= max(0.10, relevance_gate)
    return base + lb + overlap_score + core_sig_score + numeric_score + detail_score + source_trace_score + consolidation_meta_penalty(text)


def layered_recall(
    api: str,
    bank: str,
    query: str,
    mode: str,
    raw_limit: int,
    limit: int,
    cards_root: str | Path | None = None,
    *,
    include_observations: bool = True,
    local_sidecar_limit: int | None = None,
    repair_sidecar_root: str | Path | None = None,
    repair_sidecar_limit: int | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for vi, q in enumerate(query_variants(query, mode)):
        try:
            results = recall(api, bank, q, raw_limit, include_observations=include_observations)
        except Exception as e:
            results = [{"error": repr(e), "query": q}]
        for rank, r in enumerate(results, 1):
            if "error" in r:
                key = f"error::{vi}::{rank}"
                r = dict(r)
                r["score"] = -999
                r["layer"] = "error"
                merged[key] = r
                continue
            key = result_key(r)
            scored = dict(r)
            scored["layer"] = classify_doc(scored.get("document_id") or "")
            scored["score"] = score_result(scored, query=query, mode=mode, rank=rank, variant_index=vi)
            scored["source_query_variant"] = vi
            if key not in merged or scored["score"] > merged[key].get("score", -1e9):
                merged[key] = scored
    ranked_nonlocal = sorted(merged.values(), key=lambda x: x.get("score", -1e9), reverse=True)
    result = ranked_nonlocal[:limit]

    # Source-preserving sidecar: local canonical observations are appended after
    # the normal Hindsight top-N, so they do not replace non-local source evidence.
    if cards_root and mode in ("high-level", "mixed"):
        if local_sidecar_limit is None:
            local_sidecar_limit = 2 if mode == "high-level" else 1
        local_sidecar_limit = max(0, int(local_sidecar_limit))
        if local_sidecar_limit:
            seen = {result_key(r) for r in result}
            added = 0
            for item in local_card_candidates(cards_root, query, mode, raw_limit):
                key = result_key(item)
                if key in seen:
                    continue
                item = dict(item)
                item["_sidecar"] = True
                item["sidecar_reason"] = "local_canonical_source_preserving_augmentation"
                result.append(item)
                seen.add(key)
                added += 1
                if added >= local_sidecar_limit:
                    break

    # Approved repair-zone sidecar: even more conservative than local canonical;
    # append only after non-local and local-canonical evidence, never replacing it.
    if repair_sidecar_root and mode in ("high-level", "mixed"):
        if repair_sidecar_limit is None:
            repair_sidecar_limit = 1
        repair_sidecar_limit = max(0, int(repair_sidecar_limit))
        if repair_sidecar_limit:
            seen = {result_key(r) for r in result}
            added = 0
            for item in repair_sidecar_candidates(repair_sidecar_root, query, mode, raw_limit):
                key = result_key(item)
                if key in seen:
                    continue
                item = dict(item)
                item["_sidecar"] = True
                item["sidecar_reason"] = "approved_repair_source_preserving_augmentation"
                result.append(item)
                seen.add(key)
                added += 1
                if added >= repair_sidecar_limit:
                    break
    return result


def render_markdown(query: str, mode: str, results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"# Layered Hindsight Recall")
    lines.append("")
    lines.append(f"query: {query}")
    lines.append(f"mode: {mode}")
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"## {i}. ERROR")
            lines.append(f"- {r.get('error')}")
            continue
        text = (r.get("text") or "").replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + "..."
        lines.append(f"## {i}. score={r.get('score', 0):.1f} layer={r.get('layer')} type={r.get('type')}")
        lines.append(f"- document: `{r.get('document_id')}`")
        if r.get("mentioned_at"):
            lines.append(f"- mentioned_at: {r.get('mentioned_at')}")
        lines.append(f"- text: {text}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Layered Hindsight recall helper")
    ap.add_argument("query", nargs="?", help="Recall query")
    ap.add_argument("--query", dest="query_kw", help="Recall query (alternative to positional)")
    ap.add_argument("--mode", choices=["high-level", "evidence", "mixed"], default="mixed")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--raw-limit", type=int, default=40, help="Results to fetch per query variant before local rerank")
    ap.add_argument("--limit", type=int, default=10, help="Final result count")
    ap.add_argument("--cards-root", default=str(DEFAULT_CARDS_ROOT), help="Local v2 canonical cards root; used by default when the directory exists")
    ap.add_argument("--use-local-cards", action="store_true", help="Force-enable local v2 canonical cards even if the root is non-standard")
    ap.add_argument("--no-local-cards", action="store_true", help="Disable local v2 canonical cards and use only Hindsight recall")
    ap.add_argument("--no-hindsight-observations", action="store_true", help="Do not request fact_type=observation from Hindsight; useful for raw baseline eval")
    ap.add_argument("--local-sidecar-limit", type=int, default=None, help="Extra local_canonical results appended after normal non-local limit; default high-level=2, mixed=1, evidence=0")
    ap.add_argument("--repair-sidecar-root", default=str(DEFAULT_REPAIR_SIDECAR_ROOT), help="Approved repair-zone sidecar root; appended after normal Hindsight/local results when enabled and relevant")
    ap.add_argument("--use-repair-sidecar", action="store_true", help="Force-enable approved repair sidecar even if root is non-standard")
    ap.add_argument("--no-repair-sidecar", action="store_true", help="Disable approved repair sidecar augmentation")
    ap.add_argument("--repair-sidecar-limit", type=int, default=None, help="Extra approved repair results appended after normal/local results; default=1 for high-level/mixed")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    query = args.query_kw or args.query
    if not query:
        ap.error("query is required")
    cards_root_path = Path(args.cards_root).expanduser()
    use_cards = not args.no_local_cards and (args.use_local_cards or cards_root_path.exists())
    cards_root = str(cards_root_path) if use_cards else None
    repair_root_path = Path(args.repair_sidecar_root).expanduser()
    use_repair = not args.no_repair_sidecar and (args.use_repair_sidecar or repair_root_path.exists())
    repair_sidecar_root = str(repair_root_path) if use_repair else None
    results = layered_recall(
        args.api, args.bank, query, args.mode, args.raw_limit, args.limit,
        cards_root=cards_root,
        include_observations=not args.no_hindsight_observations,
        local_sidecar_limit=args.local_sidecar_limit,
        repair_sidecar_root=repair_sidecar_root,
        repair_sidecar_limit=args.repair_sidecar_limit,
    )
    if args.json:
        print(json.dumps({"query": query, "mode": args.mode, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(query, args.mode, results))


if __name__ == "__main__":
    main()
