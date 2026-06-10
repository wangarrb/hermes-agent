#!/usr/bin/env python3
"""Plan bounded LLM scoring batches for Hindsight review backlog.

This script is intentionally non-mutating by default:
- no Hindsight API calls
- no production writes
- no LLM calls

It turns a content-free review backlog/sample JSONL into scorer batches.  One
batch is treated as one future LLM call, so cost can be capped before any paid
provider is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "hindsight-review-backlog-llm-scorer-plan-v1"
SCORE_SCHEMA_VERSION = "hindsight-review-backlog-llm-score-v1"
CONFIRM_SCORE_TOKEN = "score-review-backlog"
DEFAULT_BATCH_SIZE = int(os.environ.get("HINDSIGHT_REVIEW_SCORER_BATCH_SIZE", "5") or "5")
# User decision 2026-05-09: weekly review should default to 10 packages/calls.
DEFAULT_MAX_LLM_CALLS = int(os.environ.get("HINDSIGHT_REVIEW_SCORER_MAX_LLM_CALLS", "10") or "10")
DEFAULT_CADENCE = os.environ.get("HINDSIGHT_REVIEW_SCORER_CADENCE", "weekly") or "weekly"
DEFAULT_LLM_MODEL = os.environ.get("HINDSIGHT_REVIEW_SCORER_LLM_MODEL", "MiniMax-M2.7")
DEFAULT_LLM_BASE_URL = os.environ.get("HINDSIGHT_REVIEW_SCORER_LLM_BASE_URL", "https://api.minimaxi.com/v1")
DEFAULT_LLM_API_KEY_ENV = os.environ.get("HINDSIGHT_REVIEW_SCORER_LLM_API_KEY_ENV", "MINIMAX_API_KEY")

VALUE_CLASSES = {
    "experiment_result",
    "durable_decision",
    "error_root_cause",
    "tool_lesson",
    "environment_fact",
    "project_state",
    "user_preference",
    "open_question",
    "low_signal",
}
ROUTES = {"wait", "raw_only", "whole_session", "windowed", "manual_review", "repair_note_candidate", "cluster_revisit"}
RISKS = {"low", "medium", "high"}
ANOMALIES = {
    "credential_like",
    "internal_planning_leak",
    "context_compaction",
    "tool_log_heavy",
    "overlong_multi_scope",
    "possible_hallucination",
    "date_ambiguous",
}
CREDENTIAL_RE = re.compile(
    r"(?i)((api[_ -]?key|secret|password|passwd|token)\s*[:=]\s*[A-Za-z0-9._\-/+]{12,}|bearer\s+[a-z0-9._\-]{16,}|sk-[a-z0-9_\-]{12,}|AKIA[0-9A-Z]{12,})"
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:  # pragma: no cover - defensive CLI path
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _positive_int(value: int | None, *, default: int) -> int:
    if value is None:
        return default
    return max(1, int(value))


def _non_negative_int(value: int | None, *, default: int) -> int:
    if value is None:
        return default
    return max(0, int(value))


def plan_scorer_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int | None = None,
    max_llm_calls: int | None = None,
    cadence: str = DEFAULT_CADENCE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group review-backlog rows into capped scorer batches.

    Cost semantics: one batch == one future LLM call.  `max_llm_calls` therefore
    caps the number of batches emitted.  Rows beyond the cap are deferred, never
    dropped.
    """
    batch_size = _positive_int(batch_size, default=DEFAULT_BATCH_SIZE)
    max_llm_calls = _non_negative_int(max_llm_calls, default=DEFAULT_MAX_LLM_CALLS)
    generated_at = datetime.now(timezone.utc).isoformat()

    total_records = len(rows)
    unbounded_llm_calls = math.ceil(total_records / batch_size) if total_records else 0
    max_records_this_run = batch_size * max_llm_calls
    planned_rows = rows[:max_records_this_run] if max_records_this_run > 0 else []
    deferred_rows = rows[len(planned_rows):]

    batches: list[dict[str, Any]] = []
    for start in range(0, len(planned_rows), batch_size):
        records = planned_rows[start:start + batch_size]
        call_index = len(batches) + 1
        batches.append({
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "batch_id": f"review-scorer-{generated_at[:10]}-{call_index:04d}",
            "cadence": cadence,
            "llm_call_index": call_index,
            "record_count": len(records),
            "one_llm_call_per_batch": True,
            "hindsight_submit_allowed": False,
            "production_mutation_allowed": False,
            "records": records,
        })

    planned_record_ids = [str(r.get("document_id") or "") for r in planned_rows]
    deferred_record_ids = [str(r.get("document_id") or "") for r in deferred_rows]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "cadence": cadence,
        "batch_size": batch_size,
        "max_llm_calls": max_llm_calls,
        "max_batches": max_llm_calls,
        "one_llm_call_per_batch": True,
        "input_records": total_records,
        "unbounded_llm_calls": unbounded_llm_calls,
        "llm_calls_planned": len(batches),
        "batches_planned": len(batches),
        "records_planned": len(planned_rows),
        "records_deferred_by_call_cap": len(deferred_rows),
        "capped_by_max_llm_calls": len(deferred_rows) > 0,
        "planned_document_ids": planned_record_ids,
        "deferred_document_ids": deferred_record_ids,
        "llm_call_allowed_by_plan": False,
        "hindsight_submit_allowed": False,
        "production_mutation_allowed": False,
    }
    return batches, summary


def read_dotenv(path: Path = Path.home() / ".hermes" / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def get_llm_key(api_key_env: str) -> str:
    key = (os.environ.get(api_key_env) or read_dotenv().get(api_key_env, "")).strip()
    if not key or key in {"***", "[REDACTED]"}:
        raise SystemExit(f"{api_key_env} missing; aborting before LLM call")
    return key


def extract_json_object(text: str) -> dict[str, Any] | None:
    s = re.sub(r"<think>.*?</think>", "", (text or "").strip(), flags=re.S | re.I).strip()
    candidates = [m.group(1).strip() for m in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", s, flags=re.S)]
    candidates.append(s)
    decoder = json.JSONDecoder()
    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        for m in re.finditer(r"\{", cand):
            try:
                obj, _ = decoder.raw_decode(cand[m.start():])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def clamp_score(value: Any) -> int:
    try:
        return max(0, min(5, int(round(float(value)))))
    except Exception:
        return 0


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_topic(value: Any) -> str:
    parts = [str(v).strip() for v in as_list(value) if str(v).strip()]
    return "+".join(parts)[:160] if parts else "<unspecified>"


def normalize_score(record: dict[str, Any], raw: dict[str, Any] | None, *, batch_id: str, llm_call_index: int) -> dict[str, Any]:
    raw = raw or {}
    nested_scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    merged = {**nested_scores, **raw}
    value_classes = [str(v) for v in as_list(merged.get("value_classes")) if str(v) in VALUE_CLASSES]
    anomalies = [str(v) for v in as_list(merged.get("anomalies")) if str(v) in ANOMALIES]
    route = str(merged.get("recommended_route") or (record.get("review") or {}).get("recommended_route") or "wait")
    if route not in ROUTES:
        route = "wait"
    risk = str(merged.get("retainability_risk") or "medium")
    if risk not in RISKS:
        risk = "medium"
    scores = {
        "value_level": clamp_score(merged.get("value_level")),
        "information_density": clamp_score(merged.get("information_density")),
        "durability": clamp_score(merged.get("durability")),
        "actionability": clamp_score(merged.get("actionability")),
    }
    scores_normalized = {k: round(v / 5.0, 4) for k, v in scores.items()}
    score_total_0_20 = sum(scores.values())
    score_mean_0_1 = round(sum(scores_normalized.values()) / len(scores_normalized), 4)
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_id": record.get("document_id"),
        "parent_document_id": record.get("parent_document_id"),
        "event_date": record.get("event_date"),
        "content_sha256": record.get("content_sha256"),
        "scorer": {
            "batch_id": batch_id,
            "llm_call_index": llm_call_index,
            "llm_call_status": "scored",
            "sidecar_only": True,
        },
        "scores": scores,
        "scores_normalized": scores_normalized,
        "score_total_0_20": score_total_0_20,
        "score_mean_0_1": score_mean_0_1,
        "topic": safe_topic(merged.get("topic")),
        "value_classes": value_classes,
        "retainability_risk": risk,
        "recommended_route": route,
        "anomalies": anomalies,
        "reason_brief": str(merged.get("reason_brief") or "")[:500],
        "suggested_spans": [v for v in as_list(merged.get("suggested_spans")) if isinstance(v, (dict, str))][:10],
        "hindsight_submit_allowed": False,
        "production_mutation_allowed": False,
    }


def redact_sensitive_text(text: str) -> str:
    text = re.sub(r"(?i)(api[_ -]?key|secret|password|passwd|token)(\s*[:=]\s*)([^\s,;\]}\"']{6,})", r"\1\2[REDACTED]", text or "")
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]{12,}", r"\1[REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9._-]{8,}", "sk-[REDACTED]", text)
    return text


def rehydrate_record_content(record: dict[str, Any]) -> tuple[str, str | None]:
    if record.get("content"):
        return str(record.get("content") or ""), None
    if record.get("content_preview"):
        return str(record.get("content_preview") or ""), "content_preview_only"
    source = record.get("source") or {}
    json_path = source.get("json_path") or (record.get("metadata") or {}).get("json_path")
    if not json_path:
        return "", "missing_source_path"
    try:
        import sys
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        import hindsight_session_retain_runner as retain_runner  # type: ignore
        shim = {
            "document_id": record.get("document_id"),
            "metadata": {"json_path": json_path},
            "bank_target": (record.get("manifest_decision") or {}).get("bank_target"),
        }
        hydrated = retain_runner.rehydrate_record(shim)
        return str(hydrated.get("content") or ""), None
    except Exception as exc:
        return "", f"rehydrate_failed:{type(exc).__name__}"


def prepare_prompt_records(records: list[dict[str, Any]], *, max_record_chars: int = 6000) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for rec in records:
        anomalies = list(rec.get("deterministic_anomalies") or [])
        content, content_error = rehydrate_record_content(rec)
        if "credential_like" in anomalies or CREDENTIAL_RE.search(content or ""):
            skipped.append({"document_id": rec.get("document_id"), "reason": "credential_like"})
            continue
        if not content:
            skipped.append({"document_id": rec.get("document_id"), "reason": content_error or "missing_content"})
            continue
        excerpt = redact_sensitive_text(content)[:max(1, max_record_chars)]
        prompt_records.append({
            "document_id": rec.get("document_id"),
            "event_date": rec.get("event_date"),
            "content_sha256": rec.get("content_sha256"),
            "topic_key": rec.get("topic_key"),
            "current_retain_outcome": rec.get("current_retain_outcome"),
            "review": rec.get("review"),
            "deterministic_anomalies": anomalies,
            "value_class_guess": rec.get("value_class_guess") or [],
            "content_excerpt": excerpt,
            "content_excerpt_chars": len(excerpt),
            "content_truncated": len(content) > len(excerpt),
            "_record": rec,
        })
    return prompt_records, skipped


def build_scorer_messages(batch: dict[str, Any], prompt_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    public_records = [{k: v for k, v in rec.items() if k != "_record"} for rec in prompt_records]
    required_document_ids = [str(rec.get("document_id") or "") for rec in public_records if rec.get("document_id")]
    system = (
        "你是 Hindsight review-backlog 的价值评分器。只判断是否值得后续重捞/修复，不抽取事实正本。"
        "必须只输出 JSON，不要输出思维链。低分不代表删除，只表示降低优先级或等待同主题聚类。"
        "必须覆盖输入里的每一个 document_id；不要只评分第一条。"
    )
    user = {
        "task": "score_review_backlog_batch",
        "batch_id": batch.get("batch_id"),
        "required_document_ids": required_document_ids,
        "required_score_count": len(required_document_ids),
        "schema": {
            "scores": [{
                "document_id": "must be one of required_document_ids; emit exactly one score per required_document_id",
                "value_level": "0-5 integer",
                "information_density": "0-5 integer",
                "durability": "0-5 integer",
                "actionability": "0-5 integer",
                "topic": "short controlled label or labels",
                "value_classes": sorted(VALUE_CLASSES),
                "retainability_risk": sorted(RISKS),
                "recommended_route": sorted(ROUTES),
                "anomalies": sorted(ANOMALIES),
                "reason_brief": "one short source-backed reason; no chain-of-thought",
                "suggested_spans": "optional message/block ranges only",
            }]
        },
        "records": public_records,
        "constraints": [
            "sidecar only: do not rewrite facts as memory content",
            "do not invent facts not visible in source excerpt",
            "do not mark low score as delete",
            "prefer wait/raw_only for unclear or noisy evidence",
            "return scores.length == required_score_count",
            "return exactly one score object for every document_id in required_document_ids",
            "if a record is unclear, still include it with recommended_route='wait' and low scores",
        ],
        "final_check_before_answer": {
            "all_required_document_ids_must_appear_once": required_document_ids,
            "scores_length_must_equal": len(required_document_ids),
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)}]


def _scores_from_document_id_map(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows: list[dict[str, Any]] = []
    for doc_id, payload in value.items():
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row.setdefault("document_id", doc_id)
        rows.append(row)
    return rows


def scores_from_llm_obj(obj: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not obj:
        return []
    value = obj.get("scores")
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        return _scores_from_document_id_map(value)
    for key in ("scores_by_document_id", "score_by_document_id", "documents"):
        mapped = _scores_from_document_id_map(obj.get(key))
        if mapped:
            return mapped
    if isinstance(obj.get("value"), list):
        return [x for x in obj["value"] if isinstance(x, dict)]
    if obj.get("document_id"):
        return [obj]
    return []


def make_openai_llm_fn(*, model: str, base_url: str, api_key_env: str) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    def _call(messages: list[dict[str, str]]) -> dict[str, Any]:
        import requests
        key = get_llm_key(api_key_env)
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4000,
                "response_format": {"type": "json_object"},
            },
            timeout=240,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        obj = extract_json_object(raw)
        if obj is None:
            raise RuntimeError("LLM response did not contain a JSON object")
        return obj
    return _call


def score_batches(
    batches: list[dict[str, Any]],
    *,
    execute: bool = False,
    confirm: str = "",
    llm_fn: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
    max_record_chars: int = 6000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "execute": bool(execute),
        "confirm_required": CONFIRM_SCORE_TOKEN,
        "batches_input": len(batches),
        "llm_calls_made": 0,
        "scores_written": 0,
        "records_skipped_before_llm": 0,
        "records_prompted_to_llm": 0,
        "valid_scores_from_llm": 0,
        "missing_scores_from_llm": 0,
        "missing_document_ids": [],
        "score_coverage": 1.0,
        "hindsight_submit_allowed": False,
        "production_mutation_allowed": False,
    }
    if not execute:
        return [], summary
    if confirm != CONFIRM_SCORE_TOKEN:
        raise SystemExit(f"Refusing to call LLM: pass --confirm-score {CONFIRM_SCORE_TOKEN}")
    if llm_fn is None:
        raise SystemExit("Refusing to call LLM without llm_fn")

    outputs: list[dict[str, Any]] = []
    skipped_all: list[dict[str, Any]] = []
    for batch in batches:
        prompt_records, skipped = prepare_prompt_records(batch.get("records") or [], max_record_chars=max_record_chars)
        skipped_all.extend(skipped)
        if not prompt_records:
            continue
        summary["records_prompted_to_llm"] += len(prompt_records)
        messages = build_scorer_messages(batch, prompt_records)
        obj = llm_fn(messages)
        summary["llm_calls_made"] += 1
        by_doc = {str(item.get("document_id")): item for item in scores_from_llm_obj(obj) if item.get("document_id")}
        for prec in prompt_records:
            original = prec.get("_record") or {}
            doc_id = str(original.get("document_id") or "")
            raw_score = by_doc.get(doc_id)
            if raw_score is None:
                summary["missing_scores_from_llm"] += 1
                cast_missing = summary.setdefault("missing_document_ids", [])
                if isinstance(cast_missing, list) and len(cast_missing) < 500:
                    cast_missing.append(doc_id)
                raw_score = {
                    "document_id": doc_id,
                    "recommended_route": "wait",
                    "reason_brief": "LLM response omitted this document_id; defaulted to wait.",
                }
            else:
                summary["valid_scores_from_llm"] += 1
            outputs.append(normalize_score(
                original,
                raw_score,
                batch_id=str(batch.get("batch_id") or ""),
                llm_call_index=int(batch.get("llm_call_index") or summary["llm_calls_made"]),
            ))
    summary["records_skipped_before_llm"] = len(skipped_all)
    summary["skipped_records"] = skipped_all[:200]
    summary["scores_written"] = len(outputs)
    prompted = int(summary.get("records_prompted_to_llm") or 0)
    valid = int(summary.get("valid_scores_from_llm") or 0)
    summary["score_coverage"] = round(valid / prompted, 4) if prompted else 1.0
    summary["coverage_ok"] = summary.get("missing_scores_from_llm") == 0
    return outputs, summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "--backlog", dest="input", required=True, help="Input review backlog/sample JSONL")
    ap.add_argument("--output-batches", help="Output scorer batch JSONL; each line is one planned LLM call")
    ap.add_argument("--score-output", help="Output scorer sidecar JSONL. Without --execute-score this will be an empty dry-run output.")
    ap.add_argument("--summary-json", help="Output summary JSON")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Records per scorer batch/LLM call; default {DEFAULT_BATCH_SIZE}")
    ap.add_argument("--max-llm-calls", "--max-batches", dest="max_llm_calls", type=int, default=DEFAULT_MAX_LLM_CALLS, help=f"Max scorer batches / future LLM calls; default {DEFAULT_MAX_LLM_CALLS}")
    ap.add_argument("--cadence", default=DEFAULT_CADENCE, help=f"Cadence label for audit metadata; default {DEFAULT_CADENCE}")
    ap.add_argument("--execute-score", action="store_true", help="Actually call the configured LLM and write scorer sidecar output")
    ap.add_argument("--confirm-score", default="", help=f"Required token for --execute-score: {CONFIRM_SCORE_TOKEN}")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    ap.add_argument("--llm-api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    ap.add_argument("--max-record-chars", type=int, default=6000, help="Max source chars per record sent to scorer prompt")
    ap.add_argument("--json", action="store_true", help="Print summary JSON to stdout")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    batches, summary = plan_scorer_batches(
        rows,
        batch_size=args.batch_size,
        max_llm_calls=args.max_llm_calls,
        cadence=args.cadence,
    )
    summary["input"] = args.input
    if args.output_batches:
        write_jsonl(args.output_batches, batches)
        summary["output_batches"] = args.output_batches

    if args.score_output or args.execute_score:
        llm_fn = None
        if args.execute_score:
            llm_fn = make_openai_llm_fn(model=args.llm_model, base_url=args.llm_base_url, api_key_env=args.llm_api_key_env)
        scores, score_summary = score_batches(
            batches,
            execute=bool(args.execute_score),
            confirm=args.confirm_score,
            llm_fn=llm_fn,
            max_record_chars=args.max_record_chars,
        )
        summary["score_summary"] = score_summary
        if args.score_output:
            write_jsonl(args.score_output, scores)
            summary["score_output"] = args.score_output

    if args.summary_json:
        p = Path(args.summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"planned {summary['batches_planned']} scorer batches "
            f"({summary['llm_calls_planned']} future LLM calls), "
            f"records={summary['records_planned']}, deferred={summary['records_deferred_by_call_cap']}"
        )
        if "score_summary" in summary:
            ss = summary["score_summary"]
            print(f"score_execute={ss['execute']} llm_calls_made={ss['llm_calls_made']} scores={ss['scores_written']}")
        if args.output_batches:
            print(f"output_batches: {args.output_batches}")
        if args.score_output:
            print(f"score_output: {args.score_output}")
        if args.summary_json:
            print(f"summary_json: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
