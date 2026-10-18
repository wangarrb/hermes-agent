#!/usr/bin/env python3
"""Deterministic A/B benchmark for accepted Egomotion4D mental models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


API = os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888")
BANK = "hermes"
HERMES_HOME = Path(
    os.environ.get("HINDSIGHT_DAILY_HERMES_HOME") or (Path.home() / ".hermes")
)
QUESTIONS_FILE = HERMES_HOME / "mental-models" / "egomotion4d" / "benchmark" / "questions.json"
PREFLIGHT_SCRIPT = HERMES_HOME / "scripts" / "hindsight_daily_noagent.py"


def build_active_context(logical_ids: list[str] | None = None) -> str:
    """Use the production fail-closed preflight for every accepted revision."""
    registry_path = HERMES_HOME / "mental-models" / "egomotion4d" / "registry.json"
    if not registry_path.exists() or not PREFLIGHT_SCRIPT.exists():
        return ""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = set(logical_ids) if logical_ids else None
    parts = []
    for logical_id, model in registry.get("models", {}).items():
        if selected is not None and logical_id not in selected:
            continue
        if not model.get("accepted_revision"):
            continue
        result = subprocess.run(
            [sys.executable, str(PREFLIGHT_SCRIPT), "--mental-model-preflight", logical_id],
            capture_output=True,
            text=True,
            timeout=45,
            env={**os.environ, "HINDSIGHT_DAILY_HERMES_HOME": str(HERMES_HOME)},
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append(result.stdout.strip())
    return "\n\n".join(parts)


def _normalize_expected_text(value: str) -> str:
    normalized = str(value).lower().replace("_", "-")
    for dash in ("‐", "‑", "‒", "–", "—", "―", "−"):
        normalized = normalized.replace(dash, "-")
    return " ".join(normalized.split())


def _matches_expected_term(text: str, term: str | list[str]) -> bool:
    normalized_text = _normalize_expected_text(text)
    alternatives = term if isinstance(term, list) else [term]
    return any(
        _normalize_expected_text(alternative) in normalized_text
        for alternative in alternatives
    )


def score_response(text: str, question: dict) -> float:
    """Score explicit preregistered terms without an LLM judge."""
    terms = question.get("required_terms")
    if terms:
        return float(all(_matches_expected_term(text, term) for term in terms))
    d_refs = list(question.get("key_d_refs", []))
    triggers = list(question.get("expected_pitfall_triggers", []))
    checks = d_refs + triggers
    if not checks:
        return 0.0
    return round(
        sum(_matches_expected_term(text, term) for term in checks) / len(checks),
        6,
    )


def format_gate_question(question: dict) -> str:
    """Expose the citation contract symmetrically to control and treatment."""
    anchors = [str(anchor) for anchor in question.get("key_d_refs", [])]
    if not anchors:
        return question["question"]
    return (
        f"{question['question']}\n\n"
        f"Required decision anchors: {', '.join(anchors)}."
    )


def summarize_results(results: list[dict]) -> dict:
    paired = [
        row
        for row in results
        if not row["A"].get("error") and not row["B"].get("error")
    ]
    if not paired:
        return {
            "total": len(results),
            "paired_complete": 0,
            "mean_score_a": None,
            "mean_score_b": None,
            "delta_b_minus_a": None,
            "verdict": "BLOCK_NO_COMPLETE_PAIRS",
        }
    mean_a = sum(row["A"]["score"] for row in paired) / len(paired)
    mean_b = sum(row["B"]["score"] for row in paired) / len(paired)
    if len(paired) != len(results):
        verdict = "BLOCK_INCOMPLETE_PAIRS"
    elif mean_b > mean_a:
        verdict = "PASS_B_BETTER"
    elif mean_b == mean_a:
        verdict = "NO_CLAIM_B_NO_GAIN"
    else:
        verdict = "REJECT_B_WORSE"
    return {
        "total": len(results),
        "paired_complete": len(paired),
        "mean_score_a": round(mean_a, 6),
        "mean_score_b": round(mean_b, 6),
        "delta_b_minus_a": round(mean_b - mean_a, 6),
        "verdict": verdict,
    }


def _reflect(query: str, timeout: int) -> tuple[str, float, str | None]:
    started = time.time()
    try:
        response = requests.post(
            f"{API}/v1/default/banks/{BANK}/reflect",
            json={
                "query": query,
                "max_tokens": 1000,
                "budget": "low",
                "exclude_mental_models": True,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            return "", time.time() - started, f"HTTP {response.status_code}"
        text = response.json().get("text", "")
        return text, time.time() - started, None if text else "empty response"
    except Exception as exc:
        return "", time.time() - started, str(exc)


def run_benchmark(
    *,
    output_dir: Path,
    timeout: int = 120,
    questions_file: Path = QUESTIONS_FILE,
    logical_ids: list[str] | None = None,
) -> tuple[dict, int]:
    benchmark = json.loads(questions_file.read_text(encoding="utf-8"))
    active_context = build_active_context(logical_ids=logical_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not active_context:
        summary = {
            "total": len(benchmark["questions"]),
            "paired_complete": 0,
            "mean_score_a": None,
            "mean_score_b": None,
            "delta_b_minus_a": None,
            "verdict": "NO_CLAIM_NO_ACCEPTED_MODELS",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary, 2

    results = []
    for question in benchmark["questions"]:
        gate_question = format_gate_question(question)
        text_a, elapsed_a, error_a = _reflect(gate_question, timeout)
        text_b, elapsed_b, error_b = _reflect(
            f"{active_context}\n\n---\n\n{gate_question}", timeout
        )
        results.append(
            {
                "qid": question["id"],
                "question": question["question"],
                "ground_truth": question["ground_truth"],
                "A": {
                    "text": text_a,
                    "elapsed": round(elapsed_a, 3),
                    "error": error_a,
                    "score": score_response(text_a, question),
                },
                "B": {
                    "text": text_b,
                    "elapsed": round(elapsed_b, 3),
                    "error": error_b,
                    "score": score_response(text_b, question),
                },
            }
        )
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = summarize_results(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary, 0 if summary["verdict"] == "PASS_B_BETTER" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERMES_HOME / "mental-models" / "egomotion4d" / "benchmark" / "latest",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument(
        "--logical-id",
        action="append",
        dest="logical_ids",
        help="Include only this accepted logical model in treatment context; repeatable.",
    )
    args = parser.parse_args()
    summary, exit_code = run_benchmark(
        output_dir=args.output_dir,
        timeout=args.timeout,
        questions_file=args.questions,
        logical_ids=args.logical_ids,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
