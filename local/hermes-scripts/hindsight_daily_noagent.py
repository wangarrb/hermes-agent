#!/usr/bin/env python3
"""
No-agent wrapper for hindsight daily pipeline.
Uses hindsight_memory_pipeline.py (the modern session/json path).
Full output -> log file; concise summary -> stdout (-> weixin delivery).

After pipeline completes, triggers an agent analysis for daily research summary.
"""

import hashlib
import json
import os
import pwd
import re
import subprocess
import sys

# Ensure localhost API calls bypass any HTTP proxy (e.g. 127.0.0.1:7890)
_no_proxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
if "127.0.0.1" not in _no_proxy and "localhost" not in _no_proxy:
    os.environ["no_proxy"] = f"127.0.0.1,localhost,{_no_proxy}".rstrip(",")
    os.environ["NO_PROXY"] = os.environ["no_proxy"]
import time
from pathlib import Path

REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir).expanduser()
# Do not use Path.home() here: Hermes CLI/profile sessions intentionally
# redirect HOME to ~/.hermes/profiles/<profile>/home, which would make a manual
# recovery run target the wrong Hindsight tree. The production daily pipeline is
# anchored at the real user home unless explicitly overridden.
HERMES_HOME = Path(os.environ.get("HINDSIGHT_DAILY_HERMES_HOME") or (REAL_HOME / ".hermes")).expanduser()
PIPELINE_SCRIPT = HERMES_HOME / "scripts" / "hindsight_memory_pipeline.py"
LOG_DIR = HERMES_HOME / "logs" / "hindsight-offline-pipeline"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = LOG_DIR / "summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
LOG_FILE = LOG_DIR / f"{TIMESTAMP}-daily-noagent.log"

# Compact research digest output. The 08:30 report embeds this; no parallel
# YYYY-MM-DD_summary.md is produced.
RESEARCH_DIGEST_DIR = Path(
    "/home/wyr/wiki/auto-maintenance/project/egomotion4d/research-digests"
)
RESEARCH_DIGEST_DIR.mkdir(parents=True, exist_ok=True)
TODAY = time.strftime("%Y-%m-%d")
# Cron runs at 00:01 — "today" has almost no data yet. Report on YESTERDAY.
from datetime import date, datetime, timedelta
import subprocess as _subprocess
REPORT_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def check_hindsight() -> bool:
    """Quick health check before running the pipeline."""
    import requests as _requests
    try:
        resp = _requests.get("http://127.0.0.1:8888/health", timeout=10)
        data = resp.json()
        return data.get("status") == "healthy" and data.get("database") == "connected"
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def get_summary() -> dict | None:
    """Check hindsight stats before/after."""
    import requests as _requests
    try:
        resp = _requests.get("http://127.0.0.1:8888/v1/default/banks/hermes/stats", timeout=10)
        return resp.json()
    except Exception:
        return None


def _get_hindsight_llm_config() -> dict:
    """Read Hindsight's LLM config from its Docker container env vars.

    Returns dict with base_url, model, api_key, provider.
    Falls back to HINDSIGHT_OFFLINE_LLM_* env vars, then to defaults.
    """
    base_url = model = api_key = provider = None

    # Known-dead provider base URLs (key expired or service unavailable).
    # If container points here, skip container config entirely and use fallback.
    _DEAD_BASE_URLS = {
        "https://opencode.ai/zen/v1",
        "https://opencode.ai/zen/go/v1",
        "https://api.minimaxi.com/v1",
    }

    # 1. Try reading from Hindsight container env vars
    try:
        result = subprocess.run(
            ["docker", "exec", "hindsight", "env"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            env_map = {}
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    k, _, v = line.partition("=")
                    env_map[k] = v
            _container_base = env_map.get("HINDSIGHT_API_LLM_BASE_URL", "")
            if _container_base not in _DEAD_BASE_URLS:
                base_url = _container_base
                model = env_map.get("HINDSIGHT_API_LLM_MODEL")
                api_key = env_map.get("HINDSIGHT_API_LLM_API_KEY")
                provider = env_map.get("HINDSIGHT_API_LLM_PROVIDER", "openai")
    except Exception:
        pass

    # 2. Fallback: host env vars (for offline pipeline)
    if not base_url:
        base_url = os.environ.get("HINDSIGHT_OFFLINE_LLM_BASE_URL")
    if not model:
        model = os.environ.get("HINDSIGHT_OFFLINE_LLM_MODEL")
    if not api_key:
        key_env = os.environ.get("HINDSIGHT_OFFLINE_LLM_API_KEY_ENV", "ONEAPI_API_KEY")
        api_key = os.environ.get(key_env)

    # 3. Final defaults (oneapi deepseek-v4-flash via local oneapi gateway)
    _FALLBACK_KEY_ENV = "ONEAPI_API_KEY"
    if not api_key:
        key_env = os.environ.get("HINDSIGHT_OFFLINE_LLM_API_KEY_ENV", _FALLBACK_KEY_ENV)
        api_key = os.environ.get(key_env)
    return {
        "base_url": base_url or os.environ.get("HINDSIGHT_OFFLINE_LLM_BASE_URL", "http://127.0.0.1:3000/v1"),
        "model": model or os.environ.get("HINDSIGHT_OFFLINE_LLM_MODEL", "xopdeepseekv4flash"),
        "api_key": api_key,
        "provider": provider or "openai",
    }


def _call_hindsight_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    *,
    response_format: bool = False,
) -> str | None:
    """Call Hindsight's current LLM (same model Hindsight uses for retain/consolidate).

    Retries up to 10 times on transient server errors (5xx, 429) with
    exponential backoff (5s → 180s max).  Non-transient errors (4xx except 429)
    fail immediately.
    """
    import requests as _requests

    cfg = _get_hindsight_llm_config()

    if not cfg["api_key"]:
        print("ERROR: cannot determine Hindsight LLM API key", flush=True)
        return None

    _BACKOFF = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
    last_exc = None
    for attempt in range(10):
        try:
            payload = {
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }
            if response_format:
                payload["response_format"] = {"type": "json_object"}
            resp = _requests.post(
                f"{cfg['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
                json=payload,
                timeout=300,
            )
            if resp.status_code in (429, 502, 503, 504):
                wait = _BACKOFF[attempt] if attempt < len(_BACKOFF) else 180
                print(f"  LLM {resp.status_code}, retry {attempt+1}/10 in {wait}s...", flush=True)
                time.sleep(wait)
                last_exc = Exception(f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (_requests.ConnectionError, _requests.Timeout) as e:
            wait = _BACKOFF[attempt] if attempt < len(_BACKOFF) else 180
            print(f"  LLM transport error ({e}), retry {attempt+1}/10 in {wait}s...", flush=True)
            time.sleep(wait)
            last_exc = e
        except Exception as e:
            if isinstance(e, _requests.HTTPError) and e.response is not None:
                code = e.response.status_code
                if code in (429, 502, 503, 504):
                    wait = _BACKOFF[attempt] if attempt < len(_BACKOFF) else 180
                    print(f"  LLM HTTP {code}, retry {attempt+1}/10 in {wait}s...", flush=True)
                    time.sleep(wait)
                    last_exc = e
                    continue
            print(f"  LLM call failed: {e}", flush=True)
            return None

    print(f"  LLM call failed after 10 retries: {last_exc}", flush=True)
    return None


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object without miscounting braces inside JSON strings."""
    import re

    cleaned = re.sub(
        r"<think>.*?</think>", "", (text or "").strip(), flags=re.S | re.I
    ).strip()
    candidates = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", cleaned, flags=re.S)
    ]
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    parsed = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                parsed.append(value)
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                parsed.append(value)
    if not parsed:
        return None
    expected = {
        "verdict",
        "conflicts",
        "stale_items",
        "unanchored_items",
        "quality_score",
        "notes",
    }
    parsed.sort(
        key=lambda value: (
            len(expected.intersection(value)),
            len(json.dumps(value, ensure_ascii=False)),
        )
    )
    return parsed[-1]


def _parse_adjudication_response(text: str) -> tuple[dict | None, str | None]:
    result = _extract_json_object(text)
    if result is None:
        return None, "no JSON object found"

    required_lists = ("conflicts", "stale_items", "unanchored_items")
    verdict = result.get("verdict")
    if verdict not in {"PASS_PUBLISH", "REJECT", "ESCALATE_D_REVIEW"}:
        return None, f"invalid verdict: {verdict!r}"
    for field in required_lists:
        if not isinstance(result.get(field), list):
            return None, f"{field} must be a list"
    score = result.get("quality_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
        return None, f"quality_score must be numeric in [0, 100]: {score!r}"
    if not isinstance(result.get("notes"), str):
        return None, "notes must be a string"
    return result, None


def _adjudication_system_prompt() -> str:
    return """You are a mental model quality adjudicator. Your job is to evaluate whether a candidate mental model content is safe to publish as the active version.

Rules:
1. Compare candidate against current D decisions. If candidate contradicts a current D, mark as CONFLICT. A status=current decision remains authoritative unless the source explicitly marks it superseded or non-current; a higher D number does not supersede an earlier current D by itself.
2. Check that actionable statements have anchors (D/task/report/artifact/commit/source-memory). A correct D anchor alone is sufficient; do not require a second report/artifact anchor.
3. Check for stale content only from explicit status or supersede evidence. Do not reject a bounded model for omitting decisions outside the declared generation scope.
4. A concise statement is not a conflict when it preserves the source condition, scope, and prohibition. Mark a conflict only when meaning changes.
5. Check for duplicates: equivalent pitfall entries should be deduplicated.
6. The active slot is comparison context, not an authority. Do not use stale active content to reject a current candidate.
7. Do NOT modify current D. If fresh evidence may推翻 D, mark as ESCALATE.

Respond with a JSON object:
{
  "verdict": "PASS_PUBLISH" | "REJECT" | "ESCALATE_D_REVIEW",
  "conflicts": [{"d_id": "Dxx", "candidate_claim": "...", "d_text": "...", "severity": "high|medium|low"}],
  "stale_items": ["description of stale items found"],
  "unanchored_items": ["description of unanchored actionable items"],
  "quality_score": 0-100,
  "notes": "brief assessment"
}"""


def _collect_today_context(today: str) -> str:
    """Collect today's experiment results, git log, and Hindsight memory as context for LLM."""
    parts = []

    # Current truth is supplied before historical artifacts so stale plans cannot
    # reopen work already closed by the KG.
    kg_root = Path("/home/wyr/.hermes/hermes_use/projects/Egomotion4D-kg")
    current_files = [kg_root / "00-card.md"] + sorted(
        (kg_root / "topics").glob("*/00-current.md")
    )
    current_parts = []
    for path in current_files:
        if not path.exists():
            continue
        try:
            current_parts.append(
                f"#### {path.relative_to(kg_root)}\n"
                + path.read_text(encoding="utf-8")[:1600]
            )
        except OSError:
            pass
    if current_parts:
        parts.append("## AUTHORITATIVE CURRENT TRUTH\n" + "\n".join(current_parts))

    # 1. Scan gpuserver_pullback experiments
    pullback_dir = Path(f"/home/wyr/code/Egomotion4D/gpuserver_pullback")
    exp_dirs = sorted(pullback_dir.glob(f"{today}-*"))
    for d in exp_dirs[:20]:  # cap at 20
        parts.append(f"\n### {d.name}")
        # Search for summary.md and metrics.json recursively under run/
        # Actual structure: run/<experiment-variant>/summary.md, run/<experiment-variant>/metrics.json
        summary_files = sorted(d.glob("run/*/summary.md"))
        metrics_files = sorted(d.glob("run/*/metrics.json"))
        # Also check direct run/summary.md (older format)
        direct_summary = d / "run" / "summary.md"
        direct_metrics = d / "run" / "metrics.json"
        if direct_summary.exists() and direct_summary not in summary_files:
            summary_files.insert(0, direct_summary)
        if direct_metrics.exists() and direct_metrics not in metrics_files:
            metrics_files.insert(0, direct_metrics)

        for sf in summary_files[:5]:  # cap variants per experiment
            try:
                variant = sf.parent.name if sf.parent.name != "run" else ""
                label = f" ({variant})" if variant else ""
                text = sf.read_text(encoding="utf-8")[:3000]
                parts.append(f"#### Summary{label}\n{text}")
            except Exception:
                pass
        for mf in metrics_files[:5]:
            try:
                variant = mf.parent.name if mf.parent.name != "run" else ""
                label = f" ({variant})" if variant else ""
                data = json.loads(mf.read_text(encoding="utf-8"))
                # Include all top-level scalar keys + nested scene/phase keys
                compact = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
                # Also flatten one level of dicts (e.g., per-scene metrics)
                for k, v in data.items():
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            if not isinstance(sv, (dict, list)):
                                compact[f"{k}.{sk}"] = sv
                parts.append(f"Metrics{label}: {json.dumps(compact, ensure_ascii=False)}")
            except Exception:
                pass

    # 2. Git log for today (with file stats for context)
    try:
        git_out = subprocess.run(
            ["git", "log", "--oneline", "--stat", "--since", f"{today} 00:00:00", "--until", f"{today} 23:59:59"],
            capture_output=True, text=True, timeout=10,
            cwd="/home/wyr/code/Egomotion4D",
        )
        if git_out.stdout.strip():
            parts.append(f"\n### Git commits ({today})\n{git_out.stdout.strip()}")
    except Exception:
        pass

    # 3. Hindsight recall — semantic memory for today's work
    # Query Hindsight REST API for recent observations related to Egomotion4D
    try:
        import requests as _req
        # Recall with date-filtered query
        for query in ["Egomotion4D experiment results", "rendering pipeline evaluation", "pose fusion optimization"]:
            try:
                resp = _req.post(
                    "http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall",
                    json={"query": query, "max_tokens": 1500},
                    timeout=15,
                )
                if resp.status_code == 200:
                    items = resp.json().get("results", [])
                    # Filter to today's items only
                    today_items = [
                        item for item in items
                        if today in (
                            item.get("mentioned_at", "")
                            or item.get("occurred_start", "")
                            or item.get("created_at", "")
                        )[:10]
                    ]
                    if today_items:
                        parts.append(f"\n### Hindsight recall: {query}")
                        for item in today_items[:5]:
                            content = item.get("content", "")[:500]
                            parts.append(f"- {content}")
            except Exception:
                pass
    except ImportError:
        pass

    # 4. Project decisions and experiment conclusions from KG
    kg_decisions = Path("/home/wyr/.hermes/hermes_use/projects/Egomotion4D-kg/10-current-decisions.md")
    if kg_decisions.exists():
        try:
            text = kg_decisions.read_text(encoding="utf-8")
            # Find decisions updated today
            lines = text.split("\n")
            today_lines = []
            capture = False
            for line in lines:
                if today in line:
                    capture = True
                if capture:
                    today_lines.append(line)
                    # Stop at next decision entry
                    if line.startswith("## D") and today not in line and len(today_lines) > 1:
                        capture = False
            if today_lines:
                parts.append(f"\n### Project decisions updated today\n" + "\n".join(today_lines[:30]))
        except Exception:
            pass

    # 5. Stable project results written to the KG on the report date.
    kg_experiments = Path(
        "/home/wyr/.hermes/hermes_use/projects/Egomotion4D-kg/30-experiments"
    )
    if kg_experiments.exists():
        for path in sorted(kg_experiments.glob(f"{today}*.md"))[:12]:
            try:
                parts.append(
                    f"\n### KG experiment: {path.name}\n"
                    + path.read_text(encoding="utf-8")[:3000]
                )
            except OSError:
                pass

    return "\n".join(parts) if parts else "(no experiment data found)"


def _research_digest_errors(content: str) -> list[str]:
    errors = []
    for section in ("## 主线", "## 已验证结果", "## 决策与转折", "## 来源"):
        if section not in content:
            errors.append(f"missing section: {section.removeprefix('## ')}")
    if "## 下一步" in content:
        errors.append("forbidden section: 下一步")
    if re.search(r"\b\d+\s*行(?:实现|测试|代码)", content):
        errors.append("implementation line-count narration is forbidden")
    if "algorithm_pitfalls" in content or "```json" in content:
        errors.append("pitfall JSON is forbidden")
    if len(content) > 8000:
        errors.append(f"digest too long: {len(content)} chars")
    return errors


def run_research_summary(
    pipeline_exit_code: int,
    before_stats: dict | None,
    after_stats: dict | None,
    *,
    report_date: str | None = None,
) -> int:
    """Generate one compact research delta for the 08:30 daily report."""
    print("\n[Research Summary]", flush=True)

    # Cron runs at 00:01, so the default source period is yesterday.
    report_date = report_date or REPORT_DATE

    # Collect context
    print("  Collecting experiment data...", flush=True)
    experiment_context = _collect_today_context(report_date)

    # Build pipeline context
    pipeline_info = f"Pipeline exit code: {pipeline_exit_code}"
    if before_stats and after_stats:
        d_doc = after_stats.get("total_documents", 0) - before_stats.get("total_documents", 0)
        d_obs = after_stats.get("total_observations", 0) - before_stats.get("total_observations", 0)
        pipeline_info += f" | Hindsight: docs={d_doc:+d} obs={d_obs:+d}"

    system_prompt = """你是自动驾驶算法研发记录整理器。根据冻结 artifact、KG 结果和 git 提交生成简洁的每日研究增量。

这不是第二份知识库：稳定知识由 KG/Graphify 管理，算法坑由 canonical Pitfall lifecycle 管理。不要分配 P-id，不要复制长背景，不要按代码行数汇报工作量。

输出格式（严格遵守，不要加多余内容）：

# YYYY-MM-DD Research Digest

## 主线
（最多2条，只写当天真正推进的主因果链）

## 已验证结果
（表格：对象 | 证据/指标 | 结论；最多8行，只收 artifact/hash/测试/指标支撑的结果）

## 决策与转折
（最多5条；写清证伪、路线关闭、claim边界或下一研究分支。无则写"无"）

## 来源
（只列最关键的 KG/plan/artifact/commit 路径，最多6条）"""

    user_prompt = f"""日期：{report_date}
{pipeline_info}

## 实验数据与提交记录

{experiment_context}

## 要求
1. 总长度不超过1200中文字；无证据的内容不写
2. 结果必须保留 claim/no-claim 边界，不把可视化或机制 PASS 写成产品收益
3. 同一结论只出现一次；不复述 Hindsight 运维、token、网络、权限等内容
4. 不生成算法坑列表或 JSON；Pitfall Discovery 是唯一检测和写入流程
5. 输入状态冲突时以 KG current、日期更晚的 evidence 和已验证结果为准
6. 不生成“下一步”或行动计划；行动所有权属于 KG current 和正式 plan"""

    print("  Calling Hindsight LLM...", flush=True)
    result = _call_hindsight_llm(system_prompt, user_prompt, max_tokens=4096)

    if not result:
        print("  LLM call failed, skipping summary", flush=True)
        return 1

    errors = _research_digest_errors(result)
    if errors:
        print(f"  Digest contract retry: {'; '.join(errors)}", flush=True)
        correction_prompt = f"""{user_prompt}

## Previous invalid output
{result}

## Contract errors
{chr(10).join('- ' + error for error in errors)}

Regenerate the complete digest and fix only these contract errors."""
        result = _call_hindsight_llm(
            system_prompt, correction_prompt, max_tokens=4096
        )
        if not result:
            print("  Digest correction failed, keeping previous canonical file", flush=True)
            return 1
        errors = _research_digest_errors(result)
        if errors:
            print(
                f"  Digest rejected after correction: {'; '.join(errors)}",
                flush=True,
            )
            return 1

    # Write the compact digest. The main daily report embeds this file.
    try:
        llm_cfg = _get_hindsight_llm_config()
        metadata = (
            "<!--\n"
            f"source_date: {report_date}\n"
            f"generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"llm: {llm_cfg['provider']}/{llm_cfg['model']}\n"
            f"base_url: {llm_cfg['base_url']}\n"
            "role: research_delta_only\n"
            "-->\n\n"
        )
        digest_file = RESEARCH_DIGEST_DIR / f"{report_date}.md"
        tmp_file = digest_file.with_suffix(".tmp")
        tmp_file.write_text(metadata + result.strip() + "\n", encoding="utf-8")
        os.replace(tmp_file, digest_file)
        print(f"  Research digest written: {digest_file}", flush=True)
    except Exception as e:
        print(f"  Failed to write summary: {e}", flush=True)
        return 2

    # Print concise output for weixin delivery
    print("\n--- Research Digest ---", flush=True)
    for line in result.strip().split("\n")[:25]:
        print(line, flush=True)

    return 0


def _current_role() -> str:
    role = os.environ.get("HERMES_PROFILE", "").strip().lower()
    allowed = {"reviewer", "planner", "implementer", "critic", "coordinator"}
    return role if role in allowed else "unknown"


def _model_evidence_sha(logical_id: str) -> str | None:
    """Return the current per-model evidence identity after source verification."""
    path = HERMES_HOME / "mental-models" / "egomotion4d" / "evidence_bundle.json"
    if not path.exists():
        return None
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        entry = bundle.get("per_model", {}).get(logical_id)
        if not entry:
            return None
        for source in entry.get("sources", {}).values():
            source_path = Path(source["path"])
            if not source_path.is_file():
                return None
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual != source.get("sha256"):
                return None
        return entry.get("evidence_sha256") or None
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _canonical_evidence_sha(entry: dict) -> str:
    payload = {
        "d_ids": sorted(set(entry.get("d_ids", []))),
        "sources": {
            key: {
                "path": value["path"],
                "sha256": value["sha256"],
            }
            for key, value in sorted(entry.get("sources", {}).items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


CURRENT_EVIDENCE_BUILD_INPUT_MANIFEST = "derived-build-inputs.json"


def _validate_current_evidence_build_input(path: Path) -> None:
    """Reject curated current-evidence files that can masquerade as a truth store."""
    if not path.name.endswith("_current_evidence.md"):
        return
    manifest_path = path.parent / CURRENT_EVIDENCE_BUILD_INPUT_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest["sources"][path.name]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{path}: missing derived build input contract in {manifest_path}"
        ) from exc
    if (
        manifest.get("schema_version") != 1
        or contract.get("authority") != "kg-current-evidence"
        or contract.get("replaceable") is not True
    ):
        raise ValueError(f"{path}: invalid derived build input contract")


def _refresh_evidence_bundle() -> dict:
    """Refresh source hashes and deterministic identities without changing acceptance."""
    path = HERMES_HOME / "mental-models" / "egomotion4d" / "evidence_bundle.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    for logical_id, entry in bundle.get("per_model", {}).items():
        spec_name = logical_id.removeprefix("egomotion4d-") + ".json"
        spec_path = path.parent / "specs" / spec_name
        if spec_path.is_file():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            decision_ids = spec.get("decision_ids")
            if decision_ids is not None:
                if not isinstance(decision_ids, list) or not all(
                    isinstance(item, str) and re.fullmatch(r"D\d+", item)
                    for item in decision_ids
                ):
                    raise ValueError(
                        f"{spec_path}: decision_ids must contain only D<number> strings"
                    )
                entry["d_ids"] = list(dict.fromkeys(decision_ids))
        for source in entry.get("sources", {}).values():
            source_path = Path(source["path"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_bytes = source_path.read_bytes()
            _validate_current_evidence_build_input(source_path)
            source["sha256"] = hashlib.sha256(source_bytes).hexdigest()
        entry["evidence_sha256"] = _canonical_evidence_sha(entry)
    bundle["schema_version"] = 2
    bundle["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    global_payload = {
        key: value["evidence_sha256"]
        for key, value in sorted(bundle.get("per_model", {}).items())
    }
    bundle["bundle_sha256"] = hashlib.sha256(
        json.dumps(global_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return bundle


def _accepted_revision(model: dict) -> dict | None:
    revision = model.get("accepted_revision")
    if not isinstance(revision, dict):
        return None
    required = {"slot", "content_sha", "source_evidence_sha", "accepted_at"}
    return revision if required.issubset(revision) else None


def _render_mental_model_index(registry: dict, generated_date: str) -> str:
    lines = [
        "# Egomotion4D Mental Models",
        "",
        f"> Registry-derived current view; generated {generated_date}.",
        "> Only PASS_PUBLISH revisions linked below are consumable.",
        "",
        "| Logical ID | State | Quality | Revision | Token cap | Current |",
        "|---|---|---:|---|---:|---|",
    ]
    for logical_id, model in sorted(registry.get("models", {}).items()):
        accepted = _accepted_revision(model)
        verdict = str(model.get("last_verdict") or "INITIAL")
        quality_match = re.search(
            r"quality=(\d+)", str(model.get("verdict_detail", ""))
        )
        quality = quality_match.group(1) if quality_match else "-"
        revision = accepted.get("content_sha", "")[:12] if accepted else "-"
        current = (
            f"[read](exports/current/{logical_id}.md)"
            if accepted is not None and verdict == "PASS_PUBLISH"
            else "-"
        )
        lines.append(
            f"| `{logical_id}` | {verdict} | {quality} | `{revision}` | "
            f"{model.get('max_tokens', '-')} | {current} |"
        )
    lines.extend(
        [
            "",
            "## Read Order",
            "",
            "1. Current code/artifacts and current KG decisions.",
            "2. The relevant accepted model under `exports/current/`.",
            "3. Canonical algorithm pitfalls in `pitfall-catalog.md`.",
            "",
            "## Knowledge Ownership",
            "",
            "- The KG is the authoritative project truth: decisions, evidence, metrics, "
            "artifacts, scope and supersession.",
            "- Accepted mental models are bounded, prompt-ready consumption caches. "
            "They summarize selected D/P anchors and never override the KG.",
            "- `sources/*_current_evidence.md` are reproducible derived build inputs, "
            "not authored conclusions. They may be replaced from current KG/evidence "
            "and must not be maintained as a third truth store.",
            "- `pitfall-catalog.md` owns detailed algorithm-pitfall lifecycle; models "
            "carry only concise routing summaries and P identifiers.",
            "- Daily maintenance reports record deltas and audit evidence, not current truth.",
            "",
            "Write or revise a conclusion in the KG first. Rebuild the source snapshot, "
            "adjudicate the candidate against current KG evidence, and publish only on "
            "PASS_PUBLISH. A conflict rejects the candidate and preserves the previous "
            "accepted model; it never rewrites the KG.",
        ]
    )
    return "\n".join(lines) + "\n"


def _model_needs_refresh(
    model: dict, current_evidence_sha: str, pending_conflicts: int
) -> bool:
    """Return whether current evidence has not yet reached a terminal candidate gate."""
    if pending_conflicts > 0:
        return True
    accepted = _accepted_revision(model)
    if accepted is not None:
        adjudicated_evidence = accepted["source_evidence_sha"]
    else:
        adjudicated_evidence = model.get("last_candidate_evidence_sha")
    return adjudicated_evidence != current_evidence_sha


def _transaction_id(
    logical_id: str,
    slot: str,
    candidate_sha: str,
    source_evidence_sha: str,
    created: str,
) -> str:
    payload = {
        "candidate_sha": candidate_sha,
        "created": created,
        "logical_id": logical_id,
        "slot": slot,
        "source_evidence_sha": source_evidence_sha,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_candidate_transaction(
    logical_id: str,
    slot: str,
    candidate_sha: str,
    source_evidence_sha: str,
    created: str,
) -> dict:
    return {
        "tx_id": _transaction_id(
            logical_id, slot, candidate_sha, source_evidence_sha, created
        ),
        "logical_id": logical_id,
        "slot": slot,
        "candidate_sha": candidate_sha,
        "source_evidence_sha": source_evidence_sha,
        "created": created,
    }


def _validate_candidate_transaction(
    transaction: dict | None,
    *,
    logical_id: str,
    slot: str,
    candidate_sha: str,
    source_evidence_sha: str,
) -> bool:
    if not isinstance(transaction, dict):
        return False
    created = transaction.get("created", "")
    expected = _make_candidate_transaction(
        logical_id, slot, candidate_sha, source_evidence_sha, created
    )
    return all(transaction.get(key) == value for key, value in expected.items())


def _model_generation_spec(logical_id: str) -> dict:
    prefix = "egomotion4d-"
    suffix = logical_id[len(prefix):] if logical_id.startswith(prefix) else logical_id
    spec_path = (
        HERMES_HOME
        / "mental-models"
        / "egomotion4d"
        / "specs"
        / f"{suffix}.json"
    )
    if not spec_path.is_file():
        return {}
    try:
        return json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _model_generation_requirements(logical_id: str) -> dict:
    spec = _model_generation_spec(logical_id)
    return {
        "required_anchors": spec.get("required_anchors", []),
        "required_terminal_marker": spec.get("required_terminal_marker", ""),
        "require_source_facts": bool(spec.get("require_source_facts", False)),
        "required_prefix": spec.get("required_prefix", ""),
        "output_language": spec.get("output_language", ""),
    }


def _render_model_source_query(logical_id: str) -> str:
    spec = _model_generation_spec(logical_id)
    query = spec.get("source_query", "")
    sections = [query.rstrip()]
    if spec.get("output_language") == "zh-CN":
        sections.append(
            "输出语言合同：必须使用简体中文输出正文。D 编号、代码符号、schema "
            "字段和必要的英文技术标识符可以原样保留；不得输出全英文正文。"
        )
    if not spec.get("inline_source_files"):
        return "\n\n".join(sections).rstrip() + "\n"
    root = HERMES_HOME / "mental-models" / "egomotion4d"
    for relative_path in spec.get("source_files", []):
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid inline source path: {relative_path!r}")
        source_path = root / relative
        content = source_path.read_text(encoding="utf-8")
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        sections.append(
            "\n".join(
                [
                    f"BEGIN AUTHORITATIVE SOURCE {relative_path} sha256={source_sha}",
                    content.rstrip(),
                    f"END AUTHORITATIVE SOURCE {relative_path}",
                ]
            )
        )
    return "\n\n".join(sections) + "\n"


def _synchronize_model_generation_contract(
    api_url: str, physical_id: str, logical_id: str
) -> str:
    import urllib.request

    spec = _model_generation_spec(logical_id)
    query = _render_model_source_query(logical_id)
    payload = {
        "source_query": query,
        "tags": spec.get("tags", ["egomotion4d"]),
        "max_tokens": spec.get("max_tokens", 2048),
    }
    request = urllib.request.Request(
        f"{api_url}/v1/default/banks/hermes/mental-models/{physical_id}",
        method="PATCH",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode(),
    )
    with urllib.request.urlopen(request, timeout=30):
        pass
    return hashlib.sha256(query.encode()).hexdigest()


def _model_benchmark_contract(logical_id: str) -> tuple[Path, list[str]]:
    root = HERMES_HOME / "mental-models" / "egomotion4d" / "benchmark"
    spec = _model_generation_spec(logical_id)
    filename = spec.get("benchmark_file", "questions.json")
    if Path(filename).name != filename:
        raise ValueError(f"benchmark_file must be a basename: {filename!r}")
    smoke_ids = spec.get(
        "smoke_ids", ["Q01", "Q04", "Q05", "Q10", "Q13", "Q15"]
    )
    if not isinstance(smoke_ids, list) or not all(
        isinstance(item, str) for item in smoke_ids
    ):
        raise ValueError("smoke_ids must be a list of strings")
    return root / filename, smoke_ids


def _candidate_completeness_errors(
    content: str,
    requirements: dict,
    *,
    source_fact_count: int | None = None,
) -> list[str]:
    required_anchors = requirements.get("required_anchors", [])
    missing = [anchor for anchor in required_anchors if anchor not in content]
    errors = []
    if missing:
        errors.append(f"missing anchors: {', '.join(missing)}")
    terminal_marker = requirements.get("required_terminal_marker", "")
    if terminal_marker and not content.rstrip().endswith(terminal_marker):
        errors.append(f"missing terminal marker: {terminal_marker}")
    if requirements.get("require_source_facts") and not source_fact_count:
        errors.append("no source facts in reflect_response.based_on")
    required_prefix = requirements.get("required_prefix", "")
    if required_prefix and not content.startswith(required_prefix):
        errors.append(f"invalid content prefix: expected {required_prefix}")
    if requirements.get("output_language") == "zh-CN":
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in content)
        if cjk_count < 20:
            errors.append("output language must be Simplified Chinese")
    return errors


def _reflect_source_fact_count(model_data: dict) -> int:
    based_on = model_data.get("reflect_response", {}).get("based_on", {})
    return sum(
        len(items)
        for key, items in based_on.items()
        if key != "mental-models" and isinstance(items, list)
    )


def _refresh_model_once(api_url: str, physical_id: str) -> tuple[str, str, dict | None]:
    """Refresh one physical model and return operation id, terminal status and data."""
    import urllib.request

    request = urllib.request.Request(
        f"{api_url}/v1/default/banks/hermes/mental-models/{physical_id}/refresh",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=b"",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        operation_id = json.loads(response.read()).get("operation_id", "?")
    for _ in range(18):
        time.sleep(10)
        try:
            with urllib.request.urlopen(
                f"{api_url}/v1/default/banks/hermes/operations/{operation_id}",
                timeout=15,
            ) as response:
                status = json.loads(response.read()).get("status", "?")
        except Exception:
            continue
        if status not in ("completed", "failed", "cancelled"):
            continue
        if status != "completed":
            return operation_id, status, None
        with urllib.request.urlopen(
            f"{api_url}/v1/default/banks/hermes/mental-models/{physical_id}",
            timeout=30,
        ) as response:
            return operation_id, status, json.loads(response.read())
    return operation_id, "timeout", None


def mental_model_preflight(logical_id: str = "egomotion4d-research-guardrails") -> int:
    _log_role_trigger(_current_role(), "mental_model_preflight_attempt", logical_id=logical_id)
    """Read registry, fetch active slot content, output to stdout.

    Used by agents before algorithm design / pre-research / root-cause analysis.
    Outputs: active model content + pitfall router summary.
    Exit 0 on success, 1 on error.
    """
    import urllib.request

    registry_path = HERMES_HOME / "mental-models" / "egomotion4d" / "registry.json"
    if not registry_path.exists():
        print(f"ERROR: registry not found at {registry_path}", file=sys.stderr)
        return 1

    with open(registry_path) as f:
        registry = json.load(f)

    model = registry.get("models", {}).get(logical_id)
    if not model:
        print(f"ERROR: logical model '{logical_id}' not in registry", file=sys.stderr)
        return 1

    active_slot = model["active_slot"]
    physical_id = model["physical_ids"][active_slot]
    last_verdict = model.get("last_verdict", "INITIAL")
    accepted = _accepted_revision(model)

    # Fail-closed gate: ONLY PASS_PUBLISH with exact content SHA match is consumable.
    # INITIAL, REJECT, ESCALATE_D_REVIEW, PASS_NO_CHANGE, hash-mismatched,
    # stale evidence, missing content_sha, or unaccepted revisions -> block + KG fallback.
    # MUST NOT expose model content for non-accepted states.
    if accepted is None:
        print(f"BLOCK: model '{logical_id}' verdict is {last_verdict} — content not consumable.", file=sys.stderr)
        print(f"ACTION: consult current KG at /home/wyr/.hermes/hermes_use/projects/Egomotion4D-kg/", file=sys.stderr)
        print(f"  00-card.md, topic 00-current.md files, and current D entries in 10-current-decisions.md", file=sys.stderr)
        return 2

    if accepted["slot"] != active_slot:
        print(f"BLOCK: model '{logical_id}' active slot does not match accepted revision.", file=sys.stderr)
        return 2
    content_sha = accepted["content_sha"]

    # Verify per-model evidence bundle identity for staleness detection.
    # Evidence bundle staleness ≠ content staleness; bundle changes when sources change.
    _evidence_current_sha = _model_evidence_sha(logical_id)
    if not _evidence_current_sha or accepted["source_evidence_sha"] != _evidence_current_sha:
        print(f"BLOCK: model '{logical_id}' evidence bundle is stale — content not consumable.", file=sys.stderr)
        print(f"ACTION: run --mental-model-daily to refresh before consumption.", file=sys.stderr)
        return 2

    # Fetch active content from Hindsight API and verify against registry content_sha.
    # Content hash mismatch is a hard block — never expose stale content.
    api_url = os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888")
    try:
        with urllib.request.urlopen(f"{api_url}/v1/default/banks/hermes/mental-models/{physical_id}", timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"ERROR: failed to fetch mental model {physical_id}: {e}", file=sys.stderr)
        return 1

    content = data.get("content", "")
    if not content:
        print(f"ERROR: mental model {physical_id} has empty content", file=sys.stderr)
        return 1

    # Verify fetched content hash against registry (hard block if mismatch)
    _actual_content_sha = hashlib.sha256(content.encode()).hexdigest()
    if _actual_content_sha != content_sha:
        print(f"BLOCK: model '{logical_id}' content SHA mismatch (registry={content_sha[:12]}, actual={_actual_content_sha[:12]}) — content not consumable.", file=sys.stderr)
        print(f"ACTION: model must be re-adjudicated before consumption.", file=sys.stderr)
        return 2

    _log_role_trigger(
        _current_role(),
        "mental_model_consumed",
        logical_id=logical_id,
        revision=content_sha,
        source_evidence_sha=accepted["source_evidence_sha"],
        context_tokens=max(1, len(content) // 4),
    )

    # Output content to stdout for agent consumption (only for accepted, verified content)
    print(f"# Mental Model Preflight: {logical_id}")
    print(f"# Active slot: {active_slot} ({physical_id})")
    print(f"# Last refreshed: {data.get('last_refreshed_at', '?')}")
    print(f"# Tags: {data.get('tags', [])}")
    print()
    print(content)

    return 0


def pitfall_lookup(keywords: str | None = None, status_filter: str = "current") -> int:
    """Search pitfall_index.json by keywords, output matching entries.

    Args:
        keywords: comma-separated search terms. If None, list all.
        status_filter: 'current' (default), 'all', 'candidate', 'superseded'.

    Usage:
        python3 hindsight_daily_noagent.py --mental-model-pitfalls depth,scale
        python3 hindsight_daily_noagent.py --mental-model-pitfalls --all
    """
    pitfall_index_path = HERMES_HOME / "mental-models" / "egomotion4d" / "pitfall_index.json"
    if not pitfall_index_path.exists():
        print("ERROR: pitfall_index.json not found", file=sys.stderr)
        return 1

    with open(pitfall_index_path) as f:
        idx = json.load(f)

    entries = idx.get("entries", [])

    # Filter by status
    if status_filter == "all":
        filtered = entries
    else:
        filtered = [e for e in entries if e.get("status") == status_filter]

    # Filter by keywords (search in title, trigger, root_cause, tags)
    if keywords:
        kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        scored = []
        for e in filtered:
            text = (e.get("title", "") + " " + e.get("trigger", "") + " " +
                    e.get("root_cause", "") + " " + " ".join(e.get("tags", []))).lower()
            score = sum(1 for kw in kw_list if kw in text)
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        filtered = [e for _, e in scored]

    # Output
    print(f"# Pitfall Lookup: {len(filtered)} entries (status={status_filter}, keywords={keywords or 'none'})")
    print()

    for e in filtered:
        print(f"## {e['p_id']}: {e.get('title', '?')}")
        print(f"  status: {e.get('status', '?')}")
        if e.get("trigger"):
            print(f"  trigger: {e['trigger'][:200]}")
        if e.get("root_cause"):
            print(f"  root_cause: {e['root_cause'][:200]}")
        if e.get("detail_locator"):
            print(f"  locator: {e['detail_locator']}")
        if e.get("alias_of"):
            print(f"  alias_of: {e['alias_of']}")
        if e.get("superseded_by"):
            print(f"  superseded_by: {e['superseded_by']}")
        print()

    return 0


def mental_model_maintain(selected_logical_id: str | None = None) -> int:
    _log_role_trigger(_current_role(), "mental_model_maintain")
    """Stage A deterministic collector for mental_model daily maintenance.

    Steps:
    1. Health/bank/queue check
    2. Read registry, active content, source watermark, pending conflicts
    3. Check if source watermark changed (skip if stale and no conflicts)
    4. Refresh inactive slot (candidate) for stale models
    5. Generate old/new diff, pitfall catalog diff
    6. Output adjudication packet for Stage B (caller decides whether to run LLM)

    Does NOT modify active slot. Does NOT run LLM (that's Stage B).
    Exit 0 on success, 1 on infra error, 2 on skip (not stale).
    """
    import urllib.request
    import hashlib

    api_url = os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888")
    registry_path = HERMES_HOME / "mental-models" / "egomotion4d" / "registry.json"
    conflicts_path = HERMES_HOME / "mental-models" / "egomotion4d" / "conflicts.jsonl"
    reports_dir = Path("/home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not registry_path.exists():
        print("ERROR: registry not found", file=sys.stderr)
        return 1

    with open(registry_path) as f:
        registry = json.load(f)

    try:
        _refresh_evidence_bundle()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: evidence bundle refresh failed: {exc}", file=sys.stderr)
        return 1

    # 1. Health check
    try:
        with urllib.request.urlopen(f"{api_url}/v1/default/banks/hermes/stats", timeout=15) as resp:
            stats = json.loads(resp.read())
    except Exception as e:
        print(f"ERROR: Hindsight API unreachable: {e}", file=sys.stderr)
        return 1

    pending = stats.get("pending_operations", 0)
    processing = stats.get("operations_by_status", {}).get("processing", 0)
    if processing > 0:
        print(f"SKIP_BUSY: {processing} operations processing, skipping maintenance")
        return 2

    # 2. Check pending conflicts
    pending_conflicts = 0
    if conflicts_path.exists():
        with open(conflicts_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                    if c.get("status") == "pending":
                        pending_conflicts += 1
                except json.JSONDecodeError:
                    continue

    # 3. For each logical model, check if source evidence stale
    # Staleness compares actual evidence bundle identity, not active/candidate content text.
    import hashlib as _hl
    evidence_path = HERMES_HOME / "mental-models" / "egomotion4d" / "evidence_bundle.json"

    today = time.strftime("%Y-%m-%d")
    report_lines = [f"# Mental Model Daily Maintenance Report ({today})", ""]

    any_stale = False
    for logical_id, model in registry.get("models", {}).items():
        if selected_logical_id and logical_id != selected_logical_id:
            report_lines.append(f"## {logical_id}: SKIP_NOT_SELECTED")
            continue
        active_slot = model["active_slot"]
        active_physical_id = model["physical_ids"][active_slot]
        inactive_slot = "b" if active_slot == "a" else "a"
        inactive_physical_id = model["physical_ids"][inactive_slot]

        current_evidence_sha = model.get("source_evidence_sha")
        if not current_evidence_sha:
            report_lines.append(f"## {logical_id}: SKIP_NO_EVIDENCE_BUNDLE")
            continue

        current_evidence_actual_sha = _model_evidence_sha(logical_id)
        if not current_evidence_actual_sha:
            report_lines.append(f"## {logical_id}: BLOCK_INVALID_EVIDENCE_BUNDLE")
            continue
        model["source_evidence_sha"] = current_evidence_actual_sha

        # Accepted and rejected candidates have separate watermarks. A rejection
        # must not mutate accepted state, but it must suppress identical retries.
        accepted = _accepted_revision(model)
        stored_watermark = (
            accepted["source_evidence_sha"]
            if accepted is not None
            else model.get("last_candidate_evidence_sha")
        )
        is_stale = _model_needs_refresh(
            model, current_evidence_actual_sha, pending_conflicts
        )

        if not is_stale:
            report_lines.append(f"## {logical_id}: SKIP_NOT_STALE (evidence={current_evidence_actual_sha[:16]})")
            continue

        any_stale = True
        report_lines.append(f"## {logical_id}: STALE (evidence watermark {stored_watermark} -> {current_evidence_actual_sha})")
        report_lines.append(f"  Active: {active_physical_id}, verdict: {model.get('last_verdict', '?')}")
        report_lines.append(f"  Inactive: {inactive_physical_id} (will refresh)")
        query_sha = _synchronize_model_generation_contract(
            api_url, inactive_physical_id, logical_id
        )
        report_lines.append(f"  Generation query_sha: {query_sha}")

        # 4. Trigger refresh of inactive slot. A single retry is allowed only
        # when Hindsight returned no source facts; content/gate failures are not retried.
        try:
            for refresh_attempt in range(2):
                op_id, status, new_data = _refresh_model_once(
                    api_url, inactive_physical_id
                )
                report_lines.append(
                    f"  Refresh attempt {refresh_attempt + 1}: "
                    f"operation_id={op_id}, status={status}"
                )
                if status != "completed" or new_data is None:
                    break

                new_content = new_data.get("content", "")
                new_content_full_sha = hashlib.sha256(new_content.encode()).hexdigest()
                report_lines.append(f"  New content_sha: {new_content_full_sha[:16]}")
                report_lines.append(f"  Content length: {len(new_content)}")
                completeness_errors = _candidate_completeness_errors(
                    new_content,
                    _model_generation_requirements(logical_id),
                    source_fact_count=_reflect_source_fact_count(new_data),
                )
                if (
                    "no source facts in reflect_response.based_on"
                    in completeness_errors
                    and refresh_attempt == 0
                ):
                    report_lines.append("  Source facts absent: retrying once")
                    continue
                if completeness_errors:
                    report_lines.append(
                        "  Candidate completeness: FAIL ("
                        + "; ".join(completeness_errors)
                        + ")"
                    )
                    model["candidate_content_sha"] = new_content_full_sha
                    model["candidate_slot"] = inactive_slot
                    model["candidate_physical_id"] = inactive_physical_id
                    model["last_candidate_verdict"] = "REJECT_INCOMPLETE"
                    model["last_candidate_evidence_sha"] = current_evidence_actual_sha
                    model.pop("candidate_transaction", None)
                    break

                report_lines.append("  Candidate completeness: PASS")
                created = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                model["candidate_content_sha"] = new_content_full_sha
                model["candidate_slot"] = inactive_slot
                model["candidate_physical_id"] = inactive_physical_id
                model["candidate_transaction"] = _make_candidate_transaction(
                    logical_id,
                    inactive_slot,
                    new_content_full_sha,
                    current_evidence_actual_sha,
                    created,
                )
                report_lines.append(
                    f"  Transaction ID: {model['candidate_transaction']['tx_id'][:12]}..."
                )
                break
        except Exception as e:
            report_lines.append(f"  ERROR refreshing: {e}")

    if not any_stale and pending_conflicts == 0:
        report_lines.insert(1, "All models up-to-date, no pending conflicts.")
        print("\n".join(report_lines))
        return 2

    # 5. Update registry: Stage A stores candidate info but does NOT mutate
    # accepted active state or accepted source watermark.
    registry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    for logical_id, model in registry.get("models", {}).items():
        if model.get("candidate_content_sha"):
            # candidate_content_sha and candidate_slot are temporary fields
            # for Stage B consumption only. They do not overwrite
            # source_watermark, active_content_sha, or last_verdict.
            model.pop("candidate_watermark", None)  # old name, remove
            pass  # Keep candidate_content_sha and candidate_slot for Stage B

    # Atomic write registry
    tmp_path = registry_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    os.rename(str(tmp_path), str(registry_path))

    # 6. Save report
    report_file = reports_dir / f"{today}-maintenance.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nReport saved: {report_file}")
    print("Stage A complete. Stage B (adjudication) pending.")

    # Pitfall candidate discovery (between Stage A and Stage B)
    print("\n--- Pitfall Candidate Discovery ---\n")
    pitfall_rc = _discover_pitfall_candidates(api_url)

    return 0


def _discover_pitfall_candidates(api_url: str) -> int:
    """Discover algorithm-level pitfall candidates from recent observations.

    Queries Hindsight for recent pitfall-like observations, uses LLM to classify
    which are algorithm-level, deduplicates against existing pitfall_index.json,
    and outputs candidate list for human/Stage B review.

    Returns 0 on success, 1 on error.
    """
    import urllib.request

    pitfall_index_path = HERMES_HOME / "mental-models" / "egomotion4d" / "pitfall_index.json"
    reports_dir = Path("/home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/reports")
    writer_dir = HERMES_HOME / "mental-models" / "egomotion4d"
    if str(writer_dir) not in sys.path:
        sys.path.insert(0, str(writer_dir))
    from pitfall_writer import PitfallIndex, PitfallStatus

    pitfall_index = PitfallIndex.load(str(pitfall_index_path))
    existing_pitfalls = [entry.to_dict() for entry in pitfall_index.entries]

    # Query recent pitfall-like observations via Hindsight recall
    pitfall_query = "algorithm pitfalls, falsified approaches, failed routes, no-gain results, closed routes, root cause analysis for algorithm design"
    try:
        recall_req = urllib.request.Request(
            f"{api_url}/v1/default/banks/hermes/memories/recall",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "query": pitfall_query,
                "max_tokens": 3000,
                "types": ["observation"],
            }).encode(),
        )
        with urllib.request.urlopen(recall_req, timeout=60) as resp:
            recall_data = json.loads(resp.read())
    except Exception as e:
        print(f"ERROR: recall failed: {e}")
        return 1

    results = recall_data.get("results", [])
    if not results:
        print("No pitfall candidates found in recent observations.")
        return 0

    # Build candidate summaries for LLM classification
    candidate_texts = []
    for i, r in enumerate(results[:20]):  # cap at 20 candidates per run
        text = r.get("text", "")[:300]
        candidate_texts.append(f"[{i+1}] {text}")

    # LLM classification: which are algorithm-level pitfalls?
    existing_summaries = "\n".join(
        f"- {p['p_id']} [{p.get('status')}]: {p.get('title', '')[:80]}"
        for p in existing_pitfalls
    )

    system_prompt = """You are an algorithm pitfall classifier. Given a list of observations from a memory bank, classify which ones are algorithm-level pitfalls vs implementation-level issues.

Algorithm-level pitfalls (INCLUDE):
- Changes algorithm representation, objective, metric, association, observer, gate or route selection
- Reveals load-bearing assumption failure, identifiability, evaluation bias, cross-domain failure
- High recurrence with symptom -> root cause -> prevention evidence chain

Non-algorithmic (EXCLUDE):
- Single function bugs, variable/path errors, CLI mistakes
- Dependency install, temp service/GPU/permission issues
- Code style, commit-specific implementation details

Respond with JSON: {"candidates": [{"index": 1, "is_algorithm": true, "title": "short title", "trigger": "when does this occur", "root_cause": "why", "status": "candidate", "duplicate_of": null, "reason": "why include/exclude"}]}"""

    user_prompt = f"""Existing canonical pitfalls (for dedup):
{existing_summaries}

New observations to classify:
{chr(10).join(candidate_texts)}

Classify each. Mark duplicates with duplicate_of=P-id."""

    llm_response = _call_hindsight_llm(system_prompt, user_prompt, max_tokens=3000)

    if not llm_response:
        print("WARNING: LLM classification failed, skipping pitfall discovery")
        return 1

    # Parse response
    try:
        def _extract_json(text):
            start = text.find('{')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError:
                            return None
            return None

        result = _extract_json(llm_response)
        if not result:
            print("WARNING: could not parse LLM pitfall response")
            return 1

        candidates = result.get("candidates", [])
        new_pitfalls = [c for c in candidates if c.get("is_algorithm") and not c.get("duplicate_of")]
        duplicates = [c for c in candidates if c.get("duplicate_of")]
        excluded = [c for c in candidates if not c.get("is_algorithm")]

        print(f"Pitfall candidates: {len(candidates)} total")
        print(f"  New algorithm-level: {len(new_pitfalls)}")
        print(f"  Duplicates of existing: {len(duplicates)}")
        print(f"  Non-algorithmic (excluded): {len(excluded)}")

        # Add candidates through the sole canonical writer. Candidate entries
        # are not consumable until a later evidence-backed adjudication creates
        # their detail record and promotes their status.
        if new_pitfalls and pitfall_index_path.exists():
            for np in new_pitfalls:
                source_index = int(np.get("index", 0)) - 1
                source_result = results[source_index] if 0 <= source_index < len(results) else {}
                source_text = source_result.get("text", "")
                p_id = pitfall_index.add_entry(
                    title=np.get("title", ""),
                    status=PitfallStatus.CANDIDATE,
                    is_algorithm_level=True,
                    trigger=np.get("trigger", ""),
                    root_cause=np.get("root_cause", ""),
                    lesson="",
                    tags=["egomotion4d"],
                    source="mental_model_daily_discovery",
                    detail_locator="",
                    source_memory_id=str(source_result.get("id") or "") or None,
                    source_content_hash=(
                        hashlib.sha256(source_text.encode()).hexdigest()
                        if source_text
                        else None
                    ),
                )
                print(f"  Added: {p_id} - {np.get('title', '')[:60]}")

            pitfall_index.dedup()
            pitfall_index.recalc_counters()
            errors = pitfall_index.validate()
            if errors:
                raise RuntimeError(f"pitfall index invalid: {errors[:5]}")
            pitfall_index.save(str(pitfall_index_path))
            print(f"  Updated: {pitfall_index_path}")

        # Write discovery report
        today = time.strftime("%Y-%m-%d")
        report_lines = [
            f"# Pitfall Discovery Report ({today})",
            f"",
            f"## Summary",
            f"- Candidates scanned: {len(candidates)}",
            f"- New algorithm-level: {len(new_pitfalls)}",
            f"- Duplicates: {len(duplicates)}",
            f"- Excluded (non-algorithmic): {len(excluded)}",
            f"",
            f"## New Pitfalls",
        ]
        for np in new_pitfalls:
            report_lines.append(f"- {np.get('title', '?')}: {np.get('trigger', '')[:80]}")
        report_lines.append(f"\n## Duplicates")
        for d in duplicates:
            report_lines.append(f"- #{d.get('index', '?')} -> {d.get('duplicate_of', '?')}")

        report_file = reports_dir / f"{today}-pitfall-discovery.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))
        print(f"  Report: {report_file}")

    except Exception as e:
        print(f"ERROR parsing pitfall candidates: {e}")
        return 1

    return 0


def _build_active_context(
    api_url: str,
    max_tokens: int = 2000,
    exclude_logical_id: str | None = None,
) -> str:
    """Build exact context from accepted (PASS_PUBLISH) active models only.

    Fetches registry, identifies PASS_PUBLISH models, reads their exact
    active slot content, and returns a concatenated summary. No fuzzy search.
    """
    import hashlib
    import urllib.request
    registry_path = HERMES_HOME / "mental-models" / "egomotion4d" / "registry.json"
    if not registry_path.exists():
        return ""
    with open(registry_path) as f:
        registry = json.load(f)

    parts = []
    for logical_id, model in registry.get("models", {}).items():
        if logical_id == exclude_logical_id:
            continue
        accepted = _accepted_revision(model)
        if accepted is None:
            continue
        active_slot = model["active_slot"]
        if accepted["slot"] != active_slot:
            continue
        current_evidence_sha = _model_evidence_sha(logical_id)
        if not current_evidence_sha or accepted["source_evidence_sha"] != current_evidence_sha:
            continue
        physical_id = model["physical_ids"][active_slot]
        content_sha = accepted["content_sha"]
        try:
            with urllib.request.urlopen(
                f"{api_url}/v1/default/banks/hermes/mental-models/{physical_id}",
                timeout=30,
            ) as resp:
                data = json.loads(resp.read())
            content = data.get("content", "")
            if content:
                # Verify content SHA matches registry
                actual_sha = hashlib.sha256(content.encode()).hexdigest()
                if actual_sha != content_sha:
                    continue
                parts.append(f"# {logical_id}")
                # Truncate to reasonable size
                token_estimate = len(content) // 4
                budget_per_model = max(200, max_tokens // max(1, len(parts)))
                trunc_len = budget_per_model * 4
                if token_estimate > budget_per_model:
                    content = content[:trunc_len] + "\n... [truncated]"
                parts.append(content)
        except Exception:
            continue

    return "\n\n".join(parts) if parts else ""


def _normalize_expected_text(value: str) -> str:
    """Normalize presentation-only differences before contract matching."""
    normalized = str(value).lower().replace("_", "-")
    for dash in ("‐", "‑", "‒", "–", "—", "―", "−"):
        normalized = normalized.replace(dash, "-")
    return " ".join(normalized.split())


def _matches_expected_term(text: str, term: str | list[str]) -> bool:
    """Match one required concept; list values are predeclared alternatives."""
    normalized_text = _normalize_expected_text(text)
    alternatives = term if isinstance(term, list) else [term]
    return any(
        _normalize_expected_text(alternative) in normalized_text
        for alternative in alternatives
    )


def _format_gate_question(question: dict) -> str:
    """Expose citation requirements already enforced by the scorer."""
    anchors = [str(anchor) for anchor in question.get("key_d_refs", [])]
    if not anchors:
        return question["question"]
    return (
        f"{question['question']}\n\n"
        f"Required decision anchors: {', '.join(anchors)}."
    )


def _run_smoke_regression(
    api_url: str,
    active_context_override: str | None = None,
    logical_id: str = "egomotion4d-research-guardrails",
) -> int:
    """Run 6-question smoke regression against active mental models.

    Blocking gate: uses exact registry-selected accepted active content only.
    Generic mental-model retrieval (reflect without exclude_mental_models) is
    FORBIDDEN for blocking gates — it may pick up inactive or stale slots.

    For each question:
    1. Load exact accepted active content from registry (only PASS_PUBLISH models)
    2. Run reflect with exclude_mental_models=true
    3. Inject accepted content as explicit context
    4. Check pass_condition

    Returns 0 if all pass, 1 if any fail, 2 on infra error.
    """
    import urllib.request

    benchmark_path, smoke_ids = _model_benchmark_contract(logical_id)
    if not benchmark_path.exists():
        print("SKIP: benchmark questions not found")
        return 2

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    smoke_questions = [q for q in benchmark["questions"] if q["id"] in smoke_ids]
    if len(smoke_questions) != len(smoke_ids):
        print(
            f"BLOCK_INVALID_SMOKE_CONTRACT: expected {len(smoke_ids)} questions, "
            f"found {len(smoke_questions)}"
        )
        return 2
    active_context = (
        active_context_override
        if active_context_override is not None
        else _build_active_context(api_url, max_tokens=2000)
    )
    if not active_context.strip():
        print("NO_CLAIM_NO_ACCEPTED_MODELS: smoke has no treatment context")
        return 0

    passed = 0
    failed = 0
    errors = 0

    for q in smoke_questions:
        qid = q["id"]
        try:
            enhanced_query = f"{active_context}\n\n{_format_gate_question(q)}"

            # Run reflect WITH exclude_mental_models=true (blocking gate rule)
            # and inject exact accepted content as explicit context
            reflect_req = urllib.request.Request(
                f"{api_url}/v1/default/banks/hermes/reflect",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "query": enhanced_query,
                    "max_tokens": 500,
                    "budget": "low",
                    "exclude_mental_models": True,
                }).encode(),
            )
            with urllib.request.urlopen(reflect_req, timeout=120) as resp:
                reflect_data = json.loads(resp.read())
            text = reflect_data.get("text", "")

            if not text or len(text) < 50:
                print(f"  {qid}: ERROR (empty response)")
                errors += 1
                continue

            key_terms = list(q.get("key_d_refs", []))
            key_terms += list(q.get("expected_pitfall_triggers", []))
            matched = sum(1 for term in key_terms if _matches_expected_term(text, term))
            total = len(key_terms)

            d_refs = list(q.get("key_d_refs", []))
            has_d_refs = all(_matches_expected_term(text, term) for term in d_refs)
            if total > 0 and has_d_refs and matched >= total * 0.5:
                print(f"  {qid}: PASS ({matched}/{total} key terms)")
                passed += 1
            else:
                print(f"  {qid}: FAIL ({matched}/{total} key terms)")
                failed += 1

        except Exception as e:
            print(f"  {qid}: ERROR ({e})")
            errors += 1

    print(f"\nSmoke regression: {passed} passed, {failed} failed, {errors} errors")

    # Write smoke report
    today = time.strftime("%Y-%m-%d")
    reports_dir = Path("/home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/reports")
    report_file = reports_dir / f"{today}-smoke-regression.md"
    with open(report_file, "w") as f:
        f.write(f"# Smoke Regression Report ({today})\n\n")
        f.write(f"- Passed: {passed}\n- Failed: {failed}\n- Errors: {errors}\n\n")
        if failed > 0 or errors > 0:
            f.write("⚠️ Some checks failed. Review model quality.\n")
        else:
            f.write("✅ All smoke checks passed.\n")

    # This result is a publication/monitoring gate, not a report-only metric.
    if failed > 0 or errors > 0:
        return 1
    return 0


def _run_all_model_smoke(api_url: str) -> int:
    """Run each accepted model's own smoke contract in isolation."""
    registry_path = HERMES_HOME / "mental-models" / "egomotion4d" / "registry.json"
    if not registry_path.exists():
        print("BLOCK: mental-model registry missing")
        return 2
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    accepted_ids = [
        logical_id
        for logical_id, model in sorted(registry.get("models", {}).items())
        if isinstance(model.get("accepted_revision"), dict)
    ]
    if not accepted_ids:
        print("NO_CLAIM_NO_ACCEPTED_MODELS: no smoke suites to run")
        return 2

    failures = 0
    for logical_id in accepted_ids:
        print(f"\n--- Target Smoke: {logical_id} ---")
        if _run_smoke_regression(api_url, logical_id=logical_id) != 0:
            failures += 1
    print(
        f"\nAll-model smoke: {len(accepted_ids) - failures}/{len(accepted_ids)} passed"
    )
    return 0 if failures == 0 else 2


def _extract_key_terms(condition: str) -> list[str]:
    """Extract key terms from a pass_condition string for smoke check.

    The condition describes what the answer must contain.
    We extract the most distinctive terms (excluding common words).
    """
    # Common words to exclude
    stop_words = {"必须", "说明", "回答", "不能", "可以", "列出", "提到", "区分",
                  "the", "must", "should", "answer", "explain", "not", "can",
                  "be", "to", "and", "or", "a", "in", "for", "is", "are"}

    # Split by common delimiters and filter
    terms = []
    for sep in ["，", ",", "、", "；", ";", " "]:
        condition = condition.replace(sep, " ")
    words = condition.split()

    for w in words:
        w = w.strip().strip(".。:：()（）").lower()
        if len(w) >= 3 and w not in stop_words:
            terms.append(w)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # Cap at 8 terms to avoid over-constraining
    return unique[:8]


def mental_model_adjudicate(selected_logical_id: str | None = None) -> int:
    _log_role_trigger(_current_role(), "mental_model_adjudicate")
    """Stage B adjudicator for mental_model daily maintenance.

    Reads Stage A report, fetches candidate content (inactive slot),
    compares against current D decisions, runs LLM-based quality check,
    and atomically publishes verdict.

    Exit 0 on PASS_PUBLISH/PASS_NO_CHANGE, 1 on error, 2 on REJECT, 3 on ESCALATE.
    """
    import urllib.request
    import hashlib

    api_url = os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888")
    registry_path = HERMES_HOME / "mental-models" / "egomotion4d" / "registry.json"
    conflicts_path = HERMES_HOME / "mental-models" / "egomotion4d" / "conflicts.jsonl"
    kg_decisions_path = REAL_HOME / ".hermes" / "hermes_use" / "projects" / "Egomotion4D-kg" / "10-current-decisions.md"
    reports_dir = Path("/home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/reports")
    today = time.strftime("%Y-%m-%d")

    if not registry_path.exists():
        print("ERROR: registry not found", file=sys.stderr)
        return 1

    with open(registry_path) as f:
        registry = json.load(f)

    # Read the complete current decision source. Character-prefix truncation can
    # silently omit the newest decisions and is forbidden for adjudication.
    d_decisions = ""
    if kg_decisions_path.exists():
        d_decisions = kg_decisions_path.read_text(encoding="utf-8")

    any_published = False
    any_rejected = False
    any_escalated = False

    for logical_id, model in registry.get("models", {}).items():
        if selected_logical_id and logical_id != selected_logical_id:
            print(f"[{logical_id}] SKIP_NOT_SELECTED")
            continue
        active_slot = model["active_slot"]
        active_physical_id = model["physical_ids"][active_slot]
        inactive_slot = "b" if active_slot == "a" else "a"
        inactive_physical_id = model["physical_ids"][inactive_slot]

        # Fetch candidate content (inactive slot)
        try:
            with urllib.request.urlopen(
                f"{api_url}/v1/default/banks/hermes/mental-models/{inactive_physical_id}",
                timeout=30,
            ) as resp:
                candidate_data = json.loads(resp.read())
        except Exception as e:
            print(f"ERROR fetching candidate {inactive_physical_id}: {e}", file=sys.stderr)
            continue

        candidate_content = candidate_data.get("content", "")
        candidate_refreshed = candidate_data.get("last_refreshed_at", "?")

        # Fetch active content for comparison
        try:
            with urllib.request.urlopen(
                f"{api_url}/v1/default/banks/hermes/mental-models/{active_physical_id}",
                timeout=30,
            ) as resp:
                active_data = json.loads(resp.read())
        except Exception as e:
            print(f"ERROR fetching active {active_physical_id}: {e}", file=sys.stderr)
            continue

        active_content = active_data.get("content", "")

        # Compare candidate and active content only after binding the candidate
        # to the exact Stage-A transaction.
        active_watermark = hashlib.sha256(active_content.encode()).hexdigest()[:16]
        candidate_watermark = hashlib.sha256(candidate_content.encode()).hexdigest()[:16]

        # Pending transaction validation: Stage B must accept exactly the
        # candidate committed by Stage A. The pending transaction ID binds:
        #   (logical_id, inactive_slot, candidate_sha, source_evidence_sha, creation_date)
        # If any field differs, the transaction is stale and must be rejected.
        candidate_full_sha = hashlib.sha256(candidate_content.encode()).hexdigest()
        expected_tx = model.get("candidate_transaction")
        if not expected_tx:
            print(f"[{logical_id}] SKIP: no pending transaction from Stage A")
            continue
        current_evidence = _model_evidence_sha(logical_id)
        if not current_evidence or not _validate_candidate_transaction(
            expected_tx,
            logical_id=logical_id,
            slot=inactive_slot,
            candidate_sha=candidate_full_sha,
            source_evidence_sha=current_evidence,
        ):
            print(f"[{logical_id}] REJECT_STALE_TRANSACTION: pending tx mismatch "
                  f"(expected slot={expected_tx.get('slot')} "
                  f"sha={expected_tx.get('candidate_sha', '')[:16]}..., "
                  f"actual slot={inactive_slot} sha={candidate_full_sha[:16]}...)")
            model.pop("candidate_transaction", None)
            any_rejected = True
            continue

        if active_watermark == candidate_watermark:
            print(f"[{logical_id}] PASS_NO_CHANGE: candidate identical to active")
            model["last_candidate_verdict"] = "PASS_NO_CHANGE"
            model["last_candidate_evidence_sha"] = current_evidence
            model.pop("candidate_transaction", None)
            model.pop("candidate_content_sha", None)
            model.pop("candidate_slot", None)
            model.pop("candidate_physical_id", None)
            continue

        print(f"[{logical_id}] Adjudicating candidate {inactive_physical_id} "
              f"({len(candidate_content)} chars, tx_id={expected_tx.get('tx_id','?')[:8]}...)")

        # LLM-based adjudication
        system_prompt = _adjudication_system_prompt()
        generation_spec = _model_generation_spec(logical_id)
        generation_contract = json.dumps(
            {
                "required_anchors": generation_spec.get("required_anchors", []),
                "required_terminal_marker": generation_spec.get(
                    "required_terminal_marker", ""
                ),
                "source_query": generation_spec.get("source_query", ""),
            },
            ensure_ascii=False,
            indent=2,
        )

        user_prompt = f"""## Declared Candidate Generation Contract:
{generation_contract}

## Current D Decisions:
{d_decisions}

## Active Mental Model Content (slot {active_slot}):
{active_content[:3000]}

## Candidate Mental Model Content (slot {inactive_slot}, refreshed {candidate_refreshed}):
{candidate_content}

Evaluate the candidate. Is it safe to publish as the new active version?"""

        llm_response = _call_hindsight_llm(
            system_prompt,
            user_prompt,
            max_tokens=4000,
            response_format=True,
        )

        if not llm_response:
            print(f"[{logical_id}] BLOCK_INFRA: LLM call failed, keeping active slot {active_slot}")
            continue

        # Parse LLM response
        verdict = "REJECT"
        conflicts = []
        quality_score = 0
        result, parse_error = _parse_adjudication_response(llm_response)
        if result is None:
            print(
                f"[{logical_id}] WARNING: invalid structured adjudication "
                f"({parse_error}), defaulting to REJECT"
            )
        else:
            verdict = result["verdict"]
            raw_conflicts = result["conflicts"]
            # Normalize conflicts: LLM may return strings or objects.
            for conflict in raw_conflicts:
                if isinstance(conflict, str):
                    conflicts.append(
                        {
                            "d_id": "",
                            "candidate_claim": conflict,
                            "severity": "medium",
                        }
                    )
                elif isinstance(conflict, dict):
                    conflicts.append(conflict)
            quality_score = result["quality_score"]

        print(f"[{logical_id}] Verdict: {verdict} (quality={quality_score})")

        if conflicts:
            print(f"  Conflicts: {len(conflicts)} found")
            for c in conflicts[:3]:
                print(f"    {c.get('d_id','?')}: {c.get('candidate_claim','')[:80]}")

        # A candidate cannot mutate accepted state until it passes the smoke
        # gate in the exact treatment context.
        if verdict == "PASS_PUBLISH":
            candidate_context = f"# {logical_id} (candidate)\n{candidate_content}"
            if _run_smoke_regression(
                api_url,
                candidate_context,
                logical_id=logical_id,
            ) != 0:
                verdict = "REJECT_SMOKE"
                print(f"[{logical_id}] REJECT_SMOKE: candidate failed publication gate")

        # Execute verdict
        if verdict == "PASS_PUBLISH":
            # Atomic switch: flip active slot and record exact content SHA
            # Only Stage B may mutate accepted active state
            inactive_slot_val = model.get("candidate_slot", inactive_slot)
            model["active_slot"] = inactive_slot_val
            model["active_content_sha"] = hashlib.sha256(candidate_content.encode()).hexdigest()
            model["content_sha"] = model["active_content_sha"]  # backward compat
            model["source_watermark"] = current_evidence
            model["last_candidate_evidence_sha"] = current_evidence
            model["accepted_revision"] = {
                "slot": inactive_slot_val,
                "content_sha": model["active_content_sha"],
                "source_evidence_sha": current_evidence,
                "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            model["last_verdict"] = "PASS_PUBLISH"
            model["last_candidate_verdict"] = "PASS_PUBLISH"
            model["verdict_detail"] = f"quality={quality_score}"
            model["last_refresh"] = candidate_refreshed
            # Clean up candidate fields and pending transaction
            model.pop("candidate_content_sha", None)
            model.pop("candidate_slot", None)
            model.pop("candidate_physical_id", None)
            model.pop("candidate_watermark", None)
            model.pop("candidate_transaction", None)
            any_published = True
            print(f"  PUBLISHED: active slot -> {inactive_slot_val}, content_sha={model['active_content_sha'][:16]}")

        elif verdict == "ESCALATE_D_REVIEW":
            # Write conflict to conflicts.jsonl, keep current active
            model["last_verdict"] = "ESCALATE_D_REVIEW"
            model["last_candidate_verdict"] = "ESCALATE_D_REVIEW"
            model["last_candidate_evidence_sha"] = current_evidence
            model["verdict_detail"] = f"quality={quality_score}, {len(conflicts)} conflicts"
            # Clean up candidate fields and pending transaction
            model.pop("candidate_content_sha", None)
            model.pop("candidate_slot", None)
            model.pop("candidate_physical_id", None)
            model.pop("candidate_watermark", None)
            model.pop("candidate_transaction", None)
            any_escalated = True

            # Append conflicts to ledger
            with open(conflicts_path, "a") as f:
                for c in conflicts:
                    conflict_entry = {
                        "date": today,
                        "logical_model": logical_id,
                        "candidate_physical_id": inactive_physical_id,
                        "d_id": c.get("d_id", ""),
                        "candidate_claim": c.get("candidate_claim", ""),
                        "d_text": c.get("d_text", ""),
                        "severity": c.get("severity", "medium"),
                        "status": "pending",
                        "suggested_action": "ESCALATE_D_REVIEW",
                    }
                    f.write(json.dumps(conflict_entry, ensure_ascii=False) + "\n")
            print(f"  ESCALATED: {len(conflicts)} conflicts written to ledger")

        else:  # REJECT or unknown
            model["last_verdict"] = "REJECT"
            model["last_candidate_verdict"] = verdict
            model["last_candidate_evidence_sha"] = current_evidence
            model["verdict_detail"] = f"quality={quality_score}"
            # Clean up candidate fields
            model.pop("candidate_content_sha", None)
            model.pop("candidate_slot", None)
            model.pop("candidate_physical_id", None)
            model.pop("candidate_watermark", None)
            model.pop("candidate_transaction", None)
            any_rejected = True
            print(f"  REJECTED: keeping active slot {active_slot}")

    # Atomic write registry
    registry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp_path = registry_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    os.rename(str(tmp_path), str(registry_path))

    # Write adjudication report
    report_lines = [
        f"# Mental Model Adjudication Report ({today})",
        "",
        f"## Summary",
        f"- Published: {'YES' if any_published else 'NO'}",
        f"- Rejected: {'YES' if any_rejected else 'NO'}",
        f"- Escalated: {'YES' if any_escalated else 'NO'}",
        "",
    ]
    report_file = reports_dir / f"{today}-adjudication.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport: {report_file}")

    # Export accepted models using revision-based policy (not date-based).
    # Only PASS_PUBLISH models get current/ consumer view and history snapshots.
    # INITIAL/REJECT/ESCALATE content stays quarantined.
    export_dir = Path("/home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/exports")
    current_dir = export_dir / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    for logical_id, model in registry.get("models", {}).items():
        accepted = _accepted_revision(model)
        if accepted is None:
            print(f"  SKIP export {logical_id}: no accepted revision")
            continue
        content_sha = accepted["content_sha"]
        active_slot = model["active_slot"]
        if accepted["slot"] != active_slot:
            print(f"  SKIP export {logical_id}: accepted slot mismatch")
            continue
        physical_id = model["physical_ids"][active_slot]

        current_evidence = _model_evidence_sha(logical_id)
        if not current_evidence or accepted["source_evidence_sha"] != current_evidence:
            print(f"  SKIP export {logical_id}: stale evidence")
            continue

        try:
            with urllib.request.urlopen(
                f"{api_url}/v1/default/banks/hermes/mental-models/{physical_id}",
                timeout=30,
            ) as resp:
                export_data = json.loads(resp.read())
            content = export_data.get("content", "")
            if not content or len(content) < 100:
                print(f"  SKIP export {logical_id}: empty or too-short content")
                continue
            if hashlib.sha256(content.encode()).hexdigest() != content_sha:
                print(f"  SKIP export {logical_id}: fetched content SHA mismatch")
                continue

            metadata_header = (
                f"<!--\n"
                f"  logical_id: {logical_id}\n"
                f"  accepted_revision: {content_sha}\n"
                f"  evidence_bundle_sha: {current_evidence}\n"
                f"  active_slot: {active_slot}\n"
                f"  verdict: PASS_PUBLISH\n"
                f"  accepted_at: {accepted['accepted_at']}\n"
                f"  evidence_stale: false\n"
                f"-->\n\n"
            )

            # Visible last-updated timestamp (ISO → readable)
            _ts_raw = accepted.get("accepted_at", "")
            try:
                from datetime import datetime as _dt
                _parsed = _dt.fromisoformat(_ts_raw.replace("Z", "+00:00"))
                _ts_display = _parsed.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                _ts_display = _ts_raw
            updated_line = f"> **最后更新**: {_ts_display}\n\n"

            # Atomic current/ consumer view
            current_file = current_dir / f"{logical_id}.md"
            tmp_current = current_dir / f".{logical_id}.tmp"
            with open(tmp_current, "w") as f:
                f.write(metadata_header + updated_line + content)
            os.rename(str(tmp_current), str(current_file))
            print(f"  Current: {current_file.name} ({len(content)} chars)")

        except Exception as e:
            print(f"  Export failed for {logical_id}: {e}")

    # Export through the canonical Pitfall writer; this script never writes
    # pitfall index/catalog bytes directly.
    pitfall_index_path = HERMES_HOME / "mental-models" / "egomotion4d" / "pitfall_index.json"
    if pitfall_index_path.exists():
        writer_dir = HERMES_HOME / "mental-models" / "egomotion4d"
        if str(writer_dir) not in sys.path:
            sys.path.insert(0, str(writer_dir))
        from pitfall_writer import PitfallIndex

        pidx = PitfallIndex.load(str(pitfall_index_path))
        catalog_file = export_dir.parent / "pitfall-catalog.md"
        pidx.export_catalog(catalog_file, generated_date=today)
        print(f"  Exported: pitfall-catalog.md ({pidx.canonical_count} current)")

    index_file = export_dir.parent / "README.md"
    tmp_index = index_file.with_suffix(".tmp")
    tmp_index.write_text(
        _render_mental_model_index(registry, today), encoding="utf-8"
    )
    os.replace(tmp_index, index_file)
    print("  Updated: mental-model README.md")

    if any_escalated:
        return 3
    elif any_rejected and not any_published:
        return 2
    return 0



def _log_role_trigger(role: str, stage: str, **metadata) -> None:
    """Log role-trigger telemetry to a structured JSONL file.

    Records which role triggered which pipeline stage at what time.
    Used for monitoring reviewer/planner/implementer/critic/coordinator
    activity patterns.
    """
    import json as _json
    telemetry_dir = LOG_DIR / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "role": role,
        "stage": stage,
        "pid": os.getpid(),
        **metadata,
    }
    log_file = telemetry_dir / f"role-triggers-{time.strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(_json.dumps(record, ensure_ascii=False) + "\n")


def _record_mental_model_feedback(
    outcome: str,
    *,
    logical_id: str,
    task_id: str,
    note: str = "",
    role: str | None = None,
) -> None:
    valid = {"helpful", "neutral", "harmful", "avoided_repeat"}
    if outcome not in valid:
        raise ValueError(f"invalid mental-model feedback outcome: {outcome}")
    _log_role_trigger(
        role or _current_role(),
        "mental_model_feedback",
        outcome=outcome,
        logical_id=logical_id,
        task_id=task_id,
        note=note[:500],
    )


def _mental_model_metrics(days: int = 28) -> dict:
    """Aggregate local consumption and outcome telemetry over a bounded window."""
    telemetry_dir = LOG_DIR / "telemetry"
    cutoff = datetime.now().astimezone() - timedelta(days=days)
    consumptions: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    context_tokens = 0
    for path in sorted(telemetry_dir.glob("role-triggers-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                timestamp = datetime.fromisoformat(record["timestamp"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if timestamp < cutoff:
                continue
            if record.get("stage") == "mental_model_consumed":
                role = record.get("role", "unknown")
                consumptions[role] = consumptions.get(role, 0) + 1
                context_tokens += int(record.get("context_tokens", 0))
            elif record.get("stage") == "mental_model_feedback":
                outcome = record.get("outcome", "unknown")
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
    feedback_total = sum(outcomes.values())
    return {
        "window_days": days,
        "total_consumptions": sum(consumptions.values()),
        "consumption_by_role": consumptions,
        "context_tokens": context_tokens,
        "feedback_by_outcome": outcomes,
        "avoided_repeat_rate": (
            outcomes.get("avoided_repeat", 0) / feedback_total if feedback_total else None
        ),
        "harmful_rate": (
            outcomes.get("harmful", 0) / feedback_total if feedback_total else None
        ),
    }


def _run_mental_model_daily(run_stage_a, run_adjudicate, run_smoke) -> int:
    """Compose the daily branches while always monitoring accepted content."""
    stage_a_rc = run_stage_a()
    if stage_a_rc == 1:
        return 1
    if stage_a_rc == 2:
        print("Stage A: no stale models, skipping Stage B")
        return run_smoke()

    print("\n--- Stage A complete, starting Stage B ---\n")
    adjudicate_rc = run_adjudicate()
    print("\n--- Smoke Regression ---\n")
    smoke_rc = run_smoke()
    if smoke_rc != 0:
        print("SMOKE REGRESSION FAILED: blocking maintenance", flush=True)
        return smoke_rc
    return adjudicate_rc



def main() -> int:
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(description="Hindsight offline pipeline wrapper")
    _ap.add_argument("--mode", choices=["daily", "full"], default="daily",
                     help="Pipeline mode: daily (default) or full (daily+weekly+wiki)")
    _ap.add_argument("--include-wiki", action="store_true",
                     help="For full mode, include wiki maintenance")
    _ap.add_argument("--skip-daily", action="store_true",
                     help="For full mode, skip daily steps (retain/daily_reflect)")
    _ap.add_argument("--mental-model-preflight", metavar="LOGICAL_ID", nargs="?", const="egomotion4d-research-guardrails",
                     help="Output active mental model content for agent preflight; does not run pipeline")
    _ap.add_argument("--mental-model-maintain", action="store_true",
                     help="Run Stage A daily maintenance (refresh inactive slots, generate report); does not run pipeline")
    _ap.add_argument("--mental-model-adjudicate", action="store_true",
                     help="Run Stage B adjudication (LLM quality check, conflict resolution, atomic publish)")
    _ap.add_argument("--mental-model-daily", action="store_true",
                     help="Run full daily cycle: Stage A (collector) + Stage B (adjudication)")
    _ap.add_argument("--mental-model-pitfalls", metavar="KEYWORDS", nargs="?", const="",
                     help="Search pitfall index by keywords (comma-separated). No args = list all current.")
    _ap.add_argument("--all-pitfalls", action="store_true",
                     help="With --mental-model-pitfalls: include superseded/rejected entries")
    _ap.add_argument("--mental-model-feedback",
                     choices=["helpful", "neutral", "harmful", "avoided_repeat"],
                     help="Record task-level outcome feedback for a consumed model")
    _ap.add_argument("--logical-id")
    _ap.add_argument("--task-id", default="")
    _ap.add_argument("--feedback-note", default="")
    _ap.add_argument("--mental-model-metrics", type=int, metavar="DAYS",
                     help="Print bounded consumption/outcome telemetry as JSON")
    _ap.add_argument("--research-digest-only", metavar="YYYY-MM-DD",
                     help="Generate one compact research digest without running the pipeline")
    _args, _ = _ap.parse_known_args()

    # Mental model preflight: output and exit, skip pipeline entirely
    if _args.mental_model_preflight is not None:
        return mental_model_preflight(_args.mental_model_preflight)

    # Pitfall lookup
    if _args.mental_model_pitfalls is not None:
        kw = _args.mental_model_pitfalls if _args.mental_model_pitfalls else None
        sf = "all" if _args.all_pitfalls else "current"
        return pitfall_lookup(kw, sf)

    if _args.mental_model_feedback:
        if not _args.task_id:
            _ap.error("--mental-model-feedback requires --task-id")
        _record_mental_model_feedback(
            _args.mental_model_feedback,
            logical_id=_args.logical_id or "egomotion4d-research-guardrails",
            task_id=_args.task_id,
            note=_args.feedback_note,
        )
        return 0

    if _args.mental_model_metrics is not None:
        print(json.dumps(_mental_model_metrics(_args.mental_model_metrics), indent=2))
        return 0

    if _args.research_digest_only:
        return run_research_summary(
            0, None, None, report_date=_args.research_digest_only
        )

    # Mental model daily maintenance: Stage A collector
    if _args.mental_model_maintain:
        return mental_model_maintain(_args.logical_id)

    # Mental model adjudication: Stage B
    if _args.mental_model_adjudicate:
        return mental_model_adjudicate(_args.logical_id)

    # Mental model full daily cycle: Stage A + Pitfall Discovery + Stage B + Smoke
    if _args.mental_model_daily:
        api_url = os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888")
        return _run_mental_model_daily(
            mental_model_maintain,
            mental_model_adjudicate,
            lambda: _run_all_model_smoke(api_url),
        )
    pipeline_mode = _args.mode
    include_wiki = _args.include_wiki
    skip_daily = _args.skip_daily

    start = time.time()
    print(f"[Hindsight {'Full' if pipeline_mode == 'full' else 'Daily'} Pipeline]", flush=True)

    if not PIPELINE_SCRIPT.exists():
        print(f"ERROR: script not found: {PIPELINE_SCRIPT}")
        return 1

    # Pre-step: auto-clean orphaned consolidation units before pipeline runs.
    # Hindsight's consolidation can leave memory_units with consolidation_failed_at
    # set but no actual pending work, causing pending_consolidation to never drop
    # and blocking the wait_native_consolidation gate forever.
    print("  Pre-clean orphaned consolidation units...", flush=True)
    try:
        fix_script = HERMES_HOME / "scripts" / "fix_orphaned_consolidation.py"
        if fix_script.exists():
            import subprocess as _sp
            fix = _sp.run(
                [sys.executable, str(fix_script), "--bypass"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "HOME": str(REAL_HOME)},
            )
            if fix.returncode == 0:
                print(f"  Clean OK: {fix.stdout.strip()}", flush=True)
            else:
                print(f"  Clean failed (non-fatal): {fix.stderr[:200]}", flush=True)
        else:
            print(f"  Clean script not found (skip)", flush=True)
    except Exception as e:
        print(f"  Pre-clean error (non-fatal): {e}", flush=True)

    # Health check
    if not check_hindsight():
        print("ERROR: Hindsight not healthy, aborting")
        return 1

    # Snapshot before
    before = get_summary()

    # Run pipeline — full output to log, no timeout (cron scheduler handles timeout)
    with open(LOG_FILE, "w", encoding="utf-8") as log_fh:
        env = {
            "HOME": str(REAL_HOME),
            "HERMES_HOME": str(HERMES_HOME),
            "HINDSIGHT_SESSION_PROFILE_MODE": "all",
            "HERMES_ACCEPT_HOOKS": "1",
            "PYTHONUNBUFFERED": "1",
        }
        proc = subprocess.run(
            [
                sys.executable, "-u",
                str(PIPELINE_SCRIPT),
                pipeline_mode,
                "--execute",
                "--confirm",
                "run-hindsight-pipeline",
            ] + (["--include-wiki"] if include_wiki else []) + (["--skip-daily"] if skip_daily else []),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env={**subprocess.os.environ, **env},
        )

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)

    # Snapshot after
    after = get_summary()

    print(f"Exit code: {proc.returncode}", flush=True)
    print(f"Elapsed: {mins}m{secs}s", flush=True)

    if before and after:
        d_doc = after.get("total_documents", 0) - before.get("total_documents", 0)
        d_obs = after.get("total_observations", 0) - before.get("total_observations", 0)
        d_node = after.get("total_nodes", 0) - before.get("total_nodes", 0)
        print(f"Changes: docs={d_doc:+d} obs={d_obs:+d} nodes={d_node:+d}")
        pend = after.get("pending_operations", 0)
        proc_count = after.get("operations_by_status", {}).get("processing", 0)
        if proc_count > 0 or pend > 0:
            print(f"Hindsight still processing: {proc_count} processing, {pend} pending")

    if proc.returncode == 143:
        print("Pipeline timed out (SIGTERM) but Hindsight internal work continues")
    elif proc.returncode != 0:
        print(f"Pipeline exited with code {proc.returncode}")
        print(f"Check log: {LOG_FILE}")
    else:
        print("Pipeline completed successfully")

    print(f"Log: {LOG_FILE}")

    # --- Phase 2: Research summary (always runs, even if pipeline had errors) ---
    # Pipeline data may still be in Hindsight even if exit code != 0.
    summary_rc = run_research_summary(proc.returncode, before, after)
    if summary_rc != 0:
        print(f"Research summary failed with code {summary_rc} (pipeline data OK)")

    # Generate Hindsight status report for wiki
    try:
        import subprocess as _sp
        report = _sp.run(
            [sys.executable, str(REAL_HOME / ".hermes" / "scripts" / "generate_hindsight_status_report.py")],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "HOME": str(REAL_HOME)},
        )
        if report.returncode == 0:
            print(f"HINDSIGHT_STATUS_REPORT {report.stdout.strip()}")
        else:
            print(f"Status report generation failed: {report.stderr[:200]}")
    except Exception as e:
        print(f"Status report generation skipped: {e}")

    # Return pipeline exit code (not summary exit code) as primary status
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
