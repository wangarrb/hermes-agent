#!/usr/bin/env python3
"""
No-agent wrapper for hindsight daily pipeline.
Uses hindsight_memory_pipeline.py (the modern session/json path).
Full output -> log file; concise summary -> stdout (-> weixin delivery).

After pipeline completes, triggers an agent analysis for daily research summary.
"""

import json
import os
import pwd
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

# Research summary output
DAILY_DIR = Path("/home/wyr/wiki/auto-maintenance/daily")
DAILY_DIR.mkdir(parents=True, exist_ok=True)
TODAY = time.strftime("%Y-%m-%d")
# Cron runs at 00:01 — "today" has almost no data yet. Report on YESTERDAY.
from datetime import date, timedelta
REPORT_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
SUMMARY_FILE = DAILY_DIR / f"{REPORT_DATE}_summary.md"


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
            base_url = env_map.get("HINDSIGHT_API_LLM_BASE_URL")
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
        key_env = os.environ.get("HINDSIGHT_OFFLINE_LLM_API_KEY_ENV", "OPENCODE_GO_API_KEY")
        api_key = os.environ.get(key_env)

    # 3. Final defaults (opencode-go is the working provider; tp-api returns 401)
    return {
        "base_url": base_url or "https://opencode.ai/zen/go/v1",
        "model": model or "deepseek-v4-flash",
        "api_key": api_key,
        "provider": provider or "openai",
    }


def _call_hindsight_llm(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str | None:
    """Call Hindsight's current LLM (same model Hindsight uses for retain/consolidate)."""
    import requests as _requests

    cfg = _get_hindsight_llm_config()

    if not cfg["api_key"]:
        print("ERROR: cannot determine Hindsight LLM API key", flush=True)
        return None

    try:
        resp = _requests.post(
            f"{cfg['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"topenrouter call failed: {e}", flush=True)
        return None


def _collect_today_context(today: str) -> str:
    """Collect today's experiment results, git log, and Hindsight memory as context for LLM."""
    parts = []

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
                    "http://127.0.0.1:8888/v1/default/banks/hermes/recall",
                    json={"query": query, "limit": 10},
                    timeout=15,
                )
                if resp.status_code == 200:
                    items = resp.json().get("results", [])
                    # Filter to today's items only
                    today_items = [
                        item for item in items
                        if today in (item.get("created_at", "") or item.get("updated_at", ""))[:10]
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

    return "\n".join(parts) if parts else "(no experiment data found)"


def run_research_summary(pipeline_exit_code: int, before_stats: dict | None, after_stats: dict | None) -> int:
    """Generate daily research summary using topenrouter (deepseek-v4-flash).

    Collects experiment results + git log, sends to LLM for analysis,
    writes output to /home/wyr/wiki/auto-maintenance/daily/YYYY-MM-DD_summary.md,
    and appends algorithm-level pitfalls to ERRORS.md.
    """
    print("\n[Research Summary]", flush=True)

    today = time.strftime("%Y-%m-%d")
    # Report on YESTERDAY (cron runs at 00:01, "today" has no data yet)
    report_date = REPORT_DATE

    # Collect context
    print("  Collecting experiment data...", flush=True)
    experiment_context = _collect_today_context(report_date)

    # Build pipeline context
    pipeline_info = f"Pipeline exit code: {pipeline_exit_code}"
    if before_stats and after_stats:
        d_doc = after_stats.get("total_documents", 0) - before_stats.get("total_documents", 0)
        d_obs = after_stats.get("total_observations", 0) - before_stats.get("total_observations", 0)
        pipeline_info += f" | Hindsight: docs={d_doc:+d} obs={d_obs:+d}"

    system_prompt = """你是一个自动驾驶算法研发的日报助手。根据提供的实验数据、git提交记录，生成详细的研发日报。

输出格式（严格遵守，不要加多余内容）：

# YYYY-MM-DD 研发日报

## 主线
（1-2句话概括当天主线方向）

## 工作详述
（按任务/阶段拆分子章节，每个子章节写清：背景、做了什么、关键结果、结论。
不要只写一句话，要展开到让没参与的人能理解当天做了什么、为什么做、结果如何。）

## 关键转折
（方向变化、证伪的假设、关闭的路线，每条1-2句话。无则写"无"）

## 关键决策
（表格：决策 | 内容 | 原因。记录当天做出的重要技术决策及其理由。
包括评审裁决、路线切换、方案选择等。无则写"无"）

## 核心数据
（关键指标简表，行数不限但每行精简。覆盖所有重要实验的指标，
不要为了省行数而遗漏关键数据。多场景/多配置分别列出。）

## 方向
（1-2句话说明下一步方向）

## 算法级坑
（只记录影响算法决策/方向的坑：错误判断、被证伪的假设、系统性偏差、metric定义偏移。
不记录：编译错误、语法错误、pip install失败、Hindsight运维、网络超时、SSH断连、文件权限等低级工程问题。
每条坑：1句话描述+1句话教训。没有算法级坑就写"无"）"""

    user_prompt = f"""日期：{report_date}
{pipeline_info}

## 实验数据与提交记录

{experiment_context}

## 要求
1. 内容要详细充实，工作详述按任务/阶段展开子章节，不能只有一两句话
2. 必须包含"关键决策"section（表格：决策|内容|原因），记录评审裁决、路线切换、方案选择
3. 核心数据行数不限，关键指标都要列全，多场景/多配置分别列出
4. 方向转换/路线切换必须写清转折原因
5. 只记录算法级坑，不记录工程/运维问题
6. 如果有算法级坑，在末尾单独输出一个 JSON 块，格式如下（没有坑就不输出）：
```json
{{"algorithm_pitfalls": [{{"title": "坑标题", "detail": "1句话描述", "lesson": "1句话教训"}}]}}
```"""

    print("  Calling Hindsight LLM...", flush=True)
    result = _call_hindsight_llm(system_prompt, user_prompt, max_tokens=8192)

    if not result:
        print("  LLM call failed, skipping summary", flush=True)
        return 1

    # Write summary file
    try:
        SUMMARY_FILE.write_text(result, encoding="utf-8")
        print(f"  Summary written: {SUMMARY_FILE}", flush=True)
    except Exception as e:
        print(f"  Failed to write summary: {e}", flush=True)
        return 2

# Extract algorithm pitfalls and append to wiki pitfalls file
    import re
    pitfall_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
    if pitfall_match:
        try:
            pitfall_data = json.loads(pitfall_match.group(1))
            pitfalls = pitfall_data.get("algorithm_pitfalls", [])
            if pitfalls:
                pitfalls_file = Path("/home/wyr/wiki/auto-maintenance/project/egomotion4d/pitfalls.md")
                # Read existing file to find next P-number
                existing = pitfalls_file.read_text(encoding="utf-8")
                next_num = 1
                import re as num_re
                nums = num_re.findall(r'## P(\d+):', existing)
                if nums:
                    next_num = max(int(n) for n in nums) + 1

                # Tag vocabulary for auto-classification
                TAG_DOMAINS = {
                    "深度融合": "#深度融合", "depth fusion": "#深度融合",
                    "深度估计": "#深度估计", "depth estimation": "#深度估计",
                    "位姿估计": "#位姿估计", "pose": "#位姿估计", "CAN": "#位姿估计", "gauge": "#位姿估计",
                    "重建": "#重建", "TSDF": "#重建", "mesh": "#重建", "surface": "#重建", "BEV": "#重建",
                    "占用": "#占用", "Occ": "#占用",
                    "评估": "#评估", "metric": "#评估", "evaluator": "#评估",
                    "数据": "#数据",
                    "对应": "#对应", "matching": "#对应", "DLT": "#对应", "RoMa": "#对应",
                    "尺度": "#尺度", "scale": "#尺度", "metric depth": "#尺度",
                    "通用": "#通用",
                }
                TAG_ERRORS = {
                    "选型错误": "#选型错误", "wrong method": "#选型错误",
                    "参数错误": "#参数错误", "wrong parameter": "#参数错误",
                    "假设错误": "#假设错误", "wrong assumption": "#假设错误", "证伪": "#假设错误",
                    "重复造轮子": "#重复造轮子",
                    "用错API": "#用错API",
                    "忽略条件": "#忽略条件", "ignored condition": "#忽略条件",
                    "过度泛化": "#过度泛化",
                }

                entries = []
                for p in pitfalls:
                    title = p.get("title", "")
                    detail = p.get("detail", "")
                    lesson = p.get("lesson", "")
                    # Auto-classify tags from title + detail
                    text = f"{title} {detail}".lower()
                    domain_tags = []
                    error_tags = []
                    for kw, tag in TAG_DOMAINS.items():
                        if kw.lower() in text:
                            domain_tags.append(tag)
                    for kw, tag in TAG_ERRORS.items():
                        if kw.lower() in text:
                            error_tags.append(tag)
                    all_tags = domain_tags + error_tags if domain_tags or error_tags else ["#通用", "#假设错误"]

                    entry = f"""
---

## P{next_num}: {title}

**坑**: {detail}
**实际**: {lesson}
**教训**: {lesson}
**同类场景**: （LLM提取，待人工补充）
**标签**: {' '.join(all_tags)}
**日期**: {report_date}
**来源**: daily-research-summary pipeline"""
                    next_num += 1
                    entries.append(entry)

                # Insert before 备选 section
                marker = "## 备选（待确认）"
                if marker in existing:
                    new_content = "\n".join(entries) + "\n\n"
                    updated = existing.replace(marker, new_content + marker)
                else:
                    updated = existing + "\n".join(entries)

                pitfalls_file.write_text(updated, encoding="utf-8")
                print(f"  Appended {len(pitfalls)} pitfalls (P{next_num - len(pitfalls)}-P{next_num - 1}) to wiki/pitfalls.md", flush=True)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Failed to parse pitfalls JSON: {e}", flush=True)

    # Print concise output for weixin delivery
    print("\n--- Research Summary ---", flush=True)
    # Remove the JSON block from printed output
    clean = re.sub(r'\n*```json.*?```\n*', '', result, flags=re.DOTALL).strip()
    for line in clean.split("\n")[:25]:
        print(line, flush=True)

    return 0


def main() -> int:
    start = time.time()
    print(f"[Hindsight Daily Pipeline]", flush=True)

    if not PIPELINE_SCRIPT.exists():
        print(f"ERROR: script not found: {PIPELINE_SCRIPT}")
        return 1

    # Health check
    if not check_hindsight():
        print("ERROR: Hindsight not healthy, aborting")
        return 1

    # Snapshot before
    before = get_summary()

    # Run pipeline — full output to log, no timeout
    with open(LOG_FILE, "w", encoding="utf-8") as log_fh:
        env = {
            "HOME": str(REAL_HOME),
            "HERMES_HOME": str(HERMES_HOME),
            "HERMES_ACCEPT_HOOKS": "1",
            "PYTHONUNBUFFERED": "1",
        }
        proc = subprocess.run(
            [
                sys.executable, "-u",
                str(PIPELINE_SCRIPT),
                "daily",
                "--execute",
                "--confirm",
                "run-hindsight-pipeline",
            ],
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
            [sys.executable, str(Path.home() / ".hermes" / "scripts" / "generate_hindsight_status_report.py")],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "HOME": str(Path.home())},
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
