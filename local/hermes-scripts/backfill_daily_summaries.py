#!/usr/bin/env python3
"""
Backfill daily research summaries from Hindsight memory + experiment results.
Usage: python3 backfill_daily_summaries.py [--start DATE] [--end DATE] [--dry-run]
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

# Config
HINDSIGHT_API = "http://localhost:8888/v1/default/banks/hermes"
WIKI_DAILY_DIR = Path("/home/wyr/wiki/auto-maintenance/daily")
GPUSERVER_PULLBACK = Path("/home/wyr/code/Egomotion4D/gpuserver_pullback")

# Get Hindsight LLM config from Docker container
def get_hindsight_llm_config():
    result = subprocess.run(
        ["docker", "exec", "hindsight", "printenv"],
        capture_output=True, text=True
    )
    env = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v

    base_url = env.get("HINDSIGHT_API_LLM_BASE_URL", "").rstrip("/")
    model = env.get("HINDSIGHT_API_LLM_MODEL", "deepseek-v4-flash")
    api_key = env.get("HINDSIGHT_API_LLM_API_KEY", "")

    if not base_url or not api_key:
        raise RuntimeError("HINDSIGHT_API_LLM_* not found in container env")

    return {"base_url": base_url, "model": model, "api_key": api_key}


def recall_memories(date_str, k=50):
    """Recall memories for a specific date via direct PG query (faster than API)."""
    import psycopg2

    try:
        conn = psycopg2.connect(
            host="127.0.0.1", port=5432, dbname="hindsight", user="hindsight"
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT f.text
            FROM memory_units f
            JOIN documents d ON f.document_id = d.id
            WHERE d.bank_id = %s
              AND DATE(d.created_at) = %s
              AND f.text IS NOT NULL
            LIMIT %s
        """, ("hermes", date_str, k))
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception as e:
        print(f"  PG query error: {e}", file=sys.stderr)
        return []


def get_experiments(date_str):
    """Get experiment results for a specific date."""
    exps = []
    for d in GPUSERVER_PULLBACK.glob(f"{date_str}*"):
        if not d.is_dir():
            continue
        summary_file = d / "run" / "summary.md"
        metrics_file = d / "run" / "metrics.json"

        exp = {"name": d.name, "path": str(d)}
        if summary_file.exists():
            exp["summary"] = summary_file.read_text(encoding="utf-8")[:2000]
        if metrics_file.exists():
            try:
                exp["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
            except:
                pass
        exps.append(exp)
    return exps


def call_llm(config, date_str, memories, experiments):
    """Call Hindsight LLM to generate daily summary."""

    # Filter memories to only keep Egomotion4D/algorithm relevant ones
    keywords = ["egomotion4d", "egomotion", "pose", "depth", "reconstruct", "fusion",
                "gtsam", "ba", "track", "dlt", "tsdf", "mesh", "scene", "metric",
                "scale", "can", "frontend", "roma", "dage", "vggt", "pi3x",
                "unidepth", "experiment", "pipeline", "phase", "evaluator",
                "sparse", "surface", "ground", "occ", "actor", "undistort",
                "camera", "intrinsic", "gauge", "coordinate"]

    relevant = []
    for m in memories:
        m_lower = m.lower()
        if any(kw in m_lower for kw in keywords):
            relevant.append(m)

    mem_block = "\n\n".join(relevant[:20]) if relevant else "（无Egomotion4D相关记忆）"

    exp_block = ""
    if experiments:
        for e in experiments:
            exp_block += f"\n### {e['name']}\n"
            if "summary" in e:
                exp_block += e["summary"][:500] + "\n"
            if "metrics" in e:
                m = e["metrics"]
                exp_block += f"Metrics: ATE={m.get('ATE_mean','?')}, RPE={m.get('RPE_mean','?')}\n"
    else:
        exp_block = "（无gpuserver实验结果）"

    prompt = f"""你是算法研发助手。根据以下信息生成 {date_str} 的每日研发总结。

## Hindsight 记忆（{len(memories)}条）
{mem_block}

## 实验结果（{len(experiments)}个）
{exp_block}

## 输出要求
1. **工作主题**：一句话概括当天主线
2. **关键进展**：2-4条，每条≤50字，含数据/结论
3. **方向转折**：如有，说明原因
4. **算法坑**：如有，按以下JSON格式输出（可多条）：
```json
{{
  "algorithm_pitfalls": [
    {{"title": "坑标题", "detail": "坑的描述", "lesson": "实际教训"}}
  ]
}}
```

只记录算法级坑（影响决策/方向），不记运维/工程问题。
表述精简，关键数据不省略。
"""

    try:
        resp = requests.post(
            f"{config['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            },
            json={
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.3
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-10", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-06-01", help="End date (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"Backfilling summaries from {args.start} to {args.end}")

    # Get LLM config once
    try:
        llm_config = get_hindsight_llm_config()
        print(f"LLM: {llm_config['model']} @ {llm_config['base_url']}")
    except Exception as e:
        print(f"Failed to get LLM config: {e}")
        sys.exit(1)

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        summary_file = WIKI_DAILY_DIR / f"{date_str}_summary.md"

        if summary_file.exists():
            print(f"{date_str}: already exists, skip")
            current += timedelta(days=1)
            continue

        print(f"\n{date_str}:")

        # Recall memories
        memories = recall_memories(date_str)
        print(f"  Memories: {len(memories)}")

        # Get experiments
        experiments = get_experiments(date_str)
        print(f"  Experiments: {len(experiments)}")

        if not memories and not experiments:
            print(f"  No data, skip")
            current += timedelta(days=1)
            continue

        if args.dry_run:
            print(f"  Would generate summary")
            current += timedelta(days=1)
            continue

        # Call LLM
        result = call_llm(llm_config, date_str, memories, experiments)
        if not result:
            print(f"  Failed to generate summary")
            current += timedelta(days=1)
            continue

        # Write summary
        summary_file.write_text(result, encoding="utf-8")
        print(f"  Written: {summary_file}")

        current += timedelta(days=1)

    print("\nDone.")


if __name__ == "__main__":
    main()
