#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PSQL = os.environ.get("HINDSIGHT_PSQL", "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql")
BANK = "hermes"
LOG = Path("/home/wyr/.hermes/logs/hindsight-observations/20260511-monitor-live.jsonl")
LOG.parent.mkdir(parents=True, exist_ok=True)


def psql(sql: str) -> str:
    cmd = [PSQL, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "hindsight", "-q", "-t", "-A", "-F", "\t", "-c", sql]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def status() -> dict:
    main = psql("""
select
  count(*) filter(where fact_type='observation') as observations,
  coalesce(max(created_at) filter(where fact_type='observation')::text,'') as last_observation,
  count(*) filter(where fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null) as unconsolidated_base,
  count(*) filter(where fact_type in ('world','experience') and consolidation_failed_at is not null) as failed_base
from memory_units where bank_id='hermes';
""").strip().split("\t")
    ops_lines = [x.split("\t") for x in psql("""
select status, operation_type, count(*) from async_operations where bank_id='hermes' group by status,operation_type order by status,operation_type;
""").strip().splitlines() if x.strip()]
    offline = psql("""
with offline_docs as (
  select d.id,coalesce(count(m.id),0) units
  from documents d left join memory_units m on m.bank_id=d.bank_id and m.document_id=d.id
  where d.bank_id='hermes' and d.id like 'hermes-offline-consolidation::%'
  group by d.id
) select count(*) total, count(*) filter(where units=0) zero, count(*) filter(where units>0) with_units, coalesce(sum(units),0) units from offline_docs;
""").strip().split("\t")
    return {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "observations": int(main[0] or 0),
        "last_observation": main[1],
        "unconsolidated_base": int(main[2] or 0),
        "failed_base": int(main[3] or 0),
        "ops": {f"{a[0]}:{a[1]}": int(a[2]) for a in ops_lines if len(a) >= 3},
        "offline_docs": {"total": int(offline[0] or 0), "zero": int(offline[1] or 0), "with_units": int(offline[2] or 0), "units": int(offline[3] or 0)},
    }


def load_rows() -> list[dict]:
    rows = []
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows


def parse_ts(ts: str):
    s = str(ts).strip().replace(" ", "T")
    # PostgreSQL may render timezone as +08; Python wants +08:00.
    if re.search(r"[+-]\d{2}$", s):
        s += ":00"
    return datetime.fromisoformat(s)


def rate_lines(rows: list[dict]) -> list[str]:
    out = []
    latest = rows[-1]
    processing = int(latest.get("ops", {}).get("processing:consolidation", 0))
    completed = int(latest.get("ops", {}).get("completed:consolidation", 0))
    last_obs_age = "unknown"
    if latest.get("last_observation"):
        try:
            last_obs_dt = parse_ts(latest["last_observation"])
            sample_dt = parse_ts(latest["ts"])
            age_min = max(0.0, (sample_dt - last_obs_dt).total_seconds() / 60)
            last_obs_age = f"{age_min:.1f}m"
        except Exception:
            pass
    for n in (10, 20, 60, len(rows)):
        sub = rows[-n:]
        if len(sub) < 2:
            continue
        mins = (parse_ts(sub[-1]["ts"]) - parse_ts(sub[0]["ts"])).total_seconds() / 60
        if mins <= 0:
            continue
        du = int(sub[0]["unconsolidated_base"]) - int(sub[-1]["unconsolidated_base"])
        do = int(sub[-1]["observations"]) - int(sub[0]["observations"])
        rate = du / mins
        latest_base = int(sub[-1]["unconsolidated_base"])
        eta = "unknown"
        if rate > 0:
            eta_min = latest_base / rate
            eta = f"{eta_min:.0f} min / {eta_min/60:.1f} h"
        out.append(
            f"window={n:>3} mins={mins:5.1f} uncon_delta={du:4} rate={rate:5.2f}/min obs_delta={do:4} obs_rate={do/mins:5.2f}/min "
            f"processing={processing} completed={completed} last_obs_age={last_obs_age} ETA={eta}"
        )
    return out


def short_error(e: Exception) -> str:
    if isinstance(e, subprocess.CalledProcessError):
        msg = (e.output or "").strip().splitlines()
        text = msg[-1] if msg else repr(e)
        return text[:220]
    return repr(e)[:220]


def docker_liveness_lines() -> list[str]:
    try:
        out = subprocess.check_output(
            ["docker", "logs", "--since", "8m", "hindsight"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except Exception as e:
        return [f"container_log_liveness_error: {short_error(e)}"]
    lines = out.splitlines()
    stats = ""
    task = ""
    events: list[str] = []
    for line in lines:
        if "[WORKER_STATS]" in line and "consolidation" in line:
            stats = line
        if "[WORKER_TASK]" in line and "type=consolidation" in line:
            task = line
        if (
            "slow llm call: scope=consolidation" in line
            or "LLM batch" in line
            or "llm_batch" in line
            or "CONSOLIDATION COMPLETE" in line
            or "Marked async operation" in line
            or "APIConnectionError" in line
        ):
            events.append(line)
    out_lines: list[str] = []
    if task:
        m = re.search(r"op=([0-9a-f-]{36}).*?age=(\d+)s.*?stage=([^ ]+)(?: stage_age=(\d+)s)?", task)
        if m:
            stage_age = f" stage_age={m.group(4)}s" if m.group(4) else ""
            out_lines.append(f"log_liveness: op={m.group(1)} age={m.group(2)}s stage={m.group(3)}{stage_age}")
        else:
            out_lines.append("log_liveness: " + task[-220:])
    if stats:
        m = re.search(r"pool: size=(\d+) limits=([^ ]+) idle=(\d+) in_use=(\d+) waiters=(\d+)", stats)
        if m:
            out_lines.append(f"api_pool: size={m.group(1)} limits={m.group(2)} idle={m.group(3)} in_use={m.group(4)} waiters={m.group(5)}")
    for ev in events[-3:]:
        out_lines.append("recent_event: " + ev[-220:])
    return out_lines or ["log_liveness: no recent consolidation log in last 8m"]


def print_dashboard(row: dict, rows: list[dict], *, cached: bool = False, error: Exception | None = None, retry_after: float = 0.0) -> None:
    now_ts = datetime.now().astimezone().isoformat(timespec="seconds")
    print(now_ts)
    print("Hindsight observations / consolidation monitor")
    print(f"live log: {LOG}")
    if cached:
        wait = max(0, int(retry_after - time.time()))
        print(f"DB_BUSY: showing last cached DB snapshot; next DB probe in {wait}s")
        if error is not None:
            print(f"last_db_error: {short_error(error)}")
        for line in docker_liveness_lines():
            print(line)
    print()
    print(f"snapshot_ts={row['ts']}")
    print(f"observations={row['observations']} last={row['last_observation']}")
    print(f"unconsolidated_base={row['unconsolidated_base']} failed_base={row['failed_base']}")
    print("ops:", json.dumps(row["ops"], ensure_ascii=False, sort_keys=True))
    print("offline_docs:", json.dumps(row["offline_docs"], ensure_ascii=False, sort_keys=True))
    print()
    for line in rate_lines(rows):
        print(line)
    print()
    print("Refresh: 30s. Attach: tmux attach -t hindsight-obs-monitor")


def main() -> int:
    retry_after = 0.0
    last_error: Exception | None = None
    while True:
        os.system("clear")
        rows = load_rows()
        row = rows[-1] if rows else None
        cached = False
        try:
            if time.time() >= retry_after:
                row = status()
                with LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows = load_rows()
                last_error = None
                retry_after = 0.0
            else:
                cached = True
            if row is None:
                raise RuntimeError("no monitor snapshot available yet")
            print_dashboard(row, rows, cached=cached, error=last_error, retry_after=retry_after)
        except Exception as e:
            last_error = e
            msg = short_error(e)
            retry_after = time.time() + (120 if "too many clients" in msg else 30)
            rows = load_rows()
            if rows:
                print_dashboard(rows[-1], rows, cached=True, error=e, retry_after=retry_after)
            else:
                print(datetime.now().astimezone().isoformat(timespec="seconds"), "ERROR", msg)
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
