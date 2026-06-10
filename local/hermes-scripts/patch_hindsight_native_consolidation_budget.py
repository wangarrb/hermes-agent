#!/usr/bin/env python3
"""Patch Hindsight native consolidation to honor a per-job memory budget.

Why: upstream `/consolidate` loops until every unconsolidated world/experience
memory in the bank is processed. On a production bank this can turn one native
consolidation operation into hundreds/thousands of LLM calls. This local patch
adds an opt-in env guard:

  HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB=100

0 or unset keeps upstream unlimited behavior.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MARKER = "# HERMES_CONSOLIDATION_MAX_MEMORIES_PER_JOB_V1"
TARGET = "/app/api/hindsight_api/engine/consolidation/consolidator.py"


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = " ".join(sh_quote(a) for a in ("docker", *args))
    full = f"newgrp docker <<'SH'\n{cmd}\nSH"
    return subprocess.run(full, shell=True, text=True, capture_output=True, check=check)


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def patch_file_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False

    if "import os\n" not in text:
        text = text.replace("import json\n", "import json\nimport os\n", 1)

    old = """    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    llm_batch_size = max(1, config.consolidation_llm_batch_size)
"""
    new = """    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    # HERMES_CONSOLIDATION_MAX_MEMORIES_PER_JOB_V1
    raw_job_budget = os.getenv("HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB", "0")
    try:
        max_memories_per_job = max(0, int(raw_job_budget))
    except Exception:
        logger.warning(
            "Invalid HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB=%r; using unlimited",
            raw_job_budget,
        )
        max_memories_per_job = 0
    llm_batch_size = max(1, config.consolidation_llm_batch_size)
"""
    if old not in text:
        raise RuntimeError("target config block not found")
    text = text.replace(old, new, 1)

    old = """    while True:
        # Fetch next batch of unconsolidated memories
        async with pool.acquire() as conn:
            t0 = time.time()
            memories = await conn.fetch(
"""
    new = """    while True:
        if max_memories_per_job and stats["memories_processed"] >= max_memories_per_job:
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} reached per-job budget "
                f"{max_memories_per_job}; leaving remaining memories for later jobs"
            )
            break

        # Fetch next batch of unconsolidated memories
        fetch_limit = max_memories_per_batch
        if max_memories_per_job:
            fetch_limit = min(max_memories_per_batch, max_memories_per_job - stats["memories_processed"])
        async with pool.acquire() as conn:
            t0 = time.time()
            memories = await conn.fetch(
"""
    if old not in text:
        raise RuntimeError("target while/fetch block not found")
    text = text.replace(old, new, 1)

    old = """                bank_id,
                max_memories_per_batch,
            )
"""
    new = """                bank_id,
                fetch_limit,
            )
"""
    if old not in text:
        raise RuntimeError("target LIMIT parameter block not found")
    text = text.replace(old, new, 1)

    return text, True


def patch_container(*, restart: bool = False) -> bool:
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "consolidator.py"
        cp_out = docker("cp", f"hindsight:{TARGET}", str(local), check=False)
        if cp_out.returncode != 0:
            raise RuntimeError(f"docker cp from container failed: {cp_out.stderr.strip()}")
        original = local.read_text(encoding="utf-8")
        try:
            patched, changed = patch_file_text(original)
        except RuntimeError as e:
            if "target" in str(e).lower() and "not found" in str(e).lower():
                print(f"Hindsight native consolidation budget patch not applicable to current image; likely upstream changed: {e}")
                return False
            raise
        if not changed:
            print("Hindsight native consolidation budget patch already present or not needed")
            return False
        local.write_text(patched, encoding="utf-8")
        cp_in = docker("cp", str(local), f"hindsight:{TARGET}", check=False)
        if cp_in.returncode != 0:
            raise RuntimeError(f"docker cp into container failed: {cp_in.stderr.strip()}")
        print("patched Hindsight native consolidation per-job budget in container")
        if restart:
            rst = docker("restart", "hindsight", check=False)
            if rst.returncode != 0:
                raise RuntimeError(f"docker restart failed: {rst.stderr.strip()}")
            print(rst.stdout.strip())
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Hindsight native consolidation per-job budget")
    parser.add_argument("--restart", action="store_true", help="restart hindsight container after patching")
    args = parser.parse_args()
    patch_container(restart=args.restart)


if __name__ == "__main__":
    main()
