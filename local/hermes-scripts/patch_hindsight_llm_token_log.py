#!/usr/bin/env python3
"""Patch Hindsight (native pip install) to log token usage for EVERY LLM call.

Why: upstream only logs token usage when duration > 10s ("slow llm call:").
Most retain/reflect/consolidation calls finish under 10s and produce no token log;
daily_report.py relies on these logs for the "Hindsight LLM 用量" table, so fast
calls were silently missing (measured 2026-09-13 00:00-02:00: 144 logged vs 240
actual bridge requests, ~40% missing; plus call_with_tools had no logging at all).

History: the original patcher (2026-06-23) targeted the Docker container via
`docker cp`. On 2026-08-13 Hindsight migrated docker -> native systemd
(pip install, hindsight.service); the patch was lost and logs reverted to
slow-only. This v2 targets the native install file directly.

Patches (idempotent, marker-guarded):
1. call(): remove `duration > 10.0` gate; prefix "slow llm call:" -> "llm call:";
   cache info moved AFTER time= so daily_report.py's regex always matches.
2. call_with_tools(): add token logging before return (upstream has none).

Usage:
  python3 patch_hindsight_llm_token_log.py            # apply patches (+py_compile)
  python3 patch_hindsight_llm_token_log.py --check    # verify markers, exit 0/1
  python3 patch_hindsight_llm_token_log.py --restart  # apply then sudo systemctl restart

NOTE: site-packages patches are overwritten by pip upgrades. After any
hindsight-api upgrade, re-run --check and re-apply if needed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import py_compile
import re
import subprocess
import sys
from pathlib import Path

TARGET = Path(
    "/home/wyr/miniconda/lib/python3.13/site-packages/hindsight_api/engine/providers/openai_compatible_llm.py"
)
MARKER_CALL = "# HERMES_LLM_TOKEN_LOG_FIX_V1"
MARKER_TOOLS = "# HERMES_LLM_TOKEN_LOG_FIX_V1_TOOLS"
SERVICE = "hindsight.service"

NEW_CALL_BLOCK = """                {marker}
                # Log EVERY LLM call (not just slow ones) for usage tracking
                if usage:
                    ratio = max(1, output_tokens) / max(1, input_tokens)
                    cache_info = f", cached_tokens={{cached_tokens}}" if cached_tokens > 0 else ""
                    logger.info(
                        f"llm call: scope={{scope}}, model={{self.provider}}/{{self.model}}, "
                        f"input_tokens={{input_tokens}}, output_tokens={{output_tokens}}, "
                        f"total_tokens={{total_tokens}}, time={{duration:.3f}}s{{cache_info}}, ratio out/in={{ratio:.2f}}"
                    )"""

TOOLS_BLOCK = """                # HERMES_LLM_TOKEN_LOG_FIX_V1_TOOLS
                # Log EVERY LLM tool call for usage tracking
                logger.info(
                    f"llm call: scope={scope}, model={self.provider}/{self.model}, "
                    f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                    f"total_tokens={input_tokens + output_tokens}, time={duration:.3f}s, "
                    f"ratio out/in={max(1, output_tokens) / max(1, input_tokens):.2f}"
                )

"""


def check() -> tuple[bool, bool]:
    content = TARGET.read_text()
    return MARKER_CALL in content, MARKER_TOOLS in content


def apply() -> int:
    content = TARGET.read_text()
    changed = False

    # --- Patch 1: call() ---
    if MARKER_CALL in content:
        print(f"[SKIP] call() already patched ({MARKER_CALL})")
    else:
        simple_re = re.compile(
            r"([ \t]+)# Log slow calls\n"
            r"([ \t]+)if duration > 10\.0 and usage:.*?"
            r"(?=\n[ \t]+if return_usage:)",
            re.DOTALL,
        )
        new_content = simple_re.sub(lambda m: NEW_CALL_BLOCK.format(marker=MARKER_CALL), content, count=1)
        if new_content == content:
            print("[ERROR] call() patch: substitution did not match")
            return 2
        content = new_content
        changed = True
        print("[OK] Patched call() — unconditional token logging")

    # --- Patch 2: call_with_tools() ---
    if MARKER_TOOLS in content:
        print(f"[SKIP] call_with_tools() already patched ({MARKER_TOOLS})")
    else:
        anchor = "                return LLMToolCallResult("
        parts = content.split(anchor)
        if len(parts) == 2:
            content = parts[0] + TOOLS_BLOCK + anchor + parts[1]
            changed = True
            print("[OK] Patched call_with_tools() — token logging before return")
        else:
            print(f"[ERROR] call_with_tools() patch: expected 1 anchor, found {len(parts) - 1}")
            return 2

    if not changed:
        print("[INFO] No changes needed — both patches already applied")
        return 0

    # backup once per day, then write + syntax check
    bak = TARGET.with_suffix(TARGET.suffix + f".bak-{_dt.date.today().isoformat()}")
    if not bak.exists():
        bak.write_text(TARGET.read_text())
        print(f"[OK] Backup: {bak}")
    TARGET.write_text(content)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        print(f"[ERROR] syntax check failed, restoring backup: {e}")
        TARGET.write_text(bak.read_text())
        return 3
    print("[OK] Written + py_compile passed")
    return 0


def restart() -> None:
    r = subprocess.run(["sudo", "-n", "systemctl", "restart", SERVICE], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"[RESTART] {SERVICE} restarted (passwordless sudo)")
        return
    r2 = subprocess.run(["systemctl", "restart", SERVICE], capture_output=True, text=True)
    print(f"[RESTART] direct attempt rc={r2.returncode}: {(r2.stderr or r2.stdout).strip()}")
    print(f"[HINT] retry manually: echo <pw> | sudo -S systemctl restart {SERVICE}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch Hindsight to log token usage for EVERY LLM call (native install)")
    ap.add_argument("--check", action="store_true", help="verify markers and exit")
    ap.add_argument("--restart", action="store_true", help="restart hindsight.service after applying")
    args = ap.parse_args()

    if args.check:
        call_ok, tools_ok = check()
        print(f"call() patched: {call_ok} | call_with_tools() patched: {tools_ok}")
        sys.exit(0 if (call_ok and tools_ok) else 1)

    rc = apply()
    if rc != 0:
        sys.exit(rc)
    if args.restart:
        restart()


if __name__ == "__main__":
    main()
