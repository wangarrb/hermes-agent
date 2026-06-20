#!/usr/bin/env python3
"""Patch Hindsight openai_compatible_llm.py to log token usage for EVERY LLM call.

Why: The upstream code only logs token usage when duration > 10s ("slow llm call:").
Most retain/reflect/consolidation calls finish under 10s and produce no token log at all.
daily_stats.py relies on these logs to capture Hindsight LLM usage for the daily report.

This patch:
1. In call(): removes the `duration > 10.0` condition — every call logs token usage.
2. In call(): changes prefix from "slow llm call:" to "llm call:" (since it's no longer slow-only).
3. In call_with_tools(): adds token logging (upstream has none).
4. Uses a compatible format: `scope=xxx, model=xxx, input_tokens=xxx, output_tokens=xxx,
   total_tokens=xxx, time=xxxs` — always putting time right after total_tokens,
   with optional fields (cached_tokens, ratio) AFTER time, so the regex in
   daily_stats.py always matches.

Format produced after patch:
  llm call: scope=reflect, model=openai/deepseek-v4-flash, input_tokens=5000,
  output_tokens=200, total_tokens=5200, time=3.5s, cached_tokens=4500,
  ratio out/in=0.04

The regex in daily_stats.py will match the prefix up to `time=3.5s` and ignore
the trailing optional fields.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

CONTAINER = "hindsight"
REMOTE = "/app/api/hindsight_api/engine/providers/openai_compatible_llm.py"
MARKER_CALL = "# HERMES_LLM_TOKEN_LOG_FIX_V1"
MARKER_TOOLS = "# HERMES_LLM_TOKEN_LOG_FIX_V1_TOOLS"


def _run(cmd: list[str], **kw) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout, r.stderr


def apply_patch(restart: bool = False) -> None:
    # Copy file out
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    _run(["docker", "cp", f"{CONTAINER}:{REMOTE}", tmp_path])

    content = Path(tmp_path).read_text()
    changed = False

    # --- Patch 1: call() method — replace "if duration > 10.0 and usage:" block ---
    if MARKER_CALL in content:
        print(f"[SKIP] call() already patched ({MARKER_CALL} found)")
    else:
        new_block = f"""                {MARKER_CALL}
                # Log EVERY LLM call (not just slow ones) for usage tracking
                if usage:
                    ratio = max(1, output_tokens) / max(1, input_tokens)
                    cached_tokens = 0
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    cache_info = f", cached_tokens={{cached_tokens}}" if cached_tokens > 0 else ""
                    logger.info(
                        f"llm call: scope={{scope}}, model={{self.provider}}/{{self.model}}, "
                        f"input_tokens={{input_tokens}}, output_tokens={{output_tokens}}, "
                        f"total_tokens={{total_tokens}}, time={{duration:.3f}}s{{cache_info}}, ratio out/in={{ratio:.2f}}"
                    )"""

        simple_re = re.compile(
            r"([ \t]+)# Log slow calls\n"
            r"([ \t]+)if duration > 10\.0 and usage:.*?"
            r'(?=\n[ \t]+if return_usage:)',
            re.DOTALL,
        )

        new_content = simple_re.sub(new_block, content)
        if new_content == content:
            print("[ERROR] call() patch: substitution did not match")
            for i, line in enumerate(content.split("\n"), 1):
                if "Log slow calls" in line:
                    print(f"  Found at line {i}: {line!r}")
                    break
        else:
            content = new_content
            changed = True
            print(f"[OK] Patched call() — unconditional token logging")

    # --- Patch 2: call_with_tools() method — add token logging before return ---
    if MARKER_TOOLS in content:
        print(f"[SKIP] call_with_tools() already patched ({MARKER_TOOLS} found)")
    else:
        tools_patch = """                # HERMES_LLM_TOKEN_LOG_FIX_V1_TOOLS
                # Log EVERY LLM tool call for usage tracking
                total_tokens = input_tokens + output_tokens
                ratio = max(1, output_tokens) / max(1, input_tokens)
                logger.info(
                    f"llm call: scope={scope}, model={self.provider}/{self.model}, "
                    f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                    f"total_tokens={total_tokens}, time={duration:.3f}s, ratio out/in={ratio:.2f}"
                )

                return LLMToolCallResult("""

        old_return = "                return LLMToolCallResult("
        # Only replace the one inside call_with_tools (after "Record OpenTelemetry span")
        # Use a targeted approach: find the second occurrence (call_with_tools)
        parts = content.split(old_return)
        if len(parts) >= 3:
            # First occurrence is in call(), second is in call_with_tools()
            # Re-join with the patch before the second occurrence
            content = old_return.join(parts[:2]) + tools_patch + old_return.join(parts[2:])
            changed = True
            print(f"[OK] Patched call_with_tools() — token logging before return")
        elif len(parts) == 2:
            # Only one occurrence — it's in call_with_tools
            content = parts[0] + tools_patch + parts[1]
            changed = True
            print(f"[OK] Patched call_with_tools() — token logging before return (single occurrence)")
        else:
            print("[ERROR] call_with_tools() patch: 'return LLMToolCallResult(' not found")

    if not changed:
        print("[INFO] No changes needed — both patches already applied")
        return

    Path(tmp_path).write_text(content)

    # Copy back
    _run(["docker", "cp", tmp_path, f"{CONTAINER}:{REMOTE}"])
    print(f"[OK] Wrote patched {REMOTE} to container {CONTAINER}")

    if restart:
        print("[RESTART] Restarting hindsight container...")
        _run(["/usr/bin/sg", "docker", "-c", "docker", "restart", CONTAINER])
        print("[RESTART] Done. Wait ~10s for container to become ready.")
    else:
        print("[INFO] Container not restarted — existing workers will use old code.")
        print("       Restart with: docker restart hindsight")
        print("       Or the pipeline will restart it automatically next run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Hindsight to log every LLM call's token usage")
    parser.add_argument("--restart", action="store_true", help="Restart container after patching")
    args = parser.parse_args()
    apply_patch(restart=args.restart)


if __name__ == "__main__":
    main()
