#!/usr/bin/env python3
"""Patch Hindsight container to robustly strip Markdown JSON fences.

Why: MiniMax sometimes returns fenced JSON for structured calls. The upstream helper
can leave a leading language tag (e.g. "json\n{...}") in some variants, causing
json.loads(...): Expecting value at char 0 and expensive retry loops.

This script patches the file inside the running/stopped Docker container. Use
--restart to restart the container so the Python process reloads the patched code.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

CONTAINER = "hindsight"
REMOTE = "/app/api/hindsight_api/engine/providers/openai_compatible_llm.py"
JSON_MARKER = "# HERMES_MINIMAX_JSON_FENCE_FIX_V4"
RATE_LIMIT_MARKER = "# HERMES_RATE_LIMIT_BACKOFF_FIX_V1"

NEW_FUNC = r'''def _strip_code_fences(content: str) -> str:
    """Strip markdown code fences from LLM response if present.

    Many LLM providers (MiniMax, some Ollama models, Claude via proxies)
    wrap JSON responses in ```json ... ``` fences even when json_object
    response format is requested. This strips the fences while preserving
    the JSON content inside. Returns the original content unchanged if
    no fences are detected.
    """
    # HERMES_MINIMAX_JSON_FENCE_FIX_V4
    # Regex-based fence stripping; handles ```json, ```JSON, CRLF, and
    # avoids leaving the language label before the JSON object. It also strips
    # <think>...</think> blocks emitted by reasoning models and removes common
    # trailing commas before } or ].
    def _repair(s: str) -> str:
        s = (s or "").strip().lstrip("\ufeff")
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()
        s = re.sub(r",\s*([}\]])", r"\1", s)
        if s and not s.startswith(("{", "[")):
            m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
            if m:
                s = m.group(1).strip()
        return s
    if not content or "```" not in content:
        return _repair(content)
    try:
        m = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", content, flags=re.DOTALL)
        if m:
            return _repair(m.group(1))
        m = re.search(r"```\s*(.*?)```", content, flags=re.DOTALL)
        if m:
            fenced = m.group(1).strip()
            fenced = re.sub(r"^(?:json|JSON|Json)\s*\n", "", fenced, count=1).strip()
            return _repair(fenced)
        return _repair(content)
    except Exception:
        return content
'''


def docker(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Use absolute path to avoid conda's ast-grep 'sg' shadowing system's newgrp 'sg'
    proc = subprocess.run(["/usr/bin/sg", "docker", "-c", cmd], text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker command failed: {cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def patch_json_fence_text(text: str) -> tuple[str, bool]:
    if JSON_MARKER in text:
        return text, False
    pattern = r"def _strip_code_fences\(content: str\) -> str:\n.*?\n\n\nclass OpenAICompatibleLLM"
    replacement = NEW_FUNC + "\n\nclass OpenAICompatibleLLM"
    new_text, n = re.subn(pattern, lambda _m: replacement, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError("target _strip_code_fences function not found")
    return new_text, True


def patch_rate_limit_backoff_text(text: str) -> tuple[str, bool]:
    """Patch APIStatusError retry backoff so 429/throttling uses a long pause.

    Hindsight already disables OpenAI SDK retries, but its own retry loop uses
    short exponential backoff. For quota-window providers this can create fast
    repeated paid calls without making progress. The patch is intentionally
    small: it only changes the APIStatusError retry sleep expression used by
    `call()`, leaving auth errors and final failure behavior unchanged.
    """
    if RATE_LIMIT_MARKER in text:
        return text, False
    pattern = re.compile(
        r"(?P<indent>[ \t]+)if attempt < max_retries:\n"
        r"(?P=indent)    backoff = min\(initial_backoff \* \(2\*\*attempt\), max_backoff\)\n"
        r"(?P=indent)    jitter = backoff \* 0\.2 \* \(2 \* \(time\.time\(\) % 1\) - 1\)\n"
        r"(?P=indent)    sleep_time = backoff \+ jitter\n"
        r"(?P=indent)    await asyncio\.sleep\(sleep_time\)\n"
        r"(?P=indent)else:\n"
        r"(?P=indent)    logger\.error\(f\"API error after \{max_retries \+ 1\} attempts: \{str\(e\)\}\"\)\n"
        r"(?P=indent)    raise\n"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}if attempt < max_retries:\n"
            f"{indent}    # HERMES_RATE_LIMIT_BACKOFF_FIX_V1\n"
            f"{indent}    error_text = str(e).lower()\n"
            f"{indent}    rate_limited = (\n"
            f"{indent}        e.status_code == 429\n"
            f"{indent}        or \"rate limit\" in error_text\n"
            f"{indent}        or \"throttl\" in error_text\n"
            f"{indent}        or \"quota\" in error_text\n"
            f"{indent}    )\n"
            f"{indent}    if rate_limited:\n"
            f"{indent}        backoff = float(os.getenv(\"HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS\", \"300\"))\n"
            f"{indent}        logger.warning(\n"
            f"{indent}            f\"Rate limited (HTTP {{e.status_code}}); sleeping {{backoff:.0f}}s before retry \"\n"
            f"{indent}            f\"(attempt {{attempt + 1}}/{{max_retries + 1}})\"\n"
            f"{indent}        )\n"
            f"{indent}    else:\n"
            f"{indent}        backoff = min(initial_backoff * (2**attempt), max_backoff)\n"
            f"{indent}        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)\n"
            f"{indent}        backoff = backoff + jitter\n"
            f"{indent}    await asyncio.sleep(backoff)\n"
            f"{indent}else:\n"
            f"{indent}    logger.error(f\"API error after {{max_retries + 1}} attempts: {{str(e)}}\")\n"
            f"{indent}    raise\n"
        )

    new_text, n = pattern.subn(repl, text, count=1)
    if n != 1:
        raise RuntimeError("target APIStatusError retry block not found")
    return new_text, True


def patch_file_text(text: str) -> tuple[str, bool]:
    text, json_changed = patch_json_fence_text(text)
    text, rate_changed = patch_rate_limit_backoff_text(text)
    return text, json_changed or rate_changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="Restart container after patch so code is loaded")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "openai_compatible_llm.py"
        docker(f"docker cp {CONTAINER}:{REMOTE} {local}")
        text = local.read_text(encoding="utf-8")
        try:
            new_text, changed = patch_file_text(text)
        except RuntimeError as e:
            if "target" in str(e).lower() and "not found" in str(e).lower():
                print(f"Hindsight JSON/backoff patch not applicable to current image; likely upstream changed: {e}")
                changed = False
                new_text = text
            else:
                raise
        if changed:
            local.write_text(new_text, encoding="utf-8")
            docker(f"docker cp {local} {CONTAINER}:{REMOTE}")
            print("patched Hindsight JSON fence parser in container")
        else:
            print("Hindsight JSON fence parser patch already present or not needed")

    if args.restart:
        docker(f"docker restart {CONTAINER}")
        print("restarted Hindsight container to load parser patch")


if __name__ == "__main__":
    main()
