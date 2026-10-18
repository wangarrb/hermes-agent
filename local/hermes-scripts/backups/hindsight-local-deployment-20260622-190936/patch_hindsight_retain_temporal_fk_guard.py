#!/usr/bin/env python3
"""Patch Hindsight retain temporal links to avoid FK races.

Root cause: create_temporal_links_batch_per_fact selects temporal neighbor
memory_units and then bulk-inserts memory_links with skip_exists_check=True.
With concurrent replace-retain, a previously selected neighbor can be deleted
before the insert, causing:

  memory_links.to_unit_id violates fk_memory_links_to_unit_id_memory_units

_bulk_insert_links already has an EXISTS-guarded path. Use that path for temporal
links. Causal links are intentionally unchanged because their endpoints are
freshly inserted units in the same write transaction.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path

MARKER = "# HERMES_RETAIN_TEMPORAL_LINK_FK_GUARD_V1"
TARGET = "/app/api/hindsight_api/engine/retain/link_utils.py"


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = " ".join(shlex.quote(a) for a in ("docker", *args))
    proc = subprocess.run(["sg", "docker", "-c", cmd], text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker command failed: {cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def patch_file_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False

    old = """        if links:
            insert_start = time_mod.time()
            await _bulk_insert_links(conn, links, bank_id=bank_id, skip_exists_check=True, ops=ops)
            _log(log_buffer, f\"      [7.4] Insert {len(links)} temporal links: {time_mod.time() - insert_start:.3f}s\")

        return len(links)
"""
    new = """        if links:
            insert_start = time_mod.time()
            # HERMES_RETAIN_TEMPORAL_LINK_FK_GUARD_V1
            # Temporal neighbors are selected before insertion. Under concurrent
            # replace-retain, a neighbor unit can be cascade-deleted between the
            # SELECT and this INSERT. Keep the EXISTS guard enabled so stale
            # neighbors are skipped instead of aborting the whole retain.
            await _bulk_insert_links(conn, links, bank_id=bank_id, skip_exists_check=False, ops=ops)
            _log(log_buffer, f\"      [7.4] Insert {len(links)} temporal links: {time_mod.time() - insert_start:.3f}s\")

        return len(links)
"""
    if old not in text:
        raise RuntimeError("target temporal link insert block not found")
    return text.replace(old, new, 1), True


def patch_container(*, restart: bool = False) -> bool:
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "link_utils.py"
        cp_out = docker("cp", f"hindsight:{TARGET}", str(local), check=False)
        if cp_out.returncode != 0:
            raise RuntimeError(f"docker cp from container failed: {cp_out.stderr.strip() or cp_out.stdout.strip()}")
        original = local.read_text(encoding="utf-8")
        try:
            patched, changed = patch_file_text(original)
        except RuntimeError as exc:
            if "not found" in str(exc):
                print(f"Hindsight retain temporal FK guard patch not applicable to current image; likely upstream changed: {exc}")
                return False
            raise
        if not changed:
            print("Hindsight retain temporal FK guard patch already present")
            return False
        local.write_text(patched, encoding="utf-8")
        cp_in = docker("cp", str(local), f"hindsight:{TARGET}", check=False)
        if cp_in.returncode != 0:
            raise RuntimeError(f"docker cp into container failed: {cp_in.stderr.strip() or cp_in.stdout.strip()}")
        print("patched Hindsight retain temporal FK guard in container")
        if restart:
            rst = docker("restart", "hindsight", check=False)
            if rst.returncode != 0:
                raise RuntimeError(f"docker restart failed: {rst.stderr.strip() or rst.stdout.strip()}")
            print(rst.stdout.strip())
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Hindsight retain temporal link FK guard")
    parser.add_argument("--restart", action="store_true", help="restart hindsight container after patching")
    args = parser.parse_args()
    patch_container(restart=args.restart)


if __name__ == "__main__":
    main()
