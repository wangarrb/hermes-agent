"""Fake watcher for testing SIGUSR1 hot-reload with REAL ``os.execve``.

This fixture mirrors the production entry-point contract of
``BaseInteractiveListener.watcher_main``:

1. Build the watcher identity.
2. If ``HERMES_KANBAN_RELOAD_IDENTITY`` matches the identity digest, STRICTLY
   adopt the inherited lock FD (fail closed on any mismatch — never fall back
   to ``acquire()`` while the inherited lock may still be held); otherwise
   acquire a fresh lock.
3. Restore the active claim from the reload handoff (without claiming or
   injecting again) when a reload nonce is present.
4. Install a flag-only SIGUSR1 handler; at the safe boundary verify the
   reload request owner (PID AND proc start-time), write a handoff, mark the
   lock FD inheritable, and ``os.execve`` itself with the same argv.
5. After exec the new process keeps the same PID and inherited lock FD,
   prints ``EXEC_GENERATION N`` (increments per exec), continues heartbeats,
   and emits exactly one ACK (idle: after a healthy tick; active: after a
   simulated claim-equality heartbeat).

Stdout markers for assertions:

    LOCK_ACQUIRED pid=...        cold-start lock acquisition
    LOCK_ADOPTED pid=... fd=...  reload lock adoption
    ADOPT_FAILED error=...       fail-closed exit path (returncode != 0)
    LOCK_REJECTED                contended cold start (returncode != 0)
    EXEC_GENERATION N            process generation (1 = first, 2 = after exec)
    HEARTBEAT N                  loop tick counter
    CLAIM_INJECT N               claim+inject count (must stay 1 across execs)
    HANDOFF_RESTORED task=...    active claim restored after exec
    HANDOFF_WRITTEN task=...     handoff snapshot before exec
    ACK_WRITTEN nonce=... ok=... success ACK (exactly once per nonce)
    FAILED_ACK nonce=... error=... failed ACK (owner_mismatch / preflight)
    EXECVE gen=N                 about to os.execve
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KANBAN = REPO / "plugins" / "kanban"
sys.path.insert(0, str(KANBAN))

from watcher_runtime import (  # type: ignore[import-not-found]
    WatcherIdentity,
    WatcherLock,
    ReloadHandoff,
    ReloadACK,
    runtime_root,
    _proc_start_time,
    read_reload_request,
    read_reload_handoff,
    write_reload_handoff,
    write_reload_ack,
    delete_reload_request,
    delete_reload_handoff,
    cleanup_reload_state,
    validate_reload_handoff,
)


def _exec_generation() -> int:
    return int(os.environ.get("HERMES_KANBAN_EXEC_GENERATION", "1"))


def _append_inject(path: str, count: int) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"inject-{count}\n")


def _read_inject(path: str) -> int:
    if not path or not Path(path).exists():
        return 0
    return len(Path(path).read_text().splitlines())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pane", required=True)
    parser.add_argument("--claim-file", default="")
    parser.add_argument("--inject-file", default="")
    parser.add_argument("--exec-limit", type=int, default=1)
    parser.add_argument("--fail-preflight", action="store_true")
    parser.add_argument("--tick-s", type=float, default=0.2)
    args = parser.parse_args(argv)

    identity = WatcherIdentity.from_values(
        args.board, args.profile, args.session, args.pane,
    )
    gen = _exec_generation()

    # ── Adopt (reload exec) or acquire (cold start) — production order ──
    inherited_fd_str = os.environ.get("HERMES_KANBAN_WATCHER_LOCK_FD", "")
    reload_identity_env = os.environ.get("HERMES_KANBAN_RELOAD_IDENTITY", "")
    reload_nonce_env = os.environ.get("HERMES_KANBAN_RELOAD_NONCE", "")
    is_reload_exec = bool(
        inherited_fd_str and reload_identity_env == identity.digest
    )

    if is_reload_exec:
        try:
            inherited_fd = int(inherited_fd_str)
            lock = WatcherLock.adopt_inherited(identity, inherited_fd)
        except (ValueError, RuntimeError) as exc:
            print(f"ADOPT_FAILED error={exc}", flush=True)
            return 1  # fail closed — no acquire fallback
        print(f"LOCK_ADOPTED pid={os.getpid()} fd={lock.fd}", flush=True)
    else:
        try:
            lock = WatcherLock.acquire(identity)
        except SystemExit:
            print("LOCK_REJECTED", flush=True)
            return 1
        print(f"LOCK_ACQUIRED pid={os.getpid()}", flush=True)

    # ── Restore active claim from handoff (reload exec only) ──
    active_task: str | None = None
    active_run_id: int | None = None
    active_generation: int | None = None
    active_claim_lock: str | None = None
    if is_reload_exec and reload_nonce_env:
        root = runtime_root()
        handoff = read_reload_handoff(root, identity.digest)
        task_state = None
        if handoff and handoff.task_id and args.claim_file:
            claim_path = Path(args.claim_file)
            if claim_path.exists():
                claim = json.loads(claim_path.read_text())
                task_state = (
                    "running", claim.get("run_id"), claim.get("generation"),
                    claim.get("claim_lock"),
                )
        handoff_error = validate_reload_handoff(
            handoff,
            nonce=reload_nonce_env,
            identity_digest=identity.digest,
            current_pid=os.getpid(),
            task_state=task_state,
        )
        if handoff_error:
            write_reload_ack(
                root, identity.digest,
                ReloadACK(
                    nonce=reload_nonce_env, pid=os.getpid(),
                    proc_start_time=_proc_start_time(os.getpid()),
                    code_revision="unknown", ok=False, error=handoff_error,
                    identity_digest=identity.digest,
                ),
            )
            print(f"HANDOFF_REJECTED error={handoff_error}", flush=True)
            delete_reload_request(root, identity.digest)
            delete_reload_handoff(root, identity.digest)
            return 1
        if handoff and handoff.task_id:
            active_task = handoff.task_id
            active_run_id = handoff.run_id
            active_generation = handoff.generation
            active_claim_lock = handoff.claim_lock
            print(
                f"HANDOFF_RESTORED task={active_task} run={active_run_id}",
                flush=True,
            )
        # The ACK (request/handoff/ack state) is rewritten by this process
        cleanup_reload_state(root, identity.digest)

    # ── Cold-start claim+inject simulation (exactly once, generation 1) ──
    inject_count = 0
    if gen == 1 and not is_reload_exec and args.claim_file:
        claim_path = Path(args.claim_file)
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_lock = f"host:{os.getpid()}:fake-interactive"
        claim_path.write_text(json.dumps({
            "task_id": "t_fake",
            "run_id": 1,
            "generation": 1,
            "claim_lock": claim_lock,
            "worker_pid": os.getpid(),
        }))
        # Simulate the active claim state (like watcher_main after
        # claim_and_inject_one) so the reload path is the ACTIVE one.
        active_task = "t_fake"
        active_run_id = 1
        active_generation = 1
        active_claim_lock = claim_lock
        inject_count += 1
        _append_inject(args.inject_file, inject_count)
        print(f"CLAIM_INJECT {inject_count}", flush=True)
    else:
        # Reload exec must NOT claim/inject again — restore prior count
        inject_count = _read_inject(args.inject_file)
        if inject_count:
            print(f"CLAIM_INJECT_RESTORED {inject_count}", flush=True)

    print(f"EXEC_GENERATION {gen}", flush=True)

    # ── SIGUSR1 handler (flag-only, no I/O in handler) ──
    reload_flag = False

    def _handle_usr1(signum: int, frame: object) -> None:  # noqa: ARG001
        nonlocal reload_flag
        reload_flag = True

    signal.signal(signal.SIGUSR1, _handle_usr1)

    ack_written = False
    heartbeat_count = 0
    exec_count = 0

    def _write_ack(
        nonce: str, *, ok: bool, error: str,
        task_id: str = "", run_id: int = 0,
    ) -> None:
        root = runtime_root()
        write_reload_ack(
            root, identity.digest,
            ReloadACK(
                nonce=nonce,
                pid=os.getpid(),
                proc_start_time=_proc_start_time(os.getpid()),
                code_revision="unknown",
                ok=ok,
                error=error,
                identity_digest=identity.digest,
                task_id=task_id,
                run_id=run_id,
            ),
        )

    def _maybe_reload() -> bool:
        """Handle a SIGUSR1 reload request at a safe boundary.

        Returns True if execve was initiated (unreachable afterwards);
        False if the old loop should continue.
        """
        nonlocal ack_written, exec_count
        root = runtime_root()
        req = read_reload_request(root, identity.digest)
        if req is None:
            print("SIGUSR1_IGNORED no_valid_request", flush=True)
            return False
        print(f"RELOAD_REQUESTED nonce={req.nonce}", flush=True)

        # Owner verification: PID AND proc start-time (PID reuse is not enough)
        meta_path = lock.lock_path.parent / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            print("SIGUSR1_IGNORED no_metadata", flush=True)
            return False
        if (
            meta.get("pid") != req.owner_pid
            or meta.get("proc_start_time") != req.owner_start_time
        ):
            print(f"FAILED_ACK nonce={req.nonce} error=owner_mismatch", flush=True)
            _write_ack(req.nonce, ok=False, error="owner_mismatch")
            delete_reload_request(root, identity.digest)
            return False

        if args.fail_preflight:
            print(f"FAILED_ACK nonce={req.nonce} error=preflight", flush=True)
            _write_ack(req.nonce, ok=False, error="preflight")
            delete_reload_request(root, identity.digest)
            delete_reload_handoff(root, identity.digest)
            return False

        # Explicit active or idle handoff; missing is always a reload failure.
        handoff = ReloadHandoff(
            nonce=req.nonce,
            identity_digest=identity.digest,
            task_id=active_task or "",
            run_id=active_run_id or 0,
            generation=active_generation or 0,
            claim_lock=active_claim_lock or "",
            worker_pid=os.getpid(),
            original_pid=os.getpid(),
        )
        write_reload_handoff(root, handoff)
        if active_task is not None:
            print(f"HANDOFF_WRITTEN task={active_task} run={active_run_id}", flush=True)

        if exec_count >= args.exec_limit:
            print(f"EXEC_SKIPPED limit={args.exec_limit}", flush=True)
            return False

        # Make the lock FD inheritable and execve ourselves
        lock_fd = lock.make_inheritable()
        env = dict(os.environ)
        env["HERMES_KANBAN_WATCHER_LOCK_FD"] = str(lock_fd)
        env["HERMES_KANBAN_RELOAD_NONCE"] = req.nonce
        env["HERMES_KANBAN_RELOAD_IDENTITY"] = identity.digest
        env["HERMES_KANBAN_EXEC_GENERATION"] = str(gen + 1)
        argv = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
        print(f"EXECVE gen={gen + 1}", flush=True)
        try:
            os.execve(argv[0], argv, env)
        except OSError as exc:
            print(f"EXECVE_FAILED {exc}", flush=True)
            lock.make_non_inheritable()
            _write_ack(req.nonce, ok=False, error="execve_failed")
            delete_reload_request(root, identity.digest)
            delete_reload_handoff(root, identity.digest)
            return False
        return True  # unreachable

    # ── Main loop: heartbeat, safe-boundary reload, ACK emission ──
    while True:
        heartbeat_count += 1
        print(f"HEARTBEAT {heartbeat_count}", flush=True)

        if reload_flag:
            reload_flag = False
            if _maybe_reload():
                return 0  # unreachable — execve replaced us

        # Idle ACK: one healthy tick after adoption is the idle requirement
        if reload_nonce_env and not ack_written and not active_task:
            _write_ack(reload_nonce_env, ok=True, error="")
            print(
                f"ACK_WRITTEN nonce={reload_nonce_env} pid={os.getpid()} ok=True",
                flush=True,
            )
            ack_written = True

        # Active ACK: simulated claim-equality heartbeat keeps the claim
        if reload_nonce_env and not ack_written and active_task:
            # Simulate heartbeat: verify the claim file still matches the
            # restored handoff (claim equality), then advance the tick.
            if args.claim_file and Path(args.claim_file).exists():
                claim = json.loads(Path(args.claim_file).read_text())
                if (
                    claim.get("task_id") == active_task
                    and claim.get("run_id") == active_run_id
                    and claim.get("generation") == active_generation
                    and claim.get("claim_lock") == active_claim_lock
                ):
                    print(
                        f"ACTIVE_HEARTBEAT task={active_task} run={active_run_id}",
                        flush=True,
                    )
                    _write_ack(
                        reload_nonce_env, ok=True, error="",
                        task_id=active_task or "", run_id=active_run_id or 0,
                    )
                    print(
                        f"ACK_WRITTEN nonce={reload_nonce_env} "
                        f"pid={os.getpid()} ok=True task={active_task}",
                        flush=True,
                    )
                    ack_written = True

        time.sleep(args.tick_s)
        if heartbeat_count > 300:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
