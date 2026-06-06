"""Shared timing and retry policy for visible Kanban listeners.

Both Hermes `/listen-kanban` and the Codex-backed kanban listener import this
module.  Keep operator-facing polling/health/cooldown constants here so changing
listener cadence affects all visible profile implementations together.
"""
from __future__ import annotations

import time
from typing import Optional

# Active hours: fast response while the user is likely watching panes.
# Quiet hours: slower polling to save quota/noise.
ACTIVE_START_HOUR = 9          # 09:00 inclusive
QUIET_START_HOUR = 1           # 01:00 inclusive; 01:00-08:59 quiet

DAY_POLL_SECONDS = 6.0
NIGHT_POLL_SECONDS = 30.0

DAY_HEALTH_CHECK_SECONDS = 60.0
NIGHT_HEALTH_CHECK_SECONDS = 1800.0

# Historical default; retained for state defaults and docs. Runtime health uses
# health_check_seconds(), not this fixed value.
LISTENER_HEALTH_CHECK_SECONDS = 240.0
LISTENER_HEALTH_CLAIM_TTL_SECONDS = 3600
LISTENER_HEALTH_EXTEND_MARGIN_SECONDS = 600

# Watcher-side timeout: disabled by default.  The coordinator audit handles
# stuck-task detection centrally via kanban event liveness + child-process
# checks, which is more reliable and doesn't require per-watcher tuning.
INTERACTIVE_TASK_TIMEOUT_SECONDS = 0

# Interactive TUI zombie detection.  This is not a raw wall-clock timeout:
# listeners only reclaim after the visible pane has continuously looked idle.
# Long-running tasks that keep the pane busy are preserved.
INTERACTIVE_IDLE_PANE_RECLAIM_SECONDS = 10 * 60

# Coordinator/audit knobs used by Hermes listener and safe for other visible
# listeners to share when they implement the same health behavior.
READY_UNCLAIMED_SECONDS = 10 * 60
DONE_AUDIT_LOOKBACK_SECONDS = 48 * 3600
DONE_AUDIT_FIRST_RUN_GRACE_SECONDS = 60
DONE_AUDIT_WATERMARK_OVERLAP_SECONDS = 5 * 60
COORDINATOR_AUDIT_MAX_REPAIRS_PER_TICK = 12

# Coordinator audit stuck-task detection.
# After IDLE_WARN_SECONDS of global kanban inactivity, the coordinator posts a
# warning comment on each stuck running task.  On the *next* audit tick (~60s
# later), if the system is still idle, the task is reclaimed.
# Tasks whose workspace has an agent process with active child processes
# (shell commands, training, downloads, …) are always skipped — the agent is
# visibly busy.
COORDINATOR_AUDIT_IDLE_WARN_SECONDS = 600   # 10 min idle → warn comment
COORDINATOR_AUDIT_IDLE_RECLAIM_SECONDS = 0  # reclaim fires on *next* idle tick after warn

# Coordinator must catch zellij/listener restarts quickly even during quiet
# hours.  This audit is deterministic and does not invoke a model.
COORDINATOR_FAST_HEALTH_CHECK_SECONDS = 60
COORDINATOR_INTERACTIVE_PANE_IDLE_RECLAIM_SECONDS = 60
COORDINATOR_INTERACTIVE_STALLED_BUSY_RECLAIM_SECONDS = 10 * 60

# Provider/API failure cooldown.  Keep this common so Hermes and Codex lanes do
# not thrash retry at different rates when a shared provider is unstable.
RETRY_COOLDOWN_SECONDS = 10 * 60
MIN_PROVIDER_FAILURE_SILENT_RETRIES = 10
READY_TASK_SCAN_LIMIT = 20


def _hour(now: Optional[float] = None) -> int:
    return time.localtime(now).tm_hour if now is not None else time.localtime().tm_hour


def is_active_hours(now: Optional[float] = None) -> bool:
    """Return True during 09:00-00:59 local time."""
    h = _hour(now)
    return h >= ACTIVE_START_HOUR or h < QUIET_START_HOUR


def poll_seconds(now: Optional[float] = None) -> float:
    """Ready-task polling interval shared by visible Kanban listeners."""
    return DAY_POLL_SECONDS if is_active_hours(now) else NIGHT_POLL_SECONDS


def health_check_seconds(now: Optional[float] = None) -> float:
    """Read-mostly health-check cadence shared by visible Kanban listeners."""
    return DAY_HEALTH_CHECK_SECONDS if is_active_hours(now) else NIGHT_HEALTH_CHECK_SECONDS


def provider_failure_text(text: str) -> bool:
    """Return True for provider/API errors that should stay quietly retried."""
    lower = (text or "").lower()
    markers = (
        "503",
        "429",
        "rate limit",
        "ratelimit",
        "too many requests",
        "quota",
        "not enough",
        "notenough",
        "insufficient",
        "provider",
        "api error",
        "service unavailable",
    )
    return any(m in lower for m in markers)


# --- Agent idle detection for safe injection ---

# How long to sample, in seconds.
AGENT_IDLE_SAMPLE_SECONDS = 2.0


def agent_pid_is_busy(pid: int, *, sample_s: float = AGENT_IDLE_SAMPLE_SECONDS) -> bool:
    """Return True if *pid* is busy and should NOT be injected into.

    Uses three signals, checked in order of reliability:

    1. **Process state** (``/proc/<pid>/stat``):
       ``R`` = actively running on CPU → busy.
    2. **I/O write bytes** (``/proc/<pid>/io`` → ``wchar``):
       If the process is writing to stdout (streaming LLM output,
       rendering, etc.) the byte counter grows.  A stable or absent
       counter means nothing is being printed → idle.
    3. **CPU jiffies** (fallback):
       CPU time growing faster than 3 jiffies/s → busy.

    The I/O check is the most direct signal: it answers "is the agent
    currently printing?" rather than inferring from CPU usage.
    """
    # --- Signal 0: process state ---
    try:
        line = open(f"/proc/{pid}/stat").read()
        end = line.rfind(")")
        if end < 0:
            return True  # can't parse → assume busy
        fields = line[end + 2:].split()
        state = fields[0]  # 'R', 'S', 'D', etc.
        if state == "R":
            return True  # actively on CPU → busy
        # Read CPU jiffies for fallback
        j1 = int(fields[11]) + int(fields[12])  # utime + stime (indices 11,12 after comm)
    except Exception:
        return True  # can't read → assume busy

    # --- Signal 1: I/O write bytes ---
    def _read_wchar(p: int) -> int | None:
        try:
            for line in open(f"/proc/{p}/io"):
                if line.startswith("wchar:"):
                    return int(line.split(":", 1)[1].strip())
        except Exception:
            pass
        return None

    wc1 = _read_wchar(pid)

    time.sleep(sample_s)

    # --- Signal 1 (after sample): check I/O growth ---
    wc2 = _read_wchar(pid)
    if wc1 is not None and wc2 is not None:
        wchar_delta = wc2 - wc1
        if wchar_delta > 0:
            return True  # wrote bytes → agent is outputting → busy
        # wchar stable + not 'R' state → genuinely idle
        return False

    # --- Signal 2 (fallback): CPU jiffies ---
    try:
        line = open(f"/proc/{pid}/stat").read()
        end = line.rfind(")")
        if end < 0:
            return True
        fields = line[end + 2:].split()
        j2 = int(fields[11]) + int(fields[12])
    except Exception:
        return True

    delta = j2 - j1
    # > 3 jiffies/s → busy.  At HZ=100, 3 jiffies = 30ms CPU/s.
    # Any real work (streaming tokens, shell command) easily exceeds this.
    return (delta / sample_s) > 3.0
