"""Shared support for the fork's skill-guided ``/mycompress`` command."""

from __future__ import annotations

import re
from typing import Optional

_KEEP_RE = re.compile(
    r"(?:^|\s)(?:-n|--keep-last-rounds)\s+(\d+)(?=\s|$)"
)


def parse_mycompress_args(raw_args: str) -> tuple[Optional[int], str]:
    """Return optional retained user-round count and remaining focus text.

    ``/mycompress`` accepts at most one ``-n N`` or ``--keep-last-rounds N``
    flag. A zero count is invalid because it cannot form a useful protected
    transcript tail.
    """
    raw_args = (raw_args or "").strip()
    matches = list(_KEEP_RE.finditer(raw_args))
    if len(matches) > 1:
        raise ValueError("use --keep-last-rounds only once")
    if not matches:
        return None, raw_args

    keep_rounds = int(matches[0].group(1))
    if keep_rounds < 1:
        raise ValueError("--keep-last-rounds must be at least 1")
    focus = (raw_args[:matches[0].start()] + raw_args[matches[0].end():]).strip()
    return keep_rounds, focus


def load_mycompress_focus(
    user_focus: str,
    *,
    task_id: str | None = None,
    runtime_note: str = "",
) -> str:
    """Load the no-slash skill body and append an optional user focus.

    The skill is intentionally ``no_slash`` because the explicit command
    registry owns ``/mycompress``. Loading the payload directly avoids falling
    through to normal skill-command discovery and avoids activation scaffolding
    becoming part of the summary prompt.
    """
    from agent.skill_commands import _load_skill_payload

    loaded = _load_skill_payload("mycompress", task_id=task_id)
    if not loaded:
        raise LookupError("unable to load mycompress skill")
    payload, _skill_dir, _skill_name = loaded
    content = str(payload.get("content") or "").strip()
    if content.startswith("---"):
        _frontmatter, marker, body = content.partition("\n---")
        if marker:
            content = body.lstrip("\n")
    if not content:
        raise LookupError("mycompress skill has no usable body")
    if user_focus:
        content = f"{content}\n\nUser focus: {user_focus}"
    if runtime_note:
        content = f"{content}\n\n[Runtime: {runtime_note}]"
    return content
