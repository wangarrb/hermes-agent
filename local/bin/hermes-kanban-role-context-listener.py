#!/usr/bin/env python3
"""Compatibility entry point for the shared Kanban role-context listener."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HERMES_KANBAN_ROOT = REPOSITORY_ROOT / "plugins" / "kanban"
if str(DEFAULT_HERMES_KANBAN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_HERMES_KANBAN_ROOT))

from role_context import render_effective_role_context  # noqa: E402


# Keep the old import name for local callers while using the shared renderer.
render_role_context = render_effective_role_context


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    plugin_root = DEFAULT_HERMES_KANBAN_ROOT
    listener_root = plugin_root / "hermes_listener"
    for path in (str(plugin_root), str(listener_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import hermes_kanban_interactive as upstream_listener

    return upstream_listener.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
