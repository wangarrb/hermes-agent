# Local Orchestration Assets

This directory is the single source of truth for local Hermes orchestration
assets that are not part of upstream Hermes proper.

Managed here:

- `bin/`: Kanban launcher and reset wrappers exposed from `~/bin`.
- `zellij/`: Zellij config and layouts used by the multi-agent Kanban sessions.
- `hermes-scripts/`: local Hindsight/Kanban/operator scripts exposed as
  `~/.hermes/scripts`.
- `hermes-skills/`: local Kanban and Hindsight skills exposed under
  `~/.hermes/skills`.
- `agent-plugins/`: local plugin marketplace metadata.

Not managed here:

- Kanban databases, board state, logs, sessions, Hindsight banks, model caches,
  credentials, generated exports, and backup archives.
- Third-party binaries in `~/.local/bin`.

Run `local/bin/install-links.sh` after changing paths or on a new machine. It
moves any pre-existing runtime file or directory into
`~/.local/state/hermes-agent/local-link-backups/<timestamp>/` before creating
symlinks, so live paths do not keep separate editable copies.
