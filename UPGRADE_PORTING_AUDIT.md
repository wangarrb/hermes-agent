# Hermes 0.19 fork-porting audit

Target: `v2026.7.20` (Hermes `0.19.0`)
Source fork: `59be84468`
Method: clean upstream worktree plus semantic, test-backed migration.  No merge/rebase
is used because the fork and `origin/main` have no valid merge-base.  The live fork is
not modified during this work.

## Recovery evidence

- Source commits are protected by local branch `backup/pre-v0.19-20260722`.
- Hermes state snapshot `20260722-024811` exists before porting.
- No rsync `--delete`, file deletion, service restart, or live-watcher action is used.
- The target worktree is isolated at
  `/home/wyr/.hermes/worktrees/hermes-v0.19.0-20260722`.

## Fork changes after the prior 0.18.2 migration

| Source commits | Decision | Evidence / target action |
|---|---|---|
| `59be84468`, `25e478431`, `1f4dc34af` | keep | The Hindsight governance implementation is under `local/`; the complete directory is copied into the isolated target. |
| `4b52b20a4`, `de347fe37`, `4a6e65af0`, `d644371d9`, `6e488c826`, `4eab9f60e`, `a1981db39`, `1e94cebbe`, `8b641ee3`, `af7abf314`, `496028c4f` and prior watcher commits | keep | Fork listener modules and watcher launch/utility assets are copied without replacing the v0.19 dashboard/systemd assets.  Core Kanban rework/control API is separately compatibility-gated before cutover. |
| `54170994e`, `c7ed8020c` | not ported | v0.19 has native GLM-5.2 metadata (1M context) and the currently configured provider inventory has no active XOP GLM provider requiring the obsolete static fallback. |
| `a39be9de0`, `0eaf331af` | not ported | These are stale hard-coded/offline fallback choices.  Current runtime configuration controls the model, and this upgrade must not revive an old fallback path. |
| `cacc716d9`, `583151229` | keep | Daily-report and session-analysis scripts are in `local/`, copied intact. |
| `3c67be153`, `7e451b80e` | keep | `agent/context_compressor.py` is ported to calculate threshold from full context without subtracting requested output tokens; relevant tests pass. |
| `155ad56bd` | not separately ported | v0.19 contains the equivalent custom-provider 401/403 retry and credential-pool handling in `agent_runtime_helpers.py`, `credential_pool.py`, and `conversation_loop.py`. |
| `6e28a238` carried core fork features | keep where still required | Chinese holographic extraction, CCH user-agent routing, Bailian literal alias, media tools, and custom assets are ported or copied.  The target's native provider resolution already covers custom-provider `key_env`; its fork duplicate was deliberately not ported. |
| `/mycompress` command patches | kept | Added a thin CLI/gateway adapter. `agent/mycompress.py` loads the `no_slash` skill body directly, parses `-n` / `--keep-last-rounds`, and invokes v0.19's native partial compressor with explicit focus/tail overrides. `64` focused tests and an installed-skill smoke pass. |
| `/resume` table columns | superseded | v0.19's native session table retains the richer title/preview display.  The old context-token field is not supplied by the v0.19 session-listing contract, so copying it would produce a misleading empty column. |

## Ported and verified code/assets

- `agent/context_compressor.py`: full-context compression threshold.
- `agent/auxiliary_client.py`: Alibaba/Bailian aliases.
- `run_agent.py`: CCH Codex-compatible User-Agent routing.
- `plugins/memory/holographic/__init__.py`: Chinese preference, decision and
  learning fact extraction.
- `plugins/kanban/` fork listener pieces; `plugins/cron/`; fork-only
  `hermes_cli/kanban_listener.py` and `hermes_cli/kanban_lifecycle.py`.
- `local/`, `tools/media_tools.py`, and `tests/tools/test_media_tools.py`.
- `agent/mycompress.py`, CLI/gateway command registration and routing: the no-slash
  skill body is retained as compression focus without a duplicate compressor.

## Release validation and cutover gates

The target exposes `return-for-rework` / `control-ack`, generation-aware
complete/block fences, and the DB control-message contract without replacing the
v0.19 attachment/artifact/goal-completion paths.

- Isolated target environment: `uv lock --check` and
  `uv sync --frozen --extra dev` passed; `.venv-v019` is Hermes `0.19.0`.
- Final per-file isolated Kanban release suite: 43 files, **1062 passed**, 0 failed.
- Final provider/compression/memory/media/local regression group: **591 passed**,
  1 skipped; the only warning is a pre-existing `datetime.utcnow()` deprecation in
  the local mental-model governance script.
- Target parsed both live profile configurations (default and `lj`) without
  printing values; shell/Python syntax checks passed.
- Frontend checks completed in the isolated target: `ui-tui` 1207 passed + 1
  skipped; `web` 97 passed.

The only remaining steps are a local reproducibility commit and maintenance-window
cutover.  CCH retains only its host-specific User-Agent; its historical direct-stream
transport remains deliberately unported absent an isolated configured smoke failure.
