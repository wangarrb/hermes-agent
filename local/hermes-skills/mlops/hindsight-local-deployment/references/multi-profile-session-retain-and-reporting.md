# Multi-profile Hermes Session Retain and Reporting

## Context

Hermes profiles keep isolated runtime state. The default paths do not include kanban or other profile work:

- default sessions: `~/.hermes/sessions/`
- default state DB: `~/.hermes/state.db`
- profile sessions: `~/.hermes/profiles/<profile>/sessions/`
- profile state DB: `~/.hermes/profiles/<profile>/state.db`
- profile config: `~/.hermes/profiles/<profile>/config.yaml`

Visible kanban workflows commonly create `coordinator`, `planner`, `implementer`, and `critic`, but future profiles must be handled by discovery rather than hardcoding names.

## Session Manifest Rules

`hindsight_session_manifest.py` should default to:

```bash
python3 $HERMES_HOME/scripts/hindsight_session_manifest.py \
  --bank-target hermes \
  --profile-mode hindsight \
  --json
```

`--profile-mode hindsight` means:

1. Always include default `~/.hermes/sessions/`.
2. Scan `~/.hermes/profiles/*/sessions/`.
3. Include only profiles whose `config.yaml` has `memory.provider: hindsight`.

Other modes:

- `--profile-mode all`: include every profile with a `sessions/` directory; useful for debug/backfill.
- `--profile-mode none`: default sessions only; useful for reproducing legacy behavior.

## Document IDs and Tags

Preserve default-profile document IDs to avoid invalidating existing incremental submit state:

```text
default: hermes-session::<session_id>
profile: hermes-session::<profile>::<session_id>
```

For profile documents, add metadata and tags:

```text
metadata.source_profile=<profile>
metadata.source_label=hermes-profile:<profile>
tags include profile:<profile>, source:kanban-profile
```

Default documents should use:

```text
metadata.source_profile=default
metadata.source_label=hermes
tags include profile:default, source:hermes-session
```

This prevents collisions when two profile session files have the same filename stem or stale embedded `session_id`.

## Retain Runner Rehydration

Lean manifests omit content, so `hindsight_session_retain_runner.py` reconstructs records from `metadata.json_path`. When rehydrating, it must pass `source_profile=metadata.source_profile` back into `records_from_json_file`; otherwise profile document IDs become default IDs and the runner cannot match the reviewed manifest record.

Regression test shape:

```text
session_profile_001.json + source_profile=planner
=> document_id hermes-session::planner::profile_001
=> rehydrate from lean manifest returns the same document_id and content
```

## Pipeline Integration

`hindsight_memory_pipeline.py daily/full` should pass `--profile-mode <session_profile_mode>` into the manifest step. Default is `hindsight` and the CLI should expose:

```text
--session-profile-mode hindsight|all|none
```

Do not make operators manually run one manifest per profile; that causes gaps and inconsistent incremental state.

## Daily Report Integration

Daily reports must aggregate model usage from:

1. `~/.hermes/state.db`
2. every `~/.hermes/profiles/*/state.db`
3. Hindsight/offline pipeline stats
4. auxiliary compression logs

For state DBs, find sessions active in the reporting window using `messages.timestamp`, not only `sessions.started_at`, because long-running sessions can cross day boundaries.

## Verification Checklist

- [ ] Manifest dry-run JSON contains `sources[]` for default and all Hindsight-backed profiles.
- [ ] Manifest summary includes `by_profile`.
- [ ] `sqlite_sources` includes per-profile `sessions_count` and `messages_count`.
- [ ] Sample profile document ID is namespaced: `hermes-session::<profile>::...`.
- [ ] Sample profile tags include `profile:<profile>` and `source:kanban-profile`.
- [ ] Pipeline plan shows manifest command contains `--profile-mode hindsight`.
- [ ] Retain-runner tests cover profile lean-manifest rehydration.
- [ ] Daily-report tests cover globbing future `profiles/*/state.db`, not hardcoded kanban names.

## Pitfalls

1. Hardcoding only `coordinator/planner/implementer/critic` misses future kanban profiles.
2. Renaming default document IDs causes unnecessary full re-retain; preserve legacy default IDs.
3. Forgetting `source_profile` during rehydration makes lean manifests fail to match profile document IDs.
4. Filtering daily stats only by `sessions.started_at` misses long sessions with today’s messages.
5. Treating `memory.provider=holographic` profiles as Hindsight retain sources by default mixes independent memory systems; include them only with explicit `--profile-mode all`.
