# Hindsight Offline V2 Genericization Audit

Purpose: turn the current working Offline Hindsight V2 pipeline into a reusable skill without baking one machine, one user, or one project into the procedure.

## Executive Summary

The V2 architecture is reusable:

```
raw sessions -> controlled retain / fact extraction -> daily processed-fact consolidation -> weekly/global consolidation -> local canonical cards -> layered recall/eval gate -> optional direct observation publish
```

The reusable part is the data-flow and fail-closed verification. The non-reusable part is the current set of paths, provider names, benchmark files, local model choices, project aliases, and document prefixes.

For a generic skill, treat those as configuration, not as workflow rules.

## Hardcoded / Preset Concept Inventory

### Environment defaults

Current scripts and docs commonly assume:

- `~/.hermes` as Hermes home
- `~/.hermes/scripts` for scripts
- `~/.hermes/hindsight/offline_reflect` as offline root
- `~/.hermes/hindsight/offline_reflect/v2_cards` as local cards root
- `http://127.0.0.1:8888` as Hindsight API
- `bank_id=hermes`
- PostgreSQL via `/tmp:5432`, user/database `hindsight`
- local pg0 binary path such as `$HOME/.pg0/installation/18.1.0/bin/psql`
- optional Python fallback path such as `$HOME/miniconda/envs/hindsight/bin/python`

Generic rule: expose these as env / CLI / config. In skill text, mark them as examples only.

### Provider / model defaults

Current examples mention:

- `minimax`, `MiniMax-M2.7`, `https://api.minimaxi.com/v1`
- `glm` / Bailian / DashScope
- `deepseek`
- `ollama`, `qwen3.5:9b-local`, `qwen2:7b-instruct`
- `BAAI/bge-small-en-v1.5`

Generic rule: provider profiles are interchangeable execution profiles. The skill should define required profile fields:

```yaml
provider_profiles:
  paid_quality:
    llm_provider: openai|minimax|ollama|custom
    llm_model: <model>
    llm_base_url: <url>
    llm_api_key_env: <ENV_NAME>
  local_fallback:
    llm_provider: ollama|local|custom
    llm_model: <model>
    llm_base_url: <url>
```

Do not make MiniMax or Ollama part of the architecture definition.

### Memory taxonomy defaults

Current V2 uses these observation types:

- `canonical_observation`
- `technical_lesson`
- `user_preference`
- `project_decision`
- `tooling_lesson`
- `risk`
- `open_question`
- `method_comparison`
- `system_rule`

This taxonomy is reasonable for Hermes/Hindsight, but it is still a preset. A generic skill should call it the default taxonomy and allow extension / replacement.

Suggested config:

```yaml
taxonomy:
  observation_types:
    - user_preference
    - project_decision
    - technical_lesson
    - tooling_lesson
    - risk
    - open_question
    - method_comparison
    - system_rule
  type_diversity_order:
    - user_preference
    - project_decision
    - technical_lesson
    - tooling_lesson
    - risk
    - open_question
    - method_comparison
    - system_rule
```

### Project / entity aliases

Current reducer supports `v2_aliases.json`; this is the right direction. Keep project names out of code and main skill text.

Generic rule:

- Put aliases in local `v2_aliases.json` or an equivalent config file.
- Do not encode names such as project names, frontends, experiments, model families, or paper IDs into scripts.
- Keep heuristic alias normalization generic: strip generic suffix/prefix only, then allow local alias map to override.

### Benchmarks and gate thresholds

Current flow uses:

- generic benchmark JSONL
- local benchmark JSONL
- term recall / expected layer hits / case-level regression checks
- example defaults such as top-k, raw-limit, local-card cap, regression tolerance

Generic rule:

- Generic benchmark validates universal queries: preferences, decisions, risks, tooling lessons, offline pipeline health.
- Local benchmark validates the user's/project's important retrieval cases.
- Gate must require both to pass before publish.
- Thresholds are defaults, not universal truth. Keep them parameterized.

### Document IDs and prefixes

Current prefixes include:

- `hermes-offline-canonical::`
- `hermes-offline-consolidation::daily::`
- `hermes-offline-consolidation::weekly::`
- `hermes-sqlite::`
- `offline-v2-card::`
- `offline-v2-observation::`

Generic rule:

- Keep a namespace prefix to make deletion/replacement safe.
- Prefix must be configurable per bank / deployment.
- Direct publish may only replace IDs under the configured canonical prefix.

## Generic Pipeline Contract

A reusable Offline V2 pipeline should expose these phases:

1. `retain`: import raw session chunks into Hindsight as source facts.
2. `daily_consolidate`: consolidate processed facts, not raw transcripts by default.
3. `weekly_consolidate`: merge daily outputs across topic/history.
4. `reduce`: deterministic local card generation; no LLM; no DB writes.
5. `eval`: compare baseline vs local cards on generic and local benchmarks.
6. `gate`: fail closed unless metrics improve and regressions are within limit.
7. `publish`: optional direct DB write of canonical observations, with backup and prefix-only replacement.
8. `audit`: verify queue health, observation layer presence, recall mix, language consistency, alias fragmentation.

Safety invariants:

- Raw transcript must not go directly into daily/weekly consolidation by default.
- Local card generation must not call LLM or Hindsight API.
- Publish must be explicit, backed up, and prefix-scoped.
- Metadata for direct DB writes must be JSON-safe: strings/numbers/bools only; encode list/dict as JSON strings or omit nulls if required by Hindsight recall path.
- Gate failure must block publish unless the caller explicitly forces it and accepts the risk.

## What to Keep in Main SKILL.md vs References

Keep in SKILL.md:

- When to use the pipeline.
- The canonical phase order.
- The safety invariants.
- The minimum commands for dry-run, local cards, eval, gate, publish.
- The verification checklist.

Move to references:

- Provider-specific quirks.
- Local machine paths.
- Project-specific benchmark examples.
- Entity alias maps and observed alias fragmentation examples.
- Historical run reports and measured numbers.
- Case studies such as Egomotion4D / DA3 / Pi3X / Ollama GPU thresholds.

## Recommended Generic Config Shape

See `templates/hindsight-offline-v2-config.yaml` for a config skeleton.

Essential keys:

```yaml
paths:
  hermes_home: ~/.hermes
  script_dir: ~/.hermes/scripts
  offline_root: ~/.hermes/hindsight/offline_reflect
  cards_root: ~/.hermes/hindsight/offline_reflect/v2_cards

hindsight:
  api_url: http://127.0.0.1:8888
  bank_id: hermes
  db_dsn: dbname=hindsight user=hindsight host=/tmp port=5432
  canonical_prefix: hermes-offline-canonical::

benchmarks:
  generic: ~/.hermes/hindsight/eval/benchmark_queries.jsonl
  local: ~/.hermes/hindsight/eval/benchmark_queries.local.jsonl

quality_gate:
  min_pairs: 2
  min_term_recall_delta: 0.001
  min_layer_hit_delta: 0
  max_case_term_regressions: 2
  case_regression_tolerance: 0.05
```

## Script-Level Improvements Worth Doing Later

These are not required for the skill update, but would make the code itself more generic:

1. Add a shared `--config` option to V2 scripts.
2. Resolve default paths from config/env before falling back to `Path.home() / ".hermes"`.
3. Replace hardcoded psql path with `HINDSIGHT_PSQL_BIN` or PATH lookup.
4. Replace hardcoded canonical prefixes with CLI/config fields.
5. Move taxonomy and boost weights into a config block.
6. Make `query_variants`, `MEASUREMENT_WORDS`, and `layer_boost` tunable.
7. Make local-card cap configurable.
8. Emit config snapshot into every run report for reproducibility.
9. Validate config at startup and print all effective defaults.
10. Split local/project benchmarks from generic benchmarks by naming convention and config.

## Review Checklist Before Calling the Skill Generic

- [ ] Main flow contains no project names as requirements.
- [ ] Paths/ports/bank/provider are overrideable.
- [ ] Alias map is external.
- [ ] Taxonomy is declared as default, not universal.
- [ ] Generic and local benchmarks are separated.
- [ ] Publish prefix is configurable and scoped.
- [ ] Direct DB writes create backups and verify recall afterwards.
- [ ] Metadata is sanitized to avoid recall 500s.
- [ ] Raw consolidation remains opt-in only.
- [ ] Final audit reports queue, DB counts, observation counts, alias fragmentation, and recall mix.
