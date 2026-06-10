# 2026-05-07 P0 hardening notes

## What changed
- Weekly paid LLM runs now do a dry-run budget check before submit.
- Budget mode is JSON on stdout; construction/logging goes to stderr.
- Content-addressed skip key moved to v2: it ignores period metadata and includes stable content/source/version fields.
- Weekly backfill completeness is unit-based, not file-existence-based.
- V2 publish now has a preflight that fails before any side effect if confirmation or safety flags are missing.

## Useful verification patterns
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest $HOME/.hermes/scripts/tests -q
python3 -m py_compile $HOME/.hermes/scripts/offline_hindsight_reflect_consolidate.py $HOME/.hermes/scripts/hindsight_offline_cron_runner.py $HOME/.hermes/scripts/hindsight_offline_v2_rebuild.py
python3 $HOME/.hermes/scripts/hindsight_offline_cron_runner.py weekly --dry-run-budget-only --week-mode current --prefilter safe --poll 10 --timeout 60
```

## Pitfalls
- Do not treat `daily/<date>/*.md` as complete just because a file exists; compare expected unit keys from the daily builder against the markdown outputs.
- Do not let `history-through-Wxx` contaminate skip keys; it is metadata, not identity.
- Do not rely on post-facto publish failure to protect state; preflight must block before reduce/eval/publish side effects.
- If `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is omitted in this repo, pytest may import a mismatched plugin and fail before tests run.
