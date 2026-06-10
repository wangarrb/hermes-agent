# Hindsight session/json lightweight candidate filter

Context: session/json retain into clean Hindsight banks should avoid obvious low-value sessions without adding a heavy semantic classifier. This came from a MiniMax session/json retain smoke where stronger retain quality helped, but blind `--limit N` still selected poor candidates.

## Principle

Keep the selector deterministic and lightweight. It should prevent obvious noise from entering paid retain, not decide all semantic value.

Skip / avoid production for clear low-signal or circular-memory sessions:
- too short / empty after deterministic cleaning
- greetings: `hi`, `hello`, `你好`, `您好`, `嗨`, `哈喽`, `在吗`
- identity checks with short assistant intro: `你是谁`, `who aru u`, `who r u`, `who are you`, `who are u`
- pure continuation/ack: `继续`, `继续吧`, `接着`, `ok`, `okay`, `好`, `收到`, `明白`, `嗯`, `谢谢`
- short assistant boilerplate after such prompts, e.g. “在，老王，有什么要我处理的？”, “好，我继续处理。” or “我是臭臭，自动驾驶领域 AI 助手。”
- route memory-recall/context-bootstrap sessions to `manual_review` rather than production, e.g. `回忆Egomotion4D`, `刚聊到哪儿了`, `刚才聊了什么` when the session is primarily reconstructing prior memory/context instead of producing fresh source evidence
- route sessions containing credential material (`api key`, `token`, `secret`, `password`, `sk-...`) to `manual_review:secret_or_credential_material`; never let key material enter production retain
- drop deterministic handoff noise such as model-switch self-identification notes and preserved task-list snippets before tagging/scoring

Do not skip semantic continuation prompts:
- “继续讨论 Hindsight native consolidation 和 recall cache” should remain eligible and flow through normal tag/action logic.

## Implementation notes

Current scripts:
- `$HOME/.hermes/scripts/hindsight_session_manifest.py`
- `$HOME/.hermes/scripts/hindsight_session_manifest_selector.py` — production-only, non-mutating curated manifest selector for small smoke/e2e runs; writes selected manifest + summary and does not call Hindsight or rehydrate full content.
- `$HOME/.hermes/scripts/hindsight_session_retain_runner.py`

Manifest metadata records:
- `candidate_filter_version=lightweight-candidate-filter-v3`

Submit-state comparison includes `candidate_filter_version` so changing selector semantics forces preview/re-retain instead of silently reusing old successful state.

Manifest summary should include `by_reason` counts so audit can distinguish:
- `skip:empty_or_too_short`
- `skip:low_signal_short_or_chitchat`
- `manual_review:memory_recall_or_context_bootstrap`
- `manual_review:context_resume_or_handoff`
- `manual_review:secret_or_credential_material`
- `manual_review:no_semantic_tags`
- `production:semantic_tags_detected`

## Verification commands

Use plugin autoload disabled in this environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest $HOME/.hermes/scripts/tests -q
python3 -m py_compile $HOME/.hermes/scripts/hindsight_session_manifest.py $HOME/.hermes/scripts/hindsight_session_retain_runner.py $HOME/.hermes/scripts/hindsight_minimax_import.py
```

Dry-run should not switch paid provider or write submit-state:

```bash
python3 $HOME/.hermes/scripts/hindsight_session_manifest.py --bank-target hermes_v3 --json
python3 $HOME/.hermes/scripts/hindsight_minimax_import.py session-manifest-retain-minimax \
  --manifest $HOME/.hermes/hindsight/session_ingest/manifests/<manifest>.jsonl \
  --bank hermes_v3_minimax_dryrun_selector \
  --limit 5 --batch-size 5 \
  --submit-state /tmp/hindsight-selector-dryrun-submit-state.json
python3 $HOME/.hermes/scripts/hindsight_minimax_import.py status
```

Expected dry-run safety:
- `dry_run=true`
- `submitted_items=0`
- no submit-state file written
- provider remains normal-local/Ollama
- `pending_operations=processing_operations=failed_operations=0`

## Pitfalls

- Do not replace this with broad semantic scoring unless explicitly requested. The user prefers simple deterministic exclusion here.
- Do not use blind `--limit N` expansion after a smoke. Curate or at least audit by action/reason/tag distribution first.
- Do not classify “继续” globally as noise; only skip when the whole short conversation is low-signal.
- If filter behavior changes, bump `candidate_filter_version` and rerun dry-run before any paid execute.
