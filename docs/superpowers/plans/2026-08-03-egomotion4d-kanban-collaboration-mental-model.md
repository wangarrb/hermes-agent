# Egomotion4D Kanban Collaboration Mental Model Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add, publish, and continuously maintain one auditable `egomotion4d-kanban-collaboration` mental model derived from current Hermes/Kanban code and Egomotion4D role rules.

**Architecture:** A deterministic extractor reads an exact selector manifest and atomically creates one bounded evidence snapshot. The existing mental-model daily pipeline refreshes that snapshot before recomputing evidence identity, while the existing inactive-slot/adjudication/smoke/export lifecycle remains the sole publication authority. Tracked model contracts and a small bootstrap helper make the fifth logical model reproducible; runtime state and generated wiki exports remain outside Git except for the accepted wiki allowlist.

**Tech Stack:** Python 3 standard library, JSON, pytest, existing Hindsight HTTP API, existing Hermes mental-model registry/export pipeline.

---

## Chunk 1: Deterministic evidence and daily integration

### Task 1: Exact source selector extractor

**Files:**
- Create: `local/hermes-scripts/kanban_collaboration_evidence.py`
- Create: `local/mental-models/egomotion4d/sources/kanban_collaboration_sources.json`
- Create: `local/mental-models/egomotion4d/sources/kanban_collaboration_current_evidence.md` (deterministically generated initial snapshot)
- Create: `local/hermes-scripts/tests/test_kanban_collaboration_evidence.py`

- [ ] **Step 1: Write failing extraction tests**

Cover all four selector kinds with temporary fixtures:

```python
def test_build_snapshot_extracts_exact_selectors_and_hashes_sources(tmp_path):
    manifest = write_fixture_manifest(tmp_path, kinds=(
        "whole_file", "markdown_heading", "python_symbol", "shell_function"
    ))
    result = module.build_snapshot(manifest, tmp_path / "current.md")
    assert result["source_count"] == 4
    assert "BEGIN_CANONICAL_SOURCE" in (tmp_path / "current.md").read_text()
    assert "sha256=" in (tmp_path / "current.md").read_text()

@pytest.mark.parametrize("fault", ["missing", "duplicate", "empty"])
def test_build_snapshot_fails_closed_on_invalid_selector(tmp_path, fault):
    manifest = write_invalid_manifest(tmp_path, fault)
    with pytest.raises(ValueError):
        module.build_snapshot(manifest, tmp_path / "current.md")
```

Also cover dotted class methods such as
`BaseInteractiveListener.wait_for_stable_composer_input`, multiple selectors
from one source file, and the `usage()` shell-function block used by the real
manifest. Assert output replacement uses `os.replace`, a failed rebuild
preserves the previous output bytes, and selector order is deterministic. The
builder computes and records each current whole-file SHA256 in the snapshot;
the stable manifest does not pin hashes.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
python3 -m pytest local/hermes-scripts/tests/test_kanban_collaboration_evidence.py -q
```

Expected: FAIL because `kanban_collaboration_evidence.py` does not exist.

- [ ] **Step 3: Implement the minimal deterministic extractor**

Implement this public boundary:

```python
def build_snapshot(manifest_path: Path, output_path: Path) -> dict[str, object]:
    """Validate manifest, extract each selector exactly once, then atomically write."""
```

Manifest schema:

```json
{
  "schema_version": 1,
  "logical_id": "egomotion4d-kanban-collaboration",
  "sources": [
    {
      "name": "publisher-result-notifications-design",
      "path": "/absolute/path",
      "selectors": [{"kind": "whole_file"}]
    }
  ]
}
```

The real manifest must contain every canonical path/selector listed in the approved design. `markdown_heading` consumes the matching heading through the next heading of equal or smaller depth. `python_symbol` consumes one top-level function or one class method, including decorators, through the next symbol at the same or smaller indentation. `shell_function` consumes exactly one `name() { ... }` block using brace depth while ignoring braces inside quoted strings. Normalize only line endings and require every selector to return non-empty bytes. Render source name, path, whole-file SHA256, selector, and exact bounded bytes; end with `END_KANBAN_COLLABORATION_EVIDENCE`.

- [ ] **Step 4: Run focused tests and real-manifest dry run**

Run:

```bash
python3 -m pytest local/hermes-scripts/tests/test_kanban_collaboration_evidence.py -q
python3 local/hermes-scripts/kanban_collaboration_evidence.py \
  --manifest local/mental-models/egomotion4d/sources/kanban_collaboration_sources.json \
  --output local/mental-models/egomotion4d/sources/kanban_collaboration_current_evidence.md
tail -1 local/mental-models/egomotion4d/sources/kanban_collaboration_current_evidence.md
```

Expected: tests PASS; last line is `END_KANBAN_COLLABORATION_EVIDENCE`.

- [ ] **Step 5: Commit Task 1**

```bash
git add local/hermes-scripts/kanban_collaboration_evidence.py \
  local/hermes-scripts/tests/test_kanban_collaboration_evidence.py \
  local/mental-models/egomotion4d/sources/kanban_collaboration_sources.json \
  local/mental-models/egomotion4d/sources/kanban_collaboration_current_evidence.md
git commit -m "feat(mental-model): derive kanban collaboration evidence"
```

### Task 2: Refresh the derived snapshot in the existing daily pipeline

**Files:**
- Modify: `local/hermes-scripts/hindsight_daily_noagent.py` (`_refresh_evidence_bundle` boundary)
- Modify: `local/mental-models/egomotion4d/sources/derived-build-inputs.json`
- Modify: `local/mental-models/egomotion4d/tests/test_governance.py`

- [ ] **Step 1: Write failing pipeline tests**

Add tests asserting:

```python
def test_refresh_builds_kanban_snapshot_before_hashing(...):
    # registered collaboration model + manifest + helper
    # assert generated bytes are what evidence_bundle hashes

def test_refresh_blocks_registered_collaboration_model_without_manifest(...):
    with pytest.raises(ValueError, match="kanban collaboration manifest"):
        daily._refresh_evidence_bundle()
```

Existing models without the new logical ID must continue to refresh without requiring this manifest.

- [ ] **Step 2: Run the tests and confirm RED**

```bash
python3 -m pytest local/mental-models/egomotion4d/tests/test_governance.py \
  -k 'kanban_snapshot or collaboration_model_without_manifest' -q
```

Expected: FAIL because daily refresh does not invoke the builder.

- [ ] **Step 3: Add the minimal integration**

Before iterating evidence entries, `_refresh_evidence_bundle()` checks whether `egomotion4d-kanban-collaboration` is registered in the bundle. If registered, require `sources/kanban_collaboration_sources.json`, load the sibling maintained helper from `Path(__file__).with_name(...)`, and atomically regenerate `sources/kanban_collaboration_current_evidence.md`. Then run the existing hash validation. Register the generated snapshot in `derived-build-inputs.json` with `authority: canonical-source-extract`, `replaceable: true`, and no algorithm D anchors; extend `_validate_current_evidence_build_input()` to accept only this explicit second authority.

The current deployment is intentionally verified rather than copied:
`~/.hermes/scripts` resolves through `~/.hermes/hermes-agent` to this repo's
`local/hermes-scripts/`, so adding the sibling helper deploys it immediately.
Add a verification command that `readlink -f
~/.hermes/scripts/kanban_collaboration_evidence.py` resolves to the maintained
repo file; if that invariant is absent, stop instead of creating a second
manual script copy.

- [ ] **Step 4: Run governance and wrapper regression tests**

```bash
python3 -m pytest \
  local/mental-models/egomotion4d/tests/test_governance.py \
  local/hermes-scripts/tests/test_hindsight_mental_model_review_export.py \
  local/hermes-scripts/tests/test_hindsight_weekly_wrapper.py -q
test "$(readlink -f ~/.hermes/scripts/kanban_collaboration_evidence.py)" = \
  "/home/wyr/.hermes/hermes-agent-repo/local/hermes-scripts/kanban_collaboration_evidence.py"
```

Expected: PASS with no behavior change for the four existing models.

- [ ] **Step 5: Commit Task 2**

```bash
git add local/hermes-scripts/hindsight_daily_noagent.py \
  local/mental-models/egomotion4d/sources/derived-build-inputs.json \
  local/mental-models/egomotion4d/tests/test_governance.py
git commit -m "feat(mental-model): refresh collaboration evidence daily"
```

## Chunk 2: Model contract, runtime registration, and accepted exports

### Task 3: Add the fifth model's reproducible contract

**Files:**
- Create: `local/mental-models/egomotion4d/specs/kanban-collaboration.json`
- Create: `local/mental-models/egomotion4d/benchmark/questions-kanban-collaboration.json`
- Modify: `local/mental-models/egomotion4d/recreate_models.py`
- Modify: `local/mental-models/egomotion4d/tests/test_governance.py`
- Modify: `local/mental-models/egomotion4d/README.md`

- [ ] **Step 1: Write failing contract tests**

Assert the tracked spec has `output_language=zh-CN`, `max_tokens=4096`, required prefix, terminal marker `END_KANBAN_COLLABORATION`, inline source file, focused benchmark, and smoke IDs. Assert the benchmark includes exact positive/forbidden assertions for dispatch, notification, goal completion, workspace ownership, deterministic verdicts, route reset, safe FIFO delivery, reassign, and authority precedence. Assert `recreate_models.py --model-root local/mental-models/egomotion4d --only egomotion4d-kanban-collaboration --dry-run` selects exactly two physical IDs and makes no HTTP call.

- [ ] **Step 2: Run contract tests and confirm RED**

```bash
python3 -m pytest local/mental-models/egomotion4d/tests/test_governance.py \
  -k 'kanban_collaboration_contract or recreate_only_dry_run' -q
```

Expected: FAIL because the fifth contract and targeted bootstrap do not exist.

- [ ] **Step 3: Add the generation and benchmark contracts**

The source query must require the nine ordered sections in the approved design, current-only statements, explicit source precedence, no invented policy, and exact terminal marker. The benchmark's pass conditions must distinguish required from forbidden claims, including same-profile default-off, cross-profile default-on, durable queue on busy composer, and final reviewer acceptance as part of goal completion.

Refactor `recreate_models.py` only enough to:

```text
--only LOGICAL_ID   select one logical model
--dry-run           validate spec and print the two physical payloads
--model-root PATH   read specs/sources from PATH (default: live ~/.hermes root)
```

Keep the no-argument existing all-model behavior. Add the fifth model metadata and ensure bootstrap creates physical A/B but never writes an accepted revision.

- [ ] **Step 4: Run contract tests and dry-run**

```bash
python3 -m pytest local/mental-models/egomotion4d/tests/test_governance.py \
  -k 'kanban_collaboration_contract or recreate_only_dry_run' -q
python3 local/mental-models/egomotion4d/recreate_models.py \
  --model-root local/mental-models/egomotion4d \
  --only egomotion4d-kanban-collaboration --dry-run
```

Expected: PASS; exactly `...-a` and `...-b` are printed, with no API mutation.

- [ ] **Step 5: Commit Task 3**

```bash
git add local/mental-models/egomotion4d/specs/kanban-collaboration.json \
  local/mental-models/egomotion4d/benchmark/questions-kanban-collaboration.json \
  local/mental-models/egomotion4d/recreate_models.py \
  local/mental-models/egomotion4d/tests/test_governance.py \
  local/mental-models/egomotion4d/README.md
git commit -m "feat(mental-model): define kanban collaboration model"
```

### Task 4: Register, publish, and verify the live model

**Files:**
- Update runtime only: `~/.hermes/mental-models/egomotion4d/{registry.json,evidence_bundle.json,review_exports.json}`
- Deploy tracked files to runtime: `~/.hermes/mental-models/egomotion4d/{specs,sources,benchmark}`
- Generated wiki allowlist only: `/home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/{README.md,exports/current,exports/history,exports/review/current,exports/review/history}`

- [ ] **Step 1: Back up runtime state and record dirty-tree allowlists**

Copy the three runtime JSON owners to timestamped files under `~/.hermes/mental-models/egomotion4d/backups/`. Record `git status --short` for Hermes and wiki in `/tmp`; abort staging if any path outside the approved Hermes files or wiki allowlist becomes newly modified.

- [ ] **Step 2: Deploy tracked contracts and initialize runtime entries**

Copy the exact tracked spec, benchmark, manifest, and generated evidence snapshot into their runtime paths. Add one `evidence_bundle.per_model` entry whose sources are the generated evidence, spec, and benchmark. Add one registry entry with active slot `a`, physical IDs `...-a`/`...-b`, tags, max tokens, evidence identity, and `last_verdict: INITIAL`, but no accepted revision. Enable the logical ID in `review_exports.json` with no extra decision IDs. Write all three JSON owners atomically.

- [ ] **Step 3: Create only the two physical models**

```bash
python3 local/mental-models/egomotion4d/recreate_models.py \
  --only egomotion4d-kanban-collaboration
```

Expected: the Hindsight API creates or confirms only `egomotion4d-kanban-collaboration-a` and `-b`; existing four logical models are untouched.

- [ ] **Step 4: Run the canonical acceptance lifecycle**

```bash
python3 local/hermes-scripts/hindsight_daily_noagent.py \
  --mental-model-maintain --logical-id egomotion4d-kanban-collaboration
python3 local/hermes-scripts/hindsight_daily_noagent.py \
  --mental-model-adjudicate --logical-id egomotion4d-kanban-collaboration
python3 local/hermes-scripts/hindsight_daily_noagent.py \
  --mental-model-preflight egomotion4d-kanban-collaboration
```

Expected: maintain records an inactive-slot candidate transaction; adjudication returns PASS_PUBLISH and atomically records `accepted_revision`; preflight prints current content ending in `END_KANBAN_COLLABORATION`. If adjudication rejects, do not hand-edit accepted state: fix only evidenced contract/extraction defects and rerun the same lifecycle.

- [ ] **Step 5: Verify tests, registry identity, smoke, exports, and dirty boundaries**

```bash
python3 -m pytest local/hermes-scripts/tests/test_kanban_collaboration_evidence.py \
  local/mental-models/egomotion4d/tests/test_governance.py \
  local/hermes-scripts/tests/test_hindsight_mental_model_review_export.py -q
python3 local/hermes-scripts/hindsight_daily_noagent.py --mental-model-daily
```

Expected: all tests PASS; daily aggregate PASS; registry accepted content/evidence hashes match fetched content and current bundle; wiki index contains the fifth model; current and review exports exist; the four pre-existing accepted revisions remain unchanged. Compare current dirty status to the recorded allowlists and stage no unrelated wiki or `scripts/wechat_inject.py` change.

- [ ] **Step 6: Commit only maintained source and allowed accepted wiki outputs**

Commit Hermes source/tests separately from the wiki repository's generated allowlist. Do not push. Report both commit SHAs, accepted content/evidence SHA, current/review export paths, test totals, and any intentionally preserved dirty paths.
