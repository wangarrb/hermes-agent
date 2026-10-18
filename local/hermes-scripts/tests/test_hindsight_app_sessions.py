"""Supported App exports reuse the real arrival scanner and ledger locally."""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
app = importlib.import_module("hindsight_app_sessions")
sources = importlib.import_module("hindsight_incremental_sources")
ledger_module = importlib.import_module("hindsight_incremental_ledger")

PARENT = "11111111-1111-1111-1111-111111111111"
CHILD = "22222222-2222-2222-2222-222222222222"
SCOPE = {
    PARENT: {"parent_session_id": None},
    CHILD: {"parent_session_id": PARENT, "relation_basis": "explicit_authorized_scope"},
}


def _turn(tid, *, status="completed", day="2026-10-14"):
    return {
        "id": tid, "status": status,
        "startedAt": f"{day}T01:00:00Z", "completedAt": f"{day}T01:01:00Z" if status == "completed" else None,
        "items": [
            {"id": tid + "-u", "type": "userMessage", "text": f"Please preserve the durable technical evidence for experiment {tid}."},
            {"id": tid + "-a", "type": "agentMessage", "text": f"The experiment {tid} has verified independent source identity and timestamp evidence."},
            {"id": tid + "-tool", "type": "functionCall", "text": "private command omitted"},
        ],
    }


def _page(sid, turns, *, more=False):
    return {
        "schemaVersion": 1,
        "thread": {"id": sid, "kind": "codex", "createdAt": "2026-10-01T00:00:00Z"},
        "page": {"hasMore": more}, "turns": turns,
    }


def _capture(ledger, home, **scan_options):
    return [batch for snapshot in sources.scan_source_snapshots(home, **scan_options) for batch in ledger.capture_snapshot(snapshot)]


def test_pages_resume_into_real_ledger_without_replay_or_compression_lineage(tmp_path):
    home = tmp_path / "hermes"
    output = home / "sessions"
    output.mkdir(parents=True)
    # The production reader uses physical baseline fences. Exercise that path
    # with a real JSON-only empty profile, then admit only post-baseline items.
    baseline = sources.capture_physical_baseline(home, bank="fixture")
    physical = ledger_module.IncrementalLedger(tmp_path / "physical-ledger", "fixture", create=True)
    from hindsight_incremental_delivery import _freeze
    artifact_path, artifact_hash = _freeze(physical.root, baseline, "physical-baseline")
    reference = {
        "schema_version": sources.PHYSICAL_BASELINE_SCHEMA, "bank": "fixture",
        "hermes_home": str(home), "artifact_path": artifact_path, "artifact_hash": artifact_hash,
    }
    physical.initialize_legacy(sources.physical_baseline_snapshots(baseline), {}, [],
                              confirm="initialize-hindsight-incremental", physical_baseline=reference)
    ledger = ledger_module.IncrementalLedger(tmp_path / "ledger", "fixture", create=True)
    ledger.initialize_legacy([], {}, [], confirm="initialize-hindsight-incremental")
    parent_page = _page(PARENT, [_turn("parent")])
    child_page = _page(CHILD, [_turn("done"), _turn("running", status="inProgress")], more=True)
    sessions = app.convert_pages([parent_page, child_page, copy.deepcopy(child_page)], SCOPE)
    child = next(s for s in sessions if s["source_session_id"] == CHILD)
    assert child["app_parent_session_id"] == PARENT
    assert "parent_session_id" not in child  # App delegation is not compression ancestry.
    assert child["coverage"]["deferred_turns"] == ["running"]
    first_sync = app.sync_sessions(sessions, output, write=True)
    assert sum(row["new_messages"] for row in first_sync) == 4
    snapshots = sources.scan_source_snapshots(home)
    assert len(snapshots) == 2
    assert len({s["source_key"] for s in snapshots}) == 2
    first = _capture(ledger, home)
    assert sum(len(b["source_event_ids"]) for b in first) == 4
    physical_first = _capture(physical, home, physical_baseline=baseline)
    assert sum(len(b["source_event_ids"]) for b in physical_first) == 4
    assert _capture(physical, home, physical_baseline=baseline) == []
    assert _capture(ledger, home) == []
    assert all(row["new_messages"] == 0 for row in app.sync_sessions(sessions, output, write=True))
    assert _capture(ledger, home) == []
    # A later production cutover can seal a now-existing App snapshot. Future
    # backfill must append without rewriting that sealed raw JSON prefix.
    sealed_baseline = sources.capture_physical_baseline(home, bank="fixture")

    # A later fetch completes the deferred turn and discovers older history.
    resumed = app.convert_pages([
        _page(CHILD, [_turn("running"), _turn("late", day="2026-10-02")]),
        child_page,  # An older overlapping page must not re-defer completed work.
    ], SCOPE)
    report = app.sync_sessions(resumed, output, write=True)[0]
    assert report["new_messages"] == 4
    assert report["total_messages"] == 6
    assert report["coverage"]["deferred_turns"] == []
    grown = _capture(ledger, home)
    assert sum(len(b["source_event_ids"]) for b in grown) == 4
    physical_grown = _capture(physical, home, physical_baseline=baseline)
    assert sum(len(b["source_event_ids"]) for b in physical_grown) == 4
    assert _capture(physical, home, physical_baseline=baseline) == []
    assert _capture(ledger, home) == []
    sealed_delta = sources.scan_source_snapshots(home, physical_baseline=sealed_baseline)
    assert sum(len(s['messages']) for s in sealed_delta) == 4
    payload = json.loads(Path(grown[0]["payload_path"]).read_text())
    assert any(m["timestamp"].startswith("2026-10-02") for m in payload["messages"])
    assert all(f"parent={PARENT}" in m["content"] and f"session={CHILD}" in m["content"] for m in payload["messages"])
    assert len(sources.scan_source_snapshots(home)) == 2


def test_scope_truncation_rewrites_and_cli_overlap_fail_closed(tmp_path):
    page = _page(CHILD, [_turn("stable")])
    with pytest.raises(ValueError, match="scope"):
        app.convert_pages([page], {PARENT: SCOPE[PARENT]})
    truncated = copy.deepcopy(page)
    truncated["turns"][0]["items"][1]["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        app.convert_pages([truncated], SCOPE)
    rewritten = copy.deepcopy(page)
    rewritten["turns"][0]["items"][1]["text"] = "Rewritten technical evidence must require review before reuse."
    with pytest.raises(ValueError, match="changed"):
        app.convert_pages([page, rewritten], SCOPE)
    sessions = app.convert_pages([page], SCOPE)
    output = tmp_path / "sessions"
    app.sync_sessions(sessions, output, write=True)
    target = output / f"session_codex-app-{CHILD}.json"
    before = target.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        app.sync_sessions(app.convert_pages([rewritten], SCOPE), output, write=True)
    assert target.read_bytes() == before
    legacy = output / f"session_codex-2026-10-14T00-00-{CHILD}.json"
    legacy.write_text("{}")
    with pytest.raises(ValueError, match="CLI/app overlap"):
        app.sync_sessions(sessions, output, write=True)
    assert target.read_bytes() == before


def test_chatgpt_install_boundary_partial_resume_local_archive_and_daily(tmp_path):
    archive_module = importlib.import_module('hindsight_app_local_archive')
    scope = {CHILD: {'kind': 'chatgpt', 'parent_session_id': None}}
    old = _turn('old', day='2026-09-08')
    edge = _turn('edge', day='2026-09-09')
    edge['startedAt'] = '2026-09-08T15:59:59Z'
    edge['completedAt'] = '2026-09-08T16:00:00Z'
    current = _turn('current', day='2026-10-15')
    current['items'][1]['text'] = 'Long public input preserved locally. ' * 2000
    partial = copy.deepcopy(current)
    partial['items'][1]['truncated'] = True
    page = _page(CHILD, [old, edge, partial])
    page['thread']['kind'] = 'chatgpt'
    page['thread']['createdAt'] = '2026-09-01T00:00:00Z'
    projected = copy.deepcopy(page)
    projected['turns'][0]['items'][0].pop('text')
    projected['turns'][0]['items'][0]['content'] = [
        {'type': 'text', 'text': 'Preserve the visible user text.', 'metadata': 'not exported'},
        {'type': 'image', 'url': 'https://private.invalid/signed-attachment'},
    ]
    assert app.project_page(projected)['turns'][0]['items'][0]['content'] == [
        {'type': 'text', 'text': 'Preserve the visible user text.'}]
    convert = lambda p: app.convert_pages([p], scope, since='2026-09-09T00:00:00+08:00', allow_partial=True)
    first = convert(page)
    assert first[0]['source'] == 'chatgpt-app'
    assert len(first[0]['messages']) == 2  # edge assistant + current user
    assert first[0]['coverage']['before_boundary_items'] == 3
    assert len(first[0]['coverage']['partial_items']) == 1
    first[0]['legacy_cli_session_overlap'] = True
    first[0]['legacy_cli_export_refs'] = ['legacy-visible-export.json']
    source = tmp_path / 'staged'
    app.sync_sessions(first, source, write=True)
    # A separate newer head must not clear the unresolved partial item.
    head = _page(CHILD, [_turn('unrelated-head', status='inProgress')])
    head['thread']['kind'] = 'chatgpt'
    head['thread']['createdAt'] = page['thread']['createdAt']
    app.sync_sessions(convert(head), source, write=True)
    held = json.loads((source / f'session_chatgpt-app-{CHILD}.json').read_text())
    assert len(held['coverage']['partial_items']) == 1
    archive = tmp_path / 'archive'
    home = tmp_path / 'production'
    before, sessions = archive_module.archive_local(source, archive, home, scope)
    assert before['new_messages'] == 2 and before['new_local_documents'] == 1
    assert before['hindsight_api_calls'] == before['embedding_calls'] == before['summary_llm_calls'] == 0
    page['turns'][-1] = current
    app.sync_sessions(convert(page), source, write=True)
    grown, sessions = archive_module.archive_local(source, archive, home, scope)
    assert grown['new_messages'] == 1 and grown['new_local_documents'] == 1
    assert len(sessions[0]['messages']) == 3
    assert sessions[0]['coverage']['partial_items'] == []
    assert sessions[0]['coverage']['deferred_turns'] == ['unrelated-head']
    assert sessions[0]['legacy_cli_session_overlap'] is True
    assert sessions[0]['legacy_cli_export_refs'] == ['legacy-visible-export.json']
    replay, _ = archive_module.archive_local(source, archive, home, scope)
    assert replay['new_messages'] == replay['new_local_documents'] == 0
    daily = tmp_path / 'daily'
    daily.mkdir()
    canonical = daily / '2026-09-10.md'
    canonical.write_text('Existing base report\n')
    out = archive_module.daily_local(sessions, tmp_path / 'evidence', daily, today='2026-10-15')
    assert canonical.read_text().startswith('Existing base report\n')
    assert '| chatgpt-app' in canonical.read_text()
    assert (daily / 'app-sessions/2026-10-15.partial.md').exists()
    assert not (daily / '2026-10-16.md').exists()
    repeated = archive_module.daily_local(sessions, tmp_path / 'evidence', daily, today='2026-10-15')
    assert not any(row['changed'] for row in repeated)


def test_fuller_completed_turn_can_add_missing_user_without_rewriting_prefix(tmp_path):
    full = _page(CHILD, [_turn('fuller')])
    partial_view = copy.deepcopy(full)
    partial_view['turns'][0]['items'] = partial_view['turns'][0]['items'][1:]
    output = tmp_path / 'home/sessions'
    app.sync_sessions(app.convert_pages([partial_view], SCOPE), output, write=True)
    before = json.loads((output / f'session_codex-app-{CHILD}.json').read_text())['messages']
    baseline = sources.capture_physical_baseline(output.parent, bank='fixture')
    expanded = app.convert_pages([full, partial_view], SCOPE)
    assert app.sync_sessions(expanded, output, write=True)[0]['new_messages'] == 1
    saved = json.loads((output / f'session_codex-app-{CHILD}.json').read_text())['messages']
    assert saved[:len(before)] == before
    assert saved[-1]['role'] == 'user'
    delta = sources.scan_source_snapshots(output.parent, physical_baseline=baseline)
    assert sum(len(s['messages']) for s in delta) == 1
    assert app.sync_sessions(expanded, output, write=True)[0]['new_messages'] == 0
