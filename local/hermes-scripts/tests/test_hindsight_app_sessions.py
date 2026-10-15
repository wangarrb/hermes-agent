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
