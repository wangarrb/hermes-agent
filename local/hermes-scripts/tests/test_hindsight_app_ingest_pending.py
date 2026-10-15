"""An existing nightly chain can consume App exports without becoming a collector."""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
consumer = importlib.import_module("hindsight_app_ingest_pending")
app = importlib.import_module("hindsight_app_sessions")
local = importlib.import_module("hindsight_app_local_archive")
usage = importlib.import_module("hindsight_app_usage")
SID = "33333333-3333-3333-3333-333333333333"
SCOPE = {SID: {"kind": "codex", "parent_session_id": None}}


def _turn(tid, *, truncated=False, status="completed"):
    return {"id": tid, "status": status, "startedAt": "2026-10-14T01:00:00Z", "completedAt": "2026-10-14T01:01:00Z",
            "items": [{"id": tid + "-u", "type": "userMessage", "text": "Preserve the durable technical evidence for this experiment."},
                      {"id": tid + "-a", "type": "agentMessage", "text": "Complete public result with verified technical source evidence.", "truncated": truncated}]}


def _page(turns):
    return {"schemaVersion": 1, "thread": {"id": SID, "kind": "codex", "createdAt": "2026-08-01T00:00:00Z"},
            "page": {"hasMore": False}, "turns": turns}


def _input(archive, turns):
    (archive / "exports").mkdir(parents=True, exist_ok=True)
    (archive / "scope.json").write_text(json.dumps(SCOPE))
    (archive / "exports/page-fixture.json").write_text(json.dumps(_page(turns)))


def test_existing_snapshot_grows_once_with_real_local_ledger_and_dry_run_is_readonly(tmp_path):
    home, archive = tmp_path / "home", tmp_path / "archive"
    home.mkdir(); archive.mkdir()
    initial = app.convert_pages([_page([_turn("first")])], SCOPE, since=consumer.SINCE, allow_partial=True)
    app.sync_sessions(initial, archive / "sessions", write=True)
    local.archive_local(archive / "sessions", archive, home, SCOPE)
    _input(archive, [_turn("first"), _turn("second")])
    original_status = {"last_success_at": "2026-10-14T23:00:00+08:00", "coverage": {"listing_limit": 50, "archived_chatgpt": "denied"}}
    (archive / "collection-status.json").write_text(json.dumps(original_status))
    before = {p: p.read_bytes() for p in [archive / "collection-status.json", archive / "manifest.json", *list((archive / "sessions").iterdir())]}
    planned = consumer.consume_pending(archive, home)
    assert planned["status"] == "dry_run" and planned["new_messages"] == 2
    assert all(p.read_bytes() == body for p, body in before.items())
    first = consumer.consume_pending(archive, home, write=True)
    assert first["new_messages"] == 2 and first["new_local_documents"] == 1
    assert first["total_messages"] == 4
    repeated = consumer.consume_pending(archive, home, write=True)
    assert repeated["new_messages"] == repeated["new_local_documents"] == 0
    status = json.loads((archive / "collection-status.json").read_text())
    assert {k: status[k] for k in original_status} == original_status
    assert status["consumer"]["api_discovery_automated"] is False
    assert repeated["production_feed_written"] is False and repeated["hindsight_api_calls"] == 0
    assert not (home / "sessions").exists()
    assert list((archive / "daily/app-sessions").glob("*.md"))


def test_missing_exports_preserve_freshness_and_partial_overlap_is_local_only(tmp_path):
    home, archive = tmp_path / "home", tmp_path / "archive"
    archive.mkdir()
    original_status = {"coverage": {"visible_subset": 51, "listing_limit": 50}}
    (archive / "collection-status.json").write_text(json.dumps(original_status))
    missing = consumer.consume_pending(archive, home, write=True)
    status = json.loads((archive / "collection-status.json").read_text())
    assert missing["status"] == "collector_missing"
    assert "last_success_at" not in status and status["coverage"] == original_status["coverage"]
    assert not (archive / "local-capture").exists()
    assert "consumer" in status

    _input(archive, [_turn("partial", truncated=True), _turn("running", status="inProgress")])
    legacy = home / "sessions" / f"session_codex-old-{SID}.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")
    result = consumer.consume_pending(archive, home, write=True)
    assert result["partial_items"] == result["deferred_turns"] == 1
    assert result["legacy_cli_session_overlap"] == [SID]
    stored = json.loads(next((archive / "sessions").glob("session_*.json")).read_text())
    assert stored["message_count"] == 1
    assert stored["legacy_cli_session_overlap"] is True
    assert stored["legacy_cli_export_refs"] == [str(legacy)]
    assert "legacy_cli_session_overlap" in stored["tags"]
    assert stored["coverage"]["external_processing_allowed"] is False
    assert stored["coverage"]["partial_items"]
    assert legacy.read_text() == "{}" and list(legacy.parent.iterdir()) == [legacy]
    assert result["production_feed_written"] is False
    assert "last_success_at" not in json.loads((archive / "collection-status.json").read_text())
    daily = usage.render_app_daily([stored], "2026-10-14")
    assert "codex-app（CLI 同源）" in daily
    assert "CLI 同源或共享历史视图 1" in daily and "不得与旧 CLI 统计相加" in daily
