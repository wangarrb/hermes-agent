"""The daily hook reports local observations without pretending collection ran."""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sidecar = importlib.import_module("hindsight_app_daily_sidecar")


def test_report_reader_uses_previous_utc8_day_deduplicates_and_never_writes(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()
    snapshot = {
        "source": "codex-app", "session_id": "codex-app-example", "source_session_id": "example",
        "messages": [
            {"origin_message_id": "before", "role": "user", "timestamp": "2026-10-13T15:59:59Z", "content": "Earlier visible fact"},
            {"origin_message_id": "inside", "role": "assistant", "timestamp": "2026-10-13T16:00:00Z", "content": "Current visible evidence"},
            {"origin_message_id": "after", "role": "user", "timestamp": "2026-10-14T16:00:00Z", "content": "Next day visible fact"},
        ],
    }
    for filename in ["session_example.json", "session_copy.json"]:
        (directory / filename).write_text(json.dumps(snapshot))
    chatgpt = {"source": "chatgpt-app", "source_session_id": "ordinary-example", "session_id": "chatgpt-app-example",
               "messages": [{"origin_message_id": "ordinary-item", "role": "user", "timestamp": "2026-10-14T01:00:00Z", "content": "Visible ordinary chat input"}]}
    (directory / "session_ordinary.json").write_text(json.dumps(chatgpt))
    coverage = {"listing_limit": 50, "ordinary_chatgpt_archived": "denied", "scope": "visible_subset"}
    (tmp_path / "collection-status.json").write_text(json.dumps({"last_success_at": "2026-10-14T16:00:00Z", "coverage": coverage}))
    files = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    files.update({p: p.read_bytes() for p in directory.iterdir()})
    result = sidecar.render_daily_sidecar("2026-10-15", archive_root=tmp_path)
    assert "2026-10-14（UTC+8）" in result
    assert "| codex-app | `example` | 0 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 |" in result
    assert "| chatgpt-app | `ordinary-example` | 1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 |" in result
    assert "inside" in result and "before" not in result and "after" not in result
    assert "采集时间已到达本日结束边界（不证明完整覆盖）" in result
    assert "不代表全账号或归档全集覆盖" in result
    assert json.dumps(coverage, ensure_ascii=False, sort_keys=True) in result
    assert all(path.read_bytes() == body for path, body in files.items())
    assert set(directory.iterdir()) == {directory / name for name in ["session_example.json", "session_copy.json", "session_ordinary.json"]}


def test_missing_stale_or_broken_archive_has_explicit_unknown_coverage(tmp_path):
    assert "归档未接入" in sidecar.render_daily_sidecar("2026-10-15", archive_root=tmp_path)
    directory = tmp_path / "sessions"
    directory.mkdir()
    assert "采集新鲜度：未知" in sidecar.render_daily_sidecar("2026-10-15", archive_root=tmp_path)
    status = tmp_path / "collection-status.json"
    status.write_text(json.dumps({"last_success_at": "2026-10-14T15:59:59Z"}))
    assert "未证明覆盖本日结束边界" in sidecar.render_daily_sidecar("2026-10-15", archive_root=tmp_path)
    (directory / "session_bad.json").write_text("{")
    failed = sidecar.render_daily_sidecar("2026-10-15", archive_root=tmp_path)
    assert "本节覆盖未知" in failed and "其他日报统计不受影响" in failed
    assert "| codex-app" not in failed
