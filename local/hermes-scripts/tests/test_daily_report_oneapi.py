"""One API report coverage and consistent-snapshot contracts."""
import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("daily_report_oneapi_test", ROOT / "daily_report.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def seed(con):
    con.execute("CREATE TABLE channels(id INTEGER PRIMARY KEY, models TEXT, status INTEGER)")
    con.execute("CREATE TABLE logs(created_at INTEGER, channel_id INTEGER, type INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER)")
    con.executemany("INSERT INTO channels VALUES(?,?,?)", [(9, "auto_cheep_model,gpt-6-luna", 1),
                        (8, "auto_cheep_model", 2), (184, " other, auto_cheep_model ", 1),
                        (10, "auto_cheep_model-preview", 1), (6, "other", 1)])
    con.executemany("INSERT INTO logs VALUES(?,?,?,?,?)", [(1000, 9, 2, 10, 2), (1001, 8, 2, 20, 3),
                         (1002, 184, 2, 30, 4), (1003, 10, 2, 99, 9), (1004, 6, 2, 99, 9),
                         (1005, 9, 5, 90, 9), (1100, 9, 2, 100, 10)])
    con.commit()


def test_dynamic_exact_alias_discovery_includes_disabled_and_future_channels(tmp_path):
    mod = load_module()
    db = tmp_path / "one-api.db"
    con = sqlite3.connect(db)
    seed(con)
    con.close()
    actual = mod.parse_oneapi_auto_cheep_counts(str(db), datetime.fromtimestamp(999), datetime.fromtimestamp(1100))
    assert actual == {"requests": 3, "prompt_tokens": 60, "completion_tokens": 9}


def test_report_snapshot_sees_committed_wal_without_docker_copy(tmp_path, monkeypatch):
    mod = load_module()
    db = tmp_path / "one-api.db"
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    seed(writer)
    # Keep the connection alive, with data still committed in WAL.
    assert Path(str(db) + "-wal").stat().st_size > 0
    monkeypatch.setattr(mod, "ONEAPI_DB_PATH", db, raising=False)
    monkeypatch.setattr(mod, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("do not copy a live main file")))
    try:
        actual = mod.oneapi_auto_cheep_counts(datetime.fromtimestamp(999), datetime.fromtimestamp(1100))
        assert actual == {"requests": 3, "prompt_tokens": 60, "completion_tokens": 9}
        assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        writer.close()
