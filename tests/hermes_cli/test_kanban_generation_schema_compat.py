"""Generation schema: additive, retryable and preserving unknown fork data."""
import sqlite3
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board-home"))
    path = tmp_path / "board.db"
    yield path
    kbc._INITIALIZED_PATHS.discard(str(path.resolve()))

@pytest.mark.parametrize("existing", [False, True])
def test_generation_init_preserves_old_rows_and_rolls_forward(board, existing):
    with sqlite3.connect(board) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                     "status TEXT NOT NULL, created_at INTEGER NOT NULL)")
        conn.execute("INSERT INTO tasks VALUES ('t_a', 'original', 'ready', 17)")
        conn.execute("CREATE TABLE retained_authorization (id INTEGER, receipt BLOB)")
        conn.execute("INSERT INTO retained_authorization VALUES (1, ?)", (b"\x00owner\xff",))
        if existing:
            conn.execute("ALTER TABLE tasks ADD COLUMN generation INTEGER NOT NULL DEFAULT 1")
            conn.execute("ALTER TABLE tasks ADD COLUMN rework_hold INTEGER NOT NULL DEFAULT 0")
            conn.execute("UPDATE tasks SET generation=8, rework_hold=1")
    kb.init_db(board)
    kb.init_db(board)
    conn = kbc.connect(board)
    try:
        task = kb.get_task(conn, "t_a")
        assert task.generation == (8 if existing else 1)
        assert (task.title, task.status, task.created_at) == ("original", "ready", 17)
        assert conn.execute("SELECT receipt FROM retained_authorization").fetchone()[0] == b"\x00owner\xff"
        if existing:
            assert conn.execute("SELECT rework_hold FROM tasks").fetchone()[0] == 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        info = {r["name"]: r for r in conn.execute("PRAGMA table_info(tasks)")}
        assert info["generation"]["notnull"] == 1
        assert str(info["generation"]["dflt_value"]) == "1"
    finally:
        conn.close()

def test_generation_init_resumes_after_add_column_failure(board, monkeypatch):
    with sqlite3.connect(board) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                     "status TEXT NOT NULL, created_at INTEGER NOT NULL)")
        conn.execute("INSERT INTO tasks VALUES ('t_b', 'retained', 'todo', 23)")
    original = kbc._add_column_if_missing
    def interrupted(conn, table, column, ddl):
        if table == "tasks" and column == "generation":
            raise sqlite3.OperationalError("injected generation migration failure")
        return original(conn, table, column, ddl)
    with monkeypatch.context() as patch:
        patch.setattr(kbc, "_add_column_if_missing", interrupted)
        with pytest.raises(sqlite3.OperationalError, match="injected generation"):
            kb.init_db(board)
    assert str(board.resolve()) not in kbc._INITIALIZED_PATHS
    kb.init_db(board)
    conn = kbc.connect(board)
    try:
        task = kb.get_task(conn, "t_b")
        assert (task.title, task.generation) == ("retained", 1)
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()

def test_fresh_generation_schema_and_reinit_keep_epoch(board):
    kb.init_db(board)
    conn = kbc.connect(board)
    try:
        tid = kb.create_task(conn, title="fresh")
        assert kb.get_task(conn, tid).generation == 1
        conn.execute("UPDATE tasks SET generation=9 WHERE id=?", (tid,))
    finally:
        conn.close()
    kb.init_db(board)
    conn = kbc.connect(board)
    try:
        assert kb.get_task(conn, tid).generation == 9
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
