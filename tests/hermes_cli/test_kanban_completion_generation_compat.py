"""Completion generation fences; all external acceptance is synthetic test input."""
from pathlib import Path
from contextlib import contextmanager
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_pr_acceptance_store as store

@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "boards"))
    path = tmp_path / "board.db"
    kb.init_db(path)
    connection = kbc.connect(path)
    yield connection
    connection.close()
    kbc._INITIALIZED_PATHS.discard(str(path.resolve()))

def _events(conn, tid, kind):
    return conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind=?", (tid, kind)).fetchone()[0]

@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("receipt_ok", [False, True])
def test_completion_rejects_rework_during_acceptance(conn, monkeypatch, explicit, receipt_ok):
    tid = kb.create_task(conn, title="race", completion_contract="https://github.com/acme/repo/pull/7")
    old_run = kb.claim_task(conn, tid).current_run_id
    path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    def collect(contract, published_pr, *, assignee):
        # A real second SQLite writer commits while completion is outside its terminal txn.
        # Keep run/status identical: optional run CAS alone cannot save this case.
        rival = kbc.connect(path)
        try:
            with kb.write_txn(rival):
                rival.execute("UPDATE tasks SET generation=generation+1 WHERE id=?", (tid,))
        finally:
            rival.close()
        return {"ok": receipt_ok, "classification": "test-only", "detail": "synthetic", "recovery": "retry"}
    monkeypatch.setattr(store, "collect_acceptance", collect)
    kwargs = {"expected_generation": 1} if explicit else {}
    assert kb.complete_task(conn, tid, result="old result", expected_run_id=old_run, **kwargs) is False
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.current_run_id == old_run
    assert task.generation == 2 and task.result is None
    assert _events(conn, tid, "pr_acceptance") == 0
    assert _events(conn, tid, "completed") == 0
    assert conn.execute("SELECT ended_at FROM task_runs WHERE id=?", (old_run,)).fetchone()[0] is None
    assert conn.execute("SELECT last_failure_error FROM tasks WHERE id=?", (tid,)).fetchone()[0] is None

def test_default_kwargs_reject_generation_changed_before_terminal_sql(conn, monkeypatch):
    tid = kb.create_task(conn, title="terminal fence")
    original = kb._claim_is_live
    def mutate_in_transaction(row):
        # Deliberate same-transaction interposition, not a claim of another writer
        # entering BEGIN IMMEDIATE. This falsifies a precheck-only implementation.
        conn.execute("UPDATE tasks SET generation=generation+1 WHERE id=?", (tid,))
        return original(row)
    monkeypatch.setattr(kb, "_claim_is_live", mutate_in_transaction)
    assert kb.complete_task(conn, tid, result="stale") is False
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.get_task(conn, tid).generation == 2
    assert _events(conn, tid, "completed") == 0

@pytest.mark.parametrize("force", [False, True])
def test_stale_explicit_generation_is_not_overridden_by_force(conn, monkeypatch, force):
    tid = kb.create_task(conn, title="stale explicit")
    conn.execute("UPDATE tasks SET generation=3 WHERE id=?", (tid,))
    def forbidden(*args, **kwargs):
        pytest.fail("stale callback reached acceptance")
    monkeypatch.setattr(store, "prepare_acceptance", forbidden)
    assert kb.complete_task(conn, tid, result="stale", expected_generation=2, force=force) is False
    assert kb.get_task(conn, tid).status == "ready"

@pytest.mark.parametrize("mode", ["default", "matching", "review"])
def test_normal_completion_preserves_metadata_and_default_kwargs(conn, mode):
    tid = kb.create_task(conn, title="ordinary", assignee="coder")
    metadata = {"independent_review_outcome": "TIMEOUT", "independent_review_note": "reviewer_a reached configured deadline", "opaque": {"a": 1}}
    kwargs = {"expected_generation": 1} if mode == "matching" else {}
    if mode == "review":
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        assert kb.complete_task(conn, tid, **kwargs) is True
    else:
        assert kb.complete_task(conn, tid, result="done", metadata=metadata, **kwargs) is True
        import json
        stored = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()[0])
        assert stored == metadata
    assert kb.get_task(conn, tid).status == "done"
    assert kb.get_task(conn, tid).generation == 1
    assert _events(conn, tid, "completed") == 1


def _completion_state(conn, tid):
    task = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
    runs = [dict(row) for row in conn.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (tid,))]
    events = [dict(row) for row in conn.execute(
        "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]
    return task, runs, events


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("interleave", ["before_first_snapshot", "before_binding_txn"])
def test_old_publication_cannot_bind_replacement_epoch(conn, monkeypatch, explicit, interleave):
    # Repository-only is the canonical native publication contract, not a new URL syntax.
    tid = kb.create_task(conn, title="publication race", completion_contract="acme/repo")
    old_run = kb.claim_task(conn, tid).current_run_id
    path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    before = _completion_state(conn, tid)
    replacement = []
    collected = []
    original_snapshot = store._snapshot
    original_txn = store.write_txn

    def advance_once():
        assert not replacement
        rival = kbc.connect(path)
        try:
            with kb.write_txn(rival):
                rival.execute("UPDATE tasks SET generation=generation+1 WHERE id=?", (tid,))
            replacement.append(_completion_state(rival, tid))
        finally:
            rival.close()
        expected_task = {**before[0], "generation": 2}
        assert replacement[0] == (expected_task, before[1], before[2])

    def snapshot_after_advance(connection, task_id):
        # complete has already captured N; advance before prepare's FIRST snapshot.
        if task_id == tid and not replacement:
            advance_once()
        return original_snapshot(connection, task_id)

    @contextmanager
    def txn_after_advance(connection, **kwargs):
        # Distinct race: prepare already snapshotted N, but has not entered binding txn.
        if not replacement:
            advance_once()
        with original_txn(connection, **kwargs):
            yield

    def forbidden_collect(*args, **kwargs):
        collected.append((args, kwargs))
        pytest.fail("stale publication reached external acceptance collection")

    monkeypatch.setattr(store, "collect_acceptance", forbidden_collect)
    if interleave == "before_first_snapshot":
        monkeypatch.setattr(store, "_snapshot", snapshot_after_advance)
    else:
        monkeypatch.setattr(store, "write_txn", txn_after_advance)
    kwargs = {"expected_generation": 1} if explicit else {}
    assert kb.complete_task(conn, tid, result="old callback result", expected_run_id=old_run,
                            metadata={"published_pr": "https://github.com/acme/repo/pull/7",
                                      "old_callback": True}, **kwargs) is False
    assert len(replacement) == 1 and collected == []
    task, runs, events = _completion_state(conn, tid)
    assert task["generation"] == 2
    assert task["completion_contract"] == "acme/repo"
    assert task["status"] == "running" and task["current_run_id"] == old_run
    assert task["result"] is None and task["last_failure_error"] is None
    assert runs[0]["ended_at"] is None
    assert _events(conn, tid, "pr_acceptance") == 0
    assert _events(conn, tid, "completed") == 0
    # Includes all receipt/event rows, run metadata/outcome and task failure/result fields.
    assert (task, runs, events) == replacement[0]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("receipt_ok", [False, True])
def test_same_epoch_publication_still_binds_and_audits(conn, monkeypatch, explicit, receipt_ok):
    tid = kb.create_task(conn, title="native publication", completion_contract="acme/repo")
    run_id = kb.claim_task(conn, tid).current_run_id
    published_pr = "https://github.com/acme/repo/pull/7"
    calls = []

    def collect(contract, publication, *, assignee):
        calls.append((contract, publication, assignee))
        return {"ok": receipt_ok, "classification": "test-only", "detail": "synthetic",
                "recovery": "retry"}

    monkeypatch.setattr(store, "collect_acceptance", collect)
    kwargs = {"expected_generation": 1} if explicit else {}
    assert kb.complete_task(conn, tid, result="same epoch", expected_run_id=run_id,
                            metadata={"published_pr": published_pr}, **kwargs) is receipt_ok
    task = kb.get_task(conn, tid)
    assert task.generation == 1 and task.completion_contract == published_pr
    assert calls == [(published_pr, published_pr, None)]
    assert _events(conn, tid, "pr_acceptance") == 1
    assert _events(conn, tid, "completed") == int(receipt_ok)
    run = conn.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone()
    if receipt_ok:
        assert task.status == "done" and task.result == "same epoch"
        assert run["ended_at"] is not None
        assert task.last_failure_error is None
    else:
        assert task.status == "running" and task.result is None
        assert task.current_run_id == run_id and run["ended_at"] is None
        assert "PR acceptance test-only" in task.last_failure_error


def test_prepare_acceptance_retains_four_positional_argument_compatibility(conn, monkeypatch):
    tid = kb.create_task(conn, title="internal compatibility", completion_contract="acme/repo")
    published_pr = "https://github.com/acme/repo/pull/7"
    receipt = {"ok": True, "classification": "test-only", "detail": "synthetic", "recovery": "retry"}
    monkeypatch.setattr(store, "collect_acceptance", lambda *args, **kwargs: receipt)
    snapshot, actual_receipt = store.prepare_acceptance(conn, tid, None, {"published_pr": published_pr})
    assert snapshot == (None, "ready", published_pr, 1)
    assert actual_receipt is receipt
    assert kb.get_task(conn, tid).completion_contract == published_pr
    assert _events(conn, tid, "pr_acceptance") == 0
