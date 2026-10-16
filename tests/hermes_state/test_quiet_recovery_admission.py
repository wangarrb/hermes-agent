"""C02 (receipt r4): minimal native quiet-recovery admission + postcommit FTS defer.

RED contract from docs/schema31-receipt-v2-design-r4-amendments.md section 3
and the effective r4 machine spec (protocol.recovery.quiet_window):

  * the ONLY way into the finite recovery mode is a private continuation-only
    admission entry -- ``SessionDB.open_quiet_recovery_admission`` -- which
    opens an existing settled store with a no-create ``mode=rw`` connection,
    must not run the ordinary schema init, generation mint or automatic
    repair, and must not leave any logical persistent delta (schema rows,
    state_meta, application_id, journal_mode unchanged);
  * a missing file, an empty file or a foreign (non-Hermes-schema) sqlite file
    is refused without being created, repaired or modified;
  * generic writers (set_meta, append_message) and the FTS maintenance
    entries (optimize_fts/rebuild_fts) must refuse on the quiet handle;
    ordinary SessionDB handles must not gain the continuation entry;
  * the approved epoch_continuation write goes through the existing
    ``_execute_write`` transaction core, increments ``_write_count`` and is
    token-idempotent (same token -> verified readback, no second insert;
    same key with different bytes -> refused);
  * at the native maintenance cadence the handle only marks a postcommit FTS
    defer (``_fts_deferred_due``) instead of executing usermerge/merge, and
    ``_fts_enabled`` keeps its real value.

RED against the unmodified b26ba5c4 tree: none of this seam exists. Real
SQLite fixtures only; no mocks.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

import pytest

import hermes_state
from hermes_state import SessionDB

SID = "owner-01"


def _settle(tmp_path, sid=SID):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(sid, "cli", model="m")
    db.append_message(session_id=sid, role="user", content="hello")
    db.set_meta("probe/key", "value")
    db.close()
    return tmp_path / "state.db"


def _fingerprint(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        schema = [tuple(r) for r in conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY name")]
        meta = [tuple(r) for r in conn.execute("SELECT key, value FROM state_meta ORDER BY key")]
        return {"schema": schema, "meta": meta,
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
                "application_id": conn.execute("PRAGMA application_id").fetchone()[0],
                "user_version": conn.execute("PRAGMA user_version").fetchone()[0]}
    finally:
        conn.close()


def _continuation_request(**overrides):
    request = {"kind": "epoch_continuation",
               "token": str(uuid.uuid4()),
               "proof_sha256": "1" * 64,
               "old_epoch": "sqlite:" + "a" * 64,
               "new_epoch": "sqlite:" + "b" * 64,
               "old_generation": 1,
               "old_head_sha256": "2" * 64,
               "old_head_token_sha256": "3" * 64,
               "boundary_roster_sha256": "4" * 64,
               "alias_inventory_sha256": "5" * 64}
    request.update(overrides)
    return request


def _read_quiet(path, request=None):
    return SessionDB.open_quiet_recovery_admission(
        db_path=path, init_request=request or _continuation_request())


def test_quiet_admission_refuses_missing_empty_and_foreign_files(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises((ValueError, sqlite3.OperationalError, FileNotFoundError)):
        _read_quiet(missing)
    assert not missing.exists()

    empty = tmp_path / "empty.db"
    empty.touch()
    before = empty.read_bytes()
    with pytest.raises((ValueError, sqlite3.OperationalError, sqlite3.DatabaseError)):
        _read_quiet(empty)
    assert empty.read_bytes() == before

    foreign = tmp_path / "foreign.db"
    conn = sqlite3.connect(foreign)
    conn.execute("CREATE TABLE t (x)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    before = foreign.read_bytes()
    with pytest.raises(ValueError):
        _read_quiet(foreign)
    assert foreign.read_bytes() == before


def test_quiet_admission_open_leaves_no_logical_persistent_delta(tmp_path):
    path = _settle(tmp_path)
    before = _fingerprint(path)
    handle = _read_quiet(path)
    try:
        assert handle._quiet_recovery_admission is True
        assert _fingerprint(path) == before
    finally:
        handle.close()
    # An ordinary handle never carries the admission capability.
    db = SessionDB(db_path=path)
    try:
        assert getattr(db, "_quiet_recovery_admission", False) is False
        ordinary_entry = getattr(db, "append_epoch_continuation", None)
        if ordinary_entry is not None:  # if the name exists at all, it must refuse
            with pytest.raises((RuntimeError, ValueError, PermissionError)):
                ordinary_entry(_continuation_request())
    finally:
        db.close()


def test_generic_writers_and_fts_maintenance_refuse_on_the_quiet_handle(tmp_path):
    path = _settle(tmp_path)
    handle = _read_quiet(path)
    try:
        with pytest.raises((RuntimeError, ValueError, PermissionError)):
            handle.set_meta("generic/key", "value")
        with pytest.raises((RuntimeError, ValueError, PermissionError)):
            handle.append_message(session_id=SID, role="user", content="nope")
        with pytest.raises((RuntimeError, ValueError, PermissionError)):
            handle.optimize_fts()
        with pytest.raises((RuntimeError, ValueError, PermissionError)):
            handle.rebuild_fts()
        # Refusals must not have written anything.
        assert "generic/key" not in dict(
            (k, v) for k, v in handle._conn.execute("SELECT key, value FROM state_meta"))
    finally:
        handle.close()


def test_approved_continuation_write_is_token_idempotent_and_counted(tmp_path):
    path = _settle(tmp_path)
    handle = _read_quiet(path)
    try:
        before = handle._write_count
        first = handle.append_epoch_continuation(_continuation_request())
        assert handle._write_count == before + 1
        assert isinstance(first, dict)
        assert first.get("token") and first.get("operation_sha256")
        # Same token: verified readback, no second insert, counter unchanged.
        replay = handle.append_epoch_continuation(_continuation_request(
            token=first["token"]))
        assert replay == first
        assert handle._write_count == before + 1
        # Same key (token) with different bytes: refused, still no write.
        with pytest.raises((ValueError, RuntimeError)):
            handle.append_epoch_continuation(_continuation_request(
                token=first["token"], new_epoch="sqlite:" + "c" * 64))
        assert handle._write_count == before + 1
    finally:
        handle.close()


def test_cadence_hit_marks_postcommit_fts_defer_without_running_usermerge(tmp_path):
    path = _settle(tmp_path)
    threshold = getattr(hermes_state, "QUIET_RECOVERY_FTS_DEFER_EVERY", 1000)
    handle = _read_quiet(path)
    try:
        assert handle._fts_enabled is True
        first = handle.append_epoch_continuation(_continuation_request())
        assert getattr(handle, "_fts_deferred_due", False) is False
        # Walk one approved continuation exactly onto the cadence boundary.
        handle._write_count = threshold - 1
        fts_before = {name: count for name, count in handle._conn.execute(
            "SELECT name, (SELECT count(*) FROM sqlite_schema WHERE name = t.name) "
            "FROM sqlite_schema t WHERE name LIKE 'messages_fts%'")}
        # A fresh token (new operation) walks the counter onto the cadence.
        handle.append_epoch_continuation(_continuation_request())
        assert handle._write_count == threshold
        assert handle._fts_deferred_due is True          # deferred, not executed
        assert handle._fts_enabled is True               # real value stays on
        fts_after = {name: count for name, count in handle._conn.execute(
            "SELECT name, (SELECT count(*) FROM sqlite_schema WHERE name = t.name) "
            "FROM sqlite_schema t WHERE name LIKE 'messages_fts%'")}
        assert fts_after == fts_before                  # no FTS shadow repair ran
        # While the defer is due, FTS write entries stay refused.
        with pytest.raises((RuntimeError, ValueError, PermissionError)):
            handle.rebuild_fts()
    finally:
        handle.close()
