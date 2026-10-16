"""C02 (receipt r4): native quiet-recovery admission + postcommit FTS defer.

Contract from docs/schema31-receipt-v2-design-r4-amendments.md section 3 and the
effective r4 machine spec (protocol.recovery.quiet_window), as repaired after the
independent implementation review of 15a03d07:

  * the ONLY way into the finite recovery mode is a private continuation-only
    admission entry -- ``SessionDB.open_quiet_recovery_admission`` -- which
    opens an existing settled store with a no-create ``mode=rw`` connection,
    must not run the ordinary schema init, generation mint or automatic
    repair, and must not leave any logical persistent delta (schema rows,
    state_meta, application_id, journal_mode unchanged);
  * fail-closed settled admission: the store must already carry the settled
    schema version, the finished message_uid backfill, its generation stamp
    and a settled FTS layout, and the read-only FTS/tokenizer probes must
    succeed -- otherwise the open is refused and nothing is "repaired first";
  * a missing file, an empty file or a foreign (non-Hermes-schema) sqlite file
    is refused without being created, repaired or modified;
  * generic writers (set_meta, append_message) and the FTS maintenance
    entries (optimize_fts/rebuild_fts) must refuse on the quiet handle;
    ordinary SessionDB handles must not gain the continuation entry;
  * the handle only writes continuations of its approved init request: the
    epoch-transition bindings (kind/owner_session_id/proof_sha256/old_epoch/
    new_epoch/boundary_roster_sha256/alias_inventory_sha256) must equal the
    approved request, and the owner's stored protected head
    (``sessions.model_config._message_copy_receipts_v2``) must match the old
    coordinates (active_epoch/generation/head_sha256 and the canonical head
    token hash) or be absent for a boundary owner's first receipt;
  * the approved epoch_continuation write runs on the existing
    ``SessionDB._execute_write`` transaction core (BEGIN IMMEDIATE, commit,
    rollback, jittered lock retry, replaced-file fence) through the quiet
    hook set (no reopen, no checkpoint, no FTS usermerge/merge, no fail-open
    repair); it increments ``_write_count`` once per committed operation and
    is token-idempotent (same token -> verified readback, no second insert,
    counter unchanged; same key with different bytes -> refused), landing
    under the canonical token-index key ``receipt3/t/<SIDhex>/<opuuid>``;
  * at the native maintenance cadence the handle only marks a postcommit FTS
    defer (``_fts_deferred_due``) instead of executing usermerge/merge, and
    ``_fts_enabled``/``_trigram_available`` keep their probed real values.

Named C02 approval-surface items that stay deferred (not silently claimed):
the certificate/pin binding of the finite roster, quiet reopen/partial-crash
flows and the post-release bounded FTS pass. The real continuation payload
(operation header/journal/pages/range checkpoint/new protected head under
oldraw CAS) is implemented by milestone 2A and asserted here and in
test_receipt_v3_range_checkpoint.py.

RED against the 15a03d07 minimal slice: these repaired expectations are not
met by it. Real SQLite fixtures only; no mocks.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid

import pytest

import hermes_state
from hermes_state import SessionDB

SID = "owner-01"
OLD_EPOCH = "sqlite:" + "a" * 64
NEW_EPOCH = "sqlite:" + "b" * 64


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


def _mutate(path, sql, params=()):
    """Direct fixture mutation (settled test helper): real sqlite, no mocks."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _canonical(x) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8", "strict")


def _canon_sha(x) -> str:
    return hashlib.sha256(_canonical(x)).hexdigest()


def _continuation_request(**overrides):
    request = {"kind": "epoch_continuation",
               "token": str(uuid.uuid4()),
               "owner_session_id": SID,
               "proof_sha256": "1" * 64,
               "old_epoch": OLD_EPOCH,
               "new_epoch": NEW_EPOCH,
               "old_generation": 1,
               "old_head_sha256": "2" * 64,
               "old_head_token_sha256": "3" * 64,
               "boundary_roster_sha256": "4" * 64,
               "alias_inventory_sha256": "5" * 64}
    request.update(overrides)
    return request


def _matching_request(head, **overrides):
    """An approved request whose oldraw coordinates match *head* exactly."""
    request = _continuation_request(old_epoch=head["active_epoch"],
                                    old_generation=head["generation"],
                                    old_head_sha256=head["head_sha256"],
                                    old_head_token_sha256=_canon_sha(head))
    request.update(overrides)
    return request


def _receipt_snapshot(handle):
    return sorted((row[0], row[1]) for row in handle._conn.execute(
        "SELECT key, value FROM state_meta WHERE key LIKE 'receipt3/%'"))


def _head_raw(handle, sid=SID):
    return handle._conn.execute(
        "SELECT model_config FROM sessions WHERE id = ?", (sid,)).fetchone()[0]


def _read_quiet(path, request=None):
    request = request if request is not None else _continuation_request()
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=request)
    return handle, request


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


@pytest.mark.parametrize("mutation", [
    ("UPDATE schema_version SET version = 30", ()),                       # unsettled schema
    ("DELETE FROM state_meta WHERE key = 'message_uid_backfill'", ()),    # UID backfill unsettled
    ("INSERT INTO state_meta (key, value) VALUES ('fts_rebuild_high_water', '5')", ()),
    ("INSERT INTO state_meta (key, value) VALUES ('fts_rebuild_progress', '0')", ()),
    ("DELETE FROM state_meta WHERE key = 'db_file_generation'", ()),      # generation stamp missing
])
def test_quiet_admission_refuses_unsettled_stores(tmp_path, mutation):
    path = _settle(tmp_path)
    _mutate(path, *mutation)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _read_quiet(path)
    assert path.read_bytes() == before


def test_quiet_admission_refuses_a_store_without_fts_support(tmp_path):
    path = _settle(tmp_path)
    _mutate(path, "DROP TABLE messages_fts")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _read_quiet(path)
    assert path.read_bytes() == before


def test_quiet_admission_open_leaves_no_logical_persistent_delta(tmp_path):
    path = _settle(tmp_path)
    before = _fingerprint(path)
    handle, _request = _read_quiet(path)
    try:
        assert handle._quiet_recovery_admission is True
        assert handle._fts_enabled is True          # probed from the store, not hardcoded
        assert handle._trigram_available is True
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


def test_quiet_admission_probes_real_fts_flags(tmp_path):
    path = _settle(tmp_path)
    _mutate(path, "DROP TABLE messages_fts_trigram")
    handle, _request = _read_quiet(path)
    try:
        assert handle._fts_enabled is True           # base index probed True
        assert handle._trigram_available is False    # real probe value, not a constant
    finally:
        handle.close()


def test_generic_writers_and_fts_maintenance_refuse_on_the_quiet_handle(tmp_path):
    path = _settle(tmp_path)
    handle, _request = _read_quiet(path)
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


def test_approved_continuation_write_persists_journal_head_and_is_token_idempotent(tmp_path):
    path = _settle(tmp_path)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    handle, _request = _read_quiet(path, approved)
    try:
        before = handle._write_count
        first = handle.append_epoch_continuation(dict(approved))
        assert handle._write_count == before + 1
        assert isinstance(first, dict)
        assert first.get("token") == approved["token"] and first.get("operation_sha256")
        # The token index lands under receipt3/t/<SIDhex>/<opuuid>, never the bare token.
        sid_hex = SID.encode("utf-8").hex()
        key = f"receipt3/t/{sid_hex}/{approved['token']}"
        stored = json.loads(handle._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()[0])
        assert stored["owner_session_id"] == SID
        assert stored["operation_id"] == approved["token"]
        assert stored["operation_sha256"] == first["operation_sha256"]
        assert handle._conn.execute(
            "SELECT count(*) FROM state_meta WHERE key = ?",
            ("receipt3/t/" + approved["token"],)).fetchone()[0] == 0
        # The real journal entry and its ordered page land under the named keys.
        op = json.loads(handle._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (f"receipt3/o/{sid_hex}/00000000000000000002",)).fetchone()[0])
        assert op["kind"] == "epoch_continuation" and op["generation"] == 2
        assert op["previous_head_sha256"] == head["head_sha256"]
        assert op["continuation"] == {name: approved[name] for name in (
            "proof_sha256", "old_epoch", "new_epoch", "old_generation", "old_head_sha256",
            "old_head_token_sha256", "boundary_roster_sha256", "alias_inventory_sha256")}
        page = json.loads(handle._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (f"receipt3/p/{sid_hex}/00000000000000000002/00000000000000000000",)).fetchone()[0])
        assert page["page_ordinal"] == 0 and page["generation"] == 2
        assert page["epoch"] == NEW_EPOCH
        # The protected head advances in the same transaction.
        new_head = json.loads(_head_raw(handle))["_message_copy_receipts_v2"]
        assert (new_head["active_epoch"], new_head["epoch_ordinal"],
                new_head["generation"]) == (NEW_EPOCH, 2, 2)
        assert new_head["head_sha256"] == first["operation_sha256"]
        assert new_head["v1_binding_sha256"] is None       # preserved from the old head
        assert new_head["activation_proof_sha256"] == approved["proof_sha256"]
        snapshot, raw_before = _receipt_snapshot(handle), _head_raw(handle)
        # Same token: verified readback, no second insert, counter unchanged.
        replay = handle.append_epoch_continuation(dict(approved))
        assert replay == first
        assert handle._write_count == before + 1
        assert _receipt_snapshot(handle) == snapshot
        assert _head_raw(handle) == raw_before
        # Same key (token) with different bytes: refused, still no write.
        with pytest.raises((ValueError, RuntimeError)):
            handle.append_epoch_continuation(dict(
                approved, old_generation=approved["old_generation"] + 1))
        assert handle._write_count == before + 1
    finally:
        handle.close()


def test_continuations_outside_the_approved_transition_are_refused(tmp_path):
    path = _settle(tmp_path)
    approved = _continuation_request()
    handle, _request = _read_quiet(path, approved)
    try:
        for tampered in (
            dict(approved, token=str(uuid.uuid4()), proof_sha256="9" * 64),
            dict(approved, token=str(uuid.uuid4()), new_epoch="sqlite:" + "c" * 64),
            dict(approved, token=str(uuid.uuid4()), old_epoch="sqlite:" + "9" * 64),
            dict(approved, token=str(uuid.uuid4()), owner_session_id="other-01"),
            dict(approved, token=str(uuid.uuid4()), boundary_roster_sha256="8" * 64),
            dict(approved, token=str(uuid.uuid4()), alias_inventory_sha256="7" * 64),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                handle.append_epoch_continuation(tampered)
        assert handle._write_count == 0
        assert handle._conn.execute(
            "SELECT count(*) FROM state_meta WHERE key LIKE 'receipt3/t/%'").fetchone()[0] == 0
    finally:
        handle.close()


def _seed_head(path, head, sid=SID):
    _mutate(path, "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_message_copy_receipts_v2": head}), sid))


def _head(**overrides):
    head = {"schema_version": "hermes-message-copy-receipts/3", "protocol_revision": 3,
            "domain_sha256": "d" * 64, "owner_session_id": SID, "active_epoch": OLD_EPOCH,
            "epoch_ordinal": 1, "generation": 1, "head_sha256": "2" * 64,
            "v1_binding_sha256": None, "activation_proof_sha256": "9" * 64}
    head.update(overrides)
    return head


def test_continuation_head_cas_binds_the_write_to_the_stored_head(tmp_path):
    path = _settle(tmp_path)
    head = _head()
    _seed_head(path, head)
    matching = _continuation_request(old_epoch=head["active_epoch"],
                                     old_generation=head["generation"],
                                     old_head_sha256=head["head_sha256"],
                                     old_head_token_sha256=_canon_sha(head))
    handle, _request = _read_quiet(path, matching)
    try:
        # The stored head matches the approved old coordinates: the continuation lands.
        first = handle.append_epoch_continuation(dict(matching))
        assert handle._write_count == 1 and first.get("token") == matching["token"]
        # Any oldraw/head coordinate that no longer matches the stored head is refused.
        for tampered in (
            dict(matching, token=str(uuid.uuid4()),
                 old_generation=matching["old_generation"] + 1),
            dict(matching, token=str(uuid.uuid4()), old_head_sha256="6" * 64),
            dict(matching, token=str(uuid.uuid4()), old_head_token_sha256="7" * 64),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                handle.append_epoch_continuation(tampered)
        assert handle._write_count == 1
    finally:
        handle.close()
    # An approved owner that has no row in the settled store is refused by the CAS.
    ghost = _continuation_request(owner_session_id="ghost-99")
    handle, _request = _read_quiet(path, ghost)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            handle.append_epoch_continuation(dict(ghost))
        assert handle._write_count == 0
    finally:
        handle.close()


def test_quiet_write_runs_on_the_sessiondb_transaction_core_and_fences_replacement(tmp_path):
    # The continuation write is executed by the EXISTING SessionDB transaction core
    # (one shared implementation -- no private BEGIN IMMEDIATE copy on the handle),
    # so the core's replaced-file fence must fire from inside the write path.
    assert hermes_state.QuietRecoveryHandle._execute_write is hermes_state.SessionDB._execute_write
    path = _settle(tmp_path)
    approved = _continuation_request()
    handle, _request = _read_quiet(path, approved)
    try:
        replacement = tmp_path / "replacement.db"
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)  # the path now names a NEW file (inode)
        with pytest.raises(hermes_state.StateDbReplacedError):
            handle.append_epoch_continuation(dict(approved))
        assert handle._write_count == 0
    finally:
        handle.close()


def test_cadence_hit_marks_postcommit_fts_defer_without_running_usermerge(tmp_path):
    path = _settle(tmp_path)
    threshold = getattr(hermes_state, "QUIET_RECOVERY_FTS_DEFER_EVERY", 1000)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    handle, _request = _read_quiet(path, approved)
    try:
        assert handle._fts_enabled is True
        # Walk one committed approved-transition operation exactly onto the cadence
        # boundary (earlier owners' continuations share one recovery handle in the
        # real flow, so the counter is advanced to the boundary here).
        handle._write_count = threshold - 1
        fts_before = {name: count for name, count in handle._conn.execute(
            "SELECT name, (SELECT count(*) FROM sqlite_schema WHERE name = t.name) "
            "FROM sqlite_schema t WHERE name LIKE 'messages_fts%'")}
        first = handle.append_epoch_continuation(dict(approved))
        assert first.get("token") == approved["token"]
        assert handle._write_count == threshold
        assert handle._fts_deferred_due is True          # deferred, not executed
        assert handle._fts_enabled is True               # real value stays on
        # This was a full real payload write (journal entry + page + head).
        assert handle._conn.execute(
            "SELECT count(*) FROM state_meta WHERE key LIKE 'receipt3/o/%'").fetchone()[0] == 1
        assert handle._conn.execute(
            "SELECT count(*) FROM state_meta WHERE key LIKE 'receipt3/p/%'").fetchone()[0] == 1
        fts_after = {name: count for name, count in handle._conn.execute(
            "SELECT name, (SELECT count(*) FROM sqlite_schema WHERE name = t.name) "
            "FROM sqlite_schema t WHERE name LIKE 'messages_fts%'")}
        assert fts_after == fts_before                  # no FTS shadow repair ran
        # While the defer is due, FTS write entries stay refused.
        with pytest.raises((RuntimeError, ValueError, PermissionError)):
            handle.rebuild_fts()
    finally:
        handle.close()


def test_first_receipt_g0_continuation_and_missing_head_downgrade_refusal(tmp_path):
    # A boundary owner with no stored receipt may continue from g0: the first
    # receipt is g1 (previous_head_sha256 null only here) and installs the head.
    path = _settle(tmp_path)
    request = _continuation_request(old_generation=0, old_head_sha256=None)
    handle, _request = _read_quiet(path, request)
    try:
        result = handle.append_epoch_continuation(dict(request))
        assert result["generation"] == 1
        sid_hex = SID.encode("utf-8").hex()
        op = json.loads(handle._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (f"receipt3/o/{sid_hex}/00000000000000000001",)).fetchone()[0])
        assert op["previous_head_sha256"] is None
        assert op["kind"] == "epoch_continuation"
        head = json.loads(_head_raw(handle))["_message_copy_receipts_v2"]
        assert (head["active_epoch"], head["epoch_ordinal"], head["generation"]) == (
            NEW_EPOCH, 1, 1)
        assert head["head_sha256"] == result["operation_sha256"]
    finally:
        handle.close()
    # A missing head may never downgrade an existing journal to "no evidence":
    # the owner already carries a receipt3/o entry, so the first-receipt claim is
    # refused.
    other = tmp_path / "other"
    other.mkdir()
    path2 = _settle(other)
    _mutate(path2, "INSERT INTO state_meta (key, value) VALUES (?, ?)",
            (f"receipt3/o/{SID.encode('utf-8').hex()}/00000000000000000001", "{}"))
    request2 = _continuation_request(old_generation=0, old_head_sha256=None)
    handle, _request = _read_quiet(path2, request2)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            handle.append_epoch_continuation(dict(request2))
        assert handle._write_count == 0
    finally:
        handle.close()


# ── milestone 2B: quiet reopen + post-release bounded FTS pass ───────────────

def _ro_receipt_rows(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return sorted((row[0], row[1]) for row in conn.execute(
            "SELECT key, value FROM state_meta WHERE key LIKE 'receipt3/%'"))
    finally:
        conn.close()


def test_quiet_reopen_resumes_partial_crash_and_enumerates_completed_tokens(tmp_path):
    path = _settle(tmp_path)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    crashed = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=dict(approved))
    first = crashed.append_epoch_continuation(dict(approved))
    # The crash: the connection dies with the process; the handle never runs
    # close() or any exit-time maintenance, and no counter or due flag was
    # persisted anywhere.
    crashed._conn.close()
    del crashed
    # Re-entry is quiet admission again, never the ordinary constructor.
    handle = hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    try:
        committed = handle.enumerate_committed_continuations()
        assert [entry["token"] for entry in committed] == [approved["token"]]
        assert [entry["generation"] for entry in committed] == [2]
        assert committed[0]["operation_sha256"] == first["operation_sha256"]
        state = handle.verify_partial_recovery_state()
        assert state["head"] is not None and state["head"]["generation"] == 2
        assert state["head"]["head_sha256"] == first["operation_sha256"]
        # A restarted in-memory counter is not a reset of persisted recovery
        # state: nothing about the committed bytes changed.
        assert handle._write_count == 0 and handle._fts_deferred_due is False
        snapshot = _receipt_snapshot(handle)
        assert handle.append_epoch_continuation(dict(approved)) == first
        assert _receipt_snapshot(handle) == snapshot
        assert handle._write_count == 0
    finally:
        handle.close()
    # A clean close resumes the same way: the recovery only ends at exit_quiet().
    handle = hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    try:
        assert handle.enumerate_committed_continuations()[0]["token"] == approved["token"]
    finally:
        handle.close()


def test_quiet_reopen_before_any_commit_continues_the_remaining_owner(tmp_path):
    path = _settle(tmp_path)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    crashed = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=dict(approved))
    crashed._conn.close()          # crash before any commit: nothing was written
    del crashed
    handle = hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    try:
        assert handle.enumerate_committed_continuations() == []
        state = handle.verify_partial_recovery_state()
        assert state["head"]["generation"] == 1        # still the frozen old head
        result = handle.append_epoch_continuation(dict(approved))   # the owner continues
        assert result["generation"] == 2 and handle._write_count == 1
        assert handle.enumerate_committed_continuations()[0]["token"] == approved["token"]
    finally:
        handle.close()


def test_quiet_reopen_refuses_inconsistent_partial_state(tmp_path):
    # (a) a committed continuation whose journal page bytes are gone
    a = tmp_path / "a"
    a.mkdir()
    path = _settle(a)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=dict(approved))
    handle.append_epoch_continuation(dict(approved))
    handle.close()
    _mutate(path, "DELETE FROM state_meta WHERE key LIKE 'receipt3/p/%'")
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    # (b) a committed continuation whose protected head was erased
    b = tmp_path / "b"
    b.mkdir()
    path = _settle(b)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=dict(approved))
    handle.append_epoch_continuation(dict(approved))
    handle.close()
    _mutate(path, "UPDATE sessions SET model_config = '{}' WHERE id = ?", (SID,))
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    # (c) orphaned receipt3/o bytes without any committed token
    c = tmp_path / "c"
    c.mkdir()
    path = _settle(c)
    _mutate(path, "INSERT INTO state_meta (key, value) VALUES (?, ?)",
            (f"receipt3/o/{SID.encode('utf-8').hex()}/00000000000000000001", "{}"))
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=_continuation_request())
    # (d) the resumed request no longer matches the committed operation's frozen
    # transition block
    d = tmp_path / "d"
    d.mkdir()
    path = _settle(d)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=dict(approved))
    handle.append_epoch_continuation(dict(approved))
    handle.close()
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.resume(
            db_path=path, init_request=dict(approved, proof_sha256="9" * 64))
    # (e) before any commit, a head that no longer carries the frozen coordinates
    e = tmp_path / "e"
    e.mkdir()
    path = _settle(e)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    _mutate(path, "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_message_copy_receipts_v2": _head(generation=7)}), SID))
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))


def test_post_release_fts_pass_runs_at_exit_after_the_accepted_boundary(tmp_path):
    path = _settle(tmp_path)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    handle = hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    try:
        handle.append_epoch_continuation(dict(approved))
        binding = handle.final_readback_binding()
        message = {"owner_session_id": binding["owner_session_id"],
                   "committed": binding["committed"], "head": binding["head"]}
        assert binding["readback_sha256"] == hashlib.sha256(_canonical(message)).hexdigest()
        assert binding["committed"][0]["token"] == approved["token"]
        snapshot = _receipt_snapshot(handle)
        assert handle._fts_deferred_due is False     # no due flag was persisted anywhere
        # The acceptance gate: a wrong binding refuses the exit and runs nothing.
        with pytest.raises(ValueError):
            handle.exit_quiet(accepted_readback_sha256="0" * 64)
        assert handle._closed is False
        assert handle._fts_usermerge_floor_applied is False
        assert _receipt_snapshot(handle) == snapshot
        report = handle.exit_quiet(accepted_readback_sha256=binding["readback_sha256"])
    finally:
        if not handle._closed:
            handle.close()
    assert handle._closed is True
    # Exactly one native bounded pass ran (500 pages/index, 4 commands/pass),
    # AFTER the accepted boundary: it touches only FTS maintenance state.
    assert report["fts_pass"]["failed"] is None
    assert report["fts_pass"]["executed"] >= 1
    assert report["write_count"] == 1                # the counter is not zeroed
    assert report["fts_enabled"] is True and report["trigram_available"] is True
    assert report["readback"]["readback_sha256"] == binding["readback_sha256"]
    # Committed continuation bytes are untouched by the pass; ordinary handles reopen.
    assert _ro_receipt_rows(path) == snapshot
    db = SessionDB(db_path=path)
    try:
        db.set_meta("post/release", "ok")
    finally:
        db.close()


def test_post_release_fts_pass_failure_never_replays_committed_continuations(tmp_path):
    path = _settle(tmp_path)
    head = _head()
    _seed_head(path, head)
    approved = _matching_request(head)
    handle = hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    first = handle.append_epoch_continuation(dict(approved))
    snapshot = _receipt_snapshot(handle)
    binding = handle.final_readback_binding()
    # The bounded pass fails for real (the connection refuses writes): the
    # failure is reported, the accepted boundary stands, and no committed
    # continuation may be replayed.
    handle._conn.execute("PRAGMA query_only = ON")
    report = handle.exit_quiet(accepted_readback_sha256=binding["readback_sha256"])
    assert handle._closed is True
    assert report["fts_pass"]["failed"] is not None
    assert report["fts_pass"]["executed"] == 0
    assert _ro_receipt_rows(path) == snapshot
    # The committed token still replays exactly through a fresh quiet reopen.
    reopened = hermes_state.QuietRecoveryHandle.resume(db_path=path, init_request=dict(approved))
    try:
        assert reopened.append_epoch_continuation(dict(approved)) == first
        assert reopened._write_count == 0
        assert _ro_receipt_rows(path) == snapshot
    finally:
        reopened.close()
