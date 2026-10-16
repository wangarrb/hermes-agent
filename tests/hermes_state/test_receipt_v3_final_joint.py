"""Receipt r4 milestone 2C: full-table joint, inverse deltas and the row-state index.

Contract from the effective r4 machine spec
(``protocol.recovery.joint_formula`` / ``identity_modes``, ``protocol.storage``
namespace, ``record_shapes.row_state_index`` / ``payload_state``,
``writer_timeline[4]``) and the design text for C02 steps 2/3/6:

  * ``receipt3/r/<birth_epoch64hex>/<rowid20>`` is a mutable, rebuildable
    accelerator with NO owner component; its reader derives the exact
    per-owner inventory from the stored journal bytes and requires exact
    inventory equality (no orphan, no missing, no stale event, strict
    key/body coordinate equality, canonical stored bytes);
  * the writer-side upsert is identity-preserving: a refresh is an in-place
    UPDATE that keeps the actual ROWID and never regresses -- ``INSERT OR
    REPLACE`` (an identity swap) is exactly what step 6 detects as a removed
    plus added row;
  * the durable-table joint binds every table (``sqlite_schema``, FTS5
    virtual tables, every shadow/config/sequence/meta table) through the
    actual identity I(row) -- the unshadowed rowid, the INTEGER PRIMARY KEY
    alias (id == ROWID verified), ALL PK components for WITHOUT ROWID, the
    virtual rowid for FTS5 -- with the complete sorted row J byte strings
    length-framed under the structure hash;
  * C02 steps 2/3: the independent read-only copy comparison and the frozen
    proof -- any logical persistent delta between the same old latest
    refuses; while no continuation exists the proof must re-verify unchanged;
  * C02 step 6: the exact identity-aware inverse -- added keys may only be
    this request's named journal/checkpoint/row-index delta, updates may only
    be an in-place row-index refresh or the protected head advance, and the
    full joint must return exactly to the original old latest. A stray key,
    a message change, a token-key deletion, a head change or an
    ``INSERT OR REPLACE`` identity swap invalidates the admission.

RED against the M2B slice: these surfaces do not exist there. Real SQLite
fixtures only; no mocks. Synthetic fixtures; no production store is touched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid

import pytest

from hermes_state import SessionDB

SID = "owner-01"
OTHER = "owner-02"
OLD_EPOCH = "sqlite:" + "a" * 64
NEW_EPOCH = "sqlite:" + "b" * 64
DOMAIN = "d" * 64


def _m():
    import hermes_state_copy_provenance as module
    return module


def _canonical(x) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8", "strict")


def _canon_sha(x) -> str:
    return hashlib.sha256(_canonical(x)).hexdigest()


def _settle(root, sid=SID):
    """One settled real store (owner session + message + a generic meta key)."""
    db = SessionDB(db_path=root / "state.db")
    db.create_session(sid, "cli", model="m")
    db.append_message(session_id=sid, role="user", content="hello")
    db.set_meta("probe/key", "value")
    db.close()
    return root / "state.db"


def _connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _mutate(path, sql, params=()):
    conn = sqlite3.connect(path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _sql(path, sql, params=()):
    _mutate(path / "state.db", sql, params)


def _head(**overrides):
    head = {"schema_version": "hermes-message-copy-receipts/3", "protocol_revision": 3,
            "domain_sha256": DOMAIN, "owner_session_id": SID, "active_epoch": OLD_EPOCH,
            "epoch_ordinal": 1, "generation": 1, "head_sha256": "2" * 64,
            "v1_binding_sha256": None, "activation_proof_sha256": "9" * 64}
    head.update(overrides)
    return head


def _seed_head(path, head, sid=SID):
    _mutate(path, "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_message_copy_receipts_v2": head}), sid))


def _request(**overrides):
    request = {"kind": "epoch_continuation", "token": str(uuid.uuid4()),
               "owner_session_id": SID, "proof_sha256": "1" * 64,
               "old_epoch": OLD_EPOCH, "new_epoch": NEW_EPOCH, "old_generation": 1,
               "old_head_sha256": "2" * 64, "old_head_token_sha256": "3" * 64,
               "boundary_roster_sha256": "4" * 64, "alias_inventory_sha256": "5" * 64}
    request.update(overrides)
    return request


def _transition(request):
    return {name: request[name] for name in _m()._CONTINUATION_KEYS}


def _row_ref(row_id, epoch=OLD_EPOCH, sid=SID):
    return {"birth_epoch": epoch, "row_id": row_id, "owner_session_id": sid}


def _payload_state(ref, seed="0"):
    return {"row_ref": ref,
            "columns": [[f"col{index:02d}",
                         hashlib.sha256(f"{seed}-{index}".encode()).hexdigest()]
                        for index in range(20)],
            "payload_sha256": hashlib.sha256(f"P-{seed}".encode()).hexdigest(),
            "feed_input_sha256": hashlib.sha256(f"Q-{seed}".encode()).hexdigest()}


def _r_entry(ref, event, seed="0"):
    return {"row_ref": ref, "latest_state": _payload_state(ref, seed), "event": event}


def _seed_r_entry(conn, ref, event, seed="0"):
    m = _m()
    key = m.encode_receipt3_row_index_key(ref["birth_epoch"], ref["row_id"])
    conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                 (key, _canonical(_r_entry(ref, event, seed)).decode("utf-8")))
    return key


def _add_owner(conn, sid):
    conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                 (sid, "cli", 0.0))


def _hand_member(conn, *, sid, token, epoch, ordinal, generation, previous, block,
                 ref=None):
    """Hand-seed one full-form journal member exactly like the writer's builders
    (header + ordered page + token record).  Returns the sealed header."""
    m = _m()
    conn.row_factory = sqlite3.Row
    request = {"kind": "epoch_continuation", "token": token, "owner_session_id": sid,
               **{name: block[name] for name in m._CONTINUATION_KEYS}}
    item = {"kind": "native_result",
            "input_ordinal": 0 if ref is not None else None, "row_ref": ref,
            "disposition": "inserted" if ref is not None else "aggregate",
            "result_count": None}
    page = m.build_journal_page(owner_session_id=sid, epoch=epoch, generation=generation,
                                page_ordinal=0, items=[item])
    contexts = m.observe_native_session_contexts(
        conn, owner_session_id=sid, actor_session_id=sid, kind="epoch_continuation",
        domain_sha256=DOMAIN)
    header = m.build_epoch_continuation_header(
        request=request, request_sha256=_canon_sha(request), domain_sha256=DOMAIN,
        epoch=epoch, epoch_ordinal=ordinal, generation=generation,
        previous_head_sha256=previous, pages=[page], native_session_contexts=contexts)
    result = {"token": token, "operation_sha256": header["operation_sha256"],
              "generation": generation}
    record = {"owner_session_id": sid, "operation_id": token,
              "request_sha256": header["request_sha256"], "operation_generation": generation,
              "operation_sha256": header["operation_sha256"],
              "result_sha256": _canon_sha(result)}
    for key, value in (
            (m.encode_receipt3_operation_key(sid, generation), _canonical(header)),
            (m.encode_receipt3_page_key(sid, generation, 0), _canonical(page)),
            (m.encode_receipt3_token_key(sid, token), _canonical(record))):
        conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                     (key, value.decode("utf-8")))
    return header


def _copy_pair(root, name="pair"):
    """_settle -> _seed_head -> copytree A/B -> quiet append on B."""
    store = root / f"{name}-store"
    store.mkdir(parents=True)
    path = _settle(store)
    head = _head()
    _seed_head(path, head)
    request = _request(old_epoch=head["active_epoch"], old_generation=head["generation"],
                       old_head_sha256=head["head_sha256"],
                       old_head_token_sha256=_canon_sha(head))
    source, replacement = root / f"{name}-old", root / f"{name}-replacement"
    shutil.copytree(store, source)
    shutil.copytree(store, replacement)
    handle = SessionDB.open_quiet_recovery_admission(
        db_path=replacement / "state.db", init_request=dict(request))
    try:
        result = handle.append_epoch_continuation(dict(request))
    finally:
        handle.close()
    return source, replacement, request, result


def _ro_conn(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _full_joint_of(path):
    conn = _ro_conn(path)
    try:
        return _m().full_table_joint(conn)
    finally:
        conn.close()


# ── receipt3/r key codec ─────────────────────────────────────────────────────

def test_row_index_key_codec_rejects_non_canonical_forms():
    m = _m()
    key = m.encode_receipt3_row_index_key(OLD_EPOCH, 7)
    assert key == f"receipt3/r/{'a' * 64}/00000000000000000007"
    assert m.parse_receipt3_row_index_key(key) == {"birth_epoch": OLD_EPOCH, "row_id": 7}
    assert m.parse_receipt3_row_index_key(
        m.encode_receipt3_row_index_key("sqlite:" + "0" * 64, 2 ** 62)) == {
            "birth_epoch": "sqlite:" + "0" * 64, "row_id": 2 ** 62}
    for epoch in ("a" * 64, "sqlite:" + "a" * 63, "sqlite:" + "a" * 65,
                  "sqlite:" + "A" * 64, "sqlite:", "sqlite:0x" + "a" * 64, "", None, 7):
        with pytest.raises(ValueError):
            m.encode_receipt3_row_index_key(epoch, 7)
    for row_id in (0, -1, True, 1.0, "7", None, m._MAX_INT64 + 1):
        with pytest.raises(ValueError):
            m.encode_receipt3_row_index_key(OLD_EPOCH, row_id)
    for bad in (
            "receipt3/r/" + "a" * 64,                                  # missing rowid
            "receipt3/r/" + "a" * 64 + "/7",                           # short rowid
            "receipt3/r/" + "a" * 64 + "/00000000000000000007/extra",  # extra component
            "receipt3/r/" + "a" * 64 + "/00000000000000000000",        # zero is not positive
            "receipt3/r/" + "a" * 64 + "/000000000000000000007",       # 21 digits
            "receipt3/r/" + "a" * 63 + "g/" + "0" * 19 + "7",          # non-hex epoch
            "receipt3/r/" + "A" * 64 + "/" + "0" * 19 + "7",           # uppercase epoch
            "receipt3/e/" + "a" * 64 + "/" + "0" * 19 + "7",           # wrong family
            "receipt3/r/" + "a" * 64 + "/" + "0" * 19 + "7 ",          # trailing space
            "/receipt3/r/" + "a" * 64 + "/" + "0" * 19 + "7"):
        with pytest.raises(ValueError):
            m.parse_receipt3_row_index_key(bad)
    for bad in (None, 7, b"receipt3/r/" + b"a" * 64 + b"/" + b"0" * 19 + b"7"):
        with pytest.raises(ValueError):
            m.parse_receipt3_row_index_key(bad)


# ── receipt3/r entry shape ───────────────────────────────────────────────────

def test_row_index_entry_shape_and_payload_state():
    m = _m()
    ref = _row_ref(7)
    entry = _r_entry(ref, [SID, 1, 0, 0])
    m.validate_row_state_index_entry(entry)                 # exact shape accepted
    m.validate_row_state_index_entry(_r_entry(
        ref, {"activation_proof_sha256": "9" * 64, "context_session_id": SID}))
    short = dict(entry["latest_state"], columns=entry["latest_state"]["columns"][:19])
    duplicate = dict(entry["latest_state"],
                     columns=[["same", "0" * 64]] * 20)
    for bad in (
            dict(entry, extra=True),                       # unknown extra key
            {name: value for name, value in entry.items() if name != "event"},
            dict(entry, event=[SID, 1, 0]),                # 3 coordinate components
            dict(entry, event=[SID, 0, 0, 0]),             # generation 0
            dict(entry, event=[SID, 1, -1, 0]),            # negative item ordinal
            dict(entry, event=[SID, 1, 0, 0, 0]),          # extra component
            dict(entry, event="event"),                    # neither list nor anchor
            dict(entry, event={"activation_proof_sha256": "9" * 64}),          # no context
            dict(entry, event={"activation_proof_sha256": "9" * 64,
                               "context_session_id": SID, "extra": 1}),
            dict(entry, event={"activation_proof_sha256": "not-a-sha",
                               "context_session_id": SID}),
            dict(entry, latest_state=short),               # 19 of the 20 columns
            dict(entry, latest_state=duplicate),           # duplicate column names
            dict(entry, latest_state=dict(entry["latest_state"], payload_sha256="Z" * 64)),
            dict(entry, latest_state=dict(entry["latest_state"], feed_input_sha256="short")),
            dict(entry, latest_state=dict(entry["latest_state"], extra=1)),
            dict(entry, row_ref=_row_ref(8))):             # state names a different row
        with pytest.raises(ValueError):
            m.validate_row_state_index_entry(bad)
    for bad_ref in (_row_ref(0), _row_ref(7, epoch="a" * 64), _row_ref(7, sid="")):
        with pytest.raises(ValueError):
            m.validate_row_state_index_entry(_r_entry(bad_ref, [SID, 1, 0, 0]))


# ── reader-side derivation of the exact inventory ────────────────────────────

def test_read_row_state_index_derives_the_exact_inventory(tmp_path):
    m = _m()
    path = _settle(tmp_path)
    conn = _connect(path)
    try:
        token = str(uuid.uuid4())
        member = _hand_member(conn, sid=SID, token=token, epoch=OLD_EPOCH, ordinal=1,
                              generation=1, previous=None, block=_transition(_request()),
                              ref=_row_ref(7))
        key = _seed_r_entry(conn, _row_ref(7), [SID, 1, 0, 0])
        conn.commit()
        assert member["generation"] == 1
        derived = m.read_row_state_index(conn, SID)
        assert [item["key"] for item in derived] == [key]
        assert derived[0]["row_ref"] == _row_ref(7)
        assert derived[0]["event"] == [SID, 1, 0, 0]
        assert derived[0]["latest_state"]["row_ref"] == _row_ref(7)
    finally:
        conn.close()
    # The reader refuses a missing / orphaned / stale / mislabelled inventory.
    scenarios = {
        "missing": lambda conn: conn.execute(
            "DELETE FROM state_meta WHERE key LIKE 'receipt3/r/%'"),
        "orphan": lambda conn: _seed_r_entry(conn, _row_ref(8), [SID, 1, 0, 0]),
        "stale": lambda conn: conn.execute(
            "UPDATE state_meta SET value = ? WHERE key LIKE 'receipt3/r/%'",
            (_canonical(_r_entry(_row_ref(7), [SID, 1, 0, 3])).decode("utf-8"),)),
        "key_body": lambda conn: conn.execute(
            "UPDATE state_meta SET value = ? WHERE key LIKE 'receipt3/r/%'",
            (_canonical(_r_entry(_row_ref(8), [SID, 1, 0, 0])).decode("utf-8"),)),
        "noncanonical_body": lambda conn: conn.execute(
            "UPDATE state_meta SET value = ' ' WHERE key LIKE 'receipt3/r/%'"),
        "bogus_key": lambda conn: conn.execute(
            "INSERT INTO state_meta (key, value) VALUES ('receipt3/r/bogus', '{}')"),
    }
    for name, tamper in scenarios.items():
        root = tmp_path / name
        root.mkdir()
        scenario_path = _settle(root)
        conn = _connect(scenario_path)
        try:
            _hand_member(conn, sid=SID, token=str(uuid.uuid4()), epoch=OLD_EPOCH,
                         ordinal=1, generation=1, previous=None,
                         block=_transition(_request()), ref=_row_ref(7))
            _seed_r_entry(conn, _row_ref(7), [SID, 1, 0, 0])
            conn.commit()
            tamper(conn)
            conn.commit()
            with pytest.raises(ValueError):
                m.read_row_state_index(conn, SID)
        finally:
            conn.close()
    path2 = tmp_path / "owner-arg"
    path2.mkdir()
    conn = _connect(_settle(path2))
    try:
        for bad_owner in ("", None, 7):
            with pytest.raises(ValueError):
                m.read_row_state_index(conn, bad_owner)
    finally:
        conn.close()


def test_read_row_state_index_attributes_entries_per_owner(tmp_path):
    m = _m()
    path = _settle(tmp_path)
    conn = _connect(path)
    try:
        block = _transition(_request())
        _hand_member(conn, sid=SID, token=str(uuid.uuid4()), epoch=OLD_EPOCH, ordinal=1,
                     generation=1, previous=None, block=block, ref=_row_ref(7))
        own_key = _seed_r_entry(conn, _row_ref(7), [SID, 1, 0, 0])
        _add_owner(conn, OTHER)
        _hand_member(conn, sid=OTHER, token=str(uuid.uuid4()), epoch=OLD_EPOCH, ordinal=1,
                     generation=1, previous=None, block=block, ref=_row_ref(9, sid=OTHER))
        foreign_key = _seed_r_entry(conn, _row_ref(9, sid=OTHER), [OTHER, 1, 0, 0])
        conn.commit()
        # Neither owner's slice is polluted by the other owner's stored entry.
        assert [item["key"] for item in m.read_row_state_index(conn, SID)] == [own_key]
        foreign = m.read_row_state_index(conn, OTHER)
        assert [item["key"] for item in foreign] == [foreign_key]
        assert foreign[0]["event"] == [OTHER, 1, 0, 0]
        # A damaged body is refused for every reader, not silently skipped.
        conn.execute("UPDATE state_meta SET value = '{}' WHERE key = ?", (foreign_key,))
        conn.commit()
        with pytest.raises(ValueError):
            m.read_row_state_index(conn, SID)
        with pytest.raises(ValueError):
            m.read_row_state_index(conn, OTHER)
    finally:
        conn.close()


# ── writer-side upsert ───────────────────────────────────────────────────────

def test_upsert_receipt3_row_index_tx_insert_refresh_and_refusals(tmp_path):
    m = _m()
    conn = _connect(_settle(tmp_path))
    try:
        entry = _r_entry(_row_ref(7), [SID, 1, 0, 0])
        key = m.upsert_receipt3_row_index_tx(conn, entry)
        assert key == m.encode_receipt3_row_index_key(OLD_EPOCH, 7)
        conn.commit()
        raw = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()[0]
        rowid = conn.execute("SELECT rowid FROM state_meta WHERE key = ?", (key,)).fetchone()[0]
        assert raw == _canonical(entry).decode("utf-8")      # canonical stored bytes
        # An in-place refresh keeps the actual ROWID and advances the event.
        newer = _r_entry(_row_ref(7), [SID, 3, 0, 0], seed="1")
        assert m.upsert_receipt3_row_index_tx(conn, newer) == key
        conn.commit()
        assert conn.execute("SELECT rowid FROM state_meta WHERE key = ?",
                            (key,)).fetchone()[0] == rowid
        assert conn.execute("SELECT value FROM state_meta WHERE key = ?",
                            (key,)).fetchone()[0] == _canonical(newer).decode("utf-8")
        # Regression / repeat / malformed events refuse without a write.
        for bad in (newer, _r_entry(_row_ref(7), [SID, 2, 0, 0], seed="2"),
                    _r_entry(_row_ref(7),
                             {"activation_proof_sha256": "9" * 64,
                              "context_session_id": SID}, seed="3")):
            with pytest.raises(ValueError):
                m.upsert_receipt3_row_index_tx(conn, bad)
        assert conn.execute("SELECT value FROM state_meta WHERE key = ?",
                            (key,)).fetchone()[0] == _canonical(newer).decode("utf-8")
        # A stored body that no longer describes its own key refuses a refresh.
        conn.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                     (_canonical(_r_entry(_row_ref(8), [SID, 3, 0, 0], seed="4")
                                 ).decode("utf-8"), key))
        conn.commit()
        with pytest.raises(ValueError):
            m.upsert_receipt3_row_index_tx(conn, _r_entry(_row_ref(7), [SID, 4, 0, 0]))
        # A new anchor row is a fresh insert, never a refresh (no such stored key).
        anchor = _r_entry(_row_ref(9),
                          {"activation_proof_sha256": "9" * 64, "context_session_id": SID})
        anchor_key = m.upsert_receipt3_row_index_tx(conn, anchor)
        assert anchor_key == m.encode_receipt3_row_index_key(OLD_EPOCH, 9)
        conn.commit()
        stored = json.loads(conn.execute("SELECT value FROM state_meta WHERE key = ?",
                                         (anchor_key,)).fetchone()[0])
        assert stored == anchor
    finally:
        conn.close()


# ── durable-table joint ──────────────────────────────────────────────────────

def test_full_table_joint_binds_every_durable_table(tmp_path):
    m = _m()
    path = _settle(tmp_path)
    conn = _ro_conn(path)
    try:
        joint = m.full_table_joint(conn)
        # sqlite_schema as its own structural inventory plus the FTS5 virtual
        # tables and every shadow/config/sequence/meta table -- no three-table
        # shortcut: exactly the sqlite_schema table list plus sqlite_schema.
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'")} | {"sqlite_schema"}
        assert set(joint["tables"]) == names
        assert joint["table_count"] == len(names)
        assert joint["tables"]["sqlite_schema"]["row_count"] > 0
        assert joint["tables"]["messages"]["identity_mode"] == \
            "messages_INTEGER_PRIMARY_KEY_ALIAS"
        assert joint["tables"]["messages_fts"]["identity_mode"] == "FTS_virtual"
        assert joint["tables"]["sessions"]["identity_mode"] == "ordinary_rowid_nonalias"
        assert "messages_fts_data" in joint["tables"]          # FTS shadow rides the rules
        assert joint["tables"]["messages_fts"]["row_count"] == \
            joint["tables"]["messages"]["row_count"]
        # Stable on re-read; any durable change moves exactly the binding digest.
        assert m.full_table_joint(conn) == joint
        before = joint["joint_sha256"]
        table_before = joint["tables"]["state_meta"]["joint_sha256"]
        assert m.table_joint(conn, "state_meta")["joint_sha256"] == table_before
    finally:
        conn.close()
    _mutate(path, "UPDATE state_meta SET value = 'changed' WHERE key = 'probe/key'")
    conn = _ro_conn(path)
    try:
        after = m.full_table_joint(conn)
        assert after["joint_sha256"] != before
        assert after["tables"]["state_meta"]["joint_sha256"] != table_before
        assert after["tables"]["messages"]["joint_sha256"] == \
            joint["tables"]["messages"]["joint_sha256"]
    finally:
        conn.close()


def test_table_joint_is_identity_sensitive_and_refuses_unbound_tables(tmp_path):
    m = _m()
    one = sqlite3.connect(tmp_path / "one.db")
    two = sqlite3.connect(tmp_path / "two.db")
    try:
        for conn, rowids in ((one, (3, 4)), (two, (3, 5))):
            conn.execute("CREATE TABLE u (v TEXT)")
            conn.execute("INSERT INTO u (rowid, v) VALUES (?, 'x')", (rowids[0],))
            conn.execute("INSERT INTO u (rowid, v) VALUES (?, 'x')", (rowids[1],))
            conn.commit()
        # Same visible content, different actual identity: different joints.
        assert m.table_joint(one, "u")["row_count"] == 2
        assert m.table_joint(one, "u")["joint_sha256"] != m.table_joint(two, "u")["joint_sha256"]
        one.execute("DELETE FROM u WHERE rowid = 4")
        one.commit()
        assert m.table_joint(one, "u")["row_count"] == 1
        assert m.table_joint(one, "u")["joint_sha256"] != m.table_joint(two, "u")["joint_sha256"]
        # Multiplicity is preserved: a duplicated row is not collapsed.
        two.execute("CREATE TABLE d (v TEXT)")
        two.execute("INSERT INTO d (rowid, v) VALUES (1, 'same')")
        two.commit()
        single = m.table_joint(two, "d")["joint_sha256"]
        two.execute("INSERT INTO d (rowid, v) VALUES (2, 'same')")
        two.commit()
        assert m.table_joint(two, "d")["joint_sha256"] != single
        # WITHOUT ROWID: the typed PK tuple is the identity, never an invented rowid.
        two.execute("CREATE TABLE wo (k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID")
        two.execute("INSERT INTO wo VALUES ('k', 'v')")
        two.commit()
        assert m.table_joint(two, "wo")["identity_mode"] == "WITHOUT_ROWID"
        assert m.table_joint(two, "wo")["row_count"] == 1
        # A table whose three rowid aliases are all shadowed is refused, never
        # silently identified by its declared PK.
        two.execute("CREATE TABLE shadow (rowid TEXT, _rowid_ TEXT, oid TEXT, x TEXT)")
        two.commit()
        with pytest.raises(ValueError):
            m.table_joint(two, "shadow")
        # Generated columns and non-pinned virtual modules are not bound.
        two.execute("CREATE TABLE gen (a INTEGER, b INTEGER GENERATED ALWAYS AS (a + 1))")
        two.execute("CREATE VIRTUAL TABLE odd USING fts4(a)")
        two.commit()
        with pytest.raises(ValueError):
            m.table_joint(two, "gen")
        with pytest.raises(ValueError):
            m.table_joint(two, "odd")
        with pytest.raises(ValueError):
            m.table_joint(two, "missing_table")
    finally:
        one.close()
        two.close()


def test_joint_detects_row_id_value_association_swap_row_count_unchanged(tmp_path):
    """B07: exchanging the row-id/value association of two stored rows -- with
    the row count, the identity set and the value multiset all unchanged and no
    column added or removed -- must move the JOINT digest, for an ordinary
    non-alias rowid table and for an FTS5 shadow table.  A digest that bound
    the sorted values and the identities separately, instead of the whole
    association, would miss exactly this tamper."""
    m = _m()
    store = tmp_path / "store"
    store.mkdir()
    path = _settle(store)
    db = SessionDB(db_path=path)
    try:
        db.append_message(session_id=SID, role="user", content="one two three four five")
    finally:
        db.close()
    _sql(store, "INSERT INTO state_meta (rowid, key, value) VALUES (901, 'swap/a', 'A')")
    _sql(store, "INSERT INTO state_meta (rowid, key, value) VALUES (902, 'swap/b', 'B')")

    def meta_view(directory):
        conn = _ro_conn(directory / "state.db")
        try:
            rows = conn.execute("SELECT rowid, value FROM state_meta "
                                "WHERE rowid IN (901, 902) ORDER BY rowid").fetchall()
            return [row[0] for row in rows], sorted(row[1] for row in rows)
        finally:
            conn.close()

    def docsize_view(directory):
        conn = _ro_conn(directory / "state.db")
        try:
            rows = conn.execute("SELECT id, sz FROM messages_fts_docsize "
                                "ORDER BY id").fetchall()
            return [row[0] for row in rows], sorted(row[1] for row in rows)
        finally:
            conn.close()

    def joints_of(directory):
        conn = _ro_conn(directory / "state.db")
        try:
            return (m.table_joint(conn, "state_meta"),
                    m.table_joint(conn, "messages_fts_docsize"),
                    m.full_table_joint(conn))
        finally:
            conn.close()

    meta_ids, meta_values = meta_view(store)
    doc_ids, doc_sizes = docsize_view(store)
    assert len(doc_ids) == 2 and doc_sizes[0] != doc_sizes[1]
    meta_before, docsize_before, full_before = joints_of(store)

    # Swap #1 -- an ordinary rowid table with a non-alias TEXT primary key:
    # the two rows exchange their values; ids, row count and the value
    # multiset stay exactly the same.
    meta_swapped = tmp_path / "meta-swap"
    shutil.copytree(store, meta_swapped)
    _sql(meta_swapped, "UPDATE state_meta SET value = CASE rowid WHEN 901 THEN 'B' "
                       "WHEN 902 THEN 'A' END WHERE rowid IN (901, 902)")
    assert meta_view(meta_swapped) == (meta_ids, meta_values)
    meta_after, _, full_after = joints_of(meta_swapped)
    assert meta_after["row_count"] == meta_before["row_count"]
    assert meta_after["joint_sha256"] != meta_before["joint_sha256"]
    assert full_after["joint_sha256"] != full_before["joint_sha256"]
    with pytest.raises(ValueError):
        m.verify_independent_copy_joint(store / "state.db", meta_swapped / "state.db")

    # Swap #2 -- an FTS5 shadow table: the two per-document size rows exchange
    # their stored sizes; ids and the blob multiset stay the same.
    doc_swapped = tmp_path / "docsize-swap"
    shutil.copytree(store, doc_swapped)
    _sql(doc_swapped, "UPDATE messages_fts_docsize SET sz = CASE id WHEN ? THEN ? "
                      "WHEN ? THEN ? END",
         (doc_ids[0], doc_sizes[1], doc_ids[1], doc_sizes[0]))
    assert docsize_view(doc_swapped) == (doc_ids, doc_sizes)
    _, docsize_after, full_after = joints_of(doc_swapped)
    assert docsize_after["row_count"] == docsize_before["row_count"]
    assert docsize_after["joint_sha256"] != docsize_before["joint_sha256"]
    assert full_after["joint_sha256"] != full_before["joint_sha256"]
    with pytest.raises(ValueError):
        m.verify_independent_copy_joint(store / "state.db", doc_swapped / "state.db")

    # The untouched copy still verifies: the refusals are the swap, not a
    # broken fixture.
    intact = tmp_path / "intact"
    shutil.copytree(store, intact)
    result = m.verify_independent_copy_joint(store / "state.db", intact / "state.db")
    assert result["joint_sha256"] == full_before["joint_sha256"]


# ── C02 steps 2/3: copy comparison and frozen proof ──────────────────────────

def test_independent_copy_joint_accepts_equal_copies_and_refuses_deltas(tmp_path):
    m = _m()
    store = tmp_path / "store"
    store.mkdir()
    path = _settle(store)
    copies = {}
    for name in ("equal", "value", "row", "table", "index"):
        copies[name] = tmp_path / name
        shutil.copytree(store, copies[name])
    result = m.verify_independent_copy_joint(path, copies["equal"] / "state.db")
    assert result["table_count"] == len(_full_joint_of(path)["tables"])
    assert result["joint_sha256"] == _full_joint_of(path)["joint_sha256"]
    assert result["schema_sha256"] == \
        _full_joint_of(path)["tables"]["sqlite_schema"]["joint_sha256"]
    _sql(copies["value"], "UPDATE state_meta SET value = 'changed' WHERE key = 'probe/key'")
    _sql(copies["row"], "INSERT INTO state_meta (key, value) VALUES ('extra/key', 'x')")
    _sql(copies["table"], "CREATE TABLE extra (x)")
    _sql(copies["index"], "CREATE INDEX probe_idx ON state_meta(key)")
    for name in ("value", "row", "table", "index"):
        with pytest.raises(ValueError):
            m.verify_independent_copy_joint(path, copies[name] / "state.db")


def test_recovery_proof_freeze_and_reverify(tmp_path):
    m = _m()
    path = _settle(tmp_path)
    head = _head()
    _seed_head(path, head)
    request = _request(old_epoch=head["active_epoch"], old_generation=head["generation"],
                       old_head_sha256=head["head_sha256"],
                       old_head_token_sha256=_canon_sha(head))
    init_request = {"kind": request["kind"], "token": request["token"],
                    "owner_session_id": request["owner_session_id"],
                    "proof_sha256": request["proof_sha256"],
                    "old_epoch": request["old_epoch"], "new_epoch": request["new_epoch"],
                    "old_generation": request["old_generation"],
                    "old_head_sha256": request["old_head_sha256"],
                    "old_head_token_sha256": request["old_head_token_sha256"],
                    "boundary_roster_sha256": request["boundary_roster_sha256"],
                    "alias_inventory_sha256": request["alias_inventory_sha256"]}
    proof = m.freeze_recovery_proof(path, owner_session_id=SID, init_request=init_request)
    assert set(proof) == {"owner_session_id", "init_request_sha256", "roster_sha256",
                          "joint_sha256", "table_count", "file_sha256", "file_identity",
                          "recovery_proof_sha256"}
    assert proof["joint_sha256"] == _full_joint_of(path)["joint_sha256"]
    verified = m.verify_frozen_proof(path, proof)
    assert verified == {"verified": True, "owner_session_id": SID,
                        "joint_sha256": proof["joint_sha256"],
                        "table_count": proof["table_count"]}
    # The frozen inode no longer matches its proof after any durable change.
    _mutate(path, "INSERT INTO state_meta (key, value) VALUES ('later/key', 'x')")
    with pytest.raises(ValueError):
        m.verify_frozen_proof(path, proof)
    # A proof whose own bytes no longer verify refuses before any read.
    broken = dict(proof)
    broken["joint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.verify_frozen_proof(path, broken)
    # Even a self-consistent rewrite of the frozen fields refuses: the joint and
    # the file identity are re-measured, never trusted.
    rewritten = dict(proof)
    rewritten["joint_sha256"] = "0" * 64
    frozen = {key: value for key, value in rewritten.items()
              if key != "recovery_proof_sha256"}
    rewritten["recovery_proof_sha256"] = m.H("quiet-recovery-proof", frozen)
    with pytest.raises(ValueError):
        m.verify_frozen_proof(path, rewritten)
    # A malformed or foreign-shaped init request is refused at freeze time.
    for bad in (dict(init_request, extra=True),
                {name: value for name, value in init_request.items() if name != "token"},
                dict(init_request, token="not-a-uuid"),
                dict(init_request, old_epoch=init_request["new_epoch"]),
                dict(init_request, owner_session_id=OTHER),
                dict(init_request, old_head_sha256=None)):
        with pytest.raises(ValueError):
            m.freeze_recovery_proof(path, owner_session_id=SID, init_request=bad)
    with pytest.raises(ValueError):
        m.freeze_recovery_proof(path, owner_session_id="", init_request=init_request)


# ── C02 step 6: identity-aware inverse deltas ────────────────────────────────

def test_inverse_recovery_joint_happy_path_with_real_continuation(tmp_path):
    m = _m()
    source, replacement, request, result = _copy_pair(tmp_path)
    transition = _transition(request)
    report = m.verify_inverse_recovery_joint(source / "state.db", replacement / "state.db",
                                             owner_session_id=SID,
                                             approved_transition=transition)
    source_joint = _full_joint_of(source / "state.db")
    assert report["source_joint_sha256"] == source_joint["joint_sha256"]
    assert report["reconstructed_joint_sha256"] == report["source_joint_sha256"]
    assert report["table_count"] == source_joint["table_count"]
    assert report["removed_rows"] == []
    assert report["committed"] == [{"token": request["token"],
                                    "generation": 2,
                                    "operation_sha256": result["operation_sha256"]}]
    sid_hex = SID.encode("utf-8").hex()
    assert report["added_keys"] == sorted([
        m.encode_receipt3_token_key(SID, request["token"]),
        m.encode_receipt3_operation_key(SID, 2),
        m.encode_receipt3_page_key(SID, 2, 0)])
    assert [row["kind"] for row in report["updated_rows"]] == ["protected_head"]
    assert report["updated_rows"][0]["session_id"] == SID
    assert report["reversal_sha256"] != report["source_joint_sha256"]
    # The same acceptance repeats read-only (idempotent, no writes).
    again = m.verify_inverse_recovery_joint(source / "state.db", replacement / "state.db",
                                            owner_session_id=SID,
                                            approved_transition=transition)
    assert again == report
    with pytest.raises(ValueError):
        m.verify_inverse_recovery_joint(source / "state.db", replacement / "state.db",
                                        owner_session_id=SID,
                                        approved_transition={k: v for k, v in
                                                             transition.items()
                                                             if k != "proof_sha256"})


def test_inverse_recovery_joint_refuses_stray_deltas(tmp_path):
    m = _m()
    source, replacement, request, _result = _copy_pair(tmp_path)
    transition = _transition(request)

    def stray_key(path):
        _sql(path, "INSERT INTO state_meta (key, value) VALUES ('stray/key', 'x')")

    def messages_change(path):
        _sql(path, "UPDATE messages SET content = 'tampered' WHERE session_id = ?", (SID,))

    def message_removed(path):
        _sql(path, "DELETE FROM messages WHERE session_id = ?", (SID,))

    def token_key_removed(path):
        _sql(path, "DELETE FROM state_meta WHERE key = ?",
             (m.encode_receipt3_token_key(SID, request["token"]),))

    def journal_key_removed(path):
        _sql(path, "DELETE FROM state_meta WHERE key = ?",
             (m.encode_receipt3_page_key(SID, 2, 0),))

    def head_tampered(path):
        conn = _connect(path / "state.db")
        try:
            raw = conn.execute("SELECT model_config FROM sessions WHERE id = ?",
                               (SID,)).fetchone()[0]
            config = json.loads(raw)
            config["_message_copy_receipts_v2"]["head_sha256"] = "7" * 64
            conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?",
                         (json.dumps(config), SID))
            conn.commit()
        finally:
            conn.close()

    def head_removed(path):
        conn = _connect(path / "state.db")
        try:
            raw = conn.execute("SELECT model_config FROM sessions WHERE id = ?",
                               (SID,)).fetchone()[0]
            config = json.loads(raw)
            del config["_message_copy_receipts_v2"]
            conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?",
                         (json.dumps(config), SID))
            conn.commit()
        finally:
            conn.close()

    def identity_swap(path):
        # INSERT OR REPLACE on a stored row swaps its actual ROWID: a refresh
        # must be an in-place UPDATE, so this shows up as removed + added.
        before = _sql_rowid(path)
        _sql(path, "INSERT OR REPLACE INTO state_meta (key, value) VALUES ('probe/key', 'value')")
        assert _sql_rowid(path) != before      # the tamper really moved the identity

    def _sql_rowid(path):
        conn = _ro_conn(path / "state.db")
        try:
            return conn.execute(
                "SELECT rowid FROM state_meta WHERE key = 'probe/key'").fetchone()[0]
        finally:
            conn.close()

    def foreign_added(path):
        _sql(path, "INSERT INTO state_meta (key, value) VALUES (?, '{}')",
             (f"receipt3/o/{OTHER.encode('utf-8').hex()}/{'0' * 19}2",))

    for tamper in (stray_key, messages_change, message_removed, token_key_removed,
                   journal_key_removed, head_tampered, head_removed, identity_swap,
                   foreign_added):
        copy = tmp_path / f"tamper-{tamper.__name__}"
        shutil.copytree(replacement, copy)
        tamper(copy)
        with pytest.raises(ValueError):
            m.verify_inverse_recovery_joint(source / "state.db", copy / "state.db",
                                            owner_session_id=SID,
                                            approved_transition=transition)
    # The untouched replacement still verifies: the refusals above are the
    # tamper, not a broken fixture.
    assert m.verify_inverse_recovery_joint(source / "state.db", replacement / "state.db",
                                           owner_session_id=SID,
                                           approved_transition=transition)


def test_inverse_recovery_joint_row_index_refresh_and_swap_refusal(tmp_path):
    m = _m()

    def build(name, *, refresh="update", event=None, write_refresh=True):
        """A is the old latest (stored index row only); B additionally carries the
        hand-seeded committed continuation member (build_epoch_continuation_header)
        that names the indexed row, and the index row itself is refreshed exactly
        as writer_timeline[4] demands: in place, never a new identity."""
        root = tmp_path / name
        store = root / "store"
        store.mkdir(parents=True)
        path = _settle(store)
        head = _head(generation=2, head_sha256="2" * 64)
        request = _request(old_generation=2, old_head_sha256=head["head_sha256"],
                           old_head_token_sha256=_canon_sha(head))
        transition = _transition(request)
        conn = _connect(path)
        try:
            r_key = _seed_r_entry(conn, _row_ref(7), [SID, 2, 0, 0])
            conn.commit()
        finally:
            conn.close()
        _seed_head(path, head)
        source, replacement = root / "old", root / "replacement"
        shutil.copytree(store, source)
        shutil.copytree(store, replacement)
        conn = _connect(replacement / "state.db")
        try:
            third = _hand_member(conn, sid=SID, token=str(uuid.uuid4()), epoch=NEW_EPOCH,
                                 ordinal=2, generation=3,
                                 previous=head["head_sha256"], block=transition,
                                 ref=_row_ref(7))
            assert third["generation"] == 3
            raw = json.loads(conn.execute(
                "SELECT model_config FROM sessions WHERE id = ?", (SID,)).fetchone()[0])
            raw["_message_copy_receipts_v2"].update(
                {"active_epoch": NEW_EPOCH, "epoch_ordinal": 2, "generation": 3,
                 "head_sha256": third["operation_sha256"]})
            conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?",
                         (json.dumps(raw), SID))
            if write_refresh:
                target = event if event is not None else [SID, 3, 0, 0]
                value = _canonical(_r_entry(_row_ref(7), target)).decode("utf-8")
                if refresh == "update":
                    conn.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                                 (value, r_key))
                else:
                    conn.execute("INSERT OR REPLACE INTO state_meta (key, value) "
                                 "VALUES (?, ?)", (r_key, value))
            conn.commit()
        finally:
            conn.close()
        return source, replacement, request, r_key

    source, replacement, request, r_key = build("refresh-ok")
    report = m.verify_inverse_recovery_joint(source / "state.db", replacement / "state.db",
                                             owner_session_id=SID,
                                             approved_transition=_transition(request))
    kinds = sorted(row["kind"] for row in report["updated_rows"])
    assert kinds == ["protected_head", "row_index"]
    row_index = next(row for row in report["updated_rows"] if row["kind"] == "row_index")
    assert row_index["key"] == r_key == m.encode_receipt3_row_index_key(OLD_EPOCH, 7)
    assert report["reconstructed_joint_sha256"] == report["source_joint_sha256"]
    assert report["removed_rows"] == []

    # INSERT OR REPLACE (an identity swap) is refused where the in-place UPDATE
    # above was accepted -- same value, different actual ROWID.
    swapped_source, swapped, swapped_request, _key = build("refresh-swap",
                                                           refresh="replace")
    conn = _ro_conn(swapped_source / "state.db")
    try:
        old_rowid = conn.execute("SELECT rowid FROM state_meta WHERE key = ?",
                                 (r_key,)).fetchone()[0]
    finally:
        conn.close()
    conn = _ro_conn(swapped / "state.db")
    try:
        new_rowid = conn.execute("SELECT rowid FROM state_meta WHERE key = ?",
                                 (r_key,)).fetchone()[0]
    finally:
        conn.close()
    assert new_rowid != old_rowid      # the tamper really moved the row identity
    with pytest.raises(ValueError):
        m.verify_inverse_recovery_joint(swapped_source / "state.db", swapped / "state.db",
                                        owner_session_id=SID,
                                        approved_transition=_transition(swapped_request))

    # A stale refresh (the committed member that names the row moves the expected
    # coordinate) and a missing refresh refuse: the index must point at the exact
    # latest committed item.
    for name, kwargs in (("refresh-stale", {"event": [SID, 3, 0, 1]}),
                         ("refresh-missing", {"write_refresh": False})):
        stale_source, stale, stale_request, _key = build(name, **kwargs)
        with pytest.raises(ValueError):
            m.verify_inverse_recovery_joint(stale_source / "state.db", stale / "state.db",
                                            owner_session_id=SID,
                                            approved_transition=_transition(stale_request))
