"""Writer-committed copy evidence; identical independent input is never ancestry.

All databases are synthetic temporary Core stores. Hash assertions independently
implement the approved physical-reader contract, not the writer's hash helper.
"""
import copy
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB

KEY = "_message_copy_receipts_v1"
SCHEMA = "hermes-message-copy-receipts/1"
MUTABLE = {"active", "compacted", "display_order"}


def digest(value):
    def blob(v):
        if isinstance(v, bytes):
            return {"sqlite_blob": v.hex()}
        if isinstance(v, dict):
            return {k: blob(x) for k, x in v.items()}
        return v
    raw = json.dumps(blob(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def immutable(row):
    return {k: v for k, v in row.items() if k not in MUTABLE}


def copy_hash(row):
    return digest({k: v for k, v in immutable(row).items() if k not in {"id", "session_id"}})


def snapshot(db):
    with db._read_ctx() as con:
        return ([dict(r) for r in con.execute("SELECT * FROM messages ORDER BY id")],
                [dict(r) for r in con.execute("SELECT * FROM sessions ORDER BY id")])


def schema(db):
    with db._read_ctx() as con:
        return (con.execute("PRAGMA schema_version").fetchone()[0],
                [tuple(r) for r in con.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name")],
                [tuple(r) for r in con.execute("PRAGMA table_info(messages)")])


def carried(db, sid="s", row_id=None):
    messages = db.get_messages(sid)
    msg = next(m for m in messages if m["id"] == row_id) if row_id else messages[-1]
    msg["_row_id"] = msg["id"]  # Explicit causal native reference, never a fingerprint lookup.
    return msg


def container(db, sid="s"):
    cfg = json.loads(db.get_session(sid)["model_config"] or "{}")
    assert KEY in cfg, "genuine native copy must commit a receipt in session metadata"
    return cfg[KEY]


def validate(db, sid="s"):
    receipt = container(db, sid)
    with sqlite3.connect(Path(db.db_path).resolve().as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        con.execute("BEGIN")
        stat = Path(db.db_path).stat()
        epoch = "sqlite:" + digest({"inode": [stat.st_dev, stat.st_ino],
                                    "application_id": con.execute("PRAGMA application_id").fetchone()[0]})
        assert set(receipt) == {"schema_version", "session_id", "source_epoch", "operations"}
        assert receipt["schema_version"] == SCHEMA
        assert receipt["session_id"] == sid and receipt["source_epoch"] == epoch
        operations, targets = set(), set()
        for op in receipt["operations"]:
            assert set(op) == {"operation_id", "kind", "copies", "receipt_sha256"}
            assert str(uuid.UUID(op["operation_id"])) == op["operation_id"]
            assert op["operation_id"] not in operations
            operations.add(op["operation_id"])
            assert op["kind"] == "inplace_compaction" and op["copies"]
            assert op["receipt_sha256"] == digest({k: v for k, v in op.items() if k != "receipt_sha256"})
            for edge in op["copies"]:
                assert set(edge) == {"source_row_id", "target_row_id", "source_row_hash", "target_row_hash", "copy_hash"}
                a, b = edge["source_row_id"], edge["target_row_id"]
                assert type(a) is int and type(b) is int and 0 < a < b
                assert b not in targets
                targets.add(b)
                old = dict(con.execute("SELECT * FROM messages WHERE id=?", (a,)).fetchone())
                new = dict(con.execute("SELECT * FROM messages WHERE id=?", (b,)).fetchone())
                assert old["session_id"] == new["session_id"] == sid
                assert edge["source_row_hash"] == digest(immutable(old))
                assert edge["target_row_hash"] == digest(immutable(new))
                assert edge["copy_hash"] == copy_hash(old) == copy_hash(new)
                assert {k: v for k, v in immutable(old).items() if k not in {"id", "session_id"}} == {
                    k: v for k, v in immutable(new).items() if k not in {"id", "session_id"}}
        con.rollback()
    # Fixed content-free fields only; no payload or provider config can enter the receipt.
    assert "secret synthetic payload" not in json.dumps(receipt)
    return receipt


@pytest.fixture(autouse=True)
def synthetic_pragmas(tmp_path, monkeypatch):
    # Real FULL-sync initialization took minutes on this fixture filesystem.
    # This proves transactional atomicity, not power-failure durability.
    import hermes_state
    original = hermes_state.apply_database_pragmas

    def apply(con, *args, **kwargs):
        files = con.execute("PRAGMA database_list").fetchall()
        assert all(not row[2] or Path(row[2]).resolve().is_relative_to(tmp_path.resolve()) for row in files)
        original(con, *args, **kwargs)
        con.execute("PRAGMA synchronous=OFF")

    monkeypatch.setattr(hermes_state, "apply_database_pragmas", apply)


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    store.create_session("s", "cli", model_config={"provider": "synthetic", "unrelated": {"keep": True}})
    store.append_message("s", "user", content="secret synthetic payload", timestamp=100.0,
                         api_content="synthetic sidecar", display_metadata={"attachment": "synthetic"})
    yield store
    store.close()


@pytest.mark.parametrize("tail_count", [0, 1])
def test_exact_carried_copy_chain_is_atomic_content_free_and_schema_neutral(db, tail_count):
    before_schema = schema(db)
    old = snapshot(db)[0][0]
    msg = carried(db)
    old_id = msg["_row_id"]
    for generation in range(2):
        prior = copy.deepcopy(container(db)) if generation else None
        count = db.archive_and_compact("s", [{"role": "assistant", "content": "summary", "_compressed_summary": True}, msg],
                                       tail_count=tail_count, model_config_patch={"generation": generation})
        receipt = validate(db)
        assert count == db.get_session("s")["message_count"] == 2
        assert len(receipt["operations"]) == generation + 1
        if prior:
            assert receipt["operations"][:-1] == prior["operations"]
        edge = receipt["operations"][-1]["copies"][0]
        assert edge["source_row_id"] == old_id
        assert edge["target_row_id"] == msg["_row_id"]
        old_id = msg["_row_id"]
        msg = carried(db, row_id=old_id)
    assert schema(db) == before_schema
    assert immutable(snapshot(db)[0][0]) == immutable(old)
    assert len(snapshot(db)[0]) == 5


def test_sql_watermark_clones_exact_tail_and_preserves_late_old_date_events(db):
    watermark = db.get_active_message_watermark("s")
    tail = db.append_message("s", "assistant", content="during summary", timestamp=1.0,
                             tool_calls=[{"id": "synthetic-call", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                             token_count=7, reasoning="synthetic reasoning", api_content="all fields")
    source = next(r for r in snapshot(db)[0] if r["id"] == tail)
    assert db.archive_and_compact("s", [{"role": "user", "content": "summary"}], watermark=watermark) == 2
    receipt = validate(db)
    edges = receipt["operations"][0]["copies"]
    assert len(edges) == 1 and edges[0]["source_row_id"] == tail
    clone = next(r for r in snapshot(db)[0] if r["id"] == edges[0]["target_row_id"])
    assert copy_hash(clone) == copy_hash(source)
    assert clone["timestamp"] == 1.0
    assert db.get_session("s")["tool_call_count"] == 1
    prior = copy.deepcopy(receipt)
    independent = db.append_message("s", "user", content="secret synthetic payload", timestamp=100.0,
                                    api_content="synthetic sidecar", display_metadata={"attachment": "synthetic"})
    assert container(db) == prior
    assert independent not in {e["target_row_id"] for op in receipt["operations"] for e in op["copies"]}
    assert db.get_session("s")["message_count"] == 3


@pytest.mark.parametrize("reference", ["absent", "missing", "wrong_sid", "bool", "str", "changed_content", "changed_sidecar", "changed_token_count"])
def test_absent_or_inexact_reference_never_fabricates_receipt(db, reference):
    msg = carried(db)
    if reference == "absent":
        msg.pop("_row_id")
    elif reference == "missing":
        msg["_row_id"] = 987654321
    elif reference == "wrong_sid":
        db.create_session("other", "cli")
        msg["_row_id"] = db.append_message("other", "user", content=msg["content"], timestamp=msg["timestamp"])
    elif reference == "bool":
        msg["_row_id"] = True
    elif reference == "str":
        msg["_row_id"] = str(msg["_row_id"])
    elif reference == "changed_content":
        msg["content"] = "not the old event"
    elif reference == "changed_sidecar":
        msg["api_content"] = "not the old sidecar"
    else:
        msg["token_count"] = 12
    db.archive_and_compact("s", [msg])
    assert KEY not in json.loads(db.get_session("s")["model_config"])
    assert db.get_session("s")["message_count"] == 1


def test_independent_identical_append_has_no_receipt_even_with_equal_display_identity(db):
    msg = carried(db)
    msg.pop("_row_id")
    new_id = db.append_message("s", "user", content=msg["content"], timestamp=msg["timestamp"],
                               api_content=msg["api_content"], display_metadata=msg["display_metadata"])
    rows = snapshot(db)[0]
    assert copy_hash(rows[0]) == copy_hash(rows[1])
    assert rows[0]["display_identity"] == rows[1]["display_identity"]
    assert rows[0]["display_order"] == rows[1]["display_order"]
    assert KEY not in json.loads(db.get_session("s")["model_config"])
    # A later *genuine* copy of this independent event retains its own exact parent.
    db.archive_and_compact("s", [carried(db, row_id=new_id)])
    assert validate(db)["operations"][0]["copies"][0]["source_row_id"] == new_id


@pytest.mark.parametrize("update", ["create", "meta", "model", "runtime", "patch"])
def test_metadata_updates_preserve_writer_owned_history(db, update):
    db.archive_and_compact("s", [carried(db)])
    before = copy.deepcopy(validate(db))
    if update == "create":
        db.create_session("s", "cli", model_config={"provider": "next"})
    elif update == "meta":
        db.update_session_meta("s", json.dumps({"provider": "next", "_reset_from": "older"}))
    elif update == "model":
        db.update_session_model("s", "next", provider="next")
    elif update == "runtime":
        db.update_session_runtime_lock("s", model="next", provider="next")
    else:
        db.patch_session_model_config("s", {"provider": "next", "unrelated": None})
    assert validate(db) == before
    # A legitimate read-back/resume may supply the exact protected bytes, never replace them.
    cfg = json.loads(db.get_session("s")["model_config"])
    db.create_session("s", "cli", model_config=cfg)
    db.update_session_meta("s", json.dumps(cfg))
    assert validate(db) == before


@pytest.mark.parametrize("write", ["new_create", "other_sid", "patch_clear", "patch_replace", "meta_replace", "create_replace", "compaction_patch"])
def test_user_metadata_cannot_install_overwrite_or_clear_owned_receipts(db, write):
    db.archive_and_compact("s", [carried(db)])
    receipt = copy.deepcopy(validate(db))
    before = snapshot(db)
    with pytest.raises(ValueError, match="copy receipt"):
        if write == "new_create":
            receipt["session_id"] = "new"
            db.create_session("new", "cli", model_config={KEY: receipt})
        elif write == "other_sid":
            db.create_session("other", "cli", model_config={KEY: receipt}, parent_session_id="s")
        elif write == "patch_clear":
            db.patch_session_model_config("s", {KEY: None})
        elif write == "patch_replace":
            db.patch_session_model_config("s", {KEY: {}})
        elif write == "meta_replace":
            db.update_session_meta("s", json.dumps({KEY: {}}))
        elif write == "create_replace":
            db.create_session("s", "cli", model_config={KEY: {}})
        else:
            db.archive_and_compact("s", [carried(db)], model_config_patch={KEY: None})
    assert snapshot(db) == before
    assert validate(db) == container(db)


@pytest.mark.parametrize("damage", ["null", "shape", "sid", "epoch", "hash", "uuid", "duplicate", "bool_id", "cycle", "extra"])
def test_malformed_owned_containers_fail_closed_without_message_changes(db, damage):
    db.archive_and_compact("s", [carried(db)])
    cfg = json.loads(db.get_session("s")["model_config"])
    receipt = cfg[KEY]
    op = receipt["operations"][0]
    if damage == "null":
        cfg[KEY] = None
    elif damage == "shape":
        receipt.pop("schema_version")
    elif damage == "sid":
        receipt["session_id"] = "another"
    elif damage == "epoch":
        receipt["source_epoch"] = "sqlite:" + "0" * 64
    elif damage == "hash":
        op["receipt_sha256"] = "0" * 64
    elif damage == "uuid":
        op["operation_id"] = None
    elif damage == "duplicate":
        receipt["operations"].append(copy.deepcopy(op))
    elif damage == "bool_id":
        op["copies"][0]["source_row_id"] = True
    elif damage == "cycle":
        op["copies"][0]["source_row_id"] = op["copies"][0]["target_row_id"]
    else:
        receipt["payload"] = "not allowed"
    if damage in {"uuid", "bool_id", "cycle"}:
        op["receipt_sha256"] = digest({k: v for k, v in op.items() if k != "receipt_sha256"})
    db._execute_write(lambda con: con.execute("UPDATE sessions SET model_config=? WHERE id='s'", (json.dumps(cfg),)))
    before = snapshot(db)
    with pytest.raises(ValueError, match="copy receipt"):
        db.archive_and_compact("s", [carried(db)])
    assert snapshot(db) == before
    with pytest.raises(ValueError, match="copy receipt"):
        db.update_session_meta("s", "{}")
    assert snapshot(db) == before


def test_receipt_save_failure_rolls_back_rows_counters_config_and_live_ids(db):
    msg = carried(db)
    before_msg = copy.deepcopy(msg)
    before = snapshot(db)
    db._execute_write(lambda con: con.execute("""CREATE TRIGGER reject_copy_receipt BEFORE UPDATE OF model_config ON sessions
        WHEN json_type(NEW.model_config, '$._message_copy_receipts_v1') IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'synthetic receipt write failure'); END"""))
    with pytest.raises(sqlite3.IntegrityError, match="synthetic receipt write failure"):
        db.archive_and_compact("s", [msg], model_config_patch={"would_commit": True})
    assert snapshot(db) == before
    assert msg == before_msg
    assert KEY not in json.loads(db.get_session("s")["model_config"])


def test_native_store_binding_survives_a_b_a_home_switches(tmp_path, monkeypatch):
    import hermes_state
    from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    # Remove conftest's static DB override; use the real context-local resolver.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    set_multiplex_active(True)
    states = {}
    for home in ("A", "B", "A"):
        path = tmp_path / home
        home_token = set_hermes_home_override(path)
        secret_token = set_secret_scope({})
        db = SessionDB()
        try:
            if home not in states:
                db.create_session("s", "cli")
                db.append_message("s", "user", content=home, timestamp=100.0)
                db.archive_and_compact("s", [carried(db)])
                states[home] = copy.deepcopy(validate(db))
            else:
                assert validate(db) == states[home]
                db.archive_and_compact("s", [carried(db)])
                assert validate(db)["operations"][:-1] == states[home]["operations"]
            assert Path(db.db_path).parent == path
        finally:
            db.close()
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    assert states["A"]["source_epoch"] != states["B"]["source_epoch"]



def test_reset_placeholder_create_adds_provider_without_losing_receipts(db):
    db.update_session_meta("s", json.dumps({"_reset_from": "previous"}))
    db.archive_and_compact("s", [carried(db)])
    before = copy.deepcopy(validate(db))
    db.create_session("s", "cli", model_config={"provider": "resolved", "model": "resolved"})
    cfg = json.loads(db.get_session("s")["model_config"])
    assert cfg["provider"] == "resolved"
    assert cfg["_reset_from"] == "previous"
    assert validate(db) == before


def test_in_memory_store_does_not_mint_durable_copy_evidence():
    from hermes_state_copy_provenance import append_copy_operation, source_epoch
    with sqlite3.connect(":memory:") as con:
        assert source_epoch(con) is None
        assert append_copy_operation(con, "s", "{}", [(None, 2)]) == "{}"


def test_retry_after_precommit_fault_keeps_original_causal_references(db, monkeypatch):
    import hermes_state_messages
    original = hermes_state_messages.append_copy_operation
    attempts = []
    msg = carried(db)
    old_id = msg["_row_id"]

    def transient(con, sid, config, pairs):
        result = original(con, sid, config, pairs)
        attempts.append(pairs[0][0]["id"])
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        return result

    monkeypatch.setattr(hermes_state_messages, "append_copy_operation", transient)
    db.archive_and_compact("s", [msg])
    receipt = validate(db)
    assert attempts == [old_id, old_id]
    assert len(receipt["operations"]) == 1 and len(snapshot(db)[0]) == 2
    assert receipt["operations"][0]["copies"][0]["target_row_id"] == msg["_row_id"]


def test_malformed_json_cannot_erase_private_history(db):
    db._execute_write(lambda con: con.execute("UPDATE sessions SET model_config=? WHERE id='s'",
                                              ('{"_message_copy_receipts_v1":',)))
    before = snapshot(db)
    with pytest.raises(ValueError, match="copy receipt"):
        db.update_session_meta("s", "{}")
    with pytest.raises(ValueError, match="copy receipt"):
        db.patch_session_model_config("s", {"provider": "x"})
    assert snapshot(db) == before
