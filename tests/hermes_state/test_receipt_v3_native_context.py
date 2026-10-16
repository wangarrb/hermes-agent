"""Q01 (receipt r4): typed native_session_contexts in the v3 operation header.

RED contract from docs/schema31-receipt-v2-design-r4-amendments.md section 4
and the effective r4 machine spec (protocol.context_binding,
record_shapes.operation_header.native_session_contexts):

  * single owner -> exactly 1 context item, compression child -> exactly the
    owner+actor 2 items, sorted by session_id strict UTF-8 bytes;
  * each closed item is {session_id, native_binding_fields, lineage_markers,
    context_sha256} where native_binding_fields is the ordered typed preimage
    [[id, SQL(v)], [started_at, SQL(v)], [parent_session_id, SQL(v)],
    [source, SQL(v)]] read from the real sessions row and lineage_markers is
    [[_delegate_from, SQL(v|null)], [_branched_from, ...], [_reset_from, ...]]
    from the existing model_config projection (explicit null when absent);
  * context_sha256 = H("native-session-context",
    {domain_sha256, native_binding_fields, lineage_markers});
  * the reader reconstructs the original four values from the typed preimage
    and the original unchanged binding algorithm still reproduces the old
    binding hash (typed new hash is not passed off as the old one);
  * operation_sha256 = H("operation", ALL header fields except itself,
    including native_session_contexts, regardless of JSON property insertion
    order -- no truncation at the field position.

No full config, bank, cleaner, message body or UID enters the context item.
RED against the unmodified b26ba5c4 tree (module/API absent); real sqlite
fixtures, hashes recomputed independently, nothing mocked.
"""

from __future__ import annotations

import hashlib
import json
import uuid

import pytest

from hermes_state import SessionDB

_PREFIX = b"hermes-copy-receipt-v2\0"


def C(x) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8", "strict")


def H(label: str, x) -> str:
    return hashlib.sha256(_PREFIX + label.encode("ascii") + b"\0" + C(x)).hexdigest()


def digest(x) -> str:
    """The unchanged live source binding digest (hindsight_incremental_sources.py)."""
    return hashlib.sha256(C(x)).hexdigest()


DOMAIN = "d" * 64  # shape placeholder for domain_sha256 D (approval is separate)


def _m():
    import importlib
    return importlib.import_module("hermes_state_copy_provenance")


@pytest.fixture()
def db(tmp_path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _seed_owner(db, sid="owner-01"):
    db.create_session(sid, "cli", model="m")
    return db._conn.execute(
        "SELECT id, started_at, parent_session_id, source FROM sessions WHERE id = ?",
        (sid,)).fetchone()


def _set_markers(db, sid, **markers):
    db._conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?",
                     (json.dumps(markers), sid))
    db._conn.commit()


def _four(row):
    return [row["id"], row["started_at"], row["parent_session_id"], row["source"]]


def test_single_owner_context_is_one_typed_item(db):
    row = _seed_owner(db)
    m = _m()
    contexts = m.observe_native_session_contexts(
        db._conn, owner_session_id="owner-01", actor_session_id="owner-01",
        kind="inplace_compaction", domain_sha256=DOMAIN)
    assert [c["session_id"] for c in contexts] == ["owner-01"]
    item = contexts[0]
    assert set(item) == {"session_id", "native_binding_fields", "lineage_markers", "context_sha256"}
    assert [name for name, _ in item["native_binding_fields"]] == [
        "id", "started_at", "parent_session_id", "source"]
    decoded = [m.sql_value_decode(value) for _, value in item["native_binding_fields"]]
    assert decoded == _four(row)
    assert [name for name, _ in item["lineage_markers"]] == [
        "_delegate_from", "_branched_from", "_reset_from"]
    # Absent permitted markers project explicit null.
    assert [m.sql_value_decode(value) for _, value in item["lineage_markers"]] == [None, None, None]
    # No full config/bank/cleaner/body/UID data in the item.
    rendered = json.dumps(item)
    assert "model_config" not in rendered and "content" not in rendered


def test_compression_child_context_is_owner_plus_actor_sorted(db):
    _seed_owner(db)
    db.create_session("child-02", "cli", model="m")
    _set_markers(db, "child-02", _delegate_from="owner-01")
    m = _m()
    contexts = m.observe_native_session_contexts(
        db._conn, owner_session_id="owner-01", actor_session_id="child-02",
        kind="compression_child", domain_sha256=DOMAIN)
    expected_order = sorted(["owner-01", "child-02"], key=lambda s: s.encode("utf-8"))
    assert [c["session_id"] for c in contexts] == expected_order == ["child-02", "owner-01"]
    child = contexts[0]
    markers = {name: m.sql_value_decode(value) for name, value in child["lineage_markers"]}
    assert markers == {"_delegate_from": "owner-01", "_branched_from": None, "_reset_from": None}
    owner = contexts[1]
    assert [m.sql_value_decode(value) for _, value in owner["lineage_markers"]] == [None, None, None]


def test_context_sha256_recomputes_over_the_exact_preimage(db):
    _seed_owner(db)
    m = _m()
    item = m.observe_native_session_contexts(
        db._conn, owner_session_id="owner-01", actor_session_id="owner-01",
        kind="inplace_compaction", domain_sha256=DOMAIN)[0]
    expected = H("native-session-context", {
        "domain_sha256": DOMAIN,
        "native_binding_fields": item["native_binding_fields"],
        "lineage_markers": item["lineage_markers"]})
    assert item["context_sha256"] == expected


def test_reader_reconstructs_old_binding_from_typed_preimage(db):
    row = _seed_owner(db)
    _set_markers(db, "owner-01", _branched_from=None)
    m = _m()
    item = m.observe_native_session_contexts(
        db._conn, owner_session_id="owner-01", actor_session_id="owner-01",
        kind="payload_repair", domain_sha256=DOMAIN)[0]
    four = m.native_binding_values_from_context(item)
    assert four == _four(row)
    # The original algorithm is unchanged and independently reproduced here;
    # the typed context hash never impersonates the old binding hash.
    assert m.native_session_binding(four) == digest(
        {"id": row["id"], "started_at": row["started_at"],
         "parent_session_id": row["parent_session_id"], "source": row["source"]})
    assert item["context_sha256"] != m.native_session_binding(four)


def test_operation_sha256_is_order_independent_and_covers_every_field(db):
    _seed_owner(db)
    m = _m()
    context = m.observe_native_session_contexts(
        db._conn, owner_session_id="owner-01", actor_session_id="owner-01",
        kind="inplace_compaction", domain_sha256=DOMAIN)[0]
    header = {
        "schema_version": "hermes-receipt-operation/3", "domain_sha256": DOMAIN,
        "owner_session_id": "owner-01", "actor_session_id": "owner-01",
        "epoch": "sqlite:" + "a" * 64, "epoch_ordinal": 1, "generation": 1,
        "operation_id": str(uuid.uuid4()), "request_sha256": "1" * 64,
        "previous_head_sha256": None,
        "group": {"group_id": str(uuid.uuid4()),
                  "members": [{"owner_session_id": "owner-01",
                               "operation_id": str(uuid.uuid4()), "ordinal": 0}]},
        "kind": "inplace_compaction", "base_fence_sha256": "2" * 64,
        "page_count": 1, "item_count": 1, "pages_root_sha256": "3" * 64,
        "result_sha256": "4" * 64, "continuation": None,
        "native_session_contexts": [context]}
    sealed = m.seal_operation_header(header)
    op_sha = sealed["operation_sha256"]
    assert sealed["native_session_contexts"] == [context]
    # Independent recomputation: H("operation", all fields minus itself).
    assert op_sha == H("operation", {k: v for k, v in header.items() if k != "operation_sha256"})
    # Re-inserting the same fields in reverse order cannot change the digest.
    reversed_header = {k: header[k] for k in reversed(list(header))}
    assert m.seal_operation_header(reversed_header)["operation_sha256"] == op_sha
    # A present-but-ignored self hash value never changes the digest.
    with_bogus_self = dict(header, operation_sha256="f" * 64)
    assert m.seal_operation_header(with_bogus_self)["operation_sha256"] == op_sha
    # No truncation at the field position: every field changes the digest,
    # including the ones sorting after "operation_sha256" and the contexts.
    mutations = {
        "request_sha256": "5" * 64, "pages_root_sha256": "6" * 64,
        "result_sha256": "7" * 64, "page_count": 2,
        "previous_head_sha256": "8" * 64, "owner_session_id": "owner-02",
        "schema_version": "hermes-receipt-operation/4",
        "native_session_contexts": [],
    }
    for field, value in mutations.items():
        mutated = dict(header)
        mutated[field] = value
        assert m.seal_operation_header(mutated)["operation_sha256"] != op_sha, field
