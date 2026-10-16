"""C01 (receipt r4): immutable epoch range keys must carry the start generation.

RED contract for receipt v3 (schema/3) epoch range checkpoints, from
docs/schema31-receipt-v2-design-r4-amendments.md section 2 ("C01") and the
effective r4 machine spec (protocol.range_checkpoints,
record_shapes.epoch_range_checkpoint):

  * key codec ``receipt3/e/<SIDhex>/<epochordinal20>/<start_generation20>``:
    old short e-keys, extra components and non-canonical encodings are refused,
    never upserted as the current index;
  * body validation: count 1..1024, ``end-start+1=count``, ``end<=start+1023``;
  * epoch-local chunking: ``start=s0+1024*k`` from the actual first header
    generation ``s0`` of that epoch (not global ``g%1024``); a full block seals
    with its last operation, a non-empty partial tail seals only on epoch exit,
    an empty range is never created;
  * ``range_root_sha256`` uses the frozen B framing over the checkpoint
    coordinate followed by every member header/page C bytes in order,
    recomputed independently here;
  * the public SessionDB meta surface refuses the reserved ``receipt3/``
    namespace (set_meta including the cursor path, and list_meta_prefix), so
    no generic setter/scan can act as the protected writer or long-history
    reader.

This file is RED against the unmodified b26ba5c4 tree: the v3 codec module and
the meta guards do not exist yet. Everything runs on real sqlite fixtures with
independently recomputed hashes -- no mocks.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hermes_state import SessionDB

# ── Frozen codec primitives (r2 section "codec", repeated in the r4 spec) ───
_PREFIX = b"hermes-copy-receipt-v2\0"


def C(x) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8", "strict")


def H(label: str, x) -> str:
    return hashlib.sha256(_PREFIX + label.encode("ascii") + b"\0" + C(x)).hexdigest()


def B(label: str, parts) -> str:
    h = hashlib.sha256(_PREFIX + label.encode("ascii") + b"\0")
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


def sid_hex(sid: str) -> str:
    return sid.encode("utf-8").hex()


def enc(v: int) -> str:
    return f"{v:020d}"


SID = "owner-01"
EPOCH = "sqlite:" + "a" * 64


def _m():
    """Import the receipt provenance module that C01 must extend."""
    import importlib
    return importlib.import_module("hermes_state_copy_provenance")


# ── module contracts ────────────────────────────────────────────────────────

def test_epoch_key_roundtrip_and_canonical_form():
    m = _m()
    key = m.encode_receipt3_epoch_key(SID, 1, 1)
    assert key == f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}"
    assert m.parse_receipt3_epoch_key(key) == {
        "session_id": SID, "epoch_ordinal": 1, "start_generation": 1}
    # A new epoch continuing from g2058 (amendment example): blocks are
    # [2058,3081] and [3082,4105]; block keys never collide across epochs.
    first = m.encode_receipt3_epoch_key(SID, 3, 2058)
    second = m.encode_receipt3_epoch_key(SID, 3, 3082)
    assert first == f"receipt3/e/{sid_hex(SID)}/{enc(3)}/{enc(2058)}"
    assert second == f"receipt3/e/{sid_hex(SID)}/{enc(3)}/{enc(3082)}"
    assert first != second
    assert m.parse_receipt3_epoch_key(first) != m.parse_receipt3_epoch_key(second)


@pytest.mark.parametrize("bad", [
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}",                    # old short e-key
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}/extra",     # extra component
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)} ",          # trailing space
    f"receipt3/e/{sid_hex(SID)}/1/{enc(1)}",                  # non-canonical ordinal
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}/1",                  # non-canonical generation
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}/+{enc(1)[1:]}",      # signed encoding
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(0)}",           # non-positive generation
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(2**63)}",       # out of int64
    f"receipt3/e/{sid_hex(SID)}/{enc(2**63)}/{enc(1)}",       # out of int64 ordinal
    f"receipt3/e/zz/{enc(1)}/{enc(1)}",                       # non-hex SID
    f"receipt3/e/abc/{enc(1)}/{enc(1)}",                      # odd-length SID hex
    f"receipt3/e/{sid_hex(SID).upper()}/{enc(1)}/{enc(1)}",   # uppercase SID hex
    f"receipt3/o/{sid_hex(SID)}/{enc(1)}",                    # wrong family
    f"receipt3/e//{enc(1)}/{enc(1)}",                         # empty SID
    f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}\n",         # trailing newline
    f"receipt3/e/x{sid_hex(SID)}/{enc(1)}/{enc(1)}",          # wrong length SID hex
])
def test_epoch_key_rejections(bad):
    m = _m()
    with pytest.raises(ValueError):
        m.parse_receipt3_epoch_key(bad)


@pytest.mark.parametrize("args", [
    (SID, 0, 1), (SID, -1, 1), (SID, 2**63, 1), (SID, True, 1), (SID, "1", 1),
    (SID, 1, 0), (SID, 1, -1), (SID, 1, 2**63), (SID, 1, True), (SID, 1, "1"),
    ("", 1, 1), (None, 1, 1),
])
def test_encode_rejects_nonpositive_nonint_and_overflow(args):
    m = _m()
    with pytest.raises((ValueError, TypeError)):
        m.encode_receipt3_epoch_key(*args)


def _valid_body(**overrides):
    body = {"schema_version": "hermes-receipt-range/3", "owner_session_id": SID,
            "epoch": EPOCH, "epoch_ordinal": 1, "start_generation": 1,
            "end_generation": 1024, "before_head_sha256": None,
            "end_head_sha256": "b" * 64, "operation_count": 1024,
            "page_count": 10, "range_root_sha256": "c" * 64}
    body.update(overrides)
    return body


def test_checkpoint_body_accepts_full_block():
    m = _m()
    m.validate_epoch_range_checkpoint(_valid_body())  # must not raise


@pytest.mark.parametrize("overrides", [
    {"operation_count": 1023},                 # end-start+1 != count
    {"operation_count": 0},                    # count floor
    {"operation_count": 1025, "end_generation": 1025},   # count ceiling
    {"end_generation": 1025},                  # end-start+1 = 1025 > 1024
    {"end_generation": 0},                     # end < start
    {"start_generation": 0},                   # non-positive
    {"epoch": "a" * 64},                       # epoch must carry sqlite: prefix
    {"epoch": "sqlite:" + "A" * 64},           # uppercase hex
    {"epoch_ordinal": 0},                      # non-positive ordinal
    {"end_head_sha256": "B" * 64},             # non-lowercase hash
    {"range_root_sha256": "c" * 63},           # wrong length
    {"before_head_sha256": ""},                # null or sha256 only
    {"page_count": -1},                        # nonnegative only
    {"page_count": "10"},                      # integers only, no bool/float/str
    {"extra": 1},                              # strict exact fields
])
def test_checkpoint_body_rejections(overrides):
    m = _m()
    body = _valid_body()
    if "extra" in overrides:
        body["extra"] = overrides["extra"]
    else:
        body.update(overrides)
    with pytest.raises(ValueError):
        m.validate_epoch_range_checkpoint(body)


def test_checkpoint_body_rejects_missing_field():
    m = _m()
    body = _valid_body()
    del body["operation_count"]
    with pytest.raises(ValueError):
        m.validate_epoch_range_checkpoint(body)


def _member(generation: int, *, epoch: str = EPOCH, pages: int = 1, prev: str | None = None):
    """One synthetic sealed member: an operation header plus its ordered pages."""
    header = {"schema_version": "hermes-receipt-operation/3", "domain_sha256": "d" * 64,
              "owner_session_id": SID, "actor_session_id": SID, "epoch": epoch,
              "epoch_ordinal": 1, "generation": generation,
              "operation_id": "00000000-0000-4000-8000-" + f"{generation:012d}",
              "request_sha256": hashlib.sha256(f"req-{generation}".encode()).hexdigest(),
              "previous_head_sha256": prev,
              "kind": "inplace_compaction", "base_fence_sha256": "e" * 64,
              "page_count": pages, "item_count": pages, "pages_root_sha256": "f" * 64,
              "result_sha256": "0" * 64, "continuation": None,
              "operation_sha256": hashlib.sha256(f"op-{generation}".encode()).hexdigest()}
    return {"header": header, "pages": [{"page_ordinal": i} for i in range(pages)]}


def test_chunking_uses_epoch_local_s0_and_sealing_rules():
    m = _m()
    s0 = 2058
    block = [_member(g) for g in range(s0, s0 + 1024)]
    key, body = m.build_epoch_range_checkpoint(
        owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
        members=block, epoch_exit=False)
    assert key == f"receipt3/e/{sid_hex(SID)}/{enc(3)}/{enc(2058)}"
    assert (body["start_generation"], body["end_generation"], body["operation_count"]) == (2058, 3081, 1024)
    assert body["schema_version"] == "hermes-receipt-range/3"
    assert body["epoch"] == EPOCH and body["epoch_ordinal"] == 3
    assert body["before_head_sha256"] == block[0]["header"]["previous_head_sha256"]
    assert body["end_head_sha256"] == block[-1]["header"]["operation_sha256"]
    assert body["page_count"] == 1024

    # Second full block of the same epoch: s0+1024*k alignment, distinct key.
    block2 = [_member(g) for g in range(s0 + 1024, s0 + 2048)]
    key2, body2 = m.build_epoch_range_checkpoint(
        owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
        members=block2, epoch_exit=False)
    assert key2 == f"receipt3/e/{sid_hex(SID)}/{enc(3)}/{enc(3082)}"
    assert (body2["start_generation"], body2["end_generation"]) == (3082, 4105)

    # The open tail is NOT sealed early.
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(
            owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
            members=[_member(g) for g in range(s0, s0 + 10)], epoch_exit=False)
    # Epoch exit seals exactly one non-empty partial tail.
    key3, body3 = m.build_epoch_range_checkpoint(
        owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
        members=[_member(g) for g in range(s0, s0 + 10)], epoch_exit=True)
    assert (key3, body3["operation_count"]) == (f"receipt3/e/{sid_hex(SID)}/{enc(3)}/{enc(2058)}", 10)
    # No empty range, no start that is not s0+1024*k, no epoch crossing.
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3,
                                       s0=s0, members=[], epoch_exit=True)
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3,
                                       s0=s0, members=[_member(g) for g in range(s0 + 1, s0 + 11)],
                                       epoch_exit=True)
    other_epoch = "sqlite:" + "b" * 64
    crossing = [_member(s0), _member(s0 + 1, epoch=other_epoch)]
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3,
                                       s0=s0, members=crossing, epoch_exit=True)
    # More than one full block in a single checkpoint is not a range.
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(
            owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
            members=[_member(g) for g in range(s0, s0 + 1025)], epoch_exit=False)


def test_range_root_uses_b_framing_independent_recompute():
    m = _m()
    s0 = 1
    members = [_member(g, pages=2 if g == 2 else 1, prev=None if g == 1 else hashlib.sha256(f"op-{g - 1}".encode()).hexdigest())
               for g in range(s0, s0 + 3)]
    key, body = m.build_epoch_range_checkpoint(
        owner_session_id=SID, epoch=EPOCH, epoch_ordinal=1, s0=s0,
        members=members, epoch_exit=True)
    coordinate = {"owner_session_id": SID, "epoch": EPOCH, "epoch_ordinal": 1,
                  "start_generation": 1, "end_generation": 3,
                  "before_head_sha256": None,
                  "end_head_sha256": members[-1]["header"]["operation_sha256"],
                  "operation_count": 3, "page_count": 4}
    parts = [C(coordinate)]
    for member in members:
        parts.append(C(member["header"]))
        parts.extend(C(page) for page in member["pages"])
    assert body["page_count"] == 4
    assert body["range_root_sha256"] == B("range-checkpoint", parts)
    # Every page byte and every header byte is covered: mutate one and the root moves.
    tampered = [C(coordinate)] + [C(members[0]["header"]), C({"page_ordinal": 99})]
    for member in members[1:]:
        tampered.append(C(member["header"]))
        tampered.extend(C(page) for page in member["pages"])
    assert B("range-checkpoint", tampered) != body["range_root_sha256"]
    # The same coordinate with a different order of members is a different range root.
    reordered = [C(coordinate)]
    for member in reversed(members):
        reordered.append(C(member["header"]))
        reordered.extend(C(page) for page in member["pages"])
    assert B("range-checkpoint", reordered) != body["range_root_sha256"]
    assert key == f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}"


# ── public meta surface guards (behavioral; runs without the new module) ────

def test_set_meta_refuses_reserved_receipt3_namespace(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with pytest.raises(ValueError):
            db.set_meta(f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}", "x")
        # The cursor form (same transaction, used by scheduler bookkeeping)
        # must not bypass the guard either.
        def _write(conn):
            db.set_meta(f"receipt3/o/{sid_hex(SID)}/{enc(1)}", "x", cursor=conn.cursor())
        with pytest.raises(ValueError):
            db._execute_write(_write)
        # Nothing above may have landed in state_meta.
        landed = db._conn.execute(
            "SELECT count(*) FROM state_meta WHERE key LIKE 'receipt3/%'").fetchone()[0]
        assert landed == 0
    finally:
        db.close()


def test_list_meta_prefix_refuses_receipt3_history_scan(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # Native provenance rows (written by the protected helper) exist in the
        # table; the generic full-materializing scan is not the protected reader.
        db._conn.execute("INSERT INTO state_meta (key, value) VALUES (?, '1')",
                         (f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}",))
        db._conn.commit()
        stored = db._conn.execute(
            "SELECT count(*) FROM state_meta WHERE key LIKE 'receipt3/%'").fetchone()[0]
        assert stored == 1
        with pytest.raises(ValueError):
            db.list_meta_prefix("receipt3")
        with pytest.raises(ValueError):
            db.list_meta_prefix(f"receipt3/e/{sid_hex(SID)}")
    finally:
        db.close()


def test_meta_surface_still_serves_ordinary_keys(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.set_meta("probe/key", "value")
        assert db.get_meta("probe/key") == "value"
        assert ("probe/key", "value") in db.list_meta_prefix("probe/")
    finally:
        db.close()
