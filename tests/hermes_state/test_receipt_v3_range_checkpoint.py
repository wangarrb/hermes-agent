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
  * member operation headers must carry every consumed field and agree with
    the checkpoint coordinate (owner_session_id/epoch/epoch_ordinal); a
    missing or inconsistent header is the canonical ValueError, never a
    KeyError;
  * ``range_root_sha256`` uses the frozen B framing over the checkpoint
    coordinate followed by every member header/page C bytes in order,
    recomputed independently here;
  * the public SessionDB meta surface refuses the reserved ``receipt3/``
    namespace (set_meta including the cursor path, and list_meta_prefix), so
    no generic setter/scan can act as the protected writer or long-history
    reader.

Milestone 2A adds the persistence/read-back half of the same contract:

  * named journal entries ``receipt3/o/<SIDhex>/<G20>`` and ordered pages
    ``receipt3/p/<SIDhex>/<G20>/<page20>`` (strict codecs, canonical bytes);
  * the quiet recovery continuation persists the real payload in one
    transaction (journal entry + pages + the epoch-exit range checkpoint read
    back from the stored member bytes + the protected head) and the reader
    replays the original bytes: exact key/body coordinates, contiguous
    coverage, no overlap, exact count/root; an open tail stays unsealed, a
    closed epoch's non-empty tail must be sealed; a repeated token only
    verifies and returns the stored result (never a second insert, never an
    overwrite), and a failed transaction leaves the old state or the new state
    -- nothing in between.

This file is RED against the unmodified b26ba5c4 tree: the v3 codec module and
the meta guards do not exist yet. Everything runs on real sqlite fixtures with
independently recomputed hashes -- no mocks.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

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


# ── milestone 2A: journal key codecs and page shape ─────────────────────────

def test_journal_key_codecs_roundtrip_and_rejections():
    m = _m()
    assert m.encode_receipt3_operation_key(SID, 4) == f"receipt3/o/{sid_hex(SID)}/{enc(4)}"
    assert m.parse_receipt3_operation_key(
        f"receipt3/o/{sid_hex(SID)}/{enc(4)}") == {"session_id": SID, "generation": 4}
    assert m.encode_receipt3_page_key(SID, 4, 0) == \
        f"receipt3/p/{sid_hex(SID)}/{enc(4)}/{enc(0)}"
    assert m.parse_receipt3_page_key(
        f"receipt3/p/{sid_hex(SID)}/{enc(4)}/{enc(7)}") == {
            "session_id": SID, "generation": 4, "page_ordinal": 7}
    for bad in (f"receipt3/o/{sid_hex(SID)}/4",
                f"receipt3/o/{sid_hex(SID)}/{enc(4)}/x",
                f"receipt3/o/{sid_hex(SID)}/{enc(0)}",
                f"receipt3/o/{sid_hex(SID).upper()}/{enc(4)}",
                f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(4)}"):
        with pytest.raises(ValueError):
            m.parse_receipt3_operation_key(bad)
    for bad in (f"receipt3/p/{sid_hex(SID)}/{enc(4)}",
                f"receipt3/p/{sid_hex(SID)}/{enc(4)}/{enc(0)}/x",
                f"receipt3/p/{sid_hex(SID)}/{enc(4)}/{enc(0)} ",
                f"receipt3/p/{sid_hex(SID)}/{enc(4)}/0/.0",
                f"receipt3/r/{sid_hex(SID)}/{enc(4)}/{enc(0)}"):
        with pytest.raises(ValueError):
            m.parse_receipt3_page_key(bad)
    with pytest.raises((ValueError, TypeError)):
        m.encode_receipt3_operation_key(SID, 0)
    with pytest.raises((ValueError, TypeError)):
        m.encode_receipt3_page_key(SID, 4, -1)
    with pytest.raises((ValueError, TypeError)):
        m.encode_receipt3_page_key(SID, 4, True)


def test_journal_page_shape_and_limits():
    m = _m()
    item = {"kind": "native_result", "input_ordinal": 0, "row_ref": None,
            "disposition": "inserted", "result_count": None}
    page = m.build_journal_page(owner_session_id=SID, epoch=EPOCH, generation=4,
                                page_ordinal=0, items=[item])
    m.validate_journal_page(page)
    assert page["schema_version"] == "hermes-receipt-page/3"
    with pytest.raises(ValueError):
        m.build_journal_page(owner_session_id=SID, epoch=EPOCH, generation=4,
                             page_ordinal=0, items=[])
    with pytest.raises(ValueError):
        m.build_journal_page(owner_session_id=SID, epoch=EPOCH, generation=4,
                             page_ordinal=0, items=[{"kind": "native_result"}] * 9)
    with pytest.raises(ValueError):
        m.build_journal_page(owner_session_id=SID, epoch=EPOCH, generation=4,
                             page_ordinal=0, items=[{"kind": "mystery"}])
    with pytest.raises(ValueError):
        m.build_journal_page(owner_session_id=SID, epoch=EPOCH, generation=4,
                             page_ordinal=0, items=[{"kind": "derivation", "blob": "x" * 70000}])
    with pytest.raises(ValueError):
        m.validate_journal_page(dict(page, extra=1))
    with pytest.raises(ValueError):
        m.validate_journal_page(dict(page, page_ordinal=-1))


# ── persistence read-back fixtures (synthetic, byte-exact) ──────────────────

def _pages_root(header, pages) -> str:
    parts = [C({"owner_session_id": header["owner_session_id"], "epoch": header["epoch"],
                "generation": header["generation"]})]
    parts.extend(C(page) for page in pages)
    return B("journal-content", parts)


def _member(generation: int, *, epoch: str = EPOCH, ordinal: int = 1, pages: int = 1,
            prev: str | None = None):
    """One synthetic sealed member: a full journal header plus valid pages.

    The header is self-hashed exactly like the writer's builder (H("operation",
    all fields minus the hash itself) so the read-back path can verify it.
    """
    header = {"schema_version": "hermes-receipt-operation/3", "domain_sha256": "d" * 64,
              "owner_session_id": SID, "actor_session_id": SID, "epoch": epoch,
              "epoch_ordinal": ordinal, "generation": generation,
              "operation_id": "00000000-0000-4000-8000-" + f"{generation:012d}",
              "request_sha256": hashlib.sha256(f"req-{generation}".encode()).hexdigest(),
              "previous_head_sha256": prev,
              "kind": "inplace_compaction", "base_fence_sha256": "e" * 64,
              "page_count": pages, "item_count": pages, "pages_root_sha256": "",
              "result_sha256": "0" * 64, "continuation": None}
    page_bodies = [{"schema_version": "hermes-receipt-page/3", "owner_session_id": SID,
                    "epoch": epoch, "generation": generation, "page_ordinal": index,
                    "items": [{"kind": "native_result", "input_ordinal": index,
                               "row_ref": None, "disposition": "inserted",
                               "result_count": None}]}
                   for index in range(pages)]
    header["pages_root_sha256"] = _pages_root(header, page_bodies)
    header["operation_sha256"] = H("operation", dict(header))
    return {"header": header, "pages": page_bodies}


def _seed_member(db, member, sid=SID):
    header = member["header"]
    generation = header["generation"]
    db._conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                     (f"receipt3/o/{sid_hex(sid)}/{enc(generation)}",
                      C(header).decode("utf-8")))
    for page in member["pages"]:
        db._conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?)",
            (f"receipt3/p/{sid_hex(sid)}/{enc(generation)}/{enc(page['page_ordinal'])}",
             C(page).decode("utf-8")))
    db._conn.commit()


def _seed_chain(db, sid, epoch, ordinal, start, count, first_prev=None, pages=1):
    members = []
    prev = first_prev
    for generation in range(start, start + count):
        member = _member(generation, epoch=epoch, ordinal=ordinal, pages=pages, prev=prev)
        _seed_member(db, member, sid)
        prev = member["header"]["operation_sha256"]
        members.append(member)
    return members


def _seal_blocks(db, sid, epoch, ordinal, s0, members, *, epoch_exit):
    m = _m()
    keys = []
    for offset in range(0, len(members), 1024):
        block = members[offset:offset + 1024]
        key, body = m.build_epoch_range_checkpoint(
            owner_session_id=sid, epoch=epoch, epoch_ordinal=ordinal, s0=s0,
            members=[{"header": member["header"], "pages": member["pages"]}
                     for member in block],
            epoch_exit=epoch_exit)
        m.seal_range_checkpoint(db._conn, key, body)
        keys.append(key)
    db._conn.commit()
    return keys


def test_chunking_uses_epoch_local_s0_and_sealing_rules():
    m = _m()
    s0 = 2058
    block = [_member(g, ordinal=3) for g in range(s0, s0 + 1024)]
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
    block2 = [_member(g, ordinal=3) for g in range(s0 + 1024, s0 + 2048)]
    key2, body2 = m.build_epoch_range_checkpoint(
        owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
        members=block2, epoch_exit=False)
    assert key2 == f"receipt3/e/{sid_hex(SID)}/{enc(3)}/{enc(3082)}"
    assert (body2["start_generation"], body2["end_generation"]) == (3082, 4105)

    # The open tail is NOT sealed early.
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(
            owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
            members=[_member(g, ordinal=3) for g in range(s0, s0 + 10)], epoch_exit=False)
    # Epoch exit seals exactly one non-empty partial tail.
    key3, body3 = m.build_epoch_range_checkpoint(
        owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
        members=[_member(g, ordinal=3) for g in range(s0, s0 + 10)], epoch_exit=True)
    assert (key3, body3["operation_count"]) == (f"receipt3/e/{sid_hex(SID)}/{enc(3)}/{enc(2058)}", 10)
    # No empty range, no start that is not s0+1024*k, no epoch crossing.
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3,
                                       s0=s0, members=[], epoch_exit=True)
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3,
                                       s0=s0, members=[_member(g, ordinal=3) for g in range(s0 + 1, s0 + 11)],
                                       epoch_exit=True)
    other_epoch = "sqlite:" + "b" * 64
    crossing = [_member(s0, ordinal=3), _member(s0 + 1, epoch=other_epoch, ordinal=3)]
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3,
                                       s0=s0, members=crossing, epoch_exit=True)
    # More than one full block in a single checkpoint is not a range.
    with pytest.raises(ValueError):
        m.build_epoch_range_checkpoint(
            owner_session_id=SID, epoch=EPOCH, epoch_ordinal=3, s0=s0,
            members=[_member(g, ordinal=3) for g in range(s0, s0 + 1025)], epoch_exit=False)


def test_builder_rejects_inconsistent_or_incomplete_member_headers():
    m = _m()
    s0 = 1

    def _block():
        return [_member(g) for g in range(s0, s0 + 3)]

    def _build(members):
        return m.build_epoch_range_checkpoint(
            owner_session_id=SID, epoch=EPOCH, epoch_ordinal=1, s0=s0,
            members=members, epoch_exit=True)

    # owner_session_id must agree with the checkpoint coordinate.
    wrong_owner = _block()
    wrong_owner[1]["header"]["owner_session_id"] = "other-01"
    with pytest.raises(ValueError):
        _build(wrong_owner)
    # epoch_ordinal must agree with the checkpoint coordinate.
    wrong_ordinal = _block()
    wrong_ordinal[1]["header"]["epoch_ordinal"] = 2
    with pytest.raises(ValueError):
        _build(wrong_ordinal)
    # Every consumed member-header field is required: the canonical ValueError,
    # never a KeyError.
    for missing in ("owner_session_id", "epoch", "epoch_ordinal", "generation",
                    "page_count", "previous_head_sha256", "operation_sha256"):
        incomplete = _block()
        del incomplete[1]["header"][missing]
        with pytest.raises(ValueError):
            _build(incomplete)


def test_range_root_uses_b_framing_independent_recompute():
    m = _m()
    s0 = 1
    members = [_member(g, pages=2 if g == 2 else 1,
                       prev=None if g == 1 else hashlib.sha256(f"op-{g - 1}".encode()).hexdigest())
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


# ── milestone 2A: read-back verification of persisted ranges and journal ────

def test_reader_replays_a_sealed_epoch_and_refuses_missing_or_non_canonical_bytes(tmp_path):
    m = _m()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(SID, "cli", model="m")
        members = _seed_chain(db, SID, EPOCH, 1, 1, 3)
        key = f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}"
        _seal_blocks(db, SID, EPOCH, 1, 1, members, epoch_exit=True)

        blocks = m.verify_epoch_range_checkpoints(
            db._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=3,
            epoch_exit=True)
        assert [(block["start"], block["end"]) for block in blocks] == [(1, 3)]
        assert blocks[0]["key"] == key and blocks[0]["body"]["operation_count"] == 3

        op2 = f"receipt3/o/{sid_hex(SID)}/{enc(2)}"
        page2 = f"receipt3/p/{sid_hex(SID)}/{enc(2)}/{enc(0)}"
        op2_value = db._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (op2,)).fetchone()[0]
        page2_value = db._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (page2,)).fetchone()[0]

        def _verify():
            return m.verify_epoch_range_checkpoints(
                db._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=3,
                epoch_exit=True)

        # Non-canonical stored bytes are refused (never replayed).
        db._conn.execute("UPDATE state_meta SET value = value || ' ' WHERE key = ?", (op2,))
        db._conn.commit()
        with pytest.raises(ValueError):
            _verify()
        db._conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (op2_value, op2))
        db._conn.commit()
        # A missing page refuses (missing page bytes under a sealed range).
        db._conn.execute("DELETE FROM state_meta WHERE key = ?", (page2,))
        db._conn.commit()
        with pytest.raises(ValueError):
            _verify()
        db._conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                         (page2, page2_value))
        db._conn.commit()
        # A missing member header refuses (the chain cannot be replayed).
        db._conn.execute("DELETE FROM state_meta WHERE key = ?", (op2,))
        db._conn.commit()
        with pytest.raises(ValueError):
            _verify()
        db._conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                         (op2, op2_value))
        db._conn.commit()
        # With every original byte restored the range verifies again.
        assert [(block["start"], block["end"]) for block in _verify()] == [(1, 3)]
        # The range root is recomputed over the ORIGINAL stored bytes.
        coordinate = {"owner_session_id": SID, "epoch": EPOCH, "epoch_ordinal": 1,
                      "start_generation": 1, "end_generation": 3,
                      "before_head_sha256": None,
                      "end_head_sha256": members[-1]["header"]["operation_sha256"],
                      "operation_count": 3, "page_count": 3}
        parts = [C(coordinate)]
        for member in members:
            parts.append(C(member["header"]))
            parts.extend(C(page) for page in member["pages"])
        assert blocks[0]["body"]["range_root_sha256"] == B("range-checkpoint", parts)
    finally:
        db.close()


def test_reader_refuses_gaps_short_coverage_early_partial_and_orphans(tmp_path):
    m = _m()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(SID, "cli", model="m")
        members = _seed_chain(db, SID, EPOCH, 1, 1, 5)
        # A sealed range must not skip generations: [1,2] + [4,5] refuse (the
        # forged second range is built with its own s0=4 to reach the store).
        _seal_blocks(db, SID, EPOCH, 1, 1, members[:2], epoch_exit=True)
        _seal_blocks(db, SID, EPOCH, 1, 4, members[3:], epoch_exit=True)
        with pytest.raises(ValueError):
            m.verify_epoch_range_checkpoints(
                db._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=5,
                epoch_exit=True)
        # Drop the second range: the closed epoch's non-empty tail is unsealed.
        db._conn.execute("DELETE FROM state_meta WHERE key = ?",
                         (f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(4)}",))
        db._conn.commit()
        with pytest.raises(ValueError):
            m.verify_epoch_range_checkpoints(
                db._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=5,
                epoch_exit=True)
        # While the epoch is open, an early-sealed partial tail refuses.
        with pytest.raises(ValueError):
            m.verify_epoch_range_checkpoints(
                db._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=5,
                epoch_exit=False)
        # A store that does not reach the claimed epoch end refuses.
        with pytest.raises(ValueError):
            m.verify_epoch_range_checkpoints(
                db._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=6,
                epoch_exit=True)
    finally:
        db.close()
    # An orphaned range (sealed bytes without the original journal) refuses.
    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    db2 = SessionDB(db_path=orphan_dir / "state.db")
    try:
        db2.create_session(SID, "cli", model="m")
        member = _member(4)
        key, body = _m().build_epoch_range_checkpoint(
            owner_session_id=SID, epoch=EPOCH, epoch_ordinal=1, s0=4,
            members=[{"header": member["header"], "pages": member["pages"]}],
            epoch_exit=True)
        _m().seal_range_checkpoint(db2._conn, key, body)
        db2._conn.commit()
        with pytest.raises(ValueError):
            _m().verify_epoch_range_checkpoints(
                db2._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=4,
                epoch_exit=True)
    finally:
        db2.close()


def test_range_seal_is_exact_and_never_overwrites(tmp_path):
    m = _m()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(SID, "cli", model="m")
        members = _seed_chain(db, SID, EPOCH, 1, 1, 2)
        key, body = m.build_epoch_range_checkpoint(
            owner_session_id=SID, epoch=EPOCH, epoch_ordinal=1, s0=1,
            members=[{"header": member["header"], "pages": member["pages"]}
                     for member in members],
            epoch_exit=True)
        first = m.seal_range_checkpoint(db._conn, key, body)
        db._conn.commit()
        assert db._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()[0] == first
        # Sealing the exact same bytes again verifies and reuses the row.
        assert m.seal_range_checkpoint(db._conn, key, body) == first
        db._conn.commit()
        assert db._conn.execute(
            "SELECT count(*) FROM state_meta WHERE key = ?", (key,)).fetchone()[0] == 1
        # The same key with different bytes refuses; the stored row is untouched.
        with pytest.raises(ValueError):
            m.seal_range_checkpoint(db._conn, key, dict(body, page_count=99))
        assert db._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()[0] == first
    finally:
        db.close()


# ── milestone 2A: two consecutive epoch switches + replay/atomicity ─────────

def _canon_sha(x) -> str:
    return hashlib.sha256(C(x)).hexdigest()


def _continuation_request(**overrides):
    request = {"kind": "epoch_continuation",
               "token": str(uuid.uuid4()),
               "owner_session_id": SID,
               "proof_sha256": "1" * 64,
               "old_epoch": EPOCH,
               "new_epoch": "sqlite:" + "b" * 64,
               "old_generation": 3,
               "old_head_sha256": "2" * 64,
               "old_head_token_sha256": "3" * 64,
               "boundary_roster_sha256": "4" * 64,
               "alias_inventory_sha256": "5" * 64}
    request.update(overrides)
    return request


def _head(epoch, ordinal, generation, head_sha256, **overrides):
    head = {"schema_version": "hermes-message-copy-receipts/3", "protocol_revision": 3,
            "domain_sha256": "d" * 64, "owner_session_id": SID, "active_epoch": epoch,
            "epoch_ordinal": ordinal, "generation": generation,
            "head_sha256": head_sha256, "v1_binding_sha256": None,
            "activation_proof_sha256": "9" * 64}
    head.update(overrides)
    return head


def _mutate(path, sql, params=()):
    conn = sqlite3.connect(path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _seed_head(path, head, sid=SID):
    _mutate(path, "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_message_copy_receipts_v2": head}), sid))


def _read_stored_head(db, sid=SID):
    raw = db._conn.execute(
        "SELECT model_config FROM sessions WHERE id = ?", (sid,)).fetchone()[0]
    return json.loads(raw)["_message_copy_receipts_v2"]


def _owner_snapshot(path):
    conn = sqlite3.connect(path)
    try:
        rows = sorted(conn.execute(
            "SELECT key, value FROM state_meta WHERE key LIKE 'receipt3/%'"))
        raw = conn.execute(
            "SELECT model_config FROM sessions WHERE id = ?", (SID,)).fetchone()[0]
        return rows, raw
    finally:
        conn.close()


def test_two_consecutive_epoch_switches_seal_each_epoch_locally(tmp_path):
    """The r4 missing case: two consecutive epoch switches, each sealing only its
    own epoch's tail, read back with the original bytes (r4 example: distinct
    immutable keys, no collision across epochs)."""
    m = _m()
    epoch_a = EPOCH
    epoch_b = "sqlite:" + "b" * 64
    epoch_c = "sqlite:" + "c" * 64
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session(SID, "cli", model="m")
        members = _seed_chain(db, SID, epoch_a, 1, 1, 3)
    finally:
        db.close()
    head0 = _head(epoch_a, 1, 3, members[-1]["header"]["operation_sha256"])
    _seed_head(path, head0)

    first = _continuation_request(old_epoch=epoch_a, new_epoch=epoch_b, old_generation=3,
                                  old_head_sha256=head0["head_sha256"],
                                  old_head_token_sha256=_canon_sha(head0))
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=first)
    try:
        result1 = handle.append_epoch_continuation(dict(first))
        assert result1["generation"] == 4
    finally:
        handle.close()

    db = SessionDB(db_path=path)
    try:
        # The first switch sealed epoch A's non-empty tail in the same transaction,
        # read back from the stored member bytes.
        blocks_a = m.verify_epoch_range_checkpoints(
            db._conn, owner_session_id=SID, epoch_ordinal=1, last_generation=3,
            epoch_exit=True)
        key_a = f"receipt3/e/{sid_hex(SID)}/{enc(1)}/{enc(1)}"
        assert [(block["start"], block["end"]) for block in blocks_a] == [(1, 3)]
        assert blocks_a[0]["key"] == key_a
        coordinate = {"owner_session_id": SID, "epoch": epoch_a, "epoch_ordinal": 1,
                      "start_generation": 1, "end_generation": 3,
                      "before_head_sha256": None,
                      "end_head_sha256": members[-1]["header"]["operation_sha256"],
                      "operation_count": 3, "page_count": 3}
        parts = [C(coordinate)]
        for member in members:
            parts.append(C(member["header"]))
            parts.extend(C(page) for page in member["pages"])
        assert blocks_a[0]["body"]["range_root_sha256"] == B("range-checkpoint", parts)
        head1 = _read_stored_head(db)
        assert (head1["active_epoch"], head1["epoch_ordinal"], head1["generation"]) == (
            epoch_b, 2, 4)
        assert head1["head_sha256"] == result1["operation_sha256"]
        # The continuation op belongs to the NEW epoch and links the old head.
        op4 = json.loads(db._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (f"receipt3/o/{sid_hex(SID)}/{enc(4)}",)).fetchone()[0])
        assert op4["epoch"] == epoch_b and op4["epoch_ordinal"] == 2
        assert op4["previous_head_sha256"] == head0["head_sha256"]
        assert op4["generation"] == 4
    finally:
        db.close()

    second = _continuation_request(old_epoch=epoch_b, new_epoch=epoch_c, old_generation=4,
                                   old_head_sha256=head1["head_sha256"],
                                   old_head_token_sha256=_canon_sha(head1))
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=second)
    try:
        result2 = handle.append_epoch_continuation(dict(second))
        assert result2["generation"] == 5
    finally:
        handle.close()

    db = SessionDB(db_path=path)
    try:
        # The second switch sealed epoch B's single-op tail: a different immutable
        # key (ordinal 2, start 4), never a collision with epoch A's range.
        blocks_b = m.verify_epoch_range_checkpoints(
            db._conn, owner_session_id=SID, epoch_ordinal=2, last_generation=4,
            epoch_exit=True)
        assert [(block["start"], block["end"]) for block in blocks_b] == [(4, 4)]
        assert blocks_b[0]["key"] == f"receipt3/e/{sid_hex(SID)}/{enc(2)}/{enc(4)}"
        assert blocks_b[0]["key"] != key_a
        assert blocks_b[0]["body"]["epoch"] == epoch_b
        head2 = _read_stored_head(db)
        assert (head2["active_epoch"], head2["epoch_ordinal"], head2["generation"]) == (
            epoch_c, 3, 5)
        # Epoch C is the current open epoch: its one-op tail stays unsealed.
        assert m.verify_epoch_range_checkpoints(
            db._conn, owner_session_id=SID, epoch_ordinal=3, last_generation=5,
            epoch_exit=False) == []
    finally:
        db.close()


def test_continuation_replay_and_failed_seal_are_transaction_atomic(tmp_path):
    """The r4 missing case: a repeated token only verifies and returns the stored
    result (no second insert, no overwrite), and a failed transaction leaves the
    old state or the new state -- never a partially written one."""
    m = _m()
    epoch_a = EPOCH
    epoch_b = "sqlite:" + "b" * 64
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session(SID, "cli", model="m")
        members = _seed_chain(db, SID, epoch_a, 1, 1, 3)
    finally:
        db.close()
    head0 = _head(epoch_a, 1, 3, members[-1]["header"]["operation_sha256"])
    _seed_head(path, head0)
    request = _continuation_request(old_epoch=epoch_a, new_epoch=epoch_b, old_generation=3,
                                    old_head_sha256=head0["head_sha256"],
                                    old_head_token_sha256=_canon_sha(head0))
    key, body = m.build_epoch_range_checkpoint(
        owner_session_id=SID, epoch=epoch_a, epoch_ordinal=1, s0=1,
        members=[{"header": member["header"], "pages": member["pages"]}
                 for member in members],
        epoch_exit=True)

    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=request)
    try:
        # A conflicting pre-sealed tail (same key, different bytes) refuses the
        # whole continuation transaction.
        conflicting = dict(body, page_count=99)
        db = SessionDB(db_path=path)
        try:
            db._conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                             (key, C(conflicting).decode("utf-8")))
            db._conn.commit()
        finally:
            db.close()
        with pytest.raises(ValueError):
            handle.append_epoch_continuation(dict(request))
        assert handle._write_count == 0
        db = SessionDB(db_path=path)
        try:
            # Nothing of the refused transaction landed: no journal entry, no
            # token record, no head change, and the conflicting row is untouched.
            assert db._conn.execute(
                "SELECT count(*) FROM state_meta WHERE key = ?",
                (f"receipt3/o/{sid_hex(SID)}/{enc(4)}",)).fetchone()[0] == 0
            assert db._conn.execute(
                "SELECT count(*) FROM state_meta WHERE key LIKE ?",
                (f"receipt3/t/{sid_hex(SID)}/%",)).fetchone()[0] == 0
            head_now = _read_stored_head(db)
            assert (head_now["active_epoch"], head_now["generation"]) == (epoch_a, 3)
            assert db._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (key,)).fetchone()[0] == C(conflicting).decode("utf-8")
        finally:
            db.close()
        # Replace it with the exact expected bytes: the seal verifies and reuses
        # the byte-identical range (never an overwrite); the continuation lands.
        db = SessionDB(db_path=path)
        try:
            db._conn.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                             (C(body).decode("utf-8"), key))
            db._conn.commit()
        finally:
            db.close()
        result = handle.append_epoch_continuation(dict(request))
        assert handle._write_count == 1 and result["generation"] == 4
        # Same token again: exact read-back returns the stored result without a
        # second write (no second journal entry, no duplicated checkpoint, head
        # bytes unchanged).
        before = _owner_snapshot(path)
        replay = handle.append_epoch_continuation(dict(request))
        assert replay == result
        assert handle._write_count == 1
        assert _owner_snapshot(path) == before
        # The same token with different bytes is refused without a write.
        with pytest.raises((ValueError, RuntimeError)):
            handle.append_epoch_continuation(dict(request, old_generation=99))
        assert handle._write_count == 1
    finally:
        handle.close()
