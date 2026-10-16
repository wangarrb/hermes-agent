"""Receipt r4 milestone 2E: standalone reader consumption, C02 steps 1-4 full
orchestration + final ALL-owner readback, capacity checkpoint plan and the
full-path prefix-collision audit.

Contract from the effective r4 machine spec / design text:

  * the reader-side surface (``protocol.independent_reader`` /
    ``protocol.context_binding``) consumes the receipt3/ namespace read-only:
    within the two pre-existing module files (plus the quiet handle) the
    standalone consumer now has a full verified snapshot reader, a typed
    prior/current context comparison, immutable ``context_before_ref``
    resolution, a standalone-lineage recompute of the namespace/cron/
    eligibility facts, compression-edge closure facts, and the D admission
    gate that returns the *proposed* ``BLOCK_SOURCE_CONTEXT_POLICY`` for
    unknown/missing/drift context (never same-source acceptance, never a
    rollback of legal native writes);
  * C02 steps 1-4 full orchestration: the external holder-drain fence
    attestation is an approved input (refused when absent/unapproved), the
    independent full-table joint runs between the original latest and the
    byte/data-complete replacement (again read-only AFTER the quiet open),
    the frozen proof must bind the boundary roster, then exactly one epoch
    continuation per boundary owner (g0 / v1-only / no-receipt owners
    included) under per-owner token/head CAS; the final ALL-owner readback
    binding is derived over the complete roster and is the bytes the explicit
    quiet exit re-checks;
  * capacity checkpoints: immutable epoch-local 1..1024-op ranges with
    ``start = s0 + 1024*k``; full blocks seal with their last operation, a
    partial tail seals only on epoch exit, an open tail is never sealed early;
    the stored sealed set must equal that plan exactly and old checkpoints are
    retained, never overwritten;
  * full-path prefix-collision audit: every stored key inside the reserved
    ``receipt3/`` namespace parses under exactly one family codec, the bare
    reserved name is a finding, non-reserved lookalikes (``receipt30/...``)
    cannot collide, and encoded owner spans are checked for genuine overlaps.

RED against the 500f15a1 slice: none of these reader/roster/capacity/audit
surfaces exist there. Real SQLite fixtures only; no mocks; synthetic
fixtures; no production store is touched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid

import pytest

import hermes_state
from hermes_state import SessionDB

SID_A = "owner-A"
SID_B = "owner-B"
SID_C = "owner-C"
SID_D = "owner-D"
OLD_EPOCH = "sqlite:" + "a" * 64
NEW_EPOCH = "sqlite:" + "b" * 64
FINAL_EPOCH = "sqlite:" + "c" * 64
DOMAIN = "d" * 64
V1_RAW = json.dumps({"_message_copy_receipts": {"v1_token": "keep-me"}})
FENCE = {"kind": "external_holder_drain_fence", "approved": True,
         "holders": ["pane-1", "scanner-2"], "fence_id": "fence-m2e-1"}


def _m():
    import hermes_state_copy_provenance as module
    return module


def _canonical(x) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8", "strict")


def _canon_sha(x) -> str:
    return hashlib.sha256(_canonical(x)).hexdigest()


def _settle(base, sids=(SID_A, SID_B, SID_C)):
    db = SessionDB(db_path=base / "state.db")
    for sid in sids:
        db.create_session(sid, "cli", model="m")
        db.append_message(session_id=sid, role="user", content="hello " + sid)
    db.set_meta("probe/key", "value")
    db.close()
    path = base / "state.db"
    if SID_C in sids:
        # C is a v1-only boundary owner: existing v1 receipt config bytes and no
        # receipt3 head at all. The continuation must preserve these lexemes.
        _mutate(path, "UPDATE sessions SET model_config = ? WHERE id = ?", (V1_RAW, SID_C))
    return path


def _mutate(path, sql, params=()):
    conn = sqlite3.connect(path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _raw_config(path, sid):
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT model_config FROM sessions WHERE id = ?", (sid,)).fetchone()
    finally:
        conn.close()
    return row[0]


def _read_head(path, sid):
    raw = _raw_config(path, sid)
    return json.loads(raw)["_message_copy_receipts_v2"]


def _request(**overrides):
    request = {"kind": "epoch_continuation", "token": str(uuid.uuid4()),
               "owner_session_id": SID_A, "proof_sha256": "1" * 64,
               "old_epoch": OLD_EPOCH, "new_epoch": NEW_EPOCH, "old_generation": 0,
               "old_head_sha256": None, "old_head_token_sha256": "3" * 64,
               "boundary_roster_sha256": "6" * 64, "alias_inventory_sha256": "5" * 64}
    request.update(overrides)
    return request


def _matching_request(head, **overrides):
    request = _request(old_epoch=head["active_epoch"], old_generation=head["generation"],
                       new_epoch=FINAL_EPOCH,
                       old_head_sha256=head["head_sha256"],
                       old_head_token_sha256=_canon_sha(head))
    request.update(overrides)
    return request


def _write_initial_receipt(path, sid=SID_A, new_epoch=NEW_EPOCH):
    """One real g0 -> g1 continuation through the quiet handle (no crash)."""
    request = _request(owner_session_id=sid, new_epoch=new_epoch)
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=request)
    result = handle.append_epoch_continuation(request)
    handle.close()
    return request, result


def _continue_single(path, sid, head, new_epoch, proof="1" * 64, roster="6" * 64):
    request = _matching_request(head, owner_session_id=sid, new_epoch=new_epoch,
                                proof_sha256=proof, boundary_roster_sha256=roster)
    handle = SessionDB.open_quiet_recovery_admission(db_path=path, init_request=request)
    result = handle.append_epoch_continuation(request)
    handle.close()
    return request, result


def _receipt_keys(path):
    conn = sqlite3.connect(path)
    try:
        return sorted(row[0] for row in conn.execute(
            "SELECT key FROM state_meta WHERE key >= 'receipt3' AND key < 'receipt4'"))
    finally:
        conn.close()


def _roster_entries(requests):
    return [{"owner_session_id": request["owner_session_id"], "token": request["token"],
             "old_generation": request["old_generation"],
             "old_head_sha256": request["old_head_sha256"],
             "old_head_token_sha256": request["old_head_token_sha256"]}
            for request in requests]


def _roster_requests(path):
    """The three boundary owners of one frozen transition: a receipt owner (A
    continuing from its stored head), a plain g0 owner (B) and a v1-only owner
    (C, no receipt3 head but existing v1 config bytes)."""
    m = _m()
    head = _read_head(path, SID_A)
    roster_sha = m.boundary_roster_digest([SID_A, SID_B, SID_C])
    shared = dict(proof_sha256="1" * 64, boundary_roster_sha256=roster_sha,
                  alias_inventory_sha256="5" * 64, old_epoch=head["active_epoch"],
                  new_epoch=FINAL_EPOCH)
    requests = [_matching_request(head, owner_session_id=SID_A, **shared)]
    for sid in (SID_B, SID_C):
        requests.append(_request(owner_session_id=sid, old_epoch=head["active_epoch"],
                                 new_epoch=FINAL_EPOCH, old_generation=0,
                                 old_head_sha256=None, old_head_token_sha256="3" * 64,
                                 proof_sha256="1" * 64, boundary_roster_sha256=roster_sha,
                                 alias_inventory_sha256="5" * 64))
    return requests, roster_sha


# ── C02 steps 1-4 full orchestration + final ALL readback ────────────────────

def test_c02_roster_orchestration_end_to_end(tmp_path):
    m = _m()
    source_dir = tmp_path / "original"
    source_dir.mkdir()
    path = _settle(source_dir)
    initial_request, _initial_result = _write_initial_receipt(path)
    replacement_dir = tmp_path / "replacement"
    shutil.copytree(source_dir, replacement_dir)
    replacement = replacement_dir / "state.db"
    requests, roster_sha = _roster_requests(replacement)
    proof = m.freeze_recovery_proof(replacement, owner_session_id=SID_A,
                                    init_request=requests[0],
                                    roster_owners=[SID_A, SID_B, SID_C])
    assert proof["roster_sha256"] == roster_sha
    baseline_joint = m.verify_independent_copy_joint(path, replacement)

    handle, report = hermes_state.QuietRecoveryHandle.run_c02_recovery_orchestration(
        original_path=path, replacement_path=replacement, fence=FENCE,
        requests=requests, approved_anchors=[SID_A, SID_B, SID_C], proof=proof)

    assert report["steps"]["step1_fence"] == {"fence_id": "fence-m2e-1", "holders": 2,
                                              "approved": True}
    assert report["steps"]["step2"]["joint_sha256"] == baseline_joint["joint_sha256"]
    assert report["steps"]["step3"]["complete"] is False
    assert report["steps"]["step3"]["owners"] == 3
    assert report["steps"]["step4"] == {SID_A: 2, SID_B: 1, SID_C: 1}
    assert report["steps"]["all_readback"]["complete"] is True

    readback = handle.verify_roster_recovery_state(complete=True)
    assert readback["readback_sha256"] == report["readback"]["readback_sha256"]
    assert [entry["owner_session_id"] for entry in readback["owners"]] == \
        [SID_A, SID_B, SID_C]
    assert [len(entry["committed"]) for entry in readback["owners"]] == [1, 1, 1]

    # The ALL readback binding is the byte-level exit gate.
    with pytest.raises(ValueError):
        handle.exit_roster_quiet(accepted_readback_sha256="0" * 64)
    exit_report = handle.exit_roster_quiet(
        accepted_readback_sha256=readback["readback_sha256"])
    assert set(exit_report["owner_session_ids"]) == {SID_A, SID_B, SID_C}
    assert exit_report["fts_pass"]["failed"] is None

    # Every boundary owner landed exactly one continuation; A advanced g1 -> g2.
    head_a = _read_head(replacement, SID_A)
    assert (head_a["generation"], head_a["active_epoch"]) == (2, FINAL_EPOCH)
    config_c = _raw_config(replacement, SID_C)
    assert '"_message_copy_receipts": {"v1_token": "keep-me"}' in config_c
    assert json.loads(config_c)["_message_copy_receipts"] == {"v1_token": "keep-me"}
    assert json.loads(config_c)["_message_copy_receipts_v2"]["generation"] == 1

    expected = {
        m.encode_receipt3_token_key(SID_A, initial_request["token"]),
        m.encode_receipt3_operation_key(SID_A, 1), m.encode_receipt3_page_key(SID_A, 1, 0),
        m.encode_receipt3_token_key(SID_A, requests[0]["token"]),
        m.encode_receipt3_operation_key(SID_A, 2), m.encode_receipt3_page_key(SID_A, 2, 0),
        m.encode_receipt3_token_key(SID_B, requests[1]["token"]),
        m.encode_receipt3_operation_key(SID_B, 1), m.encode_receipt3_page_key(SID_B, 1, 0),
        m.encode_receipt3_token_key(SID_C, requests[2]["token"]),
        m.encode_receipt3_operation_key(SID_C, 1), m.encode_receipt3_page_key(SID_C, 1, 0),
        m.encode_receipt3_epoch_key(SID_A, 1, 1),
    }
    assert set(_receipt_keys(replacement)) == expected
    # A's closed epoch 1 sealed its partial tail; B/C open epochs sealed nothing.
    conn = _connect(replacement)
    try:
        plan = m.verify_capacity_checkpoint_plan(conn, owner_session_id=SID_A,
                                                 epoch_ordinal=1)
        assert plan["sealed_blocks"] == 1 and plan["verify"] == "capacity_plan_exact"
        assert m.verify_capacity_checkpoint_plan(
            conn, owner_session_id=SID_B, epoch_ordinal=1)["sealed_blocks"] == 0
    finally:
        conn.close()


def test_c02_orchestration_refuses_without_approved_fence(tmp_path):
    source_dir = tmp_path / "original"
    source_dir.mkdir()
    path = _settle(source_dir, sids=(SID_A,))
    replacement_dir = tmp_path / "replacement"
    shutil.copytree(source_dir, replacement_dir)
    replacement = replacement_dir / "state.db"
    request = _request()
    before = hashlib.sha256(replacement.read_bytes()).hexdigest()
    for bad in (dict(FENCE, approved=False),
                {key: value for key, value in FENCE.items() if key != "holders"},
                dict(FENCE, holders=[]),
                dict(FENCE, kind="not_a_fence")):
        with pytest.raises(ValueError):
            hermes_state.QuietRecoveryHandle.run_c02_recovery_orchestration(
                original_path=path, replacement_path=replacement, fence=bad,
                requests=[request], approved_anchors=[SID_A])
    assert hashlib.sha256(replacement.read_bytes()).hexdigest() == before
    assert _receipt_keys(replacement) == []


def test_c02_orchestration_refuses_proof_not_binding_roster(tmp_path):
    m = _m()
    source_dir = tmp_path / "original"
    source_dir.mkdir()
    path = _settle(source_dir)
    _write_initial_receipt(path)
    replacement_dir = tmp_path / "replacement"
    shutil.copytree(source_dir, replacement_dir)
    replacement = replacement_dir / "state.db"
    requests, _roster_sha = _roster_requests(replacement)
    # A proof frozen over a SMALLER roster than the frozen request must refuse.
    proof = m.freeze_recovery_proof(replacement, owner_session_id=SID_A,
                                    init_request=requests[0],
                                    roster_owners=[SID_A, SID_B])
    before = _receipt_keys(replacement)
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.run_c02_recovery_orchestration(
            original_path=path, replacement_path=replacement, fence=FENCE,
            requests=requests, approved_anchors=[SID_A, SID_B, SID_C], proof=proof)
    assert _receipt_keys(replacement) == before


def test_open_roster_refuses_membership_change_and_frame_drift(tmp_path):
    m = _m()
    requests, roster_sha = _roster_requests_synthetic()
    missing = tmp_path / "missing.db"
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.open_roster(
            db_path=missing, requests=requests[:2], approved_anchors=[SID_A, SID_B, SID_C])
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.open_roster(
            db_path=missing, requests=requests, approved_anchors=[SID_A, SID_B, SID_C, SID_D])
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.open_roster(
            db_path=missing, requests=[requests[0], dict(requests[0])],
            approved_anchors=[SID_A])
    drifted = [requests[0],
               dict(requests[1], proof_sha256="9" * 64),
               requests[2]]
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.open_roster(
            db_path=missing, requests=drifted, approved_anchors=[SID_A, SID_B, SID_C])
    wrong_digest = [dict(request, boundary_roster_sha256="7" * 64) for request in requests]
    with pytest.raises(ValueError):
        hermes_state.QuietRecoveryHandle.open_roster(
            db_path=missing, requests=wrong_digest, approved_anchors=[SID_A, SID_B, SID_C])
    assert roster_sha == m.boundary_roster_digest([SID_A, SID_B, SID_C])


def _roster_requests_synthetic():
    m = _m()
    roster_sha = m.boundary_roster_digest([SID_A, SID_B, SID_C])
    requests = []
    for sid in (SID_A, SID_B, SID_C):
        requests.append(_request(owner_session_id=sid, old_epoch=OLD_EPOCH,
                                 new_epoch=FINAL_EPOCH, boundary_roster_sha256=roster_sha))
    return requests, roster_sha


def test_open_roster_partial_state_and_foreign_token_refusal(tmp_path):
    m = _m()
    store = _settle(tmp_path)
    requests, _roster_sha = _roster_requests_synthetic()
    handle = hermes_state.QuietRecoveryHandle.open_roster(
        db_path=store, requests=requests, approved_anchors=[SID_A, SID_B, SID_C])
    try:
        state = handle.verify_roster_recovery_state()
        assert state["complete"] is False
        assert [entry["owner_session_id"] for entry in state["owners"]] == \
            [SID_A, SID_B, SID_C]
        assert all(entry["committed"] == [] for entry in state["owners"])
        assert state["readback_sha256"] == m.final_all_owner_readback(state["owners"])
        # A continuation token outside the approved roster never rides along.
        foreign = m.encode_receipt3_token_key(SID_D, str(uuid.uuid4()))
        _mutate(store, "INSERT INTO state_meta (key, value) VALUES (?, 'x')", (foreign,))
        with pytest.raises(ValueError):
            handle.verify_roster_recovery_state()
    finally:
        handle.close()


def test_validate_boundary_roster_shape_and_digest(tmp_path):
    m = _m()
    store = _settle(tmp_path, sids=(SID_A,))
    single = _request()
    proof = m.freeze_recovery_proof(store, owner_session_id=SID_A, init_request=single)
    assert proof["roster_sha256"] == m.boundary_roster_digest([SID_A])
    requests, roster_sha = _roster_requests_synthetic()
    entries = _roster_entries(requests)
    roster = m.validate_boundary_roster(entries, approved_anchors=[SID_A, SID_B, SID_C])
    assert roster["owners"] == [SID_A, SID_B, SID_C]
    assert roster["roster_sha256"] == roster_sha
    with pytest.raises(ValueError):
        m.validate_boundary_roster(entries, approved_anchors=[SID_A, SID_B])
    with pytest.raises(ValueError):
        m.validate_boundary_roster(entries, approved_anchors=[SID_A, SID_B, SID_C, SID_D])
    with pytest.raises(ValueError):
        m.validate_boundary_roster(entries + [entries[0]], approved_anchors=[SID_A, SID_B, SID_C])
    g0_with_head = dict(entries[1], old_head_sha256="1" * 64)
    with pytest.raises(ValueError):
        m.validate_boundary_roster([entries[0], g0_with_head, entries[2]],
                                   approved_anchors=[SID_A, SID_B, SID_C])
    with pytest.raises(ValueError):
        m.validate_boundary_roster([dict(entries[0], token="not-a-uuid"), entries[1], entries[2]],
                                   approved_anchors=[SID_A, SID_B, SID_C])


# ── standalone reader consumption ────────────────────────────────────────────

def test_reader_snapshot_consumes_owner_history(tmp_path):
    m = _m()
    store = _settle(tmp_path)
    _write_initial_receipt(store)
    head = _read_head(store, SID_A)
    _continue_single(store, SID_A, head, FINAL_EPOCH)
    conn = _connect(store)
    try:
        snapshot = m.read_receipt3_reader_snapshot(conn, SID_A)
    finally:
        conn.close()
    assert snapshot["owner_session_id"] == SID_A
    assert snapshot["head"]["generation"] == 2
    assert [operation["generation"] for operation in snapshot["operations"]] == [1, 2]
    assert [epoch["epoch_ordinal"] for epoch in snapshot["epochs"]] == [1, 2]
    assert snapshot["epochs"][0]["sealed_blocks"] == [{"start": 1, "end": 1}]
    assert snapshot["epochs"][1]["sealed_blocks"] == []
    assert sorted(token["generation"] for token in snapshot["tokens"]) == [1, 2]
    assert snapshot["key_audit"]["findings"] == []
    assert snapshot["key_audit"]["accepted"] == len(_receipt_keys(store))


def test_reader_snapshot_refuses_tampered_bytes(tmp_path):
    m = _m()
    store = _settle(tmp_path)
    _write_initial_receipt(store)
    conn = _connect(store)
    try:
        m.read_receipt3_reader_snapshot(conn, SID_A)
    finally:
        conn.close()
    token_key = [key for key in _receipt_keys(store) if "/t/" in key][0]
    _mutate(store, "UPDATE state_meta SET value = value || ' ' WHERE key = ?", (token_key,))
    conn = _connect(store)
    try:
        with pytest.raises(ValueError):
            m.read_receipt3_reader_snapshot(conn, SID_A)
    finally:
        conn.close()


def test_reader_snapshot_refuses_missing_checkpoint_and_orphan(tmp_path):
    m = _m()
    store = _settle(tmp_path)
    _write_initial_receipt(store)
    head = _read_head(store, SID_A)
    _continue_single(store, SID_A, head, FINAL_EPOCH)
    checkpoint = [key for key in _receipt_keys(store) if "/e/" in key][0]
    # A closed epoch's mandated partial tail seal: removing it refuses.
    copy = tmp_path / "missing-checkpoint.db"
    shutil.copyfile(store, copy)
    _mutate(copy, "DELETE FROM state_meta WHERE key = ?", (checkpoint,))
    conn = _connect(copy)
    try:
        with pytest.raises(ValueError):
            m.read_receipt3_reader_snapshot(conn, SID_A)
    finally:
        conn.close()
    # A token record removed leaves the journal operation unnamed: refuses.
    older_token = sorted(key for key in _receipt_keys(store) if "/t/" in key)[0]
    copy2 = tmp_path / "orphan.db"
    shutil.copyfile(store, copy2)
    _mutate(copy2, "DELETE FROM state_meta WHERE key = ?", (older_token,))
    conn = _connect(copy2)
    try:
        with pytest.raises(ValueError):
            m.read_receipt3_reader_snapshot(conn, SID_A)
    finally:
        conn.close()
    # A junk key inside the owner's hex span refuses before anything else.
    copy3 = tmp_path / "junk.db"
    shutil.copyfile(store, copy3)
    junk = "receipt3/t/" + SID_A.encode("utf-8").hex() + "/" + "0" * 20
    _mutate(copy3, "INSERT INTO state_meta (key, value) VALUES (?, 'x')", (junk,))
    conn = _connect(copy3)
    try:
        with pytest.raises(ValueError):
            m.read_receipt3_reader_snapshot(conn, SID_A)
    finally:
        conn.close()


# ── capacity checkpoint plan (protocol.capacity.concrete_bounds.checkpoint) ──

def test_capacity_checkpoint_plan_boundaries():
    m = _m()
    full = m.capacity_checkpoint_plan(owner_session_id=SID_A, epoch=NEW_EPOCH,
                                      epoch_ordinal=1, first_generation=5,
                                      last_generation=5 + 2048 - 1, epoch_exit=False)
    assert full["sealed_count"] == 2 and full["open_tail"] is None
    assert [(block["start"], block["end"]) for block in full["sealed_blocks"]] == \
        [(5, 1028), (1029, 2052)]
    exact = m.capacity_checkpoint_plan(owner_session_id=SID_A, epoch=NEW_EPOCH,
                                       epoch_ordinal=1, first_generation=1,
                                       last_generation=1024, epoch_exit=False)
    assert exact["sealed_count"] == 1 and exact["open_tail"] is None
    open_tail = m.capacity_checkpoint_plan(owner_session_id=SID_A, epoch=NEW_EPOCH,
                                           epoch_ordinal=1, first_generation=1,
                                           last_generation=1025, epoch_exit=False)
    assert open_tail["sealed_count"] == 1
    assert open_tail["open_tail"] == {"start": 1025, "end": 1025, "operation_count": 1,
                                      "sealed": False}
    closed = m.capacity_checkpoint_plan(owner_session_id=SID_A, epoch=NEW_EPOCH,
                                        epoch_ordinal=1, first_generation=1,
                                        last_generation=1025, epoch_exit=True)
    assert closed["sealed_count"] == 2 and closed["open_tail"] is None
    assert closed["sealed_blocks"][1]["operation_count"] == 1
    non_g1 = m.capacity_checkpoint_plan(owner_session_id=SID_A, epoch=NEW_EPOCH,
                                        epoch_ordinal=2, first_generation=7,
                                        last_generation=7, epoch_exit=True)
    assert non_g1["sealed_blocks"] == [{"start": 7, "end": 7, "operation_count": 1,
                                        "sealed": True}]
    with pytest.raises(ValueError):
        m.capacity_checkpoint_plan(owner_session_id=SID_A, epoch=NEW_EPOCH,
                                   epoch_ordinal=1, first_generation=5,
                                   last_generation=4, epoch_exit=False)
    with pytest.raises(ValueError):
        m.capacity_checkpoint_plan(owner_session_id=SID_A, epoch="bad",
                                   epoch_ordinal=1, first_generation=1,
                                   last_generation=1, epoch_exit=False)


def test_capacity_checkpoint_plan_matches_and_refuses_stored_drift(tmp_path):
    m = _m()
    store = _settle(tmp_path)
    _write_initial_receipt(store, new_epoch=NEW_EPOCH)
    head = _read_head(store, SID_A)
    _continue_single(store, SID_A, head, FINAL_EPOCH)
    conn = _connect(store)
    try:
        closed_plan = m.verify_capacity_checkpoint_plan(conn, owner_session_id=SID_A,
                                                        epoch_ordinal=1)
        open_plan = m.verify_capacity_checkpoint_plan(conn, owner_session_id=SID_A,
                                                      epoch_ordinal=2)
    finally:
        conn.close()
    assert closed_plan["sealed_blocks"] == 1
    assert open_plan["sealed_blocks"] == 0
    assert open_plan["open_tail"] == {"start": 2, "end": 2, "operation_count": 1,
                                      "sealed": False}
    checkpoint = [key for key in _receipt_keys(store) if "/e/" in key][0]
    _mutate(store, "DELETE FROM state_meta WHERE key = ?", (checkpoint,))
    conn = _connect(store)
    try:
        with pytest.raises(ValueError):
            m.verify_capacity_checkpoint_plan(conn, owner_session_id=SID_A, epoch_ordinal=1)
    finally:
        conn.close()


# ── full-path prefix-collision audit ─────────────────────────────────────────

def test_audit_receipt3_key_paths_exact_and_findings():
    m = _m()
    token_a, token_b = str(uuid.uuid4()), str(uuid.uuid4())
    canonical = [
        m.encode_receipt3_epoch_key(SID_A, 1, 1),
        m.encode_receipt3_epoch_key(SID_B, 2, 7),
        m.encode_receipt3_operation_key(SID_A, 1),
        m.encode_receipt3_operation_key(SID_B, 9),
        m.encode_receipt3_page_key(SID_A, 1, 0),
        m.encode_receipt3_page_key(SID_B, 9, 3),
        m.encode_receipt3_token_key(SID_A, token_a),
        m.encode_receipt3_token_key(SID_B, token_b),
        m.encode_receipt3_row_index_key("sqlite:" + "a" * 64, 3),
        m.encode_receipt3_row_index_key("sqlite:" + "b" * 64, 12),
    ]
    report = m.audit_receipt3_key_paths(canonical)
    assert report["findings"] == []
    assert report["accepted"] == len(canonical) == 10
    assert report["families"] == {"e": 2, "o": 2, "p": 2, "t": 2, "r": 2}
    # Cross-family exclusivity: no key of one family is accepted by another.
    parsers = {"e": m.parse_receipt3_epoch_key, "o": m.parse_receipt3_operation_key,
               "p": m.parse_receipt3_page_key, "t": m.parse_receipt3_token_key,
               "r": m.parse_receipt3_row_index_key}
    for key in canonical:
        accepted = []
        for family, parser in parsers.items():
            try:
                parser(key)
                accepted.append(family)
            except ValueError:
                pass
        assert len(accepted) == 1, (key, accepted)
    findings = m.audit_receipt3_key_paths([
        "receipt3",
        "receipt3/e/" + SID_A.encode("utf-8").hex() + "/1",          # unparsed (4 parts)
        "receipt3/t/" + SID_A.encode("utf-8").hex() + "/" + "0" * 20,  # unparsed (not uuid4)
        "receipt3/z/" + "f" * 64,
        "receipt30/lookalike/key",
    ])
    kinds = sorted(finding["finding"] for finding in findings["findings"])
    assert kinds == ["bare_reserved_name", "unparsed", "unparsed", "unparsed"]
    assert findings["lookalikes"] == ["receipt30/lookalike/key"]
    assert findings["accepted"] == 0


def test_receipt3_prefix_spans_detects_encoded_overlap():
    m = _m()
    disjoint = m.receipt3_prefix_spans([SID_A, SID_B, SID_C])
    assert disjoint["pairwise_disjoint"] is True and disjoint["overlaps"] == []
    assert len(disjoint["spans"]) == 12
    # "\u00ab" encodes to the hex "c2ab"; "\u00ab\x12" to "c2ab12" -- the
    # longer owner's encoded span starts inside the shorter one's span.
    overlap = m.receipt3_prefix_spans(["\xab", "\xab\x12"])
    assert overlap["pairwise_disjoint"] is False
    assert len(overlap["overlaps"]) == 4
    assert m.encode_receipt3_operation_key("\xab\x12", 1).startswith("receipt3/o/c2ab12/")
    assert m.encode_receipt3_operation_key("\xab", 1).startswith("receipt3/o/c2ab/")


# ── D prior/current context + namespace/cron/eligibility recompute ───────────

def _ctx(session_id, parent=None, source="cli", started_at=1, delegate=None,
         branch=None, reset=None):
    m = _m()
    binding = [[name, m.sql_value_tag(value)] for name, value in
               zip(("id", "started_at", "parent_session_id", "source"),
                   (session_id, started_at, parent, source))]
    markers = [[name, m.sql_value_tag(value)] for name, value in
               zip(("_delegate_from", "_branched_from", "_reset_from"),
                   (delegate, branch, reset))]
    return {"session_id": session_id, "native_binding_fields": binding,
            "lineage_markers": markers,
            "context_sha256": m.H("native-session-context", {
                "domain_sha256": DOMAIN, "native_binding_fields": binding,
                "lineage_markers": markers})}


def test_compare_context_projections_and_edits():
    m = _m()
    root = _ctx("root")
    child = _ctx("child", parent="root")
    assert m.compare_context_projections([root, child], [child, root]) == {
        "equal": True, "missing": [], "extra": [], "drift": [],
        "prior_sessions": ["child", "root"], "current_sessions": ["child", "root"]}
    changed = _ctx("child", parent="root", started_at=2)
    report = m.compare_context_projections([root, child], [root, changed])
    assert report["equal"] is False and report["drift"] == ["child"]
    report = m.compare_context_projections([root, child], [child])
    assert report["equal"] is False and report["missing"] == ["root"]
    report = m.compare_context_projections([child], [root, child])
    assert report["extra"] == ["root"]
    with pytest.raises(ValueError):
        m.compare_context_projections([root, root], [root])
    with pytest.raises(ValueError):
        m.compare_context_projections([{"session_id": "x"}], [])


def test_recompute_session_feed_facts_root_cron_and_missing_edges():
    m = _m()
    root = _ctx("root", source="cli")
    child = _ctx("child", parent="root", source="cli")
    cron = _ctx("cron-session", parent="root", source="cron")
    report = m.recompute_session_feed_facts(
        [child, cron, root], approved_scope={"session_ids": ["root"]})
    assert report["recompute_version"] == "standalone-lineage/1"
    assert report["all_qualified"] is True
    facts = report["facts"]
    assert facts["child"]["root_session_id"] == "root"
    assert facts["child"]["parent_chain"] == ["child", "root"]
    assert facts["child"]["is_cron_surface"] is False
    assert facts["cron-session"]["is_cron_surface"] is True
    assert facts["cron-session"]["namespace_qualified"] is True
    ghost = _ctx("ghostly", parent="nowhere", source="cli")
    report = m.recompute_session_feed_facts([ghost], approved_scope={"session_ids": ["root"]})
    assert report["all_qualified"] is False
    assert report["facts"]["ghostly"]["reasons"] == ["missing_parent_edge"]
    # An approved anchor terminates the walk without expanding membership.
    report = m.recompute_session_feed_facts([child], approved_scope={"session_ids": ["root"]})
    assert report["facts"]["child"]["root_session_id"] == "root"
    assert report["facts"]["child"]["namespace_qualified"] is True
    reset = _ctx("cutover", reset="anchor-x")
    report = m.recompute_session_feed_facts([reset], approved_scope={"session_ids": ["anchor-x"]})
    assert report["facts"]["cutover"]["root_session_id"] == "anchor-x"
    assert report["facts"]["cutover"]["namespace_qualified"] is True


def test_verify_compression_edge_closure_facts():
    m = _m()
    edge = {"parent_session_id": "P", "child_session_id": "C",
            "end_reason": "compression"}
    result = m.verify_compression_edge_closure(edge)
    assert result["proven"] is False and result["basis"] is None
    assert "self-signs" in result["refusal"]
    approved = m.verify_compression_edge_closure(
        edge, approved_boundary_facts=[{"parent_session_id": "P", "child_session_id": "C"}])
    assert approved == {"proven": True, "basis": "approved_boundary_fact", "refusal": None}
    native = m.verify_compression_edge_closure(
        edge, native_child_operations=[{"kind": "compression_child",
                                        "owner_session_id": "C", "actor_session_id": "P",
                                        "closure_end_reason": "compression"}])
    assert native["proven"] is True
    assert native["basis"] == "native_compression_child_closure"
    wrong_owner = m.verify_compression_edge_closure(
        edge, native_child_operations=[{"kind": "compression_child",
                                        "owner_session_id": "other",
                                        "actor_session_id": "P",
                                        "closure_end_reason": "compression"}])
    assert wrong_owner["proven"] is False
    wrong_closure = m.verify_compression_edge_closure(
        edge, native_child_operations=[{"kind": "compression_child",
                                        "owner_session_id": "C", "actor_session_id": "P",
                                        "closure_end_reason": None}])
    assert wrong_closure["proven"] is False
    with pytest.raises(ValueError):
        m.verify_compression_edge_closure({"parent_session_id": "P"})


def test_decide_source_context_policy_blocks_and_admits():
    m = _m()
    child = _ctx("child", parent="root")
    current = [child]
    scope = {"session_ids": ["root"]}
    refused = m.decide_source_context_policy(q_same=True, prior_contexts=None,
                                             current_contexts=current)
    assert refused["decision"] == "BLOCK_SOURCE_CONTEXT_POLICY"
    assert refused["q_same_sufficient"] is False and refused["writes"] == "none"
    assert refused["preserves_native_values"] is True
    admitted = m.decide_source_context_policy(
        q_same=True, prior_contexts=[child], current_contexts=current, approved_scope=scope)
    assert admitted["decision"] == "ADMIT_D_SAME_SOURCE"
    assert admitted["reasons"] == [] and admitted["proposed"] is True
    drifted = _ctx("child", parent="root", source="cron")
    blocked = m.decide_source_context_policy(
        q_same=True, prior_contexts=[child], current_contexts=[drifted], approved_scope=scope)
    assert blocked["decision"] == "BLOCK_SOURCE_CONTEXT_POLICY"
    assert blocked["checks"]["drift"] == ["child"]
    unproven = m.decide_source_context_policy(
        q_same=True, prior_contexts=[child], current_contexts=current, approved_scope=scope,
        relations=[{"parent_session_id": "root", "child_session_id": "child"}])
    assert unproven["decision"] == "BLOCK_SOURCE_CONTEXT_POLICY"
    assert "unproven_compression_edge" in unproven["reasons"]
    proven = m.decide_source_context_policy(
        q_same=True, prior_contexts=[child], current_contexts=current, approved_scope=scope,
        relations=[{"parent_session_id": "root", "child_session_id": "child"}],
        native_child_operations=[{"kind": "compression_child", "owner_session_id": "child",
                                  "actor_session_id": "root",
                                  "closure_end_reason": "compression"}])
    assert proven["decision"] == "ADMIT_D_SAME_SOURCE"
    q_changed = m.decide_source_context_policy(
        q_same=False, prior_contexts=[child], current_contexts=current, approved_scope=scope)
    assert q_changed["decision"] == "BLOCK_SOURCE_CONTEXT_POLICY"
    assert "protected_Q_change_pending_correction_policy" in q_changed["reasons"]
    with pytest.raises(ValueError):
        m.decide_source_context_policy(q_same="yes", prior_contexts=None,
                                       current_contexts=current)


def test_resolve_context_before_ref_resolution_paths(tmp_path):
    m = _m()
    store = _settle(tmp_path)
    request, _result = _write_initial_receipt(store)
    conn = _connect(store)
    try:
        snapshot = m.read_receipt3_reader_snapshot(conn, SID_A)
    finally:
        conn.close()
    header = snapshot["operations"][0]["header"]
    ref = {"owner_session_id": SID_A, "generation": 1, "context_session_id": SID_A}
    # A self-reference to the current header refuses outright.
    with pytest.raises(ValueError):
        m.resolve_context_before_ref(snapshot, ref, current_header=header)
    other = dict(header, generation=2)
    resolved = m.resolve_context_before_ref(snapshot, ref, current_header=other)
    assert resolved["resolved"] is True and resolved["basis"] == "stored_member"
    assert resolved["context"]["session_id"] == SID_A
    missing = m.resolve_context_before_ref(
        snapshot, {"owner_session_id": SID_A, "generation": 9,
                   "context_session_id": SID_A}, current_header=other)
    assert missing["resolved"] is False and missing["reason"] == "missing_prior_member"
    activation_ref = {"activation_proof_sha256": "a" * 64, "context_session_id": "some-session"}
    unresolvable = m.resolve_context_before_ref(snapshot, activation_ref, current_header=other)
    assert unresolvable["resolved"] is False
    assert unresolvable["basis"] == "approved_activation_inventory"
    with_inventory = dict(snapshot)
    with_inventory["approved_activation_contexts"] = [_ctx("some-session")]
    resolved = m.resolve_context_before_ref(with_inventory, activation_ref,
                                            current_header=other)
    assert resolved["resolved"] is True
    assert resolved["basis"] == "approved_activation_inventory"
