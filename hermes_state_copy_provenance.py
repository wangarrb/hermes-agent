"""Content-free, writer-owned evidence for exact same-session message copies.

No schema migration or history inference: only the compaction writer supplies
causal source rows, and the native transaction owns both rows and receipts.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import NoReturn

COPY_RECEIPTS_KEY = "_message_copy_receipts_v1"
COPY_RECEIPTS_SCHEMA = "hermes-message-copy-receipts/1"
_MUTABLE = {"active", "compacted", "display_order"}
_COPY_EXCLUDED = _MUTABLE | {"id", "session_id"}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CONTAINER_KEYS = {"schema_version", "session_id", "source_epoch", "operations"}
_OPERATION_KEYS = {"operation_id", "kind", "copies", "receipt_sha256"}
_EDGE_KEYS = {"source_row_id", "target_row_id", "source_row_hash", "target_row_hash", "copy_hash"}


def _sql_hash(value):
    # Match the physical SQLite reader: BLOB bytes are not a repr or a lossy string.
    if isinstance(value, bytes):
        return {"sqlite_blob": value.hex()}
    if isinstance(value, dict):
        return {k: _sql_hash(v) for k, v in value.items()}
    return value


def _digest(value):
    raw = json.dumps(_sql_hash(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_epoch(conn):
    """Bind the actual opened file, not the environment's current profile home.

    In-memory SQLite has no durable feed identity; never mint one for it.
    SessionDB's existing replacement guard fences the handle before BEGIN.
    """
    filename = next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")
    if not filename:
        return None
    stat = Path(filename).stat()
    return "sqlite:" + _digest({"inode": [stat.st_dev, stat.st_ino],
                                "application_id": conn.execute("PRAGMA application_id").fetchone()[0]})


def _invalid() -> NoReturn:
    # Do not include values, message bodies, provider config, or exception reprs.
    raise ValueError("Invalid or unauthorized message copy receipt metadata")


def _shape(value, keys):
    if type(value) is not dict or set(value) != keys:
        _invalid()


def _hash(value):
    if type(value) is not str or not _HEX.fullmatch(value):
        _invalid()


def validate_receipts(value, session_id, epoch):
    """Validate append-only container structure and exact owner; never adopt inherited evidence."""
    _shape(value, _CONTAINER_KEYS)
    if (value["schema_version"] != COPY_RECEIPTS_SCHEMA or type(value["session_id"]) is not str
            or value["session_id"] != session_id or epoch is None or value["source_epoch"] != epoch
            or type(value["operations"]) is not list):
        _invalid()
    operation_ids, targets = set(), set()
    for operation in value["operations"]:
        _shape(operation, _OPERATION_KEYS)
        op_id = operation["operation_id"]
        try:
            parsed_uuid = uuid.UUID(op_id) if type(op_id) is str else None
            canonical = str(parsed_uuid) if parsed_uuid is not None else None
        except (ValueError, AttributeError):
            _invalid()
        if (type(op_id) is not str or parsed_uuid is None or parsed_uuid.int == 0
                or canonical != op_id or op_id in operation_ids
                or operation["kind"] != "inplace_compaction"
                or type(operation["copies"]) is not list or not operation["copies"]):
            _invalid()
        operation_ids.add(op_id)
        _hash(operation["receipt_sha256"])
        for edge in operation["copies"]:
            _shape(edge, _EDGE_KEYS)
            old_id, new_id = edge["source_row_id"], edge["target_row_id"]
            # Native AUTOINCREMENT clones always point forward. This also rules out cycles.
            if (type(old_id) is not int or type(new_id) is not int or not 0 < old_id < new_id
                    or new_id in targets):
                _invalid()
            targets.add(new_id)
            for key in ("source_row_hash", "target_row_hash", "copy_hash"):
                _hash(edge[key])
        if operation["receipt_sha256"] != _digest({k: v for k, v in operation.items() if k != "receipt_sha256"}):
            _invalid()


def checked_config(conn, session_id, raw):
    """Keep legacy tolerant decoding only for config without protected evidence."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except (ValueError, TypeError):
            if COPY_RECEIPTS_KEY in raw:
                _invalid()
            parsed = {}
    else:
        parsed = raw
    config = dict(parsed) if isinstance(parsed, dict) else {}
    if COPY_RECEIPTS_KEY in config:
        validate_receipts(config[COPY_RECEIPTS_KEY], session_id, source_epoch(conn))
    return config


def inherited_child_config(conn, parent_session_id, child_session_id, incoming, *, compression_parent=False):
    """Strip only exact native-parent history; it cannot prove a child's copies."""
    if isinstance(incoming, str):
        try:
            parsed = json.loads(incoming)
        except (ValueError, TypeError):
            if COPY_RECEIPTS_KEY in incoming:
                _invalid()
            return incoming
    else:
        parsed = incoming
    if not isinstance(parsed, dict) or COPY_RECEIPTS_KEY not in parsed:
        return incoming
    value = parsed[COPY_RECEIPTS_KEY]
    if (not isinstance(value, dict) or not isinstance(parent_session_id, str)
            or not parent_session_id or parent_session_id == child_session_id
            or value.get("session_id") != parent_session_id
            or not (compression_parent is True or any(parsed.get(key) == parent_session_id for key in
                    ("_delegate_from", "_branched_from", "_reset_from")))):
        return incoming  # Ordinary update protection rejects wrong/spoofed owners.
    row = conn.execute("SELECT model_config FROM sessions WHERE id = ?", (parent_session_id,)).fetchone()
    if row is None:
        _invalid()
    parent = checked_config(conn, parent_session_id, row[0])
    if COPY_RECEIPTS_KEY not in parent or value != parent[COPY_RECEIPTS_KEY]:
        _invalid()
    return {key: val for key, val in parsed.items() if key != COPY_RECEIPTS_KEY}


def protect_config_update(conn, session_id, current, incoming):
    """Provider/create/resume writes may echo, but never install or replace owned history.

    A replacement omitting the private key preserves it. Only the native copy
    writer below can append operations, inside its message transaction.
    """
    old = checked_config(conn, session_id, current)
    new = checked_config(conn, session_id, incoming)
    if COPY_RECEIPTS_KEY in new and (
            COPY_RECEIPTS_KEY not in old or new[COPY_RECEIPTS_KEY] != old[COPY_RECEIPTS_KEY]):
        _invalid()
    if COPY_RECEIPTS_KEY in old:
        new[COPY_RECEIPTS_KEY] = old[COPY_RECEIPTS_KEY]
    return new


def copy_source(conn, session_id, row_id):
    """Resolve only an explicit, literal native reference. Missing/wrong-owner IDs prove nothing."""
    if type(row_id) is not int or row_id <= 0:
        return None
    row = conn.execute("SELECT * FROM messages WHERE id = ? AND session_id = ?", (row_id, session_id)).fetchone()
    return dict(row) if row is not None else None


def _verified_edge(conn, session_id, source, target_id):
    target = copy_source(conn, session_id, target_id)
    if source is None or target is None or source["session_id"] != session_id or source["id"] >= target_id:
        return None
    # Sidecar/prune/carrier edits are derivative rows, not exact copies, even if the old ID survived.
    old_copy = {k: v for k, v in source.items() if k not in _COPY_EXCLUDED}
    new_copy = {k: v for k, v in target.items() if k not in _COPY_EXCLUDED}
    if old_copy != new_copy:
        return None
    current_source = copy_source(conn, session_id, source["id"])
    immutable_source = {k: v for k, v in source.items() if k not in _MUTABLE}
    if current_source is None or immutable_source != {k: v for k, v in current_source.items() if k not in _MUTABLE}:
        return None
    return {"source_row_id": source["id"], "target_row_id": target_id,
            "source_row_hash": _digest(immutable_source),
            "target_row_hash": _digest({k: v for k, v in target.items() if k not in _MUTABLE}),
            "copy_hash": _digest(old_copy)}


def append_copy_operation(conn, session_id, model_config_json, pairs):
    """Return merged JSON with verified exact edges; no independent insert gets a receipt.

    Caller saves this JSON and message counters before its existing transaction
    commits. Candidate sources were captured before live `_row_id` overwrite;
    targets are re-read after checkpoint rewrites/triggers have finished.
    """
    config = checked_config(conn, session_id, model_config_json)
    epoch = source_epoch(conn)
    if epoch is None:
        return model_config_json
    edges = [edge for source, target in pairs if (edge := _verified_edge(conn, session_id, source, target)) is not None]
    if not edges:
        return model_config_json
    operation = {"operation_id": str(uuid.uuid4()), "kind": "inplace_compaction", "copies": edges}
    operation["receipt_sha256"] = _digest(operation)
    previous = config.get(COPY_RECEIPTS_KEY)
    receipt = {"schema_version": COPY_RECEIPTS_SCHEMA, "session_id": session_id, "source_epoch": epoch,
               "operations": [*(previous["operations"] if previous else []), operation]}
    validate_receipts(receipt, session_id, epoch)
    config[COPY_RECEIPTS_KEY] = receipt
    return json.dumps(config)
