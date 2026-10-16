"""Content-free, writer-owned evidence for exact same-session message copies.

Receipt v3 (schema/3) writer-owned provenance. This module is the single home of
the ``receipt3/`` protected namespace in the SessionDB store; the initial slice
implemented here is the C01 epoch range-checkpoint contract (receipt r4
amendments, section 2):

  * immutable epoch keys ``receipt3/e/<SIDhex>/<epochordinal20>/<start_generation20>``;
  * ``epoch_range_checkpoint`` body validation (schema/3, count 1..1024,
    ``end-start+1=count``, ``end<=start+1023``);
  * epoch-local chunking from the actual first header generation ``s0``
    (``start=s0+1024*k``), full blocks sealed with their last operation and a
    non-empty partial tail sealed only on epoch exit -- never early, never empty;
  * the frozen B framing of ``range_root_sha256`` over the checkpoint coordinate
    followed by every member header/page C byte in order.

Only the native writer (inside its own ``_execute_write`` transaction) may call
the builder; the generic ``SessionDB.set_meta``/``list_meta_prefix`` surfaces
refuse the reserved namespace.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import NoReturn

# Frozen codec primitives (design r2 "codec"; unchanged by r3/r4).
_BYTES_PREFIX = b"hermes-copy-receipt-v2\0"
_RECEIPT3_EPOCH_PREFIX = "receipt3/e/"
_RANGE_SCHEMA = "hermes-receipt-range/3"
_MAX_INT64 = 2**63 - 1
_CHUNK = 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_EPOCH = re.compile(r"sqlite:[0-9a-f]{64}\Z")
_SID_HEX = re.compile(r"(?:[0-9a-f]{2})+\Z")
_INT20 = re.compile(r"[0-9]{20}\Z")

_RANGE_BODY_KEYS = {
    "schema_version", "owner_session_id", "epoch", "epoch_ordinal",
    "start_generation", "end_generation", "before_head_sha256",
    "end_head_sha256", "operation_count", "page_count", "range_root_sha256",
}


def C(x) -> bytes:
    """Canonical JSON bytes: the frozen C encoding (design r2 "codec")."""
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8", "strict")


def H(label: str, x) -> str:
    """H(label,x) = sha256(prefix || ASCII(label) || NUL || C(x))."""
    return hashlib.sha256(_BYTES_PREFIX + label.encode("ascii") + b"\0" + C(x)).hexdigest()


def B(label: str, parts) -> str:
    """B(label,parts) = sha256(prefix || ASCII(label) || NUL || concat(uint64be(len)||p)).

    Length-framed so no part boundary can be moved without changing the hash.
    """
    h = hashlib.sha256(_BYTES_PREFIX + label.encode("ascii") + b"\0")
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


def _invalid() -> NoReturn:
    # Do not include values, message bodies, provider config, or exception reprs.
    raise ValueError("Invalid or unauthorized receipt metadata")


def _positive_int64(value) -> int:
    if type(value) is not int or not 0 < value <= _MAX_INT64:
        _invalid()
    return value


def _int20(value: int) -> str:
    return f"{value:020d}"


def _decode_int20(text: str) -> int:
    if type(text) is not str or not _INT20.fullmatch(text):
        _invalid()
    value = int(text)
    if not 0 < value <= _MAX_INT64:
        _invalid()
    return value


def _decode_sid_hex(text: str) -> str:
    if type(text) is not str or not text or not _SID_HEX.fullmatch(text):
        _invalid()
    try:
        session_id = bytes.fromhex(text).decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError):
        _invalid()
    if not session_id or session_id.encode("utf-8").hex() != text:
        _invalid()
    return session_id


# ── receipt3/e key codec ────────────────────────────────────────────────────

def encode_receipt3_epoch_key(owner_session_id: str, epoch_ordinal: int, start_generation: int) -> str:
    """``receipt3/e/<SIDhex>/<epochordinal20>/<start_generation20>`` (r4 C01).

    The start generation is part of the immutable key: a later epoch's blocks can
    never collide with (or be upserted over) an earlier epoch's checkpoint.
    """
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    _positive_int64(epoch_ordinal)
    _positive_int64(start_generation)
    return (_RECEIPT3_EPOCH_PREFIX + owner_session_id.encode("utf-8").hex()
            + "/" + _int20(epoch_ordinal) + "/" + _int20(start_generation))


def parse_receipt3_epoch_key(key: str) -> dict:
    """Parse an exact receipt3/e key; every non-canonical form is refused.

    Refused here: the old short e-key without start_generation, extra components,
    non-canonical integer encodings, non-hex/uppercase/odd SID hex, other families.
    """
    if type(key) is not str:
        _invalid()
    parts = key.split("/")
    if len(parts) != 5 or parts[0] != "receipt3" or parts[1] != "e":
        _invalid()
    return {"session_id": _decode_sid_hex(parts[2]),
            "epoch_ordinal": _decode_int20(parts[3]),
            "start_generation": _decode_int20(parts[4])}


# ── epoch_range_checkpoint body validation ──────────────────────────────────

def validate_epoch_range_checkpoint(body: dict) -> None:
    """Strict exact-shape validation of a sealed epoch range checkpoint body."""
    if type(body) is not dict or set(body) != _RANGE_BODY_KEYS:
        _invalid()
    if body["schema_version"] != _RANGE_SCHEMA:
        _invalid()
    if type(body["owner_session_id"]) is not str or not body["owner_session_id"]:
        _invalid()
    if type(body["epoch"]) is not str or not _EPOCH.fullmatch(body["epoch"]):
        _invalid()
    _positive_int64(body["epoch_ordinal"])
    start = _positive_int64(body["start_generation"])
    end = _positive_int64(body["end_generation"])
    if end < start or end > start + (_CHUNK - 1):
        _invalid()
    before = body["before_head_sha256"]
    if before is not None and (type(before) is not str or not _HEX.fullmatch(before)):
        _invalid()
    if not isinstance(body["end_head_sha256"], str) or not _HEX.fullmatch(body["end_head_sha256"]):
        _invalid()
    count = body["operation_count"]
    if type(count) is not int or not 1 <= count <= _CHUNK or end - start + 1 != count:
        _invalid()
    if type(body["page_count"]) is not int or body["page_count"] < 0:
        _invalid()
    if not isinstance(body["range_root_sha256"], str) or not _HEX.fullmatch(body["range_root_sha256"]):
        _invalid()


# ── range checkpoint builder (writer-owned; caller holds the transaction) ───

def build_epoch_range_checkpoint(*, owner_session_id: str, epoch: str, epoch_ordinal: int,
                                 s0: int, members: list[dict], epoch_exit: bool):
    """Seal one epoch-local range from *members* (each ``{"header": ..., "pages": [...]}``).

    Chunking is epoch-local: ``start=s0+1024*k`` for the actual first header
    generation ``s0`` of this epoch -- never global ``g%1024``. A full block seals
    with its last operation at any time; a non-empty partial tail seals only with
    ``epoch_exit`` (the continuation transaction that leaves the epoch); an empty
    range is never created. Member headers/pages are the original C-source bytes;
    the range root replays them in order.
    """
    _positive_int64(s0)
    _positive_int64(epoch_ordinal)
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    if type(epoch) is not str or not _EPOCH.fullmatch(epoch):
        _invalid()
    if type(members) is not list or not 1 <= len(members) <= _CHUNK:
        _invalid()
    headers = []
    pages_by_member = []
    for member in members:
        if type(member) is not dict or set(member) != {"header", "pages"}:
            _invalid()
        header = member["header"]
        pages = member["pages"]
        if type(header) is not dict or type(pages) is not list:
            _invalid()
        if header.get("epoch") != epoch:
            _invalid()  # a range never crosses its epoch
        if type(header.get("page_count")) is not int or header["page_count"] != len(pages):
            _invalid()
        for page in pages:
            if type(page) is not dict:
                _invalid()
        headers.append(header)
        pages_by_member.append(pages)
    generations = [_positive_int64(header["generation"]) for header in headers]
    start = generations[0]
    for previous, current in zip(generations, generations[1:]):
        if current != previous + 1:
            _invalid()  # contiguous coverage, no gaps or overlaps
    if (start - s0) < 0 or (start - s0) % _CHUNK != 0:
        _invalid()  # start must be s0+1024*k
    if len(members) < _CHUNK and not epoch_exit:
        _invalid()  # the open tail is not sealed early
    end = generations[-1]
    before_head = headers[0]["previous_head_sha256"]
    if before_head is not None and (type(before_head) is not str or not _HEX.fullmatch(before_head)):
        _invalid()
    end_head = headers[-1]["operation_sha256"]
    if type(end_head) is not str or not _HEX.fullmatch(end_head):
        _invalid()
    operation_count = len(members)
    page_count = sum(len(pages) for pages in pages_by_member)
    coordinate = {"owner_session_id": owner_session_id, "epoch": epoch,
                  "epoch_ordinal": epoch_ordinal, "start_generation": start,
                  "end_generation": end, "before_head_sha256": before_head,
                  "end_head_sha256": end_head, "operation_count": operation_count,
                  "page_count": page_count}
    parts = [C(coordinate)]
    for header, pages in zip(headers, pages_by_member):
        parts.append(C(header))
        parts.extend(C(page) for page in pages)
    body = {"schema_version": _RANGE_SCHEMA, "owner_session_id": owner_session_id,
            "epoch": epoch, "epoch_ordinal": epoch_ordinal, "start_generation": start,
            "end_generation": end, "before_head_sha256": before_head,
            "end_head_sha256": end_head, "operation_count": operation_count,
            "page_count": page_count, "range_root_sha256": B("range-checkpoint", parts)}
    validate_epoch_range_checkpoint(body)
    return encode_receipt3_epoch_key(owner_session_id, epoch_ordinal, start), body


# ── native session contexts (receipt r4, Q01 / protocol.context_binding) ────

_NATIVE_BINDING_FIELDS = ("id", "started_at", "parent_session_id", "source")
_LINEAGE_MARKERS = ("_delegate_from", "_branched_from", "_reset_from")


def sql_value_tag(value):
    """Encode one SQLite value with its SQL storage class (exact, reversible).

    Four binding fields and the permitted marker scalars keep their actual SQL
    types -- a string never impersonates an integer and a missing value is an
    explicit null, never omitted.
    """
    if value is None:
        return {"sql_null": True}
    if type(value) is bool:
        _invalid()
    if type(value) is int:
        return {"sql_integer": value}
    if type(value) is float:
        return {"sql_real": value.hex()}
    if type(value) is str:
        return {"sql_text": value}
    if type(value) is bytes:
        return {"sql_blob": value.hex()}
    _invalid()


def sql_value_decode(tag):
    """Inverse of :func:`sql_value_tag`; every other shape is refused."""
    if type(tag) is not dict or len(tag) != 1:
        _invalid()
    kind, value = next(iter(tag.items()))
    if kind == "sql_null" and value is True:
        return None
    if kind == "sql_integer" and type(value) is int and not isinstance(value, bool):
        return value
    if kind == "sql_real" and type(value) is str:
        try:
            return float.fromhex(value)
        except ValueError:
            _invalid()
    if kind == "sql_text" and type(value) is str:
        return value
    if kind == "sql_blob" and type(value) is str and _SID_HEX.fullmatch(value):
        return bytes.fromhex(value)
    _invalid()


def _projected_markers(conn, session_id):
    """The existing model_config JSON projection of the three lineage markers.

    Only the permitted TEXT scalar is taken (anything else projects explicit
    NULL); no full config ever leaves this projection.
    """
    row = conn.execute("SELECT model_config FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        _invalid()
    raw = row[0]
    try:
        config = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except (ValueError, TypeError):
        config = {}
    if not isinstance(config, dict):
        config = {}
    markers = []
    for marker in _LINEAGE_MARKERS:
        value = config.get(marker)
        markers.append([marker, sql_value_tag(value if type(value) is str else None)])
    return markers


def observe_native_session_contexts(conn, *, owner_session_id: str, actor_session_id: str,
                                    kind: str, domain_sha256: str) -> list[dict]:
    """Typed prior image for the header: 1 subject owner, or owner+actor for a
    compression child -- sorted by session_id strict UTF-8 bytes."""
    if kind == "compression_child":
        if owner_session_id == actor_session_id:
            _invalid()
        subjects = {owner_session_id, actor_session_id}
    else:
        subjects = {owner_session_id}
    contexts = []
    for session_id in sorted(subjects, key=lambda s: s.encode("utf-8")):
        row = conn.execute(
            "SELECT id, started_at, parent_session_id, source FROM sessions WHERE id = ?",
            (session_id,)).fetchone()
        if row is None or row["id"] != session_id:
            _invalid()
        binding = [[name, sql_value_tag(row[name])] for name in _NATIVE_BINDING_FIELDS]
        markers = _projected_markers(conn, session_id)
        contexts.append({
            "session_id": session_id,
            "native_binding_fields": binding,
            "lineage_markers": markers,
            "context_sha256": H("native-session-context", {
                "domain_sha256": domain_sha256,
                "native_binding_fields": binding,
                "lineage_markers": markers})})
    return contexts


def native_binding_values_from_context(context: dict) -> list:
    """Reader-side reconstruction of the original four values (in order)."""
    if type(context) is not dict:
        _invalid()
    binding = context.get("native_binding_fields")
    if type(binding) is not list or [name for name, _ in binding] != list(_NATIVE_BINDING_FIELDS):
        _invalid()
    return [sql_value_decode(value) for _, value in binding]


def native_session_binding(values) -> str:
    """The unchanged live-source algorithm on the reconstructed four values.

    Same algorithm as ``_native_session_binding`` in the standalone source
    (id/started_at/parent_session_id/source digest); the typed context hash is
    never passed off as this old binding hash.
    """
    if type(values) not in (list, tuple) or len(values) != 4:
        _invalid()
    return hashlib.sha256(C({name: value for name, value in zip(_NATIVE_BINDING_FIELDS, values)})).hexdigest()


def seal_operation_header(header: dict) -> dict:
    """Return the header with ``operation_sha256`` bound over ALL fields except
    itself (including native_session_contexts), regardless of insertion order."""
    if type(header) is not dict:
        _invalid()
    sealed = dict(header)
    sealed["operation_sha256"] = H(
        "operation", {key: value for key, value in header.items() if key != "operation_sha256"})
    return sealed
