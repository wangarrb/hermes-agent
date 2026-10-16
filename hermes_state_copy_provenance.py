"""Content-free, writer-owned evidence for exact same-session message copies.

Receipt v3 (schema/3) writer-owned provenance. This module is the single home of
the ``receipt3/`` protected namespace in the SessionDB store:

  * C01 epoch range-checkpoint contract (receipt r4 amendments, section 2):
    immutable epoch keys ``receipt3/e/<SIDhex>/<epochordinal20>/<start_generation20>``;
    ``epoch_range_checkpoint`` body validation (schema/3, count 1..1024,
    ``end-start+1=count``, ``end<=start+1023``); epoch-local chunking from the
    actual first header generation ``s0`` (``start=s0+1024*k``), full blocks
    sealed with their last operation and a non-empty partial tail sealed only
    on epoch exit -- never early, never empty; the frozen B framing of
    ``range_root_sha256`` over the checkpoint coordinate followed by every
    member header/page C byte in order; member headers must carry the consumed
    fields and agree with the checkpoint coordinate (owner_session_id, epoch,
    epoch_ordinal) -- anything else is the canonical ValueError, never a
    KeyError;
  * Q01 native session contexts (``protocol.context_binding``) with the frozen
    r4 four-kind whitelist (inplace_compaction | compression_child |
    payload_repair | epoch_continuation); unknown kinds are refused instead of
    silently projecting a single-owner context;
  * the token-index key codec ``receipt3/t/<SIDhex>/<opuuid>`` (canonical
    lower-case uuid4 operation id);
  * the epoch-continuation payload written by the quiet recovery handle: the
    named journal entry ``receipt3/o/<SIDhex>/<G20>`` and its ordered pages
    ``receipt3/p/<SIDhex>/<G20>/<page20>`` as canonical C bytes, the framed
    pages root, the "insert-or-verified-bytes" range seal and the strict
    read-back verification that replays the original header/page bytes;
  * the strict model_config marker projection: duplicate raw config keys are
    refused (never last-wins) and damaged raw that mentions the protected
    ``_message_copy_receipts_`` family is refused outright; unreadable raw
    outside that family projects explicit NULLs;
  * the C02 step 6 acceptance machinery (receipt r4 ``protocol.recovery``):
    the identity-aware durable-table joint (``joint_formula`` /
    ``identity_modes``) over EVERY table including ``sqlite_schema``, FTS
    virtual tables and all shadows, the exact identity-aware inverse-delta
    verification against the original old latest, the independent read-only
    copy comparison (steps 2/3) and the frozen proof/roster (step 3);
  * the ``receipt3/r/<birth_epoch64hex>/<rowid20>`` row-state index codec
    (``protocol.storage`` namespace), its strict body shape, its reader-side
    derivation (``read_row_state_index``) and the identity-preserving
    same-transaction upsert (``writer_timeline[4]``: never INSERT OR REPLACE).

Only the native writer (inside its own ``_execute_write`` transaction) may call
the builder; the generic ``SessionDB.set_meta``/``list_meta_prefix`` surfaces
refuse the reserved namespace. * the M2D design binding (receipt r4 amendments section 5): effective r4
    metadata, the ``r4_expected``-over-``r3_expected`` expected precedence
    resolver over the 40-row acceptance disposition, and the accurate
    disclosure fields -- auditable design metadata explicitly NOT protocol
    record bytes (see the M2D section at the end of this module).

The joint/inverse/proof readers open their own
independent ``mode=ro``/``query_only`` connections and never write.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import NoReturn

# Frozen codec primitives (design r2 "codec"; unchanged by r3/r4).
_BYTES_PREFIX = b"hermes-copy-receipt-v2\0"
_RECEIPT3_EPOCH_PREFIX = "receipt3/e/"
_RECEIPT3_TOKEN_PREFIX = "receipt3/t/"
_RANGE_SCHEMA = "hermes-receipt-range/3"
_OPERATION_SCHEMA = "hermes-receipt-operation/3"
_PAGE_SCHEMA = "hermes-receipt-page/3"
_HEAD_SCHEMA = "hermes-message-copy-receipts/3"
_PROTOCOL_REVISION = 3
# Frozen r4 domain D (hermes-message-source-payload/schema31/3 manifest hash).
_DOMAIN_SHA256 = "2b910d15bacab84fbb6a160e34b491c21f9fd268e396c4831915fe903a420367"
_MAX_PAGE_BYTES = 65536
_MAX_INT64 = 2**63 - 1
_CHUNK = 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_EPOCH = re.compile(r"sqlite:[0-9a-f]{64}\Z")
_SID_HEX = re.compile(r"(?:[0-9a-f]{2})+\Z")
_INT20 = re.compile(r"[0-9]{20}\Z")
_UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")

_RANGE_BODY_KEYS = {
    "schema_version", "owner_session_id", "epoch", "epoch_ordinal",
    "start_generation", "end_generation", "before_head_sha256",
    "end_head_sha256", "operation_count", "page_count", "range_root_sha256",
}

# Member operation headers the range builder consumes: every consumed field is
# required (canonical ValueError when missing, never a KeyError) and the
# coordinate fields must agree with the checkpoint (M07 repair).
_MEMBER_HEADER_KEYS = frozenset({
    "owner_session_id", "epoch", "epoch_ordinal", "generation", "page_count",
    "previous_head_sha256", "operation_sha256",
})


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


def _canonical_uuid4(token: str) -> str:
    if type(token) is not str or not _UUID4.fullmatch(token):
        _invalid()
    return token


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


# ── receipt3/t token-index key codec (I03 repair: protocol.storage) ─────────

def encode_receipt3_token_key(owner_session_id: str, token: str) -> str:
    """``receipt3/t/<SIDhex>/<opuuid>`` -- the fixed token-index namespace.

    The owner component is mandatory: two owners carrying the same token are
    distinct operations, and the key-range reader must be able to page the
    index per owner without a bare-token key ever existing.
    """
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    _canonical_uuid4(token)
    return _RECEIPT3_TOKEN_PREFIX + owner_session_id.encode("utf-8").hex() + "/" + token


def parse_receipt3_token_key(key: str) -> dict:
    """Parse an exact receipt3/t key; every non-canonical form is refused."""
    if type(key) is not str:
        _invalid()
    parts = key.split("/")
    if len(parts) != 4 or parts[0] != "receipt3" or parts[1] != "t":
        _invalid()
    return {"session_id": _decode_sid_hex(parts[2]),
            "operation_id": _canonical_uuid4(parts[3])}


# ── receipt3/o | receipt3/p journal codecs (epoch-continuation payload) ─────

_RECEIPT3_OPERATION_PREFIX = "receipt3/o/"
_RECEIPT3_PAGE_PREFIX = "receipt3/p/"


def _decode_int20_nonneg(text: str) -> int:
    if type(text) is not str or not _INT20.fullmatch(text):
        _invalid()
    value = int(text)
    if value < 0 or value > _MAX_INT64:
        _invalid()
    return value


def encode_receipt3_operation_key(owner_session_id: str, generation: int) -> str:
    """``receipt3/o/<SIDhex>/<G20>`` -- the immutable named operation header."""
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    _positive_int64(generation)
    return (_RECEIPT3_OPERATION_PREFIX + owner_session_id.encode("utf-8").hex()
            + "/" + _int20(generation))


def parse_receipt3_operation_key(key: str) -> dict:
    """Parse an exact receipt3/o key; non-canonical forms are refused."""
    if type(key) is not str:
        _invalid()
    parts = key.split("/")
    if len(parts) != 4 or parts[0] != "receipt3" or parts[1] != "o":
        _invalid()
    return {"session_id": _decode_sid_hex(parts[2]),
            "generation": _decode_int20(parts[3])}


def encode_receipt3_page_key(owner_session_id: str, generation: int, page_ordinal: int) -> str:
    """``receipt3/p/<SIDhex>/<G20>/<page20>`` -- ordered journal page bytes."""
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    _positive_int64(generation)
    if type(page_ordinal) is not int or not 0 <= page_ordinal <= _MAX_INT64:
        _invalid()
    return (_RECEIPT3_PAGE_PREFIX + owner_session_id.encode("utf-8").hex()
            + "/" + _int20(generation) + "/" + _int20(page_ordinal))


def parse_receipt3_page_key(key: str) -> dict:
    """Parse an exact receipt3/p key; non-canonical forms are refused."""
    if type(key) is not str:
        _invalid()
    parts = key.split("/")
    if len(parts) != 5 or parts[0] != "receipt3" or parts[1] != "p":
        _invalid()
    return {"session_id": _decode_sid_hex(parts[2]),
            "generation": _decode_int20(parts[3]),
            "page_ordinal": _decode_int20_nonneg(parts[4])}


def _receipt3_prefix_span(prefix: str, owner_session_id: str):
    """Half-open ``[lo, hi)`` key span for one owner under a receipt3/ family."""
    lo = prefix + owner_session_id.encode("utf-8").hex() + "/"
    return lo, lo[:-1] + ":"          # ':' sorts after every 20-digit suffix


# ── protected head / journal page / operation header shapes ─────────────────

_HEAD_KEYS = frozenset({
    "schema_version", "protocol_revision", "domain_sha256", "owner_session_id",
    "active_epoch", "epoch_ordinal", "generation", "head_sha256",
    "v1_binding_sha256", "activation_proof_sha256"})
_PAGE_KEYS = frozenset({
    "schema_version", "owner_session_id", "epoch", "generation", "page_ordinal", "items"})
_ITEM_KINDS = frozenset({"copy", "derivation", "native_result"})
_OPERATION_HEADER_KEYS = frozenset({
    "schema_version", "domain_sha256", "owner_session_id", "actor_session_id",
    "epoch", "epoch_ordinal", "generation", "operation_id", "request_sha256",
    "previous_head_sha256", "group", "kind", "base_fence_sha256", "page_count",
    "item_count", "pages_root_sha256", "result_sha256", "continuation",
    "operation_sha256", "native_session_contexts"})
_CONTINUATION_KEYS = frozenset({
    "proof_sha256", "old_epoch", "new_epoch", "old_generation", "old_head_sha256",
    "old_head_token_sha256", "boundary_roster_sha256", "alias_inventory_sha256"})


def _sha256_hex(value) -> str:
    if type(value) is not str or not _HEX.fullmatch(value):
        _invalid()
    return value


def _epoch_string(value) -> str:
    if type(value) is not str or not _EPOCH.fullmatch(value):
        _invalid()
    return value


def validate_receipt3_head(head: dict) -> None:
    """Strict exact-shape validation of the protected head (10 fields)."""
    if type(head) is not dict or set(head) != _HEAD_KEYS:
        _invalid()
    if head["schema_version"] != _HEAD_SCHEMA or head["protocol_revision"] != _PROTOCOL_REVISION:
        _invalid()
    _sha256_hex(head["domain_sha256"])
    if type(head["owner_session_id"]) is not str or not head["owner_session_id"]:
        _invalid()
    _epoch_string(head["active_epoch"])
    _positive_int64(head["epoch_ordinal"])
    generation = head["generation"]
    if type(generation) is not int or not 0 <= generation <= _MAX_INT64:
        _invalid()
    if head["head_sha256"] is not None:
        _sha256_hex(head["head_sha256"])
    if head["v1_binding_sha256"] is not None:
        _sha256_hex(head["v1_binding_sha256"])
    _sha256_hex(head["activation_proof_sha256"])


def _validate_row_ref(row_ref) -> None:
    """Exact shape of one actual row reference (record_shapes.row_ref)."""
    if type(row_ref) is not dict or set(row_ref) != {
            "birth_epoch", "row_id", "owner_session_id"}:
        _invalid()
    _epoch_string(row_ref["birth_epoch"])
    _positive_int64(row_ref["row_id"])
    if type(row_ref["owner_session_id"]) is not str or not row_ref["owner_session_id"]:
        _invalid()


def _validate_reference(ref) -> None:
    """One immutable prior-context reference: a stored member coordinate or an
    approved boundary anchor (record_shapes.derivation_item)."""
    if type(ref) is not dict:
        _invalid()
    keys = set(ref)
    if keys == {"owner_session_id", "generation", "context_session_id"}:
        if type(ref["owner_session_id"]) is not str or not ref["owner_session_id"]:
            _invalid()
        _positive_int64(ref["generation"])
    elif keys == {"activation_proof_sha256", "context_session_id"}:
        _sha256_hex(ref["activation_proof_sha256"])
    else:
        _invalid()
    if type(ref["context_session_id"]) is not str or not ref["context_session_id"]:
        _invalid()


def _validate_item(item) -> None:
    """Full exact shape of one journal-page item (record_shapes.copy_item /
    derivation_item / native_result_item); unknown keys and inconsistent
    dispositions are refused, never silently projected."""
    if type(item) is not dict:
        _invalid()
    kind = item.get("kind")
    if kind == "native_result":
        if set(item) != {"kind", "input_ordinal", "row_ref", "disposition", "result_count"}:
            _invalid()
        disposition = item["disposition"]
        if disposition not in ("inserted", "repaired", "adopted", "aggregate"):
            _invalid()
        input_ordinal = item["input_ordinal"]
        if input_ordinal is not None and (type(input_ordinal) is not int
                                          or not 0 <= input_ordinal <= _MAX_INT64):
            _invalid()
        row_ref = item["row_ref"]
        if row_ref is not None:
            _validate_row_ref(row_ref)
        if (disposition == "aggregate") != (input_ordinal is None and row_ref is None):
            _invalid()   # aggregate ⇔ both coordinates null; a result carries a row ref
        result_count = item["result_count"]
        if result_count is not None and (type(result_count) is not int
                                         or not -_MAX_INT64 <= result_count <= _MAX_INT64):
            _invalid()
    elif kind == "copy":
        if set(item) != {"kind", "source", "source_previous_event", "target", "mode",
                         "observation", "native_request_sha256"}:
            _invalid()
        if item["source"] != "payload_state" or item["target"] != "payload_state":
            _invalid()
        if item["mode"] not in ("inplace_compaction", "compression_child"):
            _invalid()
        if item["observation"] != ("source_pre_insert_to_target_final; "
                                   "writer enforces insert P equality"):
            _invalid()
        _sha256_hex(item["native_request_sha256"])
        event = item["source_previous_event"]
        if event is not None:
            if type(event) is not list or len(event) != 4:
                _invalid()
            owner, generation, page, index = event
            if type(owner) is not str or not owner:
                _invalid()
            _positive_int64(generation)
            if type(page) is not int or not 0 <= page <= _MAX_INT64:
                _invalid()
            if type(index) is not int or not 0 <= index <= _MAX_INT64:
                _invalid()
    elif kind == "derivation":
        if set(item) != {"kind", "writer_kind", "before", "after", "previous_event",
                         "feed_class", "native_request_sha256", "context_before_ref"}:
            _invalid()
        if item["writer_kind"] not in ("cas_owned_repair", "legacy_blank_content_repair",
                                       "set_user_message_content", "approved_core_repair"):
            _invalid()
        if item["before"] != "payload_state" or item["after"] != "payload_state":
            _invalid()
        if item["feed_class"] not in ("same", "changed"):
            _invalid()
        _sha256_hex(item["native_request_sha256"])
        _validate_reference(item["previous_event"])
        _validate_reference(item["context_before_ref"])
    else:
        _invalid()


def validate_journal_page(body: dict) -> None:
    """Strict page shape: exact fields, 1..8 fully-shaped items, canonical
    <= 65536 bytes."""
    if type(body) is not dict or set(body) != _PAGE_KEYS:
        _invalid()
    if body["schema_version"] != _PAGE_SCHEMA:
        _invalid()
    if type(body["owner_session_id"]) is not str or not body["owner_session_id"]:
        _invalid()
    _epoch_string(body["epoch"])
    _positive_int64(body["generation"])
    ordinal = body["page_ordinal"]
    if type(ordinal) is not int or not 0 <= ordinal <= _MAX_INT64:
        _invalid()
    items = body["items"]
    if type(items) is not list or not 1 <= len(items) <= 8:
        _invalid()
    for item in items:
        _validate_item(item)
    if len(C(body)) > _MAX_PAGE_BYTES:
        _invalid()


def build_journal_page(*, owner_session_id: str, epoch: str, generation: int,
                       page_ordinal: int, items: list) -> dict:
    """Build one validated journal page body (never over 64KiB canonical)."""
    body = {"schema_version": _PAGE_SCHEMA, "owner_session_id": owner_session_id,
            "epoch": epoch, "generation": generation, "page_ordinal": page_ordinal,
            "items": items}
    validate_journal_page(body)
    return body


def journal_pages_root(owner_session_id: str, epoch: str, generation: int, pages: list) -> str:
    """``B("journal-content", [C({owner,epoch,generation}), C(page0), ...])``.
    The label is the r4 "journal content root" framing over the operation
    coordinate followed by every ordered page C byte.
    """
    parts = [C({"owner_session_id": owner_session_id, "epoch": epoch,
                "generation": generation})]
    parts.extend(C(page) for page in pages)
    return B("journal-content", parts)


def _validate_native_session_context(context, domain_sha256: str) -> None:
    """Full exact shape of one typed native-session-context item: the ordered
    typed preimage, the strict marker projection and the recomputed context
    hash (record_shapes.operation_header.native_session_contexts)."""
    if type(context) is not dict or set(context) != {
            "session_id", "native_binding_fields", "lineage_markers", "context_sha256"}:
        _invalid()
    if type(context["session_id"]) is not str or not context["session_id"]:
        _invalid()
    binding = context["native_binding_fields"]
    if type(binding) is not list or len(binding) != len(_NATIVE_BINDING_FIELDS):
        _invalid()
    for entry in binding:
        if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str:
            _invalid()
    if [entry[0] for entry in binding] != list(_NATIVE_BINDING_FIELDS):
        _invalid()
    values = [sql_value_decode(tag) for _, tag in binding]
    if values[0] != context["session_id"]:
        _invalid()   # the id binding must equal the subject session_id
    markers = context["lineage_markers"]
    if type(markers) is not list or len(markers) != len(_LINEAGE_MARKERS):
        _invalid()
    for entry in markers:
        if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str:
            _invalid()
    if [entry[0] for entry in markers] != list(_LINEAGE_MARKERS):
        _invalid()
    for _, tag in markers:
        value = sql_value_decode(tag)
        if value is not None and type(value) is not str:
            _invalid()   # markers are exact permitted TEXT scalars or explicit null
    if context["context_sha256"] != H("native-session-context", {
            "domain_sha256": domain_sha256,
            "native_binding_fields": binding,
            "lineage_markers": markers}):
        _invalid()


def validate_operation_header(header: dict) -> None:
    """Strict exact-shape validation of the writer-built operation header."""
    if type(header) is not dict or set(header) != _OPERATION_HEADER_KEYS:
        _invalid()
    if header["schema_version"] != _OPERATION_SCHEMA:
        _invalid()
    _sha256_hex(header["domain_sha256"])
    if type(header["owner_session_id"]) is not str or not header["owner_session_id"]:
        _invalid()
    if type(header["actor_session_id"]) is not str or not header["actor_session_id"]:
        _invalid()
    _epoch_string(header["epoch"])
    _positive_int64(header["epoch_ordinal"])
    _positive_int64(header["generation"])
    _canonical_uuid4(header["operation_id"])
    _sha256_hex(header["request_sha256"])
    if header["previous_head_sha256"] is not None:
        _sha256_hex(header["previous_head_sha256"])
    group = header["group"]
    if type(group) is not dict or set(group) != {"group_id", "members"}:
        _invalid()
    _canonical_uuid4(group["group_id"])
    members = group["members"]
    if type(members) is not list or not 1 <= len(members) <= 2:
        _invalid()
    ordinals = []
    for member in members:
        if type(member) is not dict or set(member) != {"owner_session_id", "operation_id", "ordinal"}:
            _invalid()
        if type(member["owner_session_id"]) is not str or not member["owner_session_id"]:
            _invalid()
        _canonical_uuid4(member["operation_id"])
        ordinal = member["ordinal"]
        if type(ordinal) is not int or not 0 <= ordinal <= _MAX_INT64:
            _invalid()
        ordinals.append(ordinal)
    if ordinals != sorted(ordinals):
        _invalid()
    if header["kind"] not in _KINDS:
        _invalid()
    _sha256_hex(header["base_fence_sha256"])
    if type(header["page_count"]) is not int or not 0 <= header["page_count"] <= _MAX_INT64:
        _invalid()
    if type(header["item_count"]) is not int or not 0 <= header["item_count"] <= _MAX_INT64:
        _invalid()
    _sha256_hex(header["pages_root_sha256"])
    _sha256_hex(header["result_sha256"])
    continuation = header["continuation"]
    if header["kind"] == "epoch_continuation":
        if type(continuation) is not dict or set(continuation) != _CONTINUATION_KEYS:
            _invalid()
        _sha256_hex(continuation["proof_sha256"])
        _epoch_string(continuation["old_epoch"])
        _epoch_string(continuation["new_epoch"])
        old_generation = continuation["old_generation"]
        if type(old_generation) is not int or not 0 <= old_generation <= _MAX_INT64:
            _invalid()
        if continuation["old_head_sha256"] is not None:
            _sha256_hex(continuation["old_head_sha256"])
        elif old_generation != 0:
            _invalid()   # a null old head only belongs to the g0 -> g1 first receipt
        _sha256_hex(continuation["old_head_token_sha256"])
        _sha256_hex(continuation["boundary_roster_sha256"])
        _sha256_hex(continuation["alias_inventory_sha256"])
    elif continuation is not None:
        _invalid()
    contexts = header["native_session_contexts"]
    if type(contexts) is not list or not 1 <= len(contexts) <= 2:
        _invalid()
    seen = []
    for context in contexts:
        _validate_native_session_context(context, header["domain_sha256"])
        seen.append(context["session_id"])
    if seen != sorted(seen, key=lambda sid: sid.encode("utf-8")) or len(set(seen)) != len(seen):
        _invalid()
    if header["owner_session_id"] not in seen:
        _invalid()
    if header["kind"] == "compression_child":
        if len(contexts) != 2 or header["actor_session_id"] not in seen:
            _invalid()
    elif len(contexts) != 1:
        _invalid()
    sealed = {key: value for key, value in header.items() if key != "operation_sha256"}
    if header["operation_sha256"] != H("operation", sealed):
        _invalid()


def _deterministic_uuid4(label: str, seed: str) -> str:
    """A canonical UUID4-shaped id fixed for one label/seed pair (stable across
    ``_execute_write`` retries: never re-minted inside the retried callback)."""
    digest = hashlib.sha256(_BYTES_PREFIX + label.encode("ascii") + b"\0"
                            + seed.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


def build_epoch_continuation_header(*, request: dict, request_sha256: str,
                                    domain_sha256: str, epoch: str, epoch_ordinal: int,
                                    generation: int, previous_head_sha256,
                                    pages: list, native_session_contexts: list) -> dict:
    """Seal the full r4 operation header for one approved epoch continuation.
    The caller supplies the approved request (transition binding, old
    coordinates, CAS token), the new epoch coordinates and the ordered journal
    pages. ``base_fence_sha256`` binds the available approved-transition fence
    coordinates of this slice (the full sampled head+refs/P+lease fence of a
    production transition is a separate approval surface).
    """
    owner_session_id = request["owner_session_id"]
    token = request["token"]
    _positive_int64(generation)
    _positive_int64(epoch_ordinal)
    _epoch_string(epoch)
    if previous_head_sha256 is not None:
        _sha256_hex(previous_head_sha256)
    if type(pages) is not list:
        _invalid()
    continuation = {key: request[key] for key in _CONTINUATION_KEYS}
    result_items = [item for page in pages for item in page.get("items", [])
                    if type(item) is dict and item.get("kind") == "native_result"]
    fence = {key: request[key] for key in
             ("proof_sha256", "old_epoch", "old_generation", "old_head_sha256",
              "old_head_token_sha256", "boundary_roster_sha256", "alias_inventory_sha256")}
    header = {
        "schema_version": _OPERATION_SCHEMA,
        "domain_sha256": domain_sha256,
        "owner_session_id": owner_session_id,
        "actor_session_id": owner_session_id,
        "epoch": epoch,
        "epoch_ordinal": epoch_ordinal,
        "generation": generation,
        "operation_id": token,
        "request_sha256": request_sha256,
        "previous_head_sha256": previous_head_sha256,
        "group": {"group_id": _deterministic_uuid4("receipt3-group", token),
                  "members": [{"owner_session_id": owner_session_id,
                               "operation_id": token, "ordinal": 0}]},
        "kind": "epoch_continuation",
        "base_fence_sha256": hashlib.sha256(C(fence)).hexdigest(),
        "page_count": len(pages),
        "item_count": sum(len(page["items"]) for page in pages),
        "pages_root_sha256": journal_pages_root(owner_session_id, epoch, generation, pages),
        "result_sha256": B("native-result", [C(item) for item in result_items]),
        "continuation": continuation,
        "native_session_contexts": list(native_session_contexts)}
    sealed = seal_operation_header(header)
    validate_operation_header(sealed)
    return sealed


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
    the range root replays them in order. Member headers must carry every
    consumed field and agree with the checkpoint coordinate (owner_session_id,
    epoch, epoch_ordinal) or the canonical ValueError is raised (never a
    KeyError).
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
        if not _MEMBER_HEADER_KEYS <= set(header):
            _invalid()  # missing consumed fields: canonical ValueError, never KeyError
        if type(header["owner_session_id"]) is not str or not header["owner_session_id"]:
            _invalid()
        if header["owner_session_id"] != owner_session_id:
            _invalid()  # key/body/headers agree on the owner
        if header["epoch"] != epoch:
            _invalid()  # a range never crosses its epoch
        _positive_int64(header["epoch_ordinal"])
        if header["epoch_ordinal"] != epoch_ordinal:
            _invalid()  # key/body/headers agree on the epoch ordinal
        _positive_int64(header["generation"])
        if type(header["page_count"]) is not int or header["page_count"] != len(pages):
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
# The frozen r4 operation-header kind whitelist (M05 repair): an unknown kind is
# refused instead of silently projecting a single-owner context.
_KINDS = frozenset({"inplace_compaction", "compression_child", "payload_repair",
                    "epoch_continuation"})


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


class _DuplicateConfigKey(ValueError):
    """Raised by the strict projection hook when one raw config key repeats."""


def _unique_config_object(pairs):
    """object_pairs_hook that refuses duplicate keys (never last-wins)."""
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise _DuplicateConfigKey(key)
        obj[key] = value
    return obj


def _projected_markers(conn, session_id):
    """The existing model_config JSON projection of the three lineage markers.

    Strict projection (M06 repair): duplicate raw config keys are refused
    outright, and damaged raw that mentions the protected
    ``_message_copy_receipts_`` family is refused because a truncated
    head/bank must never be projected as "no markers". Unreadable raw outside
    that family keeps the narrow tolerance (the markers project explicit
    NULL). Only the permitted TEXT scalar is taken (anything else projects
    explicit NULL); no full config ever leaves this projection.
    """
    row = conn.execute("SELECT model_config FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        _invalid()
    raw = row[0]
    config = {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw, object_pairs_hook=_unique_config_object)
        except _DuplicateConfigKey:
            _invalid()
        except (ValueError, TypeError):
            if "_message_copy_receipts_" in raw:
                _invalid()
            parsed = None
        if parsed is not None:
            if isinstance(parsed, dict):
                config = parsed
            elif "_message_copy_receipts_" in raw:
                _invalid()
    markers = []
    for marker in _LINEAGE_MARKERS:
        value = config.get(marker)
        markers.append([marker, sql_value_tag(value if type(value) is str else None)])
    return markers


def observe_native_session_contexts(conn, *, owner_session_id: str, actor_session_id: str,
                                    kind: str, domain_sha256: str) -> list[dict]:
    """Typed prior image for the header: 1 subject owner, or owner+actor for a
    compression child -- sorted by session_id strict UTF-8 bytes.

    The kind must be one of the frozen r4 four (``_KINDS``); anything else is
    refused before any row is read.
    """
    if type(kind) is not str or kind not in _KINDS:
        _invalid()
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


# ── persistence read-back: strict readers over the receipt3/ journal ────────

def _read_canonical(conn, key: str):
    """Read one protected key; the stored bytes must be exactly canonical C."""
    row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        _invalid()
    value = row[0]
    if type(value) is not str:
        _invalid()
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_config_object)
    except _DuplicateConfigKey:
        _invalid()
    except (ValueError, TypeError):
        _invalid()
    if C(parsed).decode("utf-8") != value:
        _invalid()   # non-canonical stored bytes are refused, never replayed
    return parsed, value


def read_operation(conn, owner_session_id: str, generation: int) -> dict:
    """Read back one journal operation: header + pages + the original bytes.

    The stored header is validated to its full exact form (every field, the
    typed contexts and the self-hash); every named key, coordinate, page
    ordinal, the page/item counts, the framed pages root and the result digest
    recomputed from the actual page items are re-verified; a missing or
    non-canonical entry is the canonical ValueError.
    """
    key = encode_receipt3_operation_key(owner_session_id, generation)
    header, header_bytes = _read_canonical(conn, key)
    validate_operation_header(header)
    if header["owner_session_id"] != owner_session_id or header["generation"] != generation:
        _invalid()
    page_count = header["page_count"]
    pages, page_bytes = [], []
    for ordinal in range(page_count):
        page, raw = _read_canonical(
            conn, encode_receipt3_page_key(owner_session_id, generation, ordinal))
        validate_journal_page(page)
        if (page["owner_session_id"] != owner_session_id or page["generation"] != generation
                or page["page_ordinal"] != ordinal or page["epoch"] != header["epoch"]):
            _invalid()
        pages.append(page)
        page_bytes.append(raw)
    if sum(len(page["items"]) for page in pages) != header["item_count"]:
        _invalid()
    if journal_pages_root(owner_session_id, header["epoch"], generation, pages) \
            != header["pages_root_sha256"]:
        _invalid()
    result_items = [item for page in pages for item in page["items"]
                    if item["kind"] == "native_result"]
    if header["result_sha256"] != B("native-result", [C(item) for item in result_items]):
        _invalid()   # the result digest must recompute from the actual page items
    return {"header": header, "pages": pages, "header_bytes": header_bytes,
            "page_bytes": page_bytes}


def read_journal_members(conn, owner_session_id: str, *, epoch=None, epoch_ordinal=None,
                         up_to_generation=None, batch: int = 128) -> list:
    """Replay one owner's stored journal in key order (strict, bounded batches).

    Optionally filtered to one epoch string or epoch ordinal; ``up_to_generation``
    stops the scan at the last generation of the claimed epoch. The selected
    members must be generation-contiguous, reach the claimed last generation and
    their previous-head chain must link exactly; a gap (missing member) refuses.
    """
    if epoch is not None:
        _epoch_string(epoch)
    if epoch_ordinal is not None:
        _positive_int64(epoch_ordinal)
    if up_to_generation is not None:
        if type(up_to_generation) is not int or not 0 <= up_to_generation <= _MAX_INT64:
            _invalid()
    lo, hi = _receipt3_prefix_span(_RECEIPT3_OPERATION_PREFIX, owner_session_id)
    members = []
    last = lo
    while True:
        rows = conn.execute(
            "SELECT key, value FROM state_meta WHERE key > ? AND key < ? ORDER BY key LIMIT ?",
            (last, hi, batch)).fetchall()
        if not rows:
            break
        stop = False
        for key, _value in rows:
            last = key
            generation = parse_receipt3_operation_key(key)["generation"]
            if up_to_generation is not None and generation > up_to_generation:
                stop = True
                break
            member = read_operation(conn, owner_session_id, generation)
            header = member["header"]
            if epoch is not None and header.get("epoch") != epoch:
                continue
            if epoch_ordinal is not None and header.get("epoch_ordinal") != epoch_ordinal:
                continue
            members.append(member)
        if stop:
            break
    if members:
        generations = [member["header"]["generation"] for member in members]
        for previous, current in zip(generations, generations[1:]):
            if current != previous + 1:
                _invalid()   # a gap in the epoch chain refuses (never replay over it)
        if up_to_generation is not None and generations[-1] != up_to_generation:
            _invalid()   # the chain must reach the claimed last generation
        first_head = members[0]["header"].get("previous_head_sha256")
        if generations[0] == 1:
            if first_head is not None:
                _invalid()
        elif not (type(first_head) is str and _HEX.fullmatch(first_head)):
            _invalid()
        for previous, current in zip(members, members[1:]):
            if (current["header"].get("previous_head_sha256")
                    != previous["header"].get("operation_sha256")):
                _invalid()   # the stored head chain must link exactly
    return members


def _read_range_blocks(conn, owner_session_id: str, epoch_ordinal: int,
                       batch: int = 128) -> list:
    """Read one owner epoch's checkpoints: strict keys, canonical bodies, exact
    key/body coordinate agreement (ordered by start generation)."""
    lo = (_RECEIPT3_EPOCH_PREFIX + owner_session_id.encode("utf-8").hex()
          + "/" + _int20(epoch_ordinal) + "/")
    hi = lo[:-1] + ":"
    blocks = []
    last = lo
    while True:
        rows = conn.execute(
            "SELECT key, value FROM state_meta WHERE key > ? AND key < ? ORDER BY key LIMIT ?",
            (last, hi, batch)).fetchall()
        if not rows:
            break
        for key, _value in rows:
            last = key
            parsed = parse_receipt3_epoch_key(key)
            if (parsed["session_id"] != owner_session_id
                    or parsed["epoch_ordinal"] != epoch_ordinal):
                _invalid()
            body, _raw = _read_canonical(conn, key)
            validate_epoch_range_checkpoint(body)
            if (body["owner_session_id"] != owner_session_id
                    or body["epoch_ordinal"] != epoch_ordinal
                    or body["start_generation"] != parsed["start_generation"]):
                _invalid()
            blocks.append({"key": key, "body": body,
                           "start": body["start_generation"],
                           "end": body["end_generation"]})
    blocks.sort(key=lambda block: block["start"])
    return blocks


def _replay_block(block: dict, by_generation: dict, owner_session_id: str,
                  epoch_ordinal: int) -> None:
    """Replay one sealed range from the original stored header/page C bytes and
    require the recomputed root, counts and boundary heads to match exactly."""
    body = block["body"]
    members = []
    for generation in range(block["start"], block["end"] + 1):
        member = by_generation.get(generation)
        if member is None:
            _invalid()   # a sealed range without its original member bytes refuses
        header = member["header"]
        if (header.get("owner_session_id") != owner_session_id
                or header.get("epoch_ordinal") != epoch_ordinal
                or header.get("epoch") != body["epoch"]
                or header.get("generation") != generation):
            _invalid()
        members.append(member)
    if len(members) != body["operation_count"]:
        _invalid()
    if sum(len(member["pages"]) for member in members) != body["page_count"]:
        _invalid()
    if members[0]["header"].get("previous_head_sha256") != body["before_head_sha256"]:
        _invalid()
    if members[-1]["header"].get("operation_sha256") != body["end_head_sha256"]:
        _invalid()
    parts = [C({"owner_session_id": owner_session_id, "epoch": body["epoch"],
                "epoch_ordinal": epoch_ordinal, "start_generation": block["start"],
                "end_generation": block["end"],
                "before_head_sha256": body["before_head_sha256"],
                "end_head_sha256": body["end_head_sha256"],
                "operation_count": body["operation_count"],
                "page_count": body["page_count"]})]
    for member in members:
        parts.append(member["header_bytes"].encode("utf-8"))
        parts.extend(raw.encode("utf-8") for raw in member["page_bytes"])
    if B("range-checkpoint", parts) != body["range_root_sha256"]:
        _invalid()


def verify_epoch_range_checkpoints(conn, *, owner_session_id: str, epoch_ordinal: int,
                                   last_generation: int, epoch_exit: bool) -> list:
    """Read-back verification of one owner epoch's sealed ranges (r4 C01 reader).

    Replays every member header/page C byte and recomputes the range root; then
    enforces strict key/body coordinates, contiguous non-overlapping epoch-local
    blocks (``s0+1024*k``), exact counts, only the final block may be a partial
    tail, an unsealed tail only while the epoch is open, and the mandatory exit
    seal of a closed epoch's non-empty tail. A checkpoint is only a locator: the
    original header/page bytes are always replayed, never trusted.
    """
    _positive_int64(epoch_ordinal)
    _positive_int64(last_generation)
    members = read_journal_members(conn, owner_session_id, epoch_ordinal=epoch_ordinal)
    by_generation = {member["header"]["generation"]: member for member in members}
    blocks = _read_range_blocks(conn, owner_session_id, epoch_ordinal)
    if blocks and not members:
        _invalid()   # orphaned range: sealed bytes without the original journal
    if not members:
        return []
    s0 = members[0]["header"]["generation"]
    if members[-1]["header"]["generation"] != last_generation:
        _invalid()   # the store does not reach the claimed epoch end
    previous_end = s0 - 1
    for index, block in enumerate(blocks):
        _replay_block(block, by_generation, owner_session_id, epoch_ordinal)
        if block["start"] != previous_end + 1:
            _invalid()   # gap or overlap between epoch-local blocks
        if (block["start"] - s0) % _CHUNK != 0:
            _invalid()   # blocks are s0+1024*k aligned
        if block["end"] > last_generation:
            _invalid()   # sealed beyond the stored chain
        if index < len(blocks) - 1 and block["body"]["operation_count"] != _CHUNK:
            _invalid()   # only the final block of an epoch may be a partial tail
        previous_end = block["end"]
    if epoch_exit:
        if not blocks or blocks[-1]["end"] != last_generation:
            _invalid()   # a closed epoch's non-empty tail must be sealed
    else:
        for block in blocks:
            if block["body"]["operation_count"] != _CHUNK:
                _invalid()   # an open epoch's tail is never sealed early
    return blocks


def seal_range_checkpoint(conn, key: str, body: dict) -> str:
    """INSERT-only, exact range seal (never an overwrite).

    A missing range is inserted; an existing range must be byte-identical
    (verified and returned, no second row, no rewrite); the same key with
    different bytes refuses.
    """
    validate_epoch_range_checkpoint(body)
    parsed = parse_receipt3_epoch_key(key)
    if (parsed["session_id"] != body["owner_session_id"]
            or parsed["epoch_ordinal"] != body["epoch_ordinal"]
            or parsed["start_generation"] != body["start_generation"]):
        _invalid()
    value = C(body).decode("utf-8")
    row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)", (key, value))
        return value
    if row[0] != value:
        _invalid()   # the same key with different bytes is refused
    return value


def seal_epoch_blocks(conn, *, owner_session_id: str, epoch: str, epoch_ordinal: int,
                      up_to_generation: int, epoch_exit: bool) -> list:
    """Seal every missing epoch-local block from the original stored bytes.

    A full block seals as soon as its last operation is stored; the single
    non-empty partial tail seals only with ``epoch_exit`` (the continuation
    transaction that leaves the epoch). Every existing checkpoint is read back
    and replayed against the member bytes first; an empty range is never
    created, and a store that cannot supply the member bytes refuses.
    """
    members = read_journal_members(conn, owner_session_id, epoch=epoch,
                                   up_to_generation=up_to_generation)
    by_generation = {member["header"]["generation"]: member for member in members}
    blocks = _read_range_blocks(conn, owner_session_id, epoch_ordinal)
    if blocks and not members:
        _invalid()
    if not members:
        return []
    s0 = members[0]["header"]["generation"]
    last_member = members[-1]["header"]["generation"]
    previous_end = s0 - 1
    for index, block in enumerate(blocks):
        _replay_block(block, by_generation, owner_session_id, epoch_ordinal)
        if block["start"] != previous_end + 1 or (block["start"] - s0) % _CHUNK != 0:
            _invalid()
        if block["end"] > last_member:
            _invalid()
        if index < len(blocks) - 1 and block["body"]["operation_count"] != _CHUNK:
            _invalid()
        previous_end = block["end"]
    if blocks and blocks[-1]["body"]["operation_count"] != _CHUNK \
            and blocks[-1]["end"] != last_member:
        _invalid()   # a sealed partial tail must end at the epoch's last bytes
    sealed = []
    while previous_end < last_member:
        window = [member for member in members
                  if previous_end + 1 <= member["header"]["generation"] <= previous_end + _CHUNK]
        members_arg = [{"header": member["header"], "pages": member["pages"]}
                       for member in window]
        if len(window) == _CHUNK:
            key, body = build_epoch_range_checkpoint(
                owner_session_id=owner_session_id, epoch=epoch, epoch_ordinal=epoch_ordinal,
                s0=s0, members=members_arg, epoch_exit=False)
        elif epoch_exit and window and window[-1]["header"]["generation"] == last_member:
            key, body = build_epoch_range_checkpoint(
                owner_session_id=owner_session_id, epoch=epoch, epoch_ordinal=epoch_ordinal,
                s0=s0, members=members_arg, epoch_exit=True)
        else:
            break   # the current open tail stays unsealed
        seal_range_checkpoint(conn, key, body)
        sealed.append({"key": key, "body": body})
        previous_end = body["end_generation"]
    return sealed


# ── C02 step 5: read-only named-only back-projection of a partial recovery ──

_TOKEN_RECORD_KEYS = frozenset({
    "owner_session_id", "operation_id", "request_sha256", "operation_generation",
    "operation_sha256", "result_sha256"})


def read_partial_recovery_state(conn, owner_session_id: str, *,
                                approved_transition: dict) -> list[dict]:
    """Read-only rebuild of one owner's named-only recovery back-projection.

    Receipt r4 C02 step 5: after a crash the recovery re-reads what is actually
    committed before an owner is continued. Every ``receipt3/t/<SIDhex>/<opuuid>``
    record must be canonical, verify its own result digest and name a journal
    entry that verifies exactly against it -- including the frozen
    *approved_transition* continuation block; every ``receipt3/o|p`` key of the
    owner must be named by a committed record (no orphaned journal bytes), and
    every ``receipt3/e`` checkpoint must replay from the original stored member
    bytes. Anything else raises the canonical ValueError: a partial state is
    never silently repaired, truncated or rebaselined. Returns the committed
    operations ordered by generation.
    """
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    if type(approved_transition) is not dict or set(approved_transition) != _CONTINUATION_KEYS:
        _invalid()
    committed = []
    members_by_generation = {}
    lo, hi = _receipt3_prefix_span(_RECEIPT3_TOKEN_PREFIX, owner_session_id)
    last = lo
    while True:
        rows = conn.execute(
            "SELECT key, value FROM state_meta WHERE key > ? AND key < ? ORDER BY key LIMIT ?",
            (last, hi, 128)).fetchall()
        if not rows:
            break
        for key, _value in rows:
            last = key
            parsed = parse_receipt3_token_key(key)
            if parsed["session_id"] != owner_session_id:
                _invalid()
            stored, _raw = _read_canonical(conn, key)
            if type(stored) is not dict or set(stored) != _TOKEN_RECORD_KEYS:
                _invalid()
            if (stored["owner_session_id"] != owner_session_id
                    or stored["operation_id"] != parsed["operation_id"]):
                _invalid()
            _sha256_hex(stored["request_sha256"])
            _sha256_hex(stored["operation_sha256"])
            _sha256_hex(stored["result_sha256"])
            generation = _positive_int64(stored["operation_generation"])
            result = {"token": stored["operation_id"],
                      "operation_sha256": stored["operation_sha256"],
                      "generation": generation}
            if stored["result_sha256"] != hashlib.sha256(C(result)).hexdigest():
                _invalid()   # the stored result digest must recompute exactly
            member = read_operation(conn, owner_session_id, generation)
            header = member["header"]
            if (header["kind"] != "epoch_continuation"
                    or header["operation_id"] != stored["operation_id"]
                    or header["request_sha256"] != stored["request_sha256"]
                    or header["operation_sha256"] != stored["operation_sha256"]
                    or header["continuation"] != approved_transition):
                _invalid()
            members_by_generation[generation] = member
            committed.append({"token": stored["operation_id"], "generation": generation,
                              "operation_sha256": stored["operation_sha256"],
                              "request_sha256": stored["request_sha256"]})
    committed.sort(key=lambda item: (item["generation"], item["token"]))
    # No orphaned journal bytes: every o|p key of the owner is named by a
    # committed record (and every committed record's keys are present --
    # read_operation above already refused a missing key).
    named = set()
    for item in committed:
        member = members_by_generation[item["generation"]]
        named.add(encode_receipt3_operation_key(owner_session_id, item["generation"]))
        for ordinal in range(len(member["pages"])):
            named.add(encode_receipt3_page_key(owner_session_id, item["generation"], ordinal))
    for prefix in (_RECEIPT3_OPERATION_PREFIX, _RECEIPT3_PAGE_PREFIX):
        lo, hi = _receipt3_prefix_span(prefix, owner_session_id)
        last = lo
        while True:
            rows = conn.execute(
                "SELECT key FROM state_meta WHERE key > ? AND key < ? ORDER BY key LIMIT ?",
                (last, hi, 128)).fetchall()
            if not rows:
                break
            for (key,) in rows:
                last = key
                if key not in named:
                    _invalid()   # orphaned journal bytes refuse the partial state
    # Every epoch checkpoint of the owner replays from the original member bytes.
    lo, hi = _receipt3_prefix_span(_RECEIPT3_EPOCH_PREFIX, owner_session_id)
    epochs = set()
    last = lo
    while True:
        rows = conn.execute(
            "SELECT key FROM state_meta WHERE key > ? AND key < ? ORDER BY key LIMIT ?",
            (last, hi, 128)).fetchall()
        if not rows:
            break
        for (key,) in rows:
            last = key
            epochs.add(parse_receipt3_epoch_key(key)["epoch_ordinal"])
    for epoch_ordinal in sorted(epochs):
        members = read_journal_members(conn, owner_session_id, epoch_ordinal=epoch_ordinal)
        by_generation = {member["header"]["generation"]: member for member in members}
        for block in _read_range_blocks(conn, owner_session_id, epoch_ordinal):
            _replay_block(block, by_generation, owner_session_id, epoch_ordinal)
    return committed


# ── receipt3/r row-state index (r4 protocol.storage / record_shapes) ─────────

_RECEIPT3_ROW_PREFIX = "receipt3/r/"

_ROW_INDEX_KEYS = frozenset({"row_ref", "latest_state", "event"})
_PAYLOAD_STATE_KEYS = frozenset({"row_ref", "columns", "payload_sha256", "feed_input_sha256"})
_PAYLOAD_STATE_COLUMN_COUNT = 20


def encode_receipt3_row_index_key(birth_epoch: str, row_id: int) -> str:
    """``receipt3/r/<birth_epoch64hex>/<rowid20>`` -- the mutable row-state index.

    The epoch component is the 64 hex characters without the ``sqlite:``
    prefix (the same routing-reversible convention as the SID component); the
    row component is the actual ROWID in the same 20-digit encoding as ``G``.
    """
    _epoch_string(birth_epoch)
    row_id = _positive_int64(row_id)
    return f"{_RECEIPT3_ROW_PREFIX}{birth_epoch.split(':', 1)[1]}/{_int20(row_id)}"


def parse_receipt3_row_index_key(key: str) -> dict:
    """Parse an exact receipt3/r key; non-canonical forms are refused."""
    if type(key) is not str:
        _invalid()
    parts = key.split("/")
    if len(parts) != 4 or parts[0] != "receipt3" or parts[1] != "r":
        _invalid()
    suffix = parts[2]
    if type(suffix) is not str or not _HEX.fullmatch(suffix):
        _invalid()
    return {"birth_epoch": "sqlite:" + suffix, "row_id": _decode_int20(parts[3])}


def _validate_payload_state(state) -> None:
    """Exact shape of one payload_state (record_shapes.payload_state): the
    ordered 20-column digest list plus P and the raw-floor Q hash."""
    if type(state) is not dict or set(state) != _PAYLOAD_STATE_KEYS:
        _invalid()
    _validate_row_ref(state["row_ref"])
    columns = state["columns"]
    if type(columns) is not list or len(columns) != _PAYLOAD_STATE_COLUMN_COUNT:
        _invalid()
    names = []
    for column in columns:
        if (type(column) is not list or len(column) != 2
                or type(column[0]) is not str or not column[0]):
            _invalid()
        _sha256_hex(column[1])
        names.append(column[0])
    if len(set(names)) != _PAYLOAD_STATE_COLUMN_COUNT:
        _invalid()   # the manifest is exactly 20 distinct columns, never a repeat
    _sha256_hex(state["payload_sha256"])
    _sha256_hex(state["feed_input_sha256"])


def validate_row_state_index_entry(entry) -> None:
    """Exact shape of one receipt3/r row-state index body (record_shapes.row_index).

    The event is either the committed member coordinate ``[owner, generation,
    page, item]`` or an exact activation anchor reference; the payload state
    must describe the same row the entry does.
    """
    if type(entry) is not dict or set(entry) != _ROW_INDEX_KEYS:
        _invalid()
    _validate_row_ref(entry["row_ref"])
    _validate_payload_state(entry["latest_state"])
    event = entry["event"]
    if type(event) is list:
        if len(event) != 4:
            _invalid()
        owner, generation, page, index = event
        if type(owner) is not str or not owner:
            _invalid()
        _positive_int64(generation)
        if type(page) is not int or not 0 <= page <= _MAX_INT64:
            _invalid()
        if type(index) is not int or not 0 <= index <= _MAX_INT64:
            _invalid()
    elif type(event) is dict:
        if set(event) != {"activation_proof_sha256", "context_session_id"}:
            _invalid()
        _sha256_hex(event["activation_proof_sha256"])
        if type(event["context_session_id"]) is not str or not event["context_session_id"]:
            _invalid()
    else:
        _invalid()
    if entry["latest_state"]["row_ref"] != entry["row_ref"]:
        _invalid()   # the payload state must describe the indexed row itself


def _owner_member_row_ref_events(members: list) -> dict:
    """Derive the per-row latest committed item event from stored member bytes.

    One entry per distinct row named by a committed ``native_result`` item of
    this owner (the item's row must belong to the operating owner); the event
    is that row's latest coordinate, ranked by (epoch_ordinal, generation,
    page_ordinal, item_ordinal).
    """
    derived = {}
    for member in members:
        header = member["header"]
        for page in member["pages"]:
            for index, item in enumerate(page["items"]):
                if item["kind"] != "native_result" or item["row_ref"] is None:
                    continue
                ref = item["row_ref"]
                if ref["owner_session_id"] != header["owner_session_id"]:
                    _invalid()   # a result row belongs to the operating owner
                key = encode_receipt3_row_index_key(ref["birth_epoch"], ref["row_id"])
                rank = (header["epoch_ordinal"], header["generation"],
                        page["page_ordinal"], index)
                event = [header["owner_session_id"], header["generation"],
                         page["page_ordinal"], index]
                if key not in derived or rank > derived[key]["rank"]:
                    derived[key] = {"rank": rank, "row_ref": ref, "event": event}
    return derived


def read_row_state_index(conn, owner_session_id: str) -> list[dict]:
    """Reader-side derivation of one owner's ``receipt3/r`` row-state inventory.

    The index is a rebuildable accelerator that points to the exact latest
    committed item for a row and is NEVER acceptance evidence by itself. Every
    stored entry must parse canonically, carry the full row_ref/latest_state/
    event shape, describe exactly the row its key names, and equal the
    inventory derived here from the stored journal bytes. A missing entry, an
    orphaned entry, a non-canonical key or a non-canonical/damaged body
    refuses. The ``receipt3/r`` namespace carries no owner component, so
    entries are attributed by the owner inside their own body: another owner's
    entry is still held to the full canonical shape but is not part of this
    owner's inventory (its own request/inverse judges it). The P/V/Q values
    themselves are recomputed by the standalone reader's copy-observation
    machinery, which is a separate surface from this inventory derivation.
    """
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    derived = _owner_member_row_ref_events(read_journal_members(conn, owner_session_id))
    lo = _RECEIPT3_ROW_PREFIX
    hi = lo[:-1] + "0"
    stored = {}
    last = lo
    while True:
        rows = conn.execute(
            "SELECT key, value FROM state_meta WHERE key > ? AND key < ? ORDER BY key LIMIT ?",
            (last, hi, 128)).fetchall()
        if not rows:
            break
        for key, _value in rows:
            last = key
            parse_receipt3_row_index_key(key)
            entry, _raw = _read_canonical(conn, key)
            validate_row_state_index_entry(entry)
            if encode_receipt3_row_index_key(entry["row_ref"]["birth_epoch"],
                                             entry["row_ref"]["row_id"]) != key:
                _invalid()   # strict key/body coordinate equality
            if entry["row_ref"]["owner_session_id"] != owner_session_id:
                continue     # validated, but another owner's slice: not this inventory
            stored[key] = entry
    if set(stored) != set(derived):
        _invalid()   # exact inventory: no orphan entry, no missing entry
    result = []
    for key in sorted(derived):
        entry = stored[key]
        expected = derived[key]
        if entry["row_ref"] != expected["row_ref"] or entry["event"] != expected["event"]:
            _invalid()   # the index must point to the exact latest committed item
        result.append({"key": key, "row_ref": entry["row_ref"],
                       "latest_state": entry["latest_state"], "event": entry["event"]})
    return result


def upsert_receipt3_row_index_tx(conn, entry: dict) -> str:
    """Writer-side identity-preserving upsert of one receipt3/r index entry.

    Caller holds the transaction (same-tx rule, ``writer_timeline[4]``). A new
    row inserts under the immutable key; an existing row is refreshed ONLY
    with an in-place UPDATE -- the actual ROWID of the stored index row is
    never swapped (no INSERT OR REPLACE) -- and the stored event must not
    regress (the index points to the exact latest committed item). Returns the
    immutable key. An anchor-event refresh is refused in this revision.
    """
    validate_row_state_index_entry(entry)
    ref = entry["row_ref"]
    key = encode_receipt3_row_index_key(ref["birth_epoch"], ref["row_id"])
    body = C(entry).decode("utf-8")
    existing = conn.execute(
        "SELECT rowid, value FROM state_meta WHERE key = ?", (key,)).fetchone()
    if existing is None:
        conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)", (key, body))
        return key
    previous, _raw = _read_canonical(conn, key)
    validate_row_state_index_entry(previous)
    if previous["row_ref"] != ref:
        _invalid()
    if type(entry["event"]) is not list or type(previous["event"]) is not list:
        _invalid()   # anchor-event refreshes are a separate approval surface
    old_rank = (previous["event"][1], previous["event"][2], previous["event"][3])
    new_rank = (entry["event"][1], entry["event"][2], entry["event"][3])
    if new_rank <= old_rank:
        _invalid()   # the index only advances, never regresses or repeats
    conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (body, key))
    identity = conn.execute("SELECT rowid FROM state_meta WHERE key = ?", (key,)).fetchone()[0]
    if identity != existing[0]:
        _invalid()   # an in-place refresh must keep the actual ROWID
    return key


# ── C02 step 6 (r4): durable-table joint + identity-aware inverse deltas ────
#
# protocol.recovery.joint_formula / identity_modes: the step 6 acceptance is a
# byte-exact identity-aware joint over EVERY durable table. J(row) =
# C([I(row), [[c, SQL(row[c])] for c in cols_T]]) with the stored column
# manifest in declaration order and SQL() the exact typed value tag; the
# COMPLETE J byte strings are sorted (duplicates and multiplicity kept) and
# length-framed as B("joint-table", [C({table, columns, identity_mode,
# table_structure_sha256}), *sorted_J]); the row count is bound separately.
# The identity I(row) is the actual engine identity: the unshadowed
# rowid/_rowid_/oid for ordinary rowid tables (for a declared INTEGER PRIMARY
# KEY alias both the actual ROWID and the id column are included and must
# agree), ALL PK components in declared PK ordinal for WITHOUT ROWID tables,
# and the actual virtual rowid for pinned FTS5 virtual tables. A table whose
# three rowid aliases are all shadowed is refused instead of assuming its PK
# is the rowid. ``sqlite_schema`` is enumerated as its own structural
# inventory table (table/index/trigger/view definitions plus their actual row
# identity) and every FTS shadow table rides the same rules.

_JOINT_TABLE_LABEL = "joint-table"
_JOINT_STRUCTURE_LABEL = "joint-table-structure"
_JOINT_FULL_LABEL = "full-table-joint"
_REVERSAL_LABEL = "quiet-recovery-reversal"
_RECOVERY_PROOF_LABEL = "quiet-recovery-proof"
_RECOVERY_ROSTER_LABEL = "quiet-recovery-roster"

_ROWID_ALIASES = ("rowid", "_rowid_", "oid")
_SQLITE_SCHEMA_DDL = ("CREATE TABLE sqlite_schema(type text,name text,tbl_name text,"
                      "rootpage integer,sql text)")
_ALIAS_IDENTITY_MODE = "messages_INTEGER_PRIMARY_KEY_ALIAS"


def _open_read_only(path) -> sqlite3.Connection:
    """One independent mode=ro / query_only connection (never creates a file)."""
    conn = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=5.0)
    try:
        conn.execute("PRAGMA query_only = ON")
    except BaseException:
        conn.close()
        raise
    return conn


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _joint_spec(conn, table: str) -> dict:
    """One durable table's binding: stored column manifest (declaration order),
    actual identity mode, rowid alias/PK coordinates and structure digest."""
    row = None
    xinfo = []
    if table == "sqlite_schema":
        manifest = [[0, "type", "TEXT", 0, None, 0], [1, "name", "TEXT", 0, None, 0],
                    [2, "tbl_name", "TEXT", 0, None, 0], [3, "rootpage", "INT", 0, None, 0],
                    [4, "sql", "TEXT", 0, None, 0]]
    else:
        row = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = ?",
            (table,)).fetchone()
        if row is None or type(row[0]) is not str or not row[0]:
            _invalid()
        lower = row[0].lower()
        if re.match(r"\s*create\s+virtual\s+table", lower) and "using fts5" not in lower:
            _invalid()   # only the pinned FTS5 module is bound; unknown modules refuse
        xinfo = [tuple(entry) for entry in conn.execute(
            f"PRAGMA table_xinfo({_quote_identifier(table)})")]
        for (_cid, _name, _decl, _notnull, _dflt, _pk, hidden) in xinfo:
            if hidden not in (0, 1):
                _invalid()   # generated columns are not in this revision
        manifest = [list(entry[:6]) for entry in xinfo if entry[6] == 0]
        if not manifest:
            _invalid()
    columns = [entry[1] for entry in manifest]
    if table == "sqlite_schema":
        sql_text = _SQLITE_SCHEMA_DDL
    else:
        if row is None:
            _invalid()
        sql_text = row[0]
    declared_names = [entry[1] for entry in manifest]
    without_rowid = bool(re.search(r"without\s+rowid", sql_text.lower()))
    fts_virtual = "using fts5" in sql_text.lower()
    pk_columns = [entry[1] for entry in sorted(
        (entry for entry in manifest if entry[5] > 0), key=lambda entry: entry[5])]
    rowid_alias = None
    identity_mode = None
    declared_all = (declared_names if table == "sqlite_schema"
                    else [entry[1] for entry in xinfo])
    for candidate in _ROWID_ALIASES:
        if candidate not in declared_all:
            rowid_alias = candidate
            break
    if fts_virtual:
        # The pinned FTS5 vtab exposes the actual virtual rowid through the
        # ordinary engine names; the declared visible columns are the manifest.
        identity_mode = "FTS_virtual"
        if rowid_alias is None:
            _invalid()   # all three engine rowid aliases are shadowed
    elif without_rowid:
        if not pk_columns:
            _invalid()   # a WITHOUT ROWID table without a PK is never bound
        identity_mode = "WITHOUT_ROWID"
    else:
        if rowid_alias is None:
            _invalid()   # all three rowid aliases are shadowed; never assume the PK
        if len(pk_columns) == 1:
            decl = next(entry[2] for entry in manifest if entry[1] == pk_columns[0])
            if type(decl) is str and decl.strip().upper() == "INTEGER":
                identity_mode = _ALIAS_IDENTITY_MODE
        if identity_mode is None:
            identity_mode = "ordinary_rowid_nonalias"
    if identity_mode is None:
        _invalid()
    structure_sha256 = H(_JOINT_STRUCTURE_LABEL, {
        "table": table, "sql": sql_text, "columns": manifest,
        "without_rowid": without_rowid, "fts_virtual": fts_virtual})
    return {"table": table, "columns": columns, "identity_mode": identity_mode,
            "pk_columns": pk_columns, "rowid_alias": rowid_alias,
            "without_rowid": without_rowid, "fts_virtual": fts_virtual,
            "declared_names": declared_names, "structure_sha256": structure_sha256}


def _iter_table_rows(conn, spec: dict):
    """Yield ``(identity_object, stored_values)`` for every row of one table."""
    table = spec["table"]
    columns = spec["columns"]
    if spec["identity_mode"] == "WITHOUT_ROWID":
        select = ", ".join(_quote_identifier(column) for column in columns)
        for raw in conn.execute(f"SELECT {select} FROM {_quote_identifier(table)}"):
            values = list(raw)
            by_name = dict(zip(columns, values))
            identity = [[column, sql_value_tag(by_name[column])]
                        for column in spec["pk_columns"]]
            yield identity, values
        return
    alias = spec["rowid_alias"]
    select = ", ".join([alias] + [_quote_identifier(column) for column in columns])
    for raw in conn.execute(f"SELECT {select} FROM {_quote_identifier(table)}"):
        rowid = raw[0]
        if type(rowid) is not int:
            _invalid()
        values = list(raw[1:])
        identity = [[alias, sql_value_tag(rowid)]]
        if spec["identity_mode"] == _ALIAS_IDENTITY_MODE:
            pk_column = spec["pk_columns"][0]
            pk_value = values[columns.index(pk_column)]
            if pk_value != rowid:
                _invalid()   # the declared INTEGER PRIMARY KEY must alias the ROWID
            identity.append([pk_column, sql_value_tag(pk_value)])
        yield identity, values


def _row_joint_bytes(identity, columns, values) -> bytes:
    return C([identity, [[column, sql_value_tag(value)]
                         for column, value in zip(columns, values)]])


def _table_inventory(conn, table: str) -> dict:
    """The full-row inventory of one table: spec + (identity bytes, J bytes,
    raw values, actual rowid) per stored row."""
    spec = _joint_spec(conn, table)
    rows = []
    for identity, values in _iter_table_rows(conn, spec):
        rowid = None
        if spec["identity_mode"] != "WITHOUT_ROWID":
            rowid = sql_value_decode(identity[0][1])
        rows.append({"identity": C(identity), "rowid": rowid,
                     "joint": _row_joint_bytes(identity, spec["columns"], values),
                     "values": dict(zip(spec["columns"], values))})
    return {"spec": spec, "rows": rows}


def _inventory_joint(inventory: dict) -> str:
    """The r4 table joint digest of one inventory: the complete sorted J byte
    strings length-framed under the structure binding (count bound separately)."""
    spec = inventory["spec"]
    binding = C({"table": spec["table"], "columns": spec["columns"],
                 "identity_mode": spec["identity_mode"],
                 "table_structure_sha256": spec["structure_sha256"]})
    return B(_JOINT_TABLE_LABEL, [binding,
                                  *sorted(row["joint"] for row in inventory["rows"])])


def _full_joint_digest(joints: dict) -> str:
    """The combined digest over per-table (name, row_count, joint) triples."""
    return B(_JOINT_FULL_LABEL, [
        C([name, joints[name]["row_count"], joints[name]["joint_sha256"]])
        for name in sorted(joints)])


def _joint_table_names(conn) -> list:
    return sorted({row[0] for row in conn.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'table'")} | {"sqlite_schema"})


def table_joint(conn, table: str) -> dict:
    """One durable table's r4 joint: sorted complete row J bytes, length-framed
    with the structure binding; the row count is bound separately."""
    inventory = _table_inventory(conn, table)
    return {"table": table, "columns": inventory["spec"]["columns"],
            "identity_mode": inventory["spec"]["identity_mode"],
            "table_structure_sha256": inventory["spec"]["structure_sha256"],
            "row_count": len(inventory["rows"]),
            "joint_sha256": _inventory_joint(inventory)}


def full_table_joint(conn) -> dict:
    """Every durable table's joint plus the combined digest.

    ``sqlite_schema`` is enumerated as its own structural inventory row table;
    FTS virtual tables and every shadow/config/sequence/meta table ride the
    same rules -- no hardcoded three-table shortcut.
    """
    names = _joint_table_names(conn)
    tables = {name: table_joint(conn, name) for name in names}
    return {"tables": tables, "table_count": len(names),
            "joint_sha256": _full_joint_digest(tables)}


def _structured_row_summary(spec: dict, row: dict) -> dict:
    """A compact, content-free descriptor of one changed row."""
    summary = {"table": spec["table"]}
    if row["rowid"] is not None:
        summary["rowid"] = row["rowid"]
    if spec["table"] == "state_meta":
        summary["key"] = row["values"].get("key")
    elif spec["table"] == "sessions":
        summary["session_id"] = row["values"].get("id")
    return summary


def _classify_added_delta_key(key, *, owner_session_id: str, named: dict):
    """One added state_meta key must be exactly one named continuation delta:
    a token/journal/checkpoint key of the committed set or a derived row-index
    entry; anything else is a stray mutation and refuses."""
    if type(key) is not str or not key:
        _invalid()
    if key in named["journal"]:
        return "journal"
    if key.startswith(_RECEIPT3_EPOCH_PREFIX):
        parsed = parse_receipt3_epoch_key(key)
        if parsed["session_id"] != owner_session_id:
            _invalid()
        # Every checkpoint of this owner was already replayed from the stored
        # member bytes by read_partial_recovery_state before this point.
        return "checkpoint"
    if key.startswith(_RECEIPT3_ROW_PREFIX):
        parse_receipt3_row_index_key(key)
        if key not in named["row_index"]:
            _invalid()
        return "row_index"
    _invalid()


def _config_head(raw):
    """The stored protected head inside one sessions.model_config value; a
    damaged raw that carries receipt config is refused, never read as 'none'."""
    if raw is None:
        return None
    if type(raw) is not str or not raw.strip():
        _invalid()
    try:
        config = json.loads(raw, object_pairs_hook=_unique_config_object)
    except _DuplicateConfigKey:
        _invalid()
    except (ValueError, TypeError):
        _invalid()
    if not isinstance(config, dict):
        _invalid()
    head = config.get("_message_copy_receipts_v2")
    return None if head is None else head


def verify_inverse_recovery_joint(source_path, replacement_path, *,
                                  owner_session_id: str, approved_transition: dict) -> dict:
    """C02 step 6 (r4): the independent read-only full-table joint acceptance.

    While quiet is still in force, this reader re-verifies the committed
    continuation set from the stored bytes, derives the exact request-named
    delta (named journal/pages/checkpoints, token index, derived row index and
    the protected head bytes) and refuses ANY other durable difference --
    removals and identity swaps included, since a refresh must be an in-place
    UPDATE that keeps the actual ROWID. It then applies the exact inverse
    (dropping the added keys, restoring the old index values and the old head
    bytes) and requires every durable table's joint to equal the original old
    latest joint -- sqlite_schema, FTS virtual tables, every shadow, config and
    sequence included. Any stray mutation invalidates the admission.
    """
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    if type(approved_transition) is not dict or set(approved_transition) != _CONTINUATION_KEYS:
        _invalid()
    source_conn = _open_read_only(source_path)
    replacement_conn = _open_read_only(replacement_path)
    try:
        source_names = _joint_table_names(source_conn)
        replacement_names = _joint_table_names(replacement_conn)
        if source_names != replacement_names:
            _invalid()   # a created/dropped table is a structural mutation
        source = {name: _table_inventory(source_conn, name) for name in source_names}
        replacement = {name: _table_inventory(replacement_conn, name) for name in source_names}
        for name in source_names:
            source_spec, replacement_spec = source[name]["spec"], replacement[name]["spec"]
            if (source_spec["structure_sha256"] != replacement_spec["structure_sha256"]
                    or source_spec["columns"] != replacement_spec["columns"]
                    or source_spec["identity_mode"] != replacement_spec["identity_mode"]):
                _invalid()   # the structure binding must be bit-identical
        committed = read_partial_recovery_state(
            replacement_conn, owner_session_id, approved_transition=approved_transition)
        named = {"journal": set(), "row_index": set()}
        for item in committed:
            named["journal"].add(encode_receipt3_token_key(owner_session_id, item["token"]))
            named["journal"].add(encode_receipt3_operation_key(owner_session_id,
                                                               item["generation"]))
            member = read_operation(replacement_conn, owner_session_id, item["generation"])
            for ordinal in range(len(member["pages"])):
                named["journal"].add(encode_receipt3_page_key(
                    owner_session_id, item["generation"], ordinal))
        for entry in read_row_state_index(replacement_conn, owner_session_id):
            named["row_index"].add(entry["key"])
        added, updated, removed = [], [], []
        for name in source_names:
            source_rows = {row["identity"]: row for row in source[name]["rows"]}
            replacement_rows = {row["identity"]: row for row in replacement[name]["rows"]}
            for identity, row in replacement_rows.items():
                if identity not in source_rows:
                    added.append((name, identity, row))
                elif source_rows[identity]["joint"] != row["joint"]:
                    updated.append((name, identity, source_rows[identity], row))
            for identity, row in source_rows.items():
                if identity not in replacement_rows:
                    removed.append((name, identity, row))
        if removed:
            _invalid()   # nothing is deleted by a continuation; an INSERT OR
            # REPLACE identity swap shows up exactly here as a remove+add pair
        added_keys = []
        for name, _identity, row in added:
            if name != "state_meta":
                _invalid()   # only the reserved receipt3/ delta may appear
            key = row["values"].get("key")
            _classify_added_delta_key(key, owner_session_id=owner_session_id, named=named)
            added_keys.append(key)
        updated_rows = []
        for name, _identity, old_row, new_row in updated:
            if name == "state_meta":
                old_key = old_row["values"].get("key")
                new_key = new_row["values"].get("key")
                if old_key != new_key or type(old_key) is not str \
                        or not old_key.startswith(_RECEIPT3_ROW_PREFIX):
                    _invalid()
                if old_key not in named["row_index"]:
                    _invalid()   # only a derived row-index entry may refresh in place
                old_entry, _raw = _read_canonical(replacement_conn, old_key)
                validate_row_state_index_entry(old_entry)
                if encode_receipt3_row_index_key(
                        old_entry["row_ref"]["birth_epoch"],
                        old_entry["row_ref"]["row_id"]) != old_key:
                    _invalid()
                new_entry = json.loads(new_row["values"]["value"],
                                       object_pairs_hook=_unique_config_object) \
                    if type(new_row["values"]["value"]) is str else None
                validate_row_state_index_entry(new_entry)
                if (new_entry["row_ref"] != old_entry["row_ref"]
                        or C(new_entry).decode("utf-8") != new_row["values"]["value"]):
                    _invalid()
                updated_rows.append({"table": name, "key": old_key, "kind": "row_index"})
            elif name == "sessions":
                old_values, new_values = old_row["values"], new_row["values"]
                if old_values.get("id") != owner_session_id:
                    _invalid()   # only the approved owner's row may advance
                changed = [column for column in old_values
                           if old_values[column] != new_values[column]]
                if changed != ["model_config"]:
                    _invalid()   # exactly the protected head column advances
                if not committed:
                    _invalid()   # a head advance without a committed operation
                last = committed[-1]
                old_head = _config_head(old_values["model_config"])
                if old_head is not None:
                    validate_receipt3_head(old_head)
                    if (old_head["owner_session_id"] != owner_session_id
                            or old_head["active_epoch"] != approved_transition["old_epoch"]
                            or old_head["generation"] != approved_transition["old_generation"]
                            or old_head["head_sha256"] != approved_transition["old_head_sha256"]
                            or hashlib.sha256(C(old_head)).hexdigest()
                            != approved_transition["old_head_token_sha256"]):
                        _invalid()   # the inverse restores exactly these frozen bytes
                elif approved_transition["old_generation"] != 0:
                    _invalid()
                new_head = _config_head(new_values["model_config"])
                if new_head is None:
                    _invalid()
                validate_receipt3_head(new_head)
                if (new_head["owner_session_id"] != owner_session_id
                        or new_head["active_epoch"] != approved_transition["new_epoch"]
                        or new_head["generation"] != last["generation"]
                        or new_head["head_sha256"] != last["operation_sha256"]):
                    _invalid()
                updated_rows.append({"table": name, "rowid": new_row["rowid"],
                                     "session_id": owner_session_id, "kind": "protected_head"})
            else:
                _invalid()   # only the reserved state_meta/head surfaces may change
        # The exact inverse: drop the added keys, substitute the old values for
        # every in-place update (the actual ROWID is proven above), then require
        # every durable table's joint to equal the original old latest joint.
        source_digest = _full_joint_digest(
            {name: {"row_count": len(source[name]["rows"]),
                    "joint_sha256": _inventory_joint(source[name])}
             for name in source_names})
        reconstructed = {}
        for name in source_names:
            added_identities = {identity for table, identity, _row in added if table == name}
            updated_source = {identity: old_row
                              for table, identity, old_row, _new in updated if table == name}
            inverted = []
            for identity, row in ((r["identity"], r) for r in replacement[name]["rows"]):
                if identity in added_identities:
                    continue
                if identity in updated_source:
                    row = updated_source[identity]
                inverted.append(row["joint"])
            spec = replacement[name]["spec"]
            joint = B(_JOINT_TABLE_LABEL, [
                C({"table": spec["table"], "columns": spec["columns"],
                   "identity_mode": spec["identity_mode"],
                   "table_structure_sha256": spec["structure_sha256"]}),
                *sorted(inverted)])
            if joint != _inventory_joint(source[name]):
                _invalid()   # not exactly invertible back to the old latest joint
            reconstructed[name] = {"row_count": len(inverted), "joint_sha256": joint}
        report = {"owner_session_id": owner_session_id,
                  "table_count": len(source_names),
                  "source_joint_sha256": source_digest,
                  "reconstructed_joint_sha256": _full_joint_digest(reconstructed),
                  "added_keys": sorted(added_keys),
                  "updated_rows": updated_rows,
                  "removed_rows": [],
                  "committed": [{"token": item["token"], "generation": item["generation"],
                                 "operation_sha256": item["operation_sha256"]}
                                for item in committed]}
        if report["reconstructed_joint_sha256"] != report["source_joint_sha256"]:
            _invalid()
        report["reversal_sha256"] = H(_REVERSAL_LABEL, {
            "owner_session_id": owner_session_id,
            "source_joint_sha256": report["source_joint_sha256"],
            "added_keys": report["added_keys"],
            "updated_rows": updated_rows})
        return report
    finally:
        source_conn.close()
        replacement_conn.close()


# ── C02 steps 2/3 (r4): copy comparison + frozen proof/roster ────────────────

def verify_independent_copy_joint(source_path, replacement_path) -> dict:
    """r4 C02 steps 2/3: independent readonly comparison of the frozen old
    latest against the selected replacement -- complete joint/schema/sequence/
    header. Run BEFORE any constructor and again after the quiet open: any
    logical persistent delta refuses. Sequence rows and protected heads are
    bound by the same durable-table joint (sqlite_sequence and sessions are
    ordinary durable tables); byte-completeness of the copy itself is the
    copy method's obligation, not asserted here.
    """
    source_conn = _open_read_only(source_path)
    replacement_conn = _open_read_only(replacement_path)
    try:
        source = full_table_joint(source_conn)
        replacement = full_table_joint(replacement_conn)
    finally:
        source_conn.close()
        replacement_conn.close()
    if list(source["tables"]) != list(replacement["tables"]):
        _invalid()
    for name in source["tables"]:
        if (source["tables"][name]["table_structure_sha256"]
                != replacement["tables"][name]["table_structure_sha256"]
                or source["tables"][name]["columns"] != replacement["tables"][name]["columns"]
                or source["tables"][name]["identity_mode"]
                != replacement["tables"][name]["identity_mode"]):
            _invalid()
    if source["joint_sha256"] != replacement["joint_sha256"]:
        _invalid()   # a logical persistent delta between the same old latest
    return {"joint_sha256": source["joint_sha256"], "table_count": source["table_count"],
            "schema_sha256": source["tables"]["sqlite_schema"]["joint_sha256"]}


def _validate_recovery_init_request(init_request, owner_session_id: str) -> dict:
    """Local strict validation of the frozen quiet-recovery init request shape."""
    required = {"kind", "token", "owner_session_id", "proof_sha256", "old_epoch",
                "new_epoch", "old_generation", "old_head_sha256",
                "old_head_token_sha256", "boundary_roster_sha256",
                "alias_inventory_sha256"}
    if type(init_request) is not dict or set(init_request) != required:
        _invalid()
    if init_request["kind"] != "epoch_continuation":
        _invalid()
    if init_request["owner_session_id"] != owner_session_id:
        _invalid()
    _canonical_uuid4(init_request["token"])
    _sha256_hex(init_request["proof_sha256"])
    _epoch_string(init_request["old_epoch"])
    _epoch_string(init_request["new_epoch"])
    if init_request["old_epoch"] == init_request["new_epoch"]:
        _invalid()
    old_generation = init_request["old_generation"]
    if type(old_generation) is not int or not 0 <= old_generation <= _MAX_INT64:
        _invalid()
    if init_request["old_head_sha256"] is not None:
        _sha256_hex(init_request["old_head_sha256"])
    elif old_generation != 0:
        _invalid()
    _sha256_hex(init_request["old_head_token_sha256"])
    _sha256_hex(init_request["boundary_roster_sha256"])
    _sha256_hex(init_request["alias_inventory_sha256"])
    return {name: init_request[name] for name in sorted(required)}


def _file_sha256(path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_identity(path) -> dict:
    stat = Path(path).stat()
    return {"st_dev": stat.st_dev, "st_ino": stat.st_ino, "st_size": stat.st_size}


def freeze_recovery_proof(path, *, owner_session_id: str, init_request: dict) -> dict:
    """r4 C02 step 3: freeze the selected replacement inode's proof together
    with the finite roster/init request.

    The proof binds the full-table joint, the actual file identity and the
    frozen request; the exact certificate/pin approval over these bytes stays
    a separate non-author decision (never taken here). While no continuation
    exists yet the proof must re-verify unchanged -- including after the quiet
    admission open, which may not leave any logical persistent delta.
    """
    if type(owner_session_id) is not str or not owner_session_id:
        _invalid()
    request = _validate_recovery_init_request(init_request, owner_session_id)
    conn = _open_read_only(path)
    try:
        joint = full_table_joint(conn)
    finally:
        conn.close()
    proof = {"owner_session_id": owner_session_id,
             "init_request_sha256": hashlib.sha256(C(request)).hexdigest(),
             "roster_sha256": H(_RECOVERY_ROSTER_LABEL,
                                {"owners": [owner_session_id]}),
             "joint_sha256": joint["joint_sha256"],
             "table_count": joint["table_count"],
             "file_sha256": _file_sha256(path),
             "file_identity": _file_identity(path)}
    proof["recovery_proof_sha256"] = H(_RECOVERY_PROOF_LABEL, proof)
    return proof


def verify_frozen_proof(path, proof) -> dict:
    """Re-verify one frozen proof against the same inode (steps 2/3 window:
    before any constructor and again after the quiet open). Any joint, file or
    request change refuses; the proof is not a substitute for the step 6
    inverse acceptance once continuations exist.
    """
    if type(proof) is not dict or "recovery_proof_sha256" not in proof:
        _invalid()
    frozen = {key: value for key, value in proof.items() if key != "recovery_proof_sha256"}
    if proof["recovery_proof_sha256"] != H(_RECOVERY_PROOF_LABEL, frozen):
        _invalid()   # the proof bytes themselves do not verify
    if set(frozen) != {"owner_session_id", "init_request_sha256", "roster_sha256",
                       "joint_sha256", "table_count", "file_sha256", "file_identity"}:
        _invalid()
    _sha256_hex(frozen["init_request_sha256"])
    _sha256_hex(frozen["joint_sha256"])
    _sha256_hex(frozen["file_sha256"])
    conn = _open_read_only(path)
    try:
        joint = full_table_joint(conn)
    finally:
        conn.close()
    if (joint["joint_sha256"] != frozen["joint_sha256"]
            or joint["table_count"] != frozen["table_count"]
            or _file_sha256(path) != frozen["file_sha256"]
            or _file_identity(path) != frozen["file_identity"]):
        _invalid()   # the frozen inode no longer matches its proof
    return {"verified": True, "owner_session_id": frozen["owner_session_id"],
            "joint_sha256": joint["joint_sha256"], "table_count": joint["table_count"]}


# ---------------------------------------------------------------------------
# M2D design binding: effective r4 metadata, gate-expected precedence and the
# accurate disclosure (receipt r4 amendments section 5 rows "版本/优先级/审查
# 状态", "受影响 expected 增补" and "准确披露"; gates inventory section 2
# A1-A3). Attribution, annotated per evidence:
#
#   * acceptance-apparatus side: ``resolve_gate_expected`` /
#     ``verify_r4_gate_binding`` implement the frozen expected precedence
#     ("r4_expected if present is the effective expectation; all historical
#     r3_expected retained") for the acceptance runner; the operator-side
#     apparatus script operator/schema31_receipt_impl/r4_gate_binding.py
#     re-derives every pinned value below from the frozen JSON bytes and
#     compares (immutability, derivation and writer_timeline deltas);
#   * product-code side: the pinned binding data lives here because the
#     official test sandbox exposes only the worktree -- the acceptance
#     tests must exercise the resolver against the real frozen bytes
#     without reading the repository reports/ tree (same home as the M2C
#     step-6 acceptance machinery).
#
# These values are auditable design/acceptance metadata, explicitly NOT
# protocol record bytes ("auditable derivation metadata, not protocol
# record"): none of them enters any receipt3/ key, operation header, page,
# token key, protected head or the wire schema. r4 is a design-document
# revision, not a wire/schema/domain/epoch revision (protocol.precedence);
# the protocol revision itself is unchanged (_PROTOCOL_REVISION == 3).
# ---------------------------------------------------------------------------
RECEIPT_DESIGN_ARTIFACT_VERSION = "schema31-receipt-v2-design/4"
RECEIPT_DESIGN_REVISION = "review-candidate-r4-focused-amendments"
RECEIPT_DESIGN_STATUS = "READY_FOR_FOCUSED_INDEPENDENT_DESIGN_REVIEW"
RECEIPT_DESIGN_PRECEDENCE = "Exact named owner SHA > r4 amendments and this self-contained effective r4 JSON (contradiction stops) > unaffected pinned r3 MD SHA 22729cadf86090464e67f3a14b04b5b031b0190c940a95d1388e977980d56018. r1/r2 only exact previously enumerated unchanged references; no fallback hybrid. r4 is a design-document revision, not a wire/schema/domain/epoch revision. All historical r3_expected retained; per-row r4_expected, if present, is the effective expectation."
RECEIPT_DESIGN_EXPECTED_PRECEDENCE = "r4_expected if present else original unchanged r3_expected; all original inputs/hashes/owner/status/disposition retained"
RECEIPT_DESIGN_AMENDMENT_LINEAGE = {
    "base_json_sha256": "5d2af1296bf6ea7dd018b01c32d7b2b670b54a6ba98cc99215345397ac726ade",
    "shared_r3_md_sha256": "22729cadf86090464e67f3a14b04b5b031b0190c940a95d1388e977980d56018",
    "amendments_sha256": "7b49d7e90f2d8782d24ba4708f484b728f8d62623afbebfec228b116ca3ad879",
    "only_findings": ('R3-C01', 'R3-C02', 'R3-Q01'),
    "owner_approved_at_authoring": False,
    "implementation_at_authoring": False,
    "runtime_pass_at_authoring": False,
}
# The ten adapted gates: (acceptance_disposition index, gate id, r4_adaptation_reason).
R4_ADAPTED_GATES = (
    (17, "A18", "R3-C01/Q01"),
    (22, "A23", "R3-C02"),
    (24, "A25", "R3-C02"),
    (26, "A27", "R3-C01/C02"),
    (29, "A30", "R3-Q01"),
    (33, "B02", "R3-Q01"),
    (35, "B04", "R3-Q01"),
    (36, "B05", "R3-C02"),
    (37, "B06", "R3-C01"),
    (38, "B07", "R3-C02"),
)
R4_ADAPTED_GATE_INDICES = tuple(index for index, _gate_id, _reason in R4_ADAPTED_GATES)
R4_GATE_ROWS = 40
R4_GATE_RETAINED_FIELDS = ("id", "name", "disposition", "original_input", "r1_input_sha256",
                           "r2_expected_sha256", "r3_expected", "reason", "authority_sha256", "status")
# r4 amendments section 5 C02 row: only writer_timeline[0] and [5] were
# overridden; entries [1..4] stay byte-identical to the r3 base (checked by
# the apparatus script against both frozen JSONs).
R4_WRITER_TIMELINE_OVERRIDDEN_INDICES = (0, 5)
R4_WRITER_TIMELINE_UNCHANGED_INDICES = (1, 2, 3, 4)
R4_DISCLOSURE_PINS = {
    "source_json_sha256": "70fec4785772aabd1ca075fe8c1d65e3fbc6da1e9c216bb74fbd26bc7fac54f1",
    "base_json_sha256": "5d2af1296bf6ea7dd018b01c32d7b2b670b54a6ba98cc99215345397ac726ade",
    "amendments_sha256": "7b49d7e90f2d8782d24ba4708f484b728f8d62623afbebfec228b116ca3ad879",
    "gates_inventory_sha256": "a0e0cfd8b94e8ed08e0e39ab1777bd157c4190e2f7f53a4adfd5e839c79ef7a7",
    "known_published_gap_sha256": "d6758d684fdd23f569233096477008bb4dbba57c19668f3382e2d9b282eaf836",
    "remaining_policy_gates_sha256": "d239797e8dd41e29c86625e40f75fd6202998b61cefe25ab689773c97ac1d432",
    "finding_disposition_I02_normative_sections_sha256": "59818fc04ab9487cb7b46f557b37f693812e0bd7457f49a8c13c2653241fb796",
    "remaining_policy_gates_count": 7,
}
# Owner state as decided 2026-10-17 (reports/owner-decisions-20261017-batch.json,
# sha256 pinned): F and context policies approved as conservative pause plus
# isolated implementation per the r4 design, with explicit limits. This overlay
# records the decision; it is not an authorization to change any data and it
# does not advance any pinned r4 design byte (r4 expectations/disclosure stay
# exactly as frozen).
R4_OWNER_STATUS = {
    "decided_on": "2026-10-17",
    "owner_decision_path": "reports/owner-decisions-20261017-batch.json",
    "owner_decision_sha256": "d052ad48922b37171efc0eb5b9a1dcc4ef1919975bcd2113d817e04bd9304a6d",
    "F_policy": "APPROVED_AS_CONSERVATIVE_PAUSE",
    "context_policy": "APPROVED_AS_CONSERVATIVE_PAUSE",
    "isolated_implementation_per_r4_design": "AUTHORIZED",
    "limits": ("no DB migration", "no activation", "no production writes",
               "no legacy bridge/certificate/pin/ledger activation"),
    "native_runtime_gates": "REQUIRED_NOT_RUN",
}
R4_STILL_OPEN = (
    "owner exact r4 design approval form / any certificate or pin: separate decision, not issued",
    "quiet native recovery seam drain/activation surfaces stay sandbox-only; separate approval needed",
    "finite measured resource maxima / provisioned reserve / maintenance budget: not approved",
    "legacy bridge + certificate/pin/source/ledger activation, migration, production, publication: NOT AUTHORIZED",
    "all 39 native runtime gates: REQUIRED_NOT_RUN (a real run needs separate implementation+copy-write approval)",
)


def receipt_design_binding() -> dict:
    """A1 binding view: version / precedence / derivation metadata.

    Returns the effective-r4 metadata with an explicit annotation that none
    of it is protocol record bytes; the acceptance runner resolves gate
    expectations through ``resolve_gate_expected`` below.
    """
    return {
        "artifact_version": RECEIPT_DESIGN_ARTIFACT_VERSION,
        "revision": RECEIPT_DESIGN_REVISION,
        "status": RECEIPT_DESIGN_STATUS,
        "precedence": RECEIPT_DESIGN_PRECEDENCE,
        "expected_precedence": RECEIPT_DESIGN_EXPECTED_PRECEDENCE,
        "amendment_lineage": dict(RECEIPT_DESIGN_AMENDMENT_LINEAGE),
        "gate_rows": R4_GATE_ROWS,
        "adapted_gate_count": len(R4_ADAPTED_GATES),
        "adapted_gate_indices": R4_ADAPTED_GATE_INDICES,
        "wire_or_schema_revision": False,
        "protocol_record": False,
        "source_json_sha256": R4_DISCLOSURE_PINS["source_json_sha256"],
    }


def resolve_gate_expected(row: dict) -> dict:
    """A1/A2 acceptance binding: the effective expectation of one gate row.

    Precedence (``protocol.precedence`` / ``amendment_lineage
    .expected_precedence``): ``r4_expected`` if present is the effective
    expectation, otherwise the original unchanged ``r3_expected``. The
    historical fields are retained and returned untouched. Shape violations
    refuse with the canonical ValueError (never KeyError/TypeError): a
    non-dict row or missing id, a missing/empty/non-string ``r3_expected``,
    a present-but-empty/non-string ``r4_expected``, or a missing
    ``r4_adaptation_reason`` while ``r4_expected`` is present.
    """
    if type(row) is not dict or "id" not in row:
        _invalid()
    r3 = row.get("r3_expected")
    if type(r3) is not str or not r3:
        _invalid()
    retained = {key: row[key] for key in R4_GATE_RETAINED_FIELDS if key in row}
    if "r4_expected" not in row:
        return {"id": row["id"], "effective_expected": r3, "adapted": False,
                "r4_adaptation_reason": None, "retained": retained}
    r4 = row["r4_expected"]
    if type(r4) is not str or not r4:
        _invalid()
    reason = row.get("r4_adaptation_reason")
    if type(reason) is not str or not reason:
        _invalid()
    return {"id": row["id"], "effective_expected": r4, "adapted": True,
            "r4_adaptation_reason": reason, "retained": retained}


def verify_r4_gate_binding(rows) -> dict:
    """A2 derivation check over the full 40-row effective disposition.

    Exactly the ten ``R4_ADAPTED_GATES`` indices carry ``r4_expected`` (gate
    identity and pinned ``r4_adaptation_reason`` must agree); every
    ``r4_expected`` extends its retained ``r3_expected`` prefix
    (``r4 == r3 + " " + extra``, supplement-only, never a rewrite); every
    other row resolves through the unchanged ``r3_expected``. Anything else
    refuses -- an unadapted row may not invent an expectation and an adapted
    row may not drop or reorder one.
    """
    if type(rows) not in (list, tuple) or len(rows) != R4_GATE_ROWS:
        _invalid()
    pinned = {index: (gate_id, reason) for index, gate_id, reason in R4_ADAPTED_GATES}
    seen, ids = set(), []
    for index, row in enumerate(rows):
        resolved = resolve_gate_expected(row)
        ids.append(resolved["id"])
        if not resolved["adapted"]:
            if index in pinned:
                _invalid()   # a pinned adapted gate must carry its r4_expected
            continue
        if index not in pinned:
            _invalid()       # an unlisted row may not invent an r4_expected
        gate_id, reason = pinned[index]
        if resolved["id"] != gate_id or resolved["r4_adaptation_reason"] != reason:
            _invalid()       # adapted identity or reason drifted
        r3 = resolved["retained"]["r3_expected"]
        if not resolved["effective_expected"].startswith(r3 + " "):
            _invalid()       # r4_expected must extend, never rewrite, r3_expected
        seen.add(index)
    if seen != set(pinned) or len(set(ids)) != R4_GATE_ROWS:
        _invalid()
    return {"rows": R4_GATE_ROWS, "adapted_indices": sorted(seen),
            "adapted_ids": [pinned[i][0] for i in sorted(seen)],
            "fallback_count": R4_GATE_ROWS - len(seen)}


def receipt_design_disclosure() -> dict:
    """A3: the accurate disclosure (frozen r4 bytes + owner-status overlay).

    The frozen r4 text is kept verbatim as the authoring-time record (its
    sha256 pins are checked against the frozen JSON by the apparatus script
    and by the acceptance tests); the owner-status overlay records what the
    owner has since approved (2026-10-17: F and context conservative pause
    policies plus isolated implementation per the r4 design). Unwitnessed
    history stays unclaimed, unexecuted runtime gates stay
    REQUIRED_NOT_RUN, and this disclosure is never an authorization.
    """
    return {
        "artifact_version": RECEIPT_DESIGN_ARTIFACT_VERSION,
        "frozen_r4": {
            "known_published_gap_sha256": R4_DISCLOSURE_PINS["known_published_gap_sha256"],
            "remaining_policy_gates_count": R4_DISCLOSURE_PINS["remaining_policy_gates_count"],
            "remaining_policy_gates_sha256": R4_DISCLOSURE_PINS["remaining_policy_gates_sha256"],
            "finding_disposition_I02_normative_sections_sha256":
                R4_DISCLOSURE_PINS["finding_disposition_I02_normative_sections_sha256"],
            "source_json_sha256": R4_DISCLOSURE_PINS["source_json_sha256"],
        },
        "owner_status": dict(R4_OWNER_STATUS),
        "still_open": tuple(R4_STILL_OPEN),
        "truthfulness": ("unwitnessed earlier context/history is never claimed as verified; "
                         "unexecuted runtime gates stay REQUIRED_NOT_RUN; policy approval is not "
                         "an implementation, activation or certification decision"),
        "disclosure_only": True,
        "protocol_record": False,
    }
