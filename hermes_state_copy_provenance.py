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
    outside that family projects explicit NULLs.

Only the native writer (inside its own ``_execute_write`` transaction) may call
the builder; the generic ``SessionDB.set_meta``/``list_meta_prefix`` surfaces
refuse the reserved namespace.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
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


def validate_journal_page(body: dict) -> None:
    """Strict page shape: exact fields, 1..8 items, canonical <= 65536 bytes."""
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
        if type(item) is not dict or item.get("kind") not in _ITEM_KINDS:
            _invalid()
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
        if type(context) is not dict or set(context) != {
                "session_id", "native_binding_fields", "lineage_markers", "context_sha256"}:
            _invalid()
        if type(context["session_id"]) is not str or not context["session_id"]:
            _invalid()
        if type(context["native_binding_fields"]) is not list \
                or len(context["native_binding_fields"]) != len(_NATIVE_BINDING_FIELDS):
            _invalid()
        if type(context["lineage_markers"]) is not list \
                or len(context["lineage_markers"]) != len(_LINEAGE_MARKERS):
            _invalid()
        _sha256_hex(context["context_sha256"])
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

    Every named key, coordinate, page ordinal, the page/item counts, the framed
    pages root and the self-hashed operation id are re-verified; a missing or
    non-canonical entry is the canonical ValueError.
    """
    key = encode_receipt3_operation_key(owner_session_id, generation)
    header, header_bytes = _read_canonical(conn, key)
    if (type(header) is not dict or header.get("owner_session_id") != owner_session_id
            or header.get("generation") != generation):
        _invalid()
    page_count = header.get("page_count")
    if type(page_count) is not int or page_count < 0:
        _invalid()
    _epoch_string(header.get("epoch"))
    pages, page_bytes = [], []
    for ordinal in range(page_count):
        page, raw = _read_canonical(
            conn, encode_receipt3_page_key(owner_session_id, generation, ordinal))
        validate_journal_page(page)
        if (page["owner_session_id"] != owner_session_id or page["generation"] != generation
                or page["page_ordinal"] != ordinal or page["epoch"] != header.get("epoch")):
            _invalid()
        pages.append(page)
        page_bytes.append(raw)
    if sum(len(page["items"]) for page in pages) != header.get("item_count"):
        _invalid()
    if journal_pages_root(owner_session_id, header.get("epoch"), generation, pages) \
            != header.get("pages_root_sha256"):
        _invalid()
    sealed = {name: value for name, value in header.items() if name != "operation_sha256"}
    if header.get("operation_sha256") != H("operation", sealed):
        _invalid()
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
