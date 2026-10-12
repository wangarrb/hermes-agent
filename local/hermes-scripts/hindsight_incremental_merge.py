#!/usr/bin/env python3
"""Lossless local canonical state; no source scanning, LLM, network or DB writes.

The detailed JSONL is the state, not the compact cards. Unknown legacy DB rows
stay opaque and must not pass through the existing lossy DB publisher. Generation
manifests are content sealed, bank/root bound, and written last without replacing
or cleaning any old files. Gates belong outside this immutable tree.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Iterator

SCHEMA = 'hindsight-incremental-generation/1'
MAX_ANCESTRY_DEPTH = 128
IDENTITY_FIELDS = ('insight', 'topic', 'type', 'applicability', 'validity',
                   'valid_from', 'valid_until', 'conditions', 'scope', 'bank')
RECIPE_FIELDS = frozenset({'id', 'node_id', 'model', 'label', 'output_path', 'batch_id',
    'batch_ids', 'generated_at', 'recipe', 'provenance', 'endpoint', 'base_url',
    'unit_index', 'contributions', 'contribution_provenance'})
SET_FIELDS = frozenset({'evidence_ids', 'source_ids', 'source_documents', 'evidence',
    'tags', 'supersedes', 'conflicts', 'open_questions'})
HASH_RE = re.compile(r'[0-9a-f]{64}\Z')


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return _payload_info(_json_chunks(value))['sha256']


def _json_chunks(value: Any) -> Iterator[bytes]:
    # Never join the whole Unicode document: one non-BMP character otherwise
    # expands all its ASCII vector text to four bytes per character.
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False)
    buffer = bytearray()
    for part in encoder.iterencode(value):
        buffer.extend(part.encode('utf-8'))
        if len(buffer) >= 65536:
            yield bytes(buffer)
            buffer.clear()
    if buffer:
        yield bytes(buffer)


def _payload_info(chunks: Iterable[bytes]) -> dict:
    digest, size = hashlib.sha256(), 0
    for chunk in chunks:
        digest.update(chunk)
        size += len(chunk)
    return {'sha256': digest.hexdigest(), 'size': size}


def _artifact_chunks(kind: str, value: Any) -> Iterator[bytes]:
    if kind == 'observations':
        for row in value:
            yield from _json_chunks(row)
            yield b'\n'
    elif kind == 'card':
        yield from _json_chunks(value)
        yield b'\n'
    else:
        yield value


def _typed(row: dict) -> bool:
    return row.get('kind') != 'opaque_legacy' and all(
        isinstance(row.get(k), str) and row[k].strip() for k in IDENTITY_FIELDS[:4])


def observation_id(row: dict) -> str:
    """Exact typed conditions are identity; execution recipes are not."""
    if _typed(row):
        return 'obs:' + canonical_hash({k: row.get(k) for k in IDENTITY_FIELDS})
    if row.get('kind') == 'opaque_legacy':
        return 'legacy:' + canonical_hash({'bank': row.get('bank'), 'row': row.get('legacy_row')})
    return 'opaque:' + canonical_hash(row)


def _content(row: dict) -> dict:
    value = {k: copy.deepcopy(v) for k, v in row.items() if k not in RECIPE_FIELDS}
    for k in SET_FIELDS & value.keys():
        if isinstance(value[k], list):
            value[k] = _union(value[k])
    return value


def _contributions(row: dict) -> list[dict]:
    values = row.get('contributions')
    return copy.deepcopy(values if isinstance(values, list) else [row])


def observation_content_hash(row: dict) -> str:
    if row.get('kind') == 'opaque_legacy':
        # Exact DB content, including its unknown metadata, is semantic. Export
        # recipes live only in the wrapper and are sealed by artifact hashes.
        return canonical_hash({'bank': row.get('bank'), 'row': row.get('legacy_row')})
    if not _typed(row):
        return canonical_hash(row)
    contents = {_json(_content(v)): _content(v) for v in _contributions(row)}
    values = [contents[k] for k in sorted(contents)]
    return canonical_hash(values[0] if len(values) == 1 else values)


def _union(values: list) -> list:
    unique = {_json(v): copy.deepcopy(v) for v in values}
    return [unique[k] for k in sorted(unique)]


def _combine(rows: list[dict]) -> dict:
    by_content: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for contribution in _contributions(row):
            by_content[canonical_hash(_content(contribution))].append(contribution)
        for contribution in row.get('contribution_provenance', []):
            by_content[canonical_hash(_content(contribution))].append(contribution)
    representatives = [min(by_content[k], key=_json) for k in sorted(by_content)]
    variants = _union([v for group in by_content.values() for v in group])
    if len(variants) == 1:
        return copy.deepcopy(variants[0])
    result: dict = {}
    # A scalar disagreement is never resolved as truth: all original values
    # remain in contributions. This deterministic projection is not a verdict.
    for key in sorted({k for r in representatives for k in r} - {'contributions', 'contribution_provenance'}):
        values = [r[key] for r in representatives if key in r]
        if key in SET_FIELDS and all(isinstance(v, list) for v in values):
            result[key] = _union([item for v in values for item in v])
        else:
            result[key] = copy.deepcopy(min(values, key=_json))
    result['contributions'] = copy.deepcopy(representatives)
    if len(variants) > len(representatives):
        result['contribution_provenance'] = variants
    return result


def _combine_opaque(rows: list[dict], *, projected: bool = False) -> dict:
    """Keep complete original wrappers, not a first-wins export provenance.

    Singleton imports are unchanged for authoritative-seed equality checks.
    Multiple wrappers are a canonical set; the top-level wrapper is only its
    deterministic projection, never a claim that the other exports disappeared.
    """
    originals = []
    node_id = observation_id(rows[0])
    for row in rows:
        variants = row.get('opaque_provenance', [row])
        if not isinstance(variants, list) or not variants:
            raise ValueError('opaque provenance requires a nonempty array')
        if any(not isinstance(v, dict) or v.get('kind') != 'opaque_legacy'
               or 'opaque_provenance' in v or observation_id(v) != node_id for v in variants):
            raise ValueError('invalid opaque provenance identity')
        if 'opaque_provenance' in row:
            exact = _union(variants)
            expected = copy.deepcopy(exact[0])
            if len(exact) > 1:
                expected['opaque_provenance'] = exact
            if projected:
                expected = publication_observation(expected)
            if expected != row:
                raise ValueError('opaque aggregate differs from complete original provenance')
        originals.extend(variants)
    variants = _union(originals)
    result = copy.deepcopy(variants[0])
    if len(variants) > 1:
        result['opaque_provenance'] = variants
    return result


def merge_observations(seed: list[dict], delta: list[dict]) -> list[dict]:
    """Merge typed nodes losslessly; preserve opaque seed without interpretation.

    No-delta is a literal no-op. Duplicate contributions cannot increase evidence
    counts. Distinct conditions/validity are distinct nodes; conflicting scalar
    values and arbitrary nested evidence survive as complete contributions.
    """
    if not isinstance(seed, list) or not isinstance(delta, list):
        raise ValueError('observations must be lists')
    for row in seed + delta:
        if not isinstance(row, dict):
            raise ValueError('observation must be an object')
        _json(row)
    if not delta:
        return copy.deepcopy(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in seed + delta:
        groups[observation_id(row)].append(row)
    result = []
    for key in sorted(groups):
        rows = groups[key]
        if _typed(rows[0]):
            result.append(_combine(rows))
        elif rows[0].get('kind') == 'opaque_legacy':
            result.append(_combine_opaque(rows))
        else:
            result.append(copy.deepcopy(rows[0]))
    return result


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError(f'invalid {name} hash')
    return value


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f'invalid {name}')
    return value


def _parent(parent: Any, bank: Any, *, _ancestors: tuple[Path, ...] = ()) -> None:
    if not isinstance(parent, dict) or parent.get('bank') != bank:
        raise ValueError('parent bank mismatch')
    field = 'manifest_sha256' if parent.get('schema_version') == SCHEMA else 'seed_sha256'
    _hash(parent.get(field), 'parent')
    if field == 'manifest_sha256':
        expected = canonical_hash({k: v for k, v in parent.items() if k != field})
        if expected != parent[field]:
            raise ValueError('parent manifest hash mismatch')
        path = _absolute(_name(parent.get('manifest_path'), 'parent manifest path'))
        if _absolute(_name(parent.get('root'), 'parent root path')) != path.parent:
            raise ValueError('parent root/path mismatch')
        # A dictionary/hash binding does not prove rollback files still exist.
        # Verify every exact ancestor's index/cards before seed use or publish.
        if _validated(path, _ancestors=_ancestors)[0] != parent:
            raise ValueError('parent artifact differs from bound manifest')
    _json(parent)


def _legacy_primary_key(row: Any, bank: str) -> tuple[str, str]:
    if not isinstance(row, dict) or row.get('bank_id') != bank:
        raise ValueError('legacy row bank mismatch')
    if not isinstance(row.get('text'), str):
        raise ValueError('legacy row requires exact text')
    # Validate the original token without trimming, coercing or substituting a
    # metadata observation_id. Real database keys, not rendered text, are unique.
    return bank, _name(row.get('id'), 'legacy row id')


def import_legacy_seed(rows: list[dict], *, bank: str, provenance: dict) -> list[dict]:
    """Pure adapter for an explicitly authorized readonly export, never latest cards.

    Exported rows must already be JSON-native. Do not split published text or
    deserialize string-valued metadata into guessed structured insight fields.
    """
    _name(bank, 'bank')
    _parent(provenance, bank)
    result = []
    ids = set()
    for row in rows:
        key = _legacy_primary_key(row, bank)
        if key in ids:
            raise ValueError('duplicate legacy primary key')
        ids.add(key)
        wrapped = {'kind': 'opaque_legacy', 'bank': bank,
                   'legacy_row': copy.deepcopy(row), 'legacy_sha256': canonical_hash(row),
                   'provenance': copy.deepcopy(provenance)}
        wrapped['id'] = observation_id(wrapped)
        result.append(wrapped)
    return sorted(result, key=observation_id)


def publication_observation(row: dict) -> dict:
    """Local card projection; opaque rows require a separate lossless DB path."""
    value = copy.deepcopy(row)
    if row.get('kind') == 'opaque_legacy':
        value.update(insight=row['legacy_row']['text'], topic='legacy', type='opaque_legacy',
                     applicability='opaque legacy row; do not reinterpret')
    value['id'] = observation_id(row)
    return value


def _evidence_fields(row: dict) -> None:
    for key in SET_FIELDS & row.keys():
        if not isinstance(row[key], list):
            raise ValueError(f'{key} must be an array')
        if key in {'evidence_ids', 'source_ids', 'source_documents', 'tags', 'supersedes'} and any(
                not isinstance(v, str) or not v for v in row[key]):
            raise ValueError(f'{key} requires nonempty string values')


def _rows(rows: list[dict], bank: str | None = None, *, legacy: bool = False,
          _ancestors: tuple[Path, ...] = ()) -> None:
    if not isinstance(rows, list):
        raise ValueError('observations must be an array')
    ids, declared_ids, legacy_keys = set(), set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('observation must be an object')
        _json(row)
        if row.get('kind') == 'opaque_legacy':
            original = row.get('legacy_row')
            key = _legacy_primary_key(original, _name(row.get('bank'), 'legacy bank'))
            if key in legacy_keys:
                raise ValueError('duplicate legacy primary key')
            legacy_keys.add(key)
            if row.get('legacy_sha256') != canonical_hash(original):
                raise ValueError('legacy row hash mismatch')
            if bank and row.get('bank') != bank:
                raise ValueError('legacy bank mismatch')
            _parent(row.get('provenance'), row.get('bank'), _ancestors=_ancestors)
            if 'opaque_provenance' in row:
                _combine_opaque([row], projected=legacy)
                for variant in row['opaque_provenance']:
                    _rows([variant], bank, _ancestors=_ancestors)
        elif legacy:
            if not isinstance(row.get('insight'), str) or not row['insight'].strip():
                raise ValueError('legacy index observation requires insight')
        elif not _typed(row):
            raise ValueError('untyped row requires explicit opaque legacy seed adapter')
        if bank and row.get('bank', bank) != bank:
            raise ValueError('observation bank mismatch')
        node_id = observation_id(row)
        if node_id in ids or (row.get('id') and row['id'] in declared_ids):
            raise ValueError('duplicate observation id')
        ids.add(node_id)
        if row.get('id'):
            declared_ids.add(row['id'])
        if row.get('kind') != 'opaque_legacy':
            _evidence_fields(row)
        contributions = row.get('contributions', [])
        if 'contributions' in row and (not isinstance(contributions, list) or not contributions):
            raise ValueError('contributions require a nonempty array')
        if contributions:
            if any(not isinstance(c, dict) or not _typed(c) or observation_id(c) != node_id for c in contributions):
                raise ValueError('invalid contribution identity')
            for contribution in contributions:
                _evidence_fields(contribution)
            hashes = [canonical_hash(_content(c)) for c in contributions]
            if len(set(hashes)) != len(hashes):
                raise ValueError('duplicate contribution')
        variants = row.get('contribution_provenance', [])
        if not isinstance(variants, list) or any(
                not isinstance(c, dict) or not _typed(c) or observation_id(c) != node_id for c in variants):
            raise ValueError('invalid contribution provenance')
        for variant in variants:
            _evidence_fields(variant)
        if contributions or variants:
            expected = _combine([row])
            if row.get('id') == node_id:
                expected['id'] = node_id  # Cards add the stable ID, not a new contribution.
            if expected != row:
                raise ValueError('aggregate differs from complete original contributions')


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith('/') or '\\' in value:
        raise ValueError('unsafe artifact path')
    if any(ord(c) < 32 or ord(c) == 127 for c in value) or any(p in {'', '.', '..'} for p in value.split('/')):
        raise ValueError('unsafe artifact path')
    return value


def _absolute(path: Path | str) -> Path:
    raw = str(path)
    if (any(ord(c) < 32 or ord(c) == 127 for c in raw) or '\\' in raw
            or '//' in raw or any(p in {'.', '..'} for p in raw.split('/'))):
        raise ValueError('unsafe root path')
    value = Path(path).absolute()
    for component in (value, *value.parents):
        if component.is_symlink():
            raise ValueError(f'symlink path forbidden: {component}')
    return value


@contextmanager
def _open_read(path: Path):
    _absolute(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError(f'missing or unsafe artifact path: {path}') from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f'artifact must be a regular unaliased file: {path}')
        with os.fdopen(fd, 'rb') as stream:
            fd = -1
            yield stream
    finally:
        if fd >= 0:
            os.close(fd)


def _file_chunks(stream) -> Iterator[bytes]:
    while chunk := stream.read(65536):
        yield chunk


def _read(path: Path) -> bytes:
    with _open_read(path) as stream:
        return stream.read()


def _file_info(path: Path) -> dict:
    with _open_read(path) as stream:
        return _payload_info(_file_chunks(stream))


def _load_json(stream, *, digest=None) -> Any:
    """Decode frozen seed arrays/envelopes without a whole-document string.

    Only the outer array and array-valued envelope fields are split; every row
    still uses the stdlib decoder, preserving arbitrary nested legacy content.
    """
    import io
    prefix = stream.read(4)
    stream.seek(0)
    if digest is not None:
        class HashedReader(io.RawIOBase):
            def readable(self):
                return True
            def readinto(self, target):
                data = stream.read(len(target))
                digest.update(data)
                target[:len(data)] = data
                return len(data)
        source = io.BufferedReader(HashedReader())
    else:
        source = stream
    text = io.TextIOWrapper(source, encoding=json.detect_encoding(prefix),
                            errors='surrogatepass', newline='')
    decoder = json.JSONDecoder()
    buffer, position, eof = '', 0, False

    def more():
        nonlocal buffer, position, eof
        chunk = text.read(65536)
        buffer, position, eof = buffer[position:] + chunk, 0, not chunk

    def peek():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position] in ' \t\r\n':
                position += 1
            if position < len(buffer) or eof:
                return buffer[position:position + 1]
            more()

    def decoded():
        nonlocal position
        peek()
        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                more()
                continue
            # A scalar number may cross a read boundary (1 followed by e-20).
            if not eof and (end == len(buffer) or buffer[end] not in ' \t\r\n,:]}'):
                more()
                continue
            position = end
            return value

    def take(token):
        nonlocal position
        if peek() != token:
            raise ValueError('invalid JSON delimiter')
        position += 1

    def array():
        values = []
        take('[')
        if peek() == ']':
            take(']')
            return values
        while True:
            values.append(decoded())
            if peek() == ']':
                take(']')
                return values
            take(',')

    try:
        if peek() == '[':
            value = array()
        elif peek() == '{':
            value = {}
            take('{')
            if peek() != '}':
                while True:
                    key = decoded()
                    if not isinstance(key, str):
                        raise ValueError('JSON object key must be a string')
                    take(':')
                    value[key] = array() if peek() == '[' else decoded()
                    if peek() == '}':
                        break
                    take(',')
            take('}')
        else:
            value = decoded()
        if peek():
            raise ValueError('extra JSON data')
        return value
    finally:
        text.detach()


def _read_json(path: Path, *, expected_hash: str | None = None) -> Any:
    with _open_read(path) as stream:
        digest = hashlib.sha256() if expected_hash is not None else None
        value = _load_json(stream, digest=digest)
        if digest is not None and digest.hexdigest() != expected_hash:
            raise ValueError('artifact hash mismatch')
        return value


def _object(payload: bytes, label: str) -> dict:
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f'duplicate JSON key in {label}')
            result[k] = v
        return result
    try:
        value = json.loads(payload, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'invalid JSON in {label}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'JSON object required in {label}')
    return value


def _index(payload: bytes | Iterable[bytes]) -> list[dict]:
    # A malformed or empty line is an error, never a silently dropped fact.
    # Retain the last line until the next chunk to preserve bytes.splitlines()
    # semantics, including a CRLF split across chunks and a final lone CR.
    def lines():
        pending = b''
        for chunk in ([payload] if isinstance(payload, bytes) else payload):
            parts = (pending + chunk).splitlines(keepends=True)
            pending = parts.pop() if parts else b''
            yield from parts
        if pending:
            yield pending
    return [_object(line, f'JSONL line {i}') for i, line in enumerate(lines(), 1)]


def _read_index(path: Path, expected: dict | None = None) -> tuple[list[dict], dict]:
    digest, size = hashlib.sha256(), 0
    with _open_read(path) as stream:
        def chunks():
            nonlocal size
            for chunk in _file_chunks(stream):
                digest.update(chunk)
                size += len(chunk)
                yield chunk
        blocks, error, rows = chunks(), None, []
        try:
            rows = _index(blocks)
        except ValueError as exc:
            error = exc
            # Finish the raw digest even when parsing failed; corrupt artifacts
            # must still report the original hash/size failure before JSON errors.
            for _ in blocks:
                pass
    info = {'sha256': digest.hexdigest(), 'size': size}
    if expected is not None and info != expected:
        raise ValueError(f'artifact hash/size mismatch: {path.name}')
    if error is not None:
        raise error
    return rows, info


def _inventory(root: Path, expected: set[str]) -> None:
    actual = set()
    expected_dirs = {str(p) for name in expected for p in Path(name).parents if str(p) != '.'}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            _absolute(path)
            relative = path.relative_to(root).as_posix()
            if name in dirs and relative not in expected_dirs:
                raise ValueError(f'unexpected directory path: {relative}')
            if name in files:
                actual.add(relative)
    if actual != expected:
        raise ValueError(f'extra or missing undeclared artifacts: {sorted(actual ^ expected)}')


def _projection(rows: list[dict], *, max_obs: int = 120) -> list[dict]:
    # Existing build_cards/write_index both call insight-only dedup and truncate
    # evidence. Reuse their document shape, not their lossy state reducer.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        projection = publication_observation(row)
        grouped[projection['topic']].append(projection)
    cards = []
    for topic in sorted(grouped):
        members = sorted(grouped[topic], key=lambda o: o['id'])
        card_id = 'offline-v2-card::topic::' + canonical_hash({'topic': topic})
        cards.append({'card_id': card_id, 'schema_version': 'offline-hindsight-v2-card/0.1',
            'scope': 'topic', 'topic': topic, 'generated_at': 'sealed-local-state',
            'input_stats': {'observation_count': len(members), 'source_document_count':
                            len({d for o in members for d in o.get('source_documents', [])})},
            'executive_summary': [f'{topic}: {len(members)} sealed observations; full details in JSONL.'],
            'canonical_observations': members[:max_obs], 'entity_aliases': [],
            'conflicts': [c for o in members[:max_obs] for c in o.get('conflicts', [])],
            'open_questions': [o for o in members[:max_obs] if o.get('type') == 'open_question'],
            'evidence_index': _union([d for o in members for d in o.get('source_documents', [])])})
    return cards


def _state_hash(rows: list[dict]) -> str:
    return canonical_hash(sorted((observation_id(r), observation_content_hash(r)) for r in rows))


def _generation_hash(bank: str, parent_hash: str, state_hash: str, batches: list[str]) -> str:
    return canonical_hash({'bank': bank, 'parent_sha256': parent_hash,
                           'observations_sha256': state_hash, 'consumed_batch_ids': batches})


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'invalid {label} count')
    return value


def _check_cards(cards: list[dict], entries: list[dict], *,
                 _ancestors: tuple[Path, ...] = ()) -> None:
    ids = set()
    groups = set()
    for card, entry in zip(cards, entries):
        for field in ('card_id', 'scope', 'topic'):
            if card.get(field) != entry.get(field):
                raise ValueError('card manifest identity mismatch')
        if not card.get('card_id') or card['card_id'] in ids:
            raise ValueError('duplicate or missing card id')
        group = (card.get('scope'), card.get('topic'))
        if card.get('scope') not in {'topic', 'global'} or not isinstance(card.get('topic'), str) or group in groups:
            raise ValueError('invalid or duplicate card group')
        ids.add(card['card_id'])
        groups.add(group)
        observations = card.get('canonical_observations')
        if not isinstance(observations, list) or len(observations) != _count(entry.get('observation_count'), 'card observation'):
            raise ValueError('card observation count mismatch')
        _rows(observations, legacy=True, _ancestors=_ancestors)


def _validated(path: Path, *, _ancestors: tuple[Path, ...] = ()) -> tuple[dict, list[dict], list[dict]]:
    path = _absolute(path)
    if path in _ancestors:
        raise ValueError('generation ancestry cycle')
    if len(_ancestors) >= MAX_ANCESTRY_DEPTH:
        raise ValueError('generation ancestry depth limit exceeded')
    _ancestors = (*_ancestors, path)
    manifest = _object(_read(path), 'manifest')
    root = path.parent
    if manifest.get('schema_version') != SCHEMA:
        raise ValueError('unsupported generation manifest schema')
    if _absolute(_name(manifest.get('root'), 'root path')) != root or manifest.get('manifest_path') != str(path):
        raise ValueError('manifest root/path mismatch')
    bank = _name(manifest.get('bank'), 'bank')
    _hash(manifest.get('manifest_sha256'), 'manifest')
    if canonical_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError('manifest hash mismatch')
    _parent(manifest.get('parent'), bank, _ancestors=_ancestors)
    if canonical_hash(manifest['parent']) != _hash(manifest.get('parent_sha256'), 'parent'):
        raise ValueError('parent hash mismatch')
    batches = manifest.get('consumed_batch_ids')
    if not isinstance(batches, list) or any(not isinstance(b, str) or not b.strip() for b in batches) or batches != sorted(set(batches)):
        raise ValueError('invalid consumed batch ids')
    paths, card_objects, rows = set(), {}, None
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('missing exact artifact manifest')
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError('invalid artifact entry')
        name = _relative(entry.get('path'))
        if name == 'manifest.json' or name in paths:
            raise ValueError('duplicate artifact path')
        if entry.get('kind') not in {'observations', 'card', 'markdown'}:
            raise ValueError('invalid artifact kind')
        expected = {'sha256': _hash(entry.get('sha256'), 'artifact'),
                    'size': _count(entry.get('size'), 'artifact size')}
        if entry['kind'] == 'observations':
            rows, info = _read_index(root / name, expected)
        elif entry['kind'] == 'card':
            payload = _read(root / name)
            info = _payload_info([payload])
        else:
            info = _file_info(root / name)
        if info != expected:
            raise ValueError(f'artifact hash/size mismatch: {name}')
        if entry['kind'] == 'card':
            card_objects[name] = _object(payload, name)
            del payload
        paths.add(name)
    _inventory(root, paths | {'manifest.json'})
    index_files = [e['path'] for e in files if e['kind'] == 'observations']
    if index_files != ['observations_index.jsonl']:
        raise ValueError('exact single observation index required')
    assert rows is not None
    _rows(rows, bank, _ancestors=_ancestors)
    if len(rows) != _count(manifest.get('observation_count'), 'observation') or _state_hash(rows) != _hash(manifest.get('observations_sha256'), 'observations'):
        raise ValueError('observation count/content hash mismatch')
    digest = _generation_hash(bank, manifest['parent_sha256'], manifest['observations_sha256'], batches)
    if manifest.get('generation_id') != 'gen:' + digest or root.name != digest:
        raise ValueError('generation bank/content hash mismatch')
    entries = manifest.get('cards')
    if not isinstance(entries, list) or len(entries) != _count(manifest.get('card_count'), 'card'):
        raise ValueError('card manifest count mismatch')
    card_paths = [_relative(e.get('path')) for e in entries]
    declared_cards = [e['path'] for e in files if e['kind'] == 'card']
    if len(set(card_paths)) != len(card_paths) or set(card_paths) != set(declared_cards):
        raise ValueError('card paths differ from exact artifact manifest')
    cards = [card_objects[name] for name in card_paths]
    _check_cards(cards, entries, _ancestors=_ancestors)
    expected_cards = _projection(rows)
    if cards != expected_cards:
        raise ValueError('compact cards do not project the complete sealed index')
    expected_files = {'observations_index.jsonl': 'observations'}
    for card in expected_cards:
        stem = canonical_hash({'topic': card['topic']})
        expected_files[f'topics/{stem}.json'] = 'card'
        expected_files[f'topics/{stem}.md'] = 'markdown'
    if {e['path']: e['kind'] for e in files} != expected_files:
        raise ValueError('artifact layout differs from sealed card projection')
    return manifest, cards, rows


def validate_generation(manifest_path: Path) -> dict:
    return _validated(manifest_path)[0]


def _write(path: Path, payload: bytes | Iterable[bytes]) -> None:
    _absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        for chunk in ([payload] if isinstance(payload, bytes) else payload):
            stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())


def write_generation(root: Path, *, bank: str, parent: dict, observations: list[dict],
                     consumed_batch_ids: list[str]) -> dict:
    """Create a private immutable generation, or verify the exact existing one.

    The caller holds its bank producer lock. A partial/corrupt directory is an
    explicit blocker; retry never overwrites it or deletes evidence of failure.
    """
    root = _absolute(root)
    _name(bank, 'bank')
    _rows(observations, bank)
    if not isinstance(consumed_batch_ids, list):
        raise ValueError('consumed batch ids must be a list')
    batches = sorted({_name(b, 'batch id') for b in consumed_batch_ids})
    rows = sorted(copy.deepcopy(observations), key=observation_id)
    parent_hash, state_hash = canonical_hash(parent), _state_hash(rows)
    digest = _generation_hash(bank, parent_hash, state_hash, batches)
    directory = root / 'generations' / digest
    _absolute(directory)
    # Include the prospective child in the ancestry budget before any writes.
    _parent(parent, bank, _ancestors=(directory / 'manifest.json',))
    payloads = {'observations_index.jsonl': ('observations', rows)}
    cards, entries = _projection(rows), []
    for card in cards:
        stem = canonical_hash({'topic': card['topic']})
        name = f'topics/{stem}.json'
        payloads[name] = ('card', card)
        # This is a deliberately compact view; detailed evidence stays in JSONL.
        md = '\n'.join([f"# {card['topic']}", f"card_id: {card['card_id']}",
                        *[f"- {o['insight']}\n  applicability: {o['applicability']}" for o in card['canonical_observations']]]) + '\n'
        payloads[f'topics/{stem}.md'] = ('markdown', md.encode('utf-8'))
        entries.append({k: card[k] for k in ('card_id', 'scope', 'topic')} |
                       {'path': name, 'observation_count': len(card['canonical_observations'])})
    manifest = {'schema_version': SCHEMA, 'bank': bank, 'root': str(directory),
        'manifest_path': str(directory / 'manifest.json'), 'generation_id': 'gen:' + digest,
        'parent': copy.deepcopy(parent), 'parent_sha256': parent_hash,
        'consumed_batch_ids': batches, 'observations_sha256': state_hash,
        'observation_count': len(rows), 'card_count': len(cards), 'cards': entries,
        'files': [{'path': name, 'kind': kind, **_payload_info(_artifact_chunks(kind, payload))}
                  for name, (kind, payload) in sorted(payloads.items())]}
    manifest['manifest_sha256'] = canonical_hash(manifest)
    if directory.exists():
        existing = validate_generation(directory / 'manifest.json')
        if existing != manifest:
            raise ValueError('immutable generation differs from requested complete state')
        return existing
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ValueError('concurrent generation writer; retry after completion') from exc
    for name, (kind, payload) in sorted(payloads.items()):
        _write(directory / name, _artifact_chunks(kind, payload))
    _write(directory / 'manifest.json', _artifact_chunks('card', manifest))
    for subdir in (directory / 'topics', directory, directory.parent):
        if subdir.exists():
            fd = os.open(subdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return validate_generation(directory / 'manifest.json')


def load_manifest_artifacts(cards_root: Path, *, bank: str | None = None) -> tuple[dict, list[dict], list[dict] | None]:
    """Exact new manifests, or a complete unmixed old reducer manifest.

    Old manifests have no frozen hashes/bank: compatibility proves only exact
    inventory and row validity, not historical provenance or migration success.
    Unknown/newer schemas never fall back to legacy directory discovery.
    """
    root = _absolute(cards_root)
    manifest = _object(_read(root / 'manifest.json'), 'manifest')
    if manifest.get('schema_version') == SCHEMA:
        manifest, cards, rows = _validated(root / 'manifest.json')
        if bank is not None and manifest['bank'] != bank:
            raise ValueError('publisher manifest bank mismatch')
        return manifest, cards, rows
    if manifest.get('schema_version') is not None:
        raise ValueError('unsupported manifest schema; legacy fallback forbidden')
    if bank is not None and manifest.get('bank', bank) != bank:
        raise ValueError('publisher legacy bank mismatch')
    entries = manifest.get('cards')
    if not isinstance(entries, list) or not entries or len(entries) != _count(manifest.get('card_count'), 'card'):
        raise ValueError('incomplete legacy card manifest')
    expected = {'manifest.json'}
    cards = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('scope') not in {'topic', 'global'}:
            raise ValueError('invalid legacy card entry')
        topic = _name(entry.get('topic'), 'topic')
        # Exact old reducer filename algorithm; no glob-based identity lookup.
        slug = re.sub(r'[^a-z0-9]+', '-', topic.strip().lower()).strip('-')[:64]
        slug = slug or 'topic-zh-' + hashlib.sha1(topic.strip().encode('utf-8')).hexdigest()[:10]
        name = f"{'topics' if entry['scope'] == 'topic' else 'global'}/{slug}.json"
        if name in expected:
            raise ValueError('duplicate legacy card path')
        expected.add(name)
        cards.append(_object(_read(root / name), name))
        md = name[:-5] + '.md'
        if (root / md).exists() or (root / md).is_symlink():
            _read(root / md)
            expected.add(md)
    _check_cards(cards, entries)
    count = _count(manifest.get('observation_index_count'), 'observation index')
    index = root / 'observations_index.jsonl'
    rows = None
    if index.exists() or index.is_symlink() or count:
        rows, _ = _read_index(index)
        _rows(rows, bank, legacy=True)
        if len(rows) != count:
            raise ValueError('legacy index count mismatch')
        expected.add('observations_index.jsonl')
    _inventory(root, expected)
    return manifest, cards, rows
