#!/usr/bin/env python3
"""Read-only arrival views; dates and filesystem mtimes are never cursors.

A snapshot is a finite observed fence, not CDC. A lost/overwritten event cannot
be recovered by this reader. The ledger detects epoch changes and missing or
rewritten previously observed identities instead of replaying a new history.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from hindsight_session_manifest import (
    CLEANING_VERSION, SECRET_MATERIAL_RE, discover_session_sources,
    extract_message_text, is_noise_message,
)


class SourceChangedError(ValueError):
    """The source no longer proves the previous arrival boundary."""


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def message_hash(message: dict) -> str:
    return digest({k: message.get(k) for k in ('event_id', 'role', 'timestamp', 'origin', 'content')})


def snapshot_hash(snapshot: dict) -> str:
    """Exclude recipe/model/label/path provenance, bind all logical input fields."""
    return digest({'source_key': snapshot['source_key'], 'source_epoch': snapshot['source_epoch'],
                   'epoch_scope': snapshot.get('epoch_scope', snapshot['source_key']),
                   'scope_source_keys': snapshot.get('scope_source_keys', [snapshot['source_key']]),
                   'profile': snapshot.get('profile', 'default'),
                   'messages': [{k: m.get(k) for k in ('event_id', 'role', 'timestamp', 'origin', 'content')}
                                for m in snapshot['messages']]})


def validate_snapshot(snapshot: dict) -> None:
    for field in ('source_key', 'source_epoch'):
        if not isinstance(snapshot.get(field), str) or not snapshot[field] or any(ord(c) < 32 for c in snapshot[field]):
            raise ValueError(f'invalid {field}')
    before, after = snapshot.get('source_hash_before'), snapshot.get('source_hash_after')
    if not isinstance(before, str) or len(before) != 64 or before != after:
        raise SourceChangedError('source_changed: unstable source fence')
    if snapshot.get('snapshot_hash') != snapshot_hash(snapshot):
        raise ValueError('snapshot hash mismatch')
    seen = set()
    for msg in snapshot['messages']:
        event = msg.get('event_id')
        if not isinstance(event, str) or not event or event in seen:
            raise ValueError('invalid or duplicate event identity')
        seen.add(event)
        if msg.get('role') not in {'user', 'assistant'} or not isinstance(msg.get('content'), str):
            raise ValueError('invalid cleaned message')
        if msg.get('timestamp') in (None, '') and not msg.get('origin'):
            raise SourceChangedError('source_changed: unprovable message identity')


def checked_path(path: Path, *, root: Path | None = None) -> Path:
    if any(ord(c) < 32 or ord(c) == 127 for c in os.fspath(path)):
        raise SourceChangedError('source_changed: path controls')
    path = Path(os.path.abspath(path))
    if root is not None and not path.is_relative_to(root):
        raise SourceChangedError('source_changed: path escape')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise SourceChangedError('source_changed: symlink source')
    return path


def _read_bytes(path: Path) -> bytes:
    checked_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise SourceChangedError('source_changed: nonregular source')
        return handle.read()


def _read_json(path: Path) -> tuple[dict, str]:
    try:
        before = _read_bytes(path)
        value = json.loads(before)
        after = _read_bytes(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceChangedError('source_changed: invalid or missing JSON') from exc
    if before != after:
        raise SourceChangedError('source_changed: file changed during scan')
    if not isinstance(value, dict) or not isinstance(value.get('messages', []), list):
        raise SourceChangedError('source_changed: invalid session schema')
    return value, hashlib.sha256(before).hexdigest()


def _project_artifact(data: dict) -> dict:
    # Keep the raw fence hash, but not tools, prompts or reasoning metadata
    # that this adapter never consumes. Holding every full export evicts the
    # filesystem cache before the SQLite identity scan even starts.
    fields = ('session_id', 'id', 'session_start', 'started_at', 'parent_session_id',
              'end_reason', 'parent_end_reason', 'lineage_root', 'platform')
    message_fields = ('role', 'content', '_compressed_summary', 'timestamp', 'created_at',
                      'origin_message_id', 'platform_message_id', 'original_id', 'message_id', 'event_id')
    messages = []
    for raw in data.get('messages', []):
        if not isinstance(raw, dict):
            raise SourceChangedError('source_changed: invalid message schema')
        if raw.get('role') in {'user', 'assistant'} and not raw.get('_compressed_summary'):
            messages.append({key: raw[key] for key in message_fields if key in raw})
    return {**{key: data[key] for key in fields if key in data}, 'messages': messages}


def _timestamp(value):
    if value in (None, ''):
        return None
    try:
        # JSON snapshots sometimes stringify the SQLite epoch value.
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        try:
            parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError as exc:
            raise SourceChangedError('source_changed: invalid timestamp identity') from exc
        if parsed.tzinfo is None:
            raise SourceChangedError('source_changed: timestamp has no timezone')
        return parsed.astimezone(timezone.utc).isoformat()


def _message(raw: dict, *, codex=False) -> dict | None:
    if raw.get('role') not in {'user', 'assistant'} or raw.get('_compressed_summary'):
        return None
    content = extract_message_text(raw)
    if is_noise_message(content) or SECRET_MATERIAL_RE.search(content):
        return None
    if codex and content.startswith(('# AGENTS.md instructions', '<INSTRUCTIONS>',
                                    '<environment_context>', '<user_instructions>', '<turn_context>')):
        return None
    timestamp = _timestamp(raw.get('timestamp') if raw.get('timestamp') is not None else raw.get('created_at'))
    origin = (raw.get('origin_message_id') or raw.get('platform_message_id') or raw.get('original_id')
              or raw.get('message_id') or raw.get('event_id'))
    if not timestamp and not origin:
        raise SourceChangedError('source_changed: message identity needs timestamp or original origin')
    identity = {'role': raw['role'], 'timestamp': timestamp, 'origin': str(origin) if origin else None,
                'content': content}
    # A native origin identifies an event, not a version of its mutable body.
    # The source_key supplies the session/lineage namespace; message_hash binds
    # payload separately so a stale artifact cannot mask a rewritten origin.
    event = digest({'origin': str(origin)}) if origin else digest(identity)
    return {'event_id': event, **identity}


def _decode_db_content(content):
    """Use core's NUL sentinel codec, with its explicit script-only equivalent.

    Bare JSON-looking strings are ordinary user text, not a serialization tag.
    Invalid tagged data blocks capture rather than leaking reasoning as prose.
    """
    prefix = '\x00json:'
    if not isinstance(content, str) or not content.startswith(prefix):
        return content
    try:
        from hermes_state import SessionDB
    except ImportError:
        try:
            decoded = json.loads(content[len(prefix):])
        except (ValueError, TypeError) as exc:
            raise SourceChangedError('source_changed: invalid typed content codec') from exc
    else:
        decoded = SessionDB._decode_content(content)
    if decoded == content:
        raise SourceChangedError('source_changed: invalid typed content codec')
    return decoded


def _merge_lineage(sessions, artifacts):
    evidence = {str(sid): dict(data) for sid, data in sessions.items()}
    for sid, data, _raw_hash in artifacts:
        current = evidence.setdefault(sid, {})
        fields = {'parent_session_id': data.get('parent_session_id'), 'end_reason': data.get('end_reason'),
                  'started_at': data.get('session_start', data.get('started_at'))}
        for key, value in fields.items():
            if value in (None, ''):
                continue
            if key == 'started_at' and sid in sessions and current.get(key) is not None:
                # DB epoch owns this feed; export timestamps are provenance.
                continue
            if current.get(key) not in (None, '', value):
                # Numeric and ISO representations of a start are equivalent.
                if key != 'started_at' or _timestamp(current[key]) != _timestamp(value):
                    raise SourceChangedError('source_changed: conflicting lineage evidence')
            current[key] = value
    return evidence


def _independent_parent_edge(sid, evidence):
    """Core's stable delegation/branch/reset marker names the direct parent.

    Compression descendants inherit these markers, so presence alone is not
    enough: only an exact direct-parent match identifies a non-copy edge.
    """
    row = evidence.get(sid, {})
    config = row.get('model_config')
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except (ValueError, TypeError):
            return False
    parent = row.get('parent_session_id')
    return (isinstance(config, dict) and isinstance(parent, str) and bool(parent)
            and any(config.get(key) == parent for key in
                    ('_delegate_from', '_branched_from', '_reset_from')))


def _lineage_root(sid, evidence):
    current, seen = sid, set()
    while evidence[current].get('parent_session_id'):
        parent = str(evidence[current]['parent_session_id'])
        if parent in seen or parent == current:
            raise SourceChangedError('source_changed: cyclic compression lineage')
        if parent in evidence and _independent_parent_edge(current, evidence):
            break
        if parent not in evidence or not evidence[parent].get('end_reason'):
            raise SourceChangedError('source_changed: missing lineage parent/end_reason proof')
        seen.add(current)
        if evidence[parent]['end_reason'] != 'compression':
            break
        current = parent
    return current


def _add_event(view, message):
    previous = view.get(message['event_id'])
    if previous is not None and message_hash(previous) != message_hash(message):
        raise SourceChangedError('source_changed: conflicting versions of stable event origin')
    view[message['event_id']] = message


def _proved_db_clone(left, right, evidence, *, cutover_anchors=()):
    if left['copy_hash'] != right['copy_hash']:
        return False
    a, b = left['session_id'], right['session_id']
    if a == b:
        # Archive/live flags alone do not prove copying: a later independent
        # insert can have the same body/time. Core carried-tail originals are
        # rewind-only and excluded by the active/compacted visibility query.
        return False
    current, seen = b, set()
    while current not in cutover_anchors:
        if current in seen:
            raise SourceChangedError('source_changed: cyclic clone lineage')
        seen.add(current)
        if current not in evidence:
            raise SourceChangedError('source_changed: missing clone lineage proof')
        if _independent_parent_edge(current, evidence):
            return False
        parent = evidence[current].get('parent_session_id')
        if not parent:
            break
        parent = str(parent)
        if parent not in evidence:
            raise SourceChangedError('source_changed: missing clone lineage proof')
        if evidence[parent].get('end_reason') != 'compression':
            break
        if parent == a:
            return True
        current = parent
    return False


def _db_identity(path: Path):
    # Reuse the core inode-generation probe when installed in the repo; the
    # script-only bundle has the identical stdlib stat fallback, not SessionDB.
    try:
        from hermes_state_common import stat_db_file_identity
    except ImportError:
        st = path.stat()
        identity = (st.st_dev, st.st_ino) if st.st_dev and st.st_ino else None
    else:
        identity = stat_db_file_identity(path)
    if identity is None:
        raise SourceChangedError('source_changed: unknown SQLite epoch')
    return identity


def _read_db(path: Path, artifacts=()) -> tuple[dict, dict, str]:
    checked_path(path)
    for suffix in ('-wal', '-shm', '-journal'):
        checked_path(Path(str(path) + suffix))
    identity = _db_identity(path)
    roots, grouped, physical_rows = {}, defaultdict(dict), defaultdict(dict)
    try:
        con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        try:
            con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN')
            sessions = {r['id']: dict(r) for r in con.execute('SELECT * FROM sessions')}
            for session in sessions.values():
                session['model_config'] = _native_config_projection(session.pop('model_config', None))
            application_id = con.execute('PRAGMA application_id').fetchone()[0]
            epoch = 'sqlite:' + digest({'inode': identity, 'application_id': application_id})
            copy_roots, copy_sources, receipt_hash = _native_copy_bindings(con, sessions, epoch)
            columns = {r[1] for r in con.execute('PRAGMA table_info(messages)')}
            active = ' AND (active = 1 OR compacted = 1)' if {'active', 'compacted'} <= columns else ''
            evidence = _merge_lineage(sessions, artifacts)
            # Keep the all-role orphan check without loading tool bodies. The
            # session-ID index covers normal histories; visibility is needed
            # only for unknown sessions, including hidden legacy rows.
            for record in con.execute('SELECT DISTINCT session_id FROM messages'):
                sid = record[0]
                if sid not in sessions and con.execute(
                        'SELECT 1 FROM messages WHERE session_id IS ?' + active + ' LIMIT 1',
                        (sid,)).fetchone() is not None:
                    raise SourceChangedError('source_changed: missing session for message')
            # Filter roles before materializing payloads, then iterate inside
            # the same read-only transaction instead of retaining all history.
            # Hidden unproven rows still falsify a receipt-delete downgrade: a
            # visible row whose complete identity repeats a hidden unmatched
            # row cannot be proven a new occurrence. Collect the cheap identity
            # tuple of every yieldable row first, decode only hidden rows that
            # share such a tuple, and fail closed on a full identity match.
            cheap_columns = [c for c in ('id', 'session_id', 'role', 'active', 'compacted', 'timestamp',
                'created_at', 'tool_call_id', 'tool_calls', 'tool_name', '_compressed_summary')
                if c in columns]
            cheap_columns += [c for c in _NATIVE_ORIGIN_FIELDS if c in columns]
            cheap_columns = sorted(set(cheap_columns))
            cheap_select = ', '.join(cheap_columns)
            # Prefilter only: mutable visibility/display fields and the row id
            # are excluded; the full event identity is confirmed after decode.
            tuple_columns = [c for c in cheap_columns if c not in {'id', 'active', 'compacted', 'display_order'}]

            def identity_tuple(row):
                return tuple(row.get(column) for column in tuple_columns)

            yieldable_tuples, hidden_candidates = set(), []
            for record in con.execute('SELECT ' + cheap_select + ' FROM messages ORDER BY id'):
                row = dict(record)
                if row.get('role') not in {'user', 'assistant'} or row.get('_compressed_summary'):
                    continue
                if row.get('id') in copy_roots:
                    continue
                sid = row.get('session_id')
                if sid not in sessions or sessions[sid].get('source') == 'cron':
                    continue
                visible = not active or row.get('active') == 1 or row.get('compacted') == 1
                if visible or row.get('id') in copy_sources:
                    yieldable_tuples.add(identity_tuple(row))
                else:
                    hidden_candidates.append((row.get('id'), identity_tuple(row)))

            hidden_events = defaultdict(set)
            candidate_ids = [row_id for row_id, value in hidden_candidates if value in yieldable_tuples]
            for start in range(0, len(candidate_ids), 250):
                chunk = candidate_ids[start:start + 250]
                for record in con.execute('SELECT * FROM messages WHERE id IN ('
                                          + ','.join('?' for _ in chunk) + ')', chunk):
                    row = dict(record)
                    sid = row['session_id']
                    if sid not in sessions or sessions[sid].get('source') == 'cron':
                        continue
                    msg = _message({**row, 'content': _decode_db_content(row.get('content'))})
                    if msg and not msg['origin']:
                        hidden_events[roots.setdefault(sid, _lineage_root(sid, evidence))].add(msg['event_id'])

            for record in con.execute("SELECT * FROM messages WHERE role IN ('user','assistant') ORDER BY id"):
                row = dict(record)
                if row['id'] in copy_roots:
                    continue
                visible = not active or row['active'] == 1 or row['compacted'] == 1
                if not (visible or row['id'] in copy_sources):
                    continue
                sid = row['session_id']
                if sid not in sessions:
                    raise SourceChangedError('source_changed: missing session for message')
                if sessions[sid].get('source') == 'cron':
                    continue
                if row.get('_compressed_summary'):
                    continue
                # Integer row IDs are not message origins: protected tails are cloned.
                msg = _message({**row, 'content': _decode_db_content(row.get('content'))})
                if msg:
                    binding = _clone_binding(_physical_record(row, sqlite_row=True))
                    root = roots.setdefault(sid, _lineage_root(sid, evidence))
                    previous = physical_rows[root].get(msg['event_id'])
                    if previous is not None and not msg['origin'] and not _proved_db_clone(previous, binding, evidence):
                        raise SourceChangedError('source_changed: ambiguous identical SQLite rows without clone proof')
                    if not msg['origin'] and msg['event_id'] in hidden_events.get(root, ()):
                        raise SourceChangedError('source_changed: ambiguous hidden SQLite twin without copy proof')
                    _add_event(grouped[root], msg)
                    physical_rows[root][msg['event_id']] = binding
            # Read-only epoch token plus ledger content-coverage detects truncate /
            # reset even when a writer reuses this inode. No write of an epoch key.
            application_id = con.execute('PRAGMA application_id').fetchone()[0]
            con.rollback()
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise SourceChangedError('source_changed: invalid SQLite source') from exc
    if _db_identity(path) != identity:
        raise SourceChangedError('source_changed: SQLite replaced during snapshot')
    epoch = 'sqlite:' + digest({'inode': identity, 'application_id': application_id})
    return evidence, {'roots': roots, 'messages': {r: list(view.values()) for r, view in grouped.items()},
                      'copy_receipts_sha256': receipt_hash}, epoch


def _discover_sources(home: Path, profile_mode: str) -> list[dict]:
    if profile_mode not in {'all', 'hindsight', 'none'}:
        raise ValueError('invalid profile_mode')
    profile_root = checked_path(home / 'profiles')
    if profile_root.exists():
        for directory in profile_root.iterdir():
            checked_path(directory, root=home)
            checked_path(directory / 'config.yaml', root=home)
    sources = discover_session_sources(sessions_dir=home / 'sessions', state_db=home / 'state.db',
                                       profile_root=profile_root, profile_mode=profile_mode)
    # Existing discovery is JSON-centric; include DB-only profiles as well.
    if profile_mode != 'none' and profile_root.exists():
        from hindsight_session_manifest import _read_profile_memory_provider
        known = {s['profile'] for s in sources}
        for directory in sorted(profile_root.iterdir()):
            if not directory.is_dir() or directory.name in known or not (directory / 'state.db').exists():
                continue
            if profile_mode == 'hindsight' and _read_profile_memory_provider(directory) != 'hindsight':
                continue
            sources.append({'profile': directory.name, 'sessions_dir': directory / 'sessions',
                            'state_db': directory / 'state.db'})
    return sources


_NO_SOURCE_UNKNOWN_POLICY = object()


def scan_source_snapshots(hermes_home: Path, *, profile_mode: str = 'all',
                          physical_baseline: dict | None = None,
                          source_unknown_policy=_NO_SOURCE_UNKNOWN_POLICY,
                          source_transition_certificates: list | None = None,
                          approved_transition_sha256s: list | None = None,
                          source_compaction_successors: dict | None = None) -> list[dict]:
    """Read a finite source view, never infer historical completion or CDC."""
    if physical_baseline is not None:
        return _scan_physical_sources(hermes_home, profile_mode, physical_baseline,
                                      source_unknown_policy=source_unknown_policy,
                                      source_transition_certificates=source_transition_certificates,
                                      approved_transition_sha256s=approved_transition_sha256s,
                                      source_compaction_successors=source_compaction_successors)
    if source_transition_certificates is not None or approved_transition_sha256s is not None:
        raise ValueError('source transition requires original physical baseline')
    if source_compaction_successors is not None:
        raise ValueError('compaction successors require original physical baseline')
    if source_unknown_policy is not _NO_SOURCE_UNKNOWN_POLICY:
        raise ValueError('source unknown policy requires original physical baseline')
    home = checked_path(Path(hermes_home))
    if not home.exists():
        return []
    sources = _discover_sources(home, profile_mode)
    result = []
    for source in sources:
        profile = source['profile']
        db_path = checked_path(Path(source['state_db']), root=home)
        directory = checked_path(Path(source['sessions_dir']), root=home)
        artifacts = []
        if directory.exists():
            for path in sorted(directory.glob('session_*.json')):
                checked_path(path, root=home)
                data, raw_hash = _read_json(path)
                if data.get('platform') == 'cron' or path.name.startswith('session_cron_'):
                    continue
                sid = str(data.get('session_id') or data.get('id') or '')
                if not sid:
                    raise SourceChangedError('source_changed: missing stable session identity')
                artifacts.append((sid, _project_artifact(data), raw_hash))
        sessions, database, db_epoch = {}, {'roots': {}, 'messages': {}}, None
        if db_path.exists():
            sessions, database, db_epoch = _read_db(db_path, artifacts)
        else:
            sessions = _merge_lineage({}, artifacts)
        views, hashes, epochs = defaultdict(dict), defaultdict(list), {}
        for root, messages in database['messages'].items():
            for msg in messages:
                _add_event(views[root], msg)
            epochs[root] = db_epoch
            hashes[root].append(digest(messages))
            if database.get('copy_receipts_sha256') is not None:
                hashes[root].append(database['copy_receipts_sha256'])
        for sid, data, raw_hash in artifacts:
            root = _lineage_root(sid, sessions)
            claimed = data.get('lineage_root')
            if claimed and claimed != root:
                raise SourceChangedError('source_changed: unproved lineage root')
            parent = data.get('parent_session_id')
            if parent and data.get('parent_end_reason') and data['parent_end_reason'] != sessions[str(parent)].get('end_reason'):
                raise SourceChangedError('source_changed: conflicting lineage parent proof')
            epoch = db_epoch or 'json:' + digest({'session': root, 'start': sessions[root].get('started_at')})
            if root in epochs and epochs[root] != epoch:
                raise SourceChangedError('source_changed: conflicting source epochs')
            epochs[root] = epoch
            codex = data.get('platform') == 'backfill-codex' or sid.startswith('codex-')
            artifact_events = set()
            for raw in data.get('messages', []):
                if not isinstance(raw, dict):
                    raise SourceChangedError('source_changed: invalid message schema')
                if raw.get('timestamp') in (None, '') and raw.get('created_at') in (None, ''):
                    content = extract_message_text(raw)
                    matches = [m for m in database['messages'].get(root, [])
                               if m['role'] == raw.get('role') and m['content'] == content]
                    unique = {m['event_id']: m for m in matches}
                    if len(unique) == 1:
                        proved = next(iter(unique.values()))
                        raw = {**raw, 'timestamp': proved['timestamp'], 'origin_message_id': proved['origin']}
                msg = _message(raw, codex=codex)
                if msg:
                    if msg['event_id'] in artifact_events:
                        raise SourceChangedError('source_changed: ambiguous identical events within artifact')
                    artifact_events.add(msg['event_id'])
                    _add_event(views[root], msg)
            hashes[root].append(raw_hash)
        scope_keys = sorted(f'{profile}:hermes:{root}' for root in views)
        if db_epoch and not views:
            # Empty is still an observable fence; otherwise an in-place reset
            # could hide the old scope and later replay under new session IDs.
            empty = {'source_key': f'{profile}:sqlite:empty', 'source_epoch': db_epoch,
                     'epoch_scope': f'{profile}:sqlite', 'scope_source_keys': [], 'profile': profile,
                     'kind': 'source', 'messages': [], 'source_hash_before': digest([]),
                     'source_hash_after': digest([])}
            empty['snapshot_hash'] = snapshot_hash(empty)
            result.append(empty)
        for root in sorted(views):
            # Preserve logical first-observed order; no ordering by event date.
            messages = list(views[root].values())
            fence_hash = digest(sorted(hashes[root]))
            snapshot = {'source_key': f'{profile}:hermes:{root}', 'source_epoch': epochs[root],
                        'epoch_scope': f'{profile}:sqlite' if db_epoch else f'{profile}:hermes:{root}',
                        'scope_source_keys': scope_keys if db_epoch else [f'{profile}:hermes:{root}'],
                        'profile': profile, 'cleaning_version': CLEANING_VERSION, 'kind': 'source',
                        'messages': messages, 'source_hash_before': fence_hash,
                        'source_hash_after': fence_hash}
            snapshot['snapshot_hash'] = snapshot_hash(snapshot)
            validate_snapshot(snapshot)
            result.append(snapshot)
    return result


PHYSICAL_BASELINE_SCHEMA = 'hindsight-physical-legacy-baseline/1'
SOURCE_UNKNOWN_POLICY_SCHEMA = 'hindsight-source-approved-unknown/1'
SOURCE_TRANSITION_SCHEMA = 'hindsight-approved-sqlite-header-transition/1'
COMPACTION_SUCCESSOR_SCHEMA = 'hindsight-source-approved-compaction-successor/1'
_MUTABLE_ROW_FIELDS = {'active', 'compacted', 'display_order'}
_NATIVE_ORIGIN_FIELDS = ('origin_message_id', 'platform_message_id', 'original_id', 'message_id', 'event_id')


def _native_origin(raw):
    return next((raw.get(k) for k in _NATIVE_ORIGIN_FIELDS if raw.get(k)), None)


def _sql_hash(value):
    # display_identity is a BLOB in core. Bind its exact bytes without storing
    # them (or any raw content) in the exclusion artifact.
    if isinstance(value, bytes):
        return {'sqlite_blob': value.hex()}
    if isinstance(value, dict):
        return {k: _sql_hash(v) for k, v in value.items()}
    return value


def _raw_export_binding(raw, *, sqlite_content=False):
    if not isinstance(raw, dict):
        return None
    content = raw.get('content')
    if sqlite_content and isinstance(content, str) and content.startswith('\x00json:'):
        try:
            content = json.loads(content[len('\x00json:'):])
        except (ValueError, TypeError):
            # This is only optional copy evidence, not historical decoding.
            return None
    timestamp = raw.get('timestamp') if raw.get('timestamp') is not None else raw.get('created_at')
    if isinstance(timestamp, (int, float)):
        timestamp = float(timestamp)
    origin = _native_origin(raw)
    return digest(_sql_hash({'role': raw.get('role'), 'timestamp': timestamp, 'content': content,
                             'origin': str(origin) if origin else None,
                             '_compressed_summary': bool(raw.get('_compressed_summary'))}))


def _physical_record(raw, *, sqlite_row=False):
    if sqlite_row:
        # Mutable columns do not enter the emitted hashes, but still require
        # the same canonical-JSON validity as the former full-row pass.
        canonical_json(_sql_hash({k: v for k, v in raw.items() if k in _MUTABLE_ROW_FIELDS}))
        immutable = {k: v for k, v in raw.items() if k not in _MUTABLE_ROW_FIELDS}
        origin = _native_origin(raw)
        return {'id': raw['id'], 'session_id': raw['session_id'],
                'row_hash': digest(_sql_hash(immutable)),
                'copy_hash': digest(_sql_hash({k: v for k, v in immutable.items() if k not in {'id', 'session_id'}})),
                'export_hash': _raw_export_binding(raw, sqlite_content=True),
                'origin_hash': digest(str(origin)) if origin else None}
    return {'record_hash': digest(_sql_hash(raw)),
            'export_hash': _raw_export_binding(raw),
            'origin_hash': digest(str(_native_origin(raw))) if isinstance(raw, dict) and _native_origin(raw) else None}


def _clone_binding(record, row=None):
    # Duplicate/lineage checks need complete copy hashes, not retained bodies.
    # The optional row adds visibility flags for approved successor matching;
    # the clone proof itself compares only the copy hash.
    binding = {k: record[k] for k in ('id', 'session_id', 'copy_hash')}
    if row is not None:
        binding.update({'row_hash': record['row_hash'],
                        'active': row.get('active'), 'compacted': row.get('compacted')})
    return binding


def _location_identity(path: Path, *, directory=False):
    checked_path(path)
    try:
        st = path.stat()
    except OSError as exc:
        raise SourceChangedError('source_changed: missing physical source') from exc
    if not (stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode)):
        raise SourceChangedError('source_changed: invalid physical source type')
    if not st.st_dev or not st.st_ino:
        raise SourceChangedError('source_changed: unknown physical source epoch')
    return [st.st_dev, st.st_ino]


_COPY_RECEIPTS_KEY = '_message_copy_receipts_v1'
_COPY_RECEIPTS_SCHEMA = 'hermes-message-copy-receipts/1'
_COPY_HASH_RE = re.compile(r'[0-9a-f]{64}\Z')


def _invalid_copy_receipts() -> NoReturn:
    # Fixed metadata only: never surface configuration, row values or bodies.
    raise SourceChangedError('source_changed: invalid native message copy receipts') from None


def _raw_mentions_copy_receipts(raw):
    """True when raw may carry the protected key in any JSON-escaped form.

    Malformed or aliased config is only tolerated for legacy keys; any encoded
    spelling of the owned key (\u005F escapes, BLOBs) must fail closed instead
    of silently projecting to an empty dict.
    """
    if isinstance(raw, bytes):
        if _COPY_RECEIPTS_KEY.encode('utf-8') in raw:
            return True
        try:
            raw = raw.decode('utf-8')
        except UnicodeDecodeError:
            return False
    if not isinstance(raw, str):
        return False
    if _COPY_RECEIPTS_KEY in raw:
        return True
    decoded = re.sub(r'\\u([0-9a-fA-F]{4})', lambda match: chr(int(match.group(1), 16)), raw)
    return _COPY_RECEIPTS_KEY in decoded


def _native_config_projection(raw):
    if isinstance(raw, str):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate metadata key')
                result[key] = value
            return result
        try:
            config = json.loads(raw, object_pairs_hook=unique_object) if raw.strip() else {}
        except (ValueError, TypeError):
            if _raw_mentions_copy_receipts(raw):
                _invalid_copy_receipts()
            config = None
    else:
        config = raw
    if type(config) is not dict:
        if _raw_mentions_copy_receipts(raw):
            _invalid_copy_receipts()
        return {}
    return {key: config[key] for key in
            ('_delegate_from', '_branched_from', '_reset_from', _COPY_RECEIPTS_KEY) if key in config}


def _native_copy_bindings(con, sessions, epoch):
    """Verify only receipt-referenced rows in this same ro/query_only snapshot.

    Standalone scripts cannot import Core's writer helper. Hash complete rows
    immediately, retaining only content-free bindings; query in bounded chunks.
    Native AUTOINCREMENT edges must point forward, so no cycle is admissible.
    """
    edges, operation_ids, targets, receipts = [], set(), set(), {}

    def shape(value, keys):
        if type(value) is not dict or set(value) != set(keys):
            _invalid_copy_receipts()

    def checksum(value):
        if type(value) is not str or not _COPY_HASH_RE.fullmatch(value):
            _invalid_copy_receipts()

    for sid, session in sessions.items():
        config = session.get('model_config', {})
        if _COPY_RECEIPTS_KEY not in config:
            continue
        value = config[_COPY_RECEIPTS_KEY]
        shape(value, ('schema_version', 'session_id', 'source_epoch', 'operations'))
        if (value['schema_version'] != _COPY_RECEIPTS_SCHEMA
                or type(value['session_id']) is not str or value['session_id'] != sid
                or type(value['source_epoch']) is not str or value['source_epoch'] != epoch
                or type(value['operations']) is not list):
            _invalid_copy_receipts()
        receipts[sid] = value
        for operation in value['operations']:
            shape(operation, ('operation_id', 'kind', 'copies', 'receipt_sha256'))
            op_id = operation['operation_id']
            try:
                parsed = uuid.UUID(op_id) if type(op_id) is str else None
            except (ValueError, AttributeError):
                _invalid_copy_receipts()
            if (parsed is None or not parsed.int or str(parsed) != op_id or op_id in operation_ids
                    or operation['kind'] != 'inplace_compaction'
                    or type(operation['copies']) is not list or not operation['copies']):
                _invalid_copy_receipts()
            operation_ids.add(op_id)
            checksum(operation['receipt_sha256'])
            for edge in operation['copies']:
                shape(edge, ('source_row_id', 'target_row_id', 'source_row_hash', 'target_row_hash', 'copy_hash'))
                old, new = edge['source_row_id'], edge['target_row_id']
                if type(old) is not int or type(new) is not int or not 0 < old < new or new in targets:
                    _invalid_copy_receipts()
                targets.add(new)
                for key in ('source_row_hash', 'target_row_hash', 'copy_hash'):
                    checksum(edge[key])
                edges.append((sid, edge))
            if operation['receipt_sha256'] != digest({k: v for k, v in operation.items() if k != 'receipt_sha256'}):
                _invalid_copy_receipts()
    referenced = sorted({edge[key] for _, edge in edges for key in ('source_row_id', 'target_row_id')})
    records = {}
    for start in range(0, len(referenced), 250):
        chunk = referenced[start:start + 250]
        for row in con.execute('SELECT * FROM messages WHERE id IN (' + ','.join('?' for _ in chunk) + ')', chunk):
            records[row['id']] = _physical_record(dict(row), sqlite_row=True)
    if set(records) != set(referenced):
        _invalid_copy_receipts()
    roots = {}
    for sid, edge in sorted(edges, key=lambda pair: pair[1]['target_row_id']):
        old, new = edge['source_row_id'], edge['target_row_id']
        left, right = records[old], records[new]
        if (left['session_id'] != sid or right['session_id'] != sid
                or left['row_hash'] != edge['source_row_hash'] or right['row_hash'] != edge['target_row_hash']
                or left['copy_hash'] != right['copy_hash'] or left['copy_hash'] != edge['copy_hash']):
            _invalid_copy_receipts()
        roots[new] = roots.get(old, old)
    return roots, set(roots.values()), digest(receipts) if receipts else None


def _native_session_binding(row):
    return digest({k: row.get(k) for k in ('id', 'started_at', 'parent_session_id', 'source')})


def _sqlite_structure_hash(con):
    return digest([list(row) for row in con.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")])


def inspect_sqlite_transition_header(path: Path):
    """Read the exact prospective header; this does not approve a transition."""
    path = checked_path(path)
    identity = _location_identity(path)
    con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        con.execute('PRAGMA query_only=ON')
        con.execute('BEGIN')
        app_id = con.execute('PRAGMA application_id').fetchone()[0]
        header = {'identity': identity, 'application_id': app_id,
            'epoch': 'sqlite:' + digest({'inode': identity, 'application_id': app_id}),
            'schema_version': con.execute('PRAGMA schema_version').fetchone()[0],
            'message_columns': [r[1] for r in con.execute('PRAGMA table_info(messages)')],
            'structure_sha256': _sqlite_structure_hash(con)}
        con.rollback()
    finally:
        con.close()
    if _location_identity(path) != identity:
        raise SourceChangedError('source_changed: SQLite replaced during transition inspection')
    return header


def _read_physical_db(path: Path, *, consume=None, include_structure=False):
    """Stream all roles in one mode=ro/query_only snapshot; never SessionDB."""
    identity = _location_identity(path)
    for suffix in ('-wal', '-shm', '-journal'):
        checked_path(Path(str(path) + suffix))
    try:
        con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        try:
            con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN')
            session_columns = {r[1] for r in con.execute('PRAGMA table_info(sessions)')}
            selected = 'id, started_at, parent_session_id, end_reason, source'
            if 'model_config' in session_columns:
                selected += ', model_config'
            sessions = {r['id']: dict(r) for r in con.execute('SELECT ' + selected + ' FROM sessions')}
            # Carry only stable relation markers and fixed-schema owned receipts.
            # Provider/credential/recipe configuration never enters the
            # physical descriptor or its returned source evidence.
            for row in sessions.values():
                row['model_config'] = _native_config_projection(row.pop('model_config', None))
            columns = [r[1] for r in con.execute('PRAGMA table_info(messages)')]
            application_id = con.execute('PRAGMA application_id').fetchone()[0]
            schema_version = con.execute('PRAGMA schema_version').fetchone()[0]
            epoch = 'sqlite:' + digest({'inode': identity, 'application_id': application_id})
            copy_roots, copy_sources, receipt_hash = _native_copy_bindings(con, sessions, epoch)
            header = {'path': str(path), 'identity': identity, 'epoch': epoch,
                      'application_id': application_id, 'schema_version': schema_version, 'message_columns': columns,
                      'sessions': [{'session_id': sid, 'identity_hash': _native_session_binding(row)}
                                   for sid, row in sorted(sessions.items())], 'rows': []}
            if include_structure:
                header['structure_sha256'] = _sqlite_structure_hash(con)
                header['native_copy_receipt_sessions'] = sum(
                    _COPY_RECEIPTS_KEY in row.get('model_config', {}) for row in sessions.values())
            if consume is not None:
                consume(None, sessions, {**header, 'copy_roots': copy_roots,
                        'copy_sources': copy_sources, 'copy_receipts_sha256': receipt_hash})
            for record in con.execute('SELECT * FROM messages ORDER BY id'):
                row = dict(record)
                if consume is None:
                    header['rows'].append(_physical_record(row, sqlite_row=True))
                else:
                    consume(row, sessions, header)
            con.rollback()
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise SourceChangedError('source_changed: invalid SQLite physical source') from exc
    if _location_identity(path) != identity:
        raise SourceChangedError('source_changed: SQLite replaced during physical snapshot')
    return header, sessions


def _artifact_binding(data):
    # Mutable counters/model/export metadata are not source identity. Full
    # raw messages remain prefix-bound separately, including unretained roles.
    return digest({k: data.get(k) for k in ('session_id', 'id', 'session_start', 'started_at',
                                           'parent_session_id', 'lineage_root', 'platform')})


def _physical_artifact(path: Path):
    identity = _location_identity(path)
    data, raw_hash = _read_json(path)
    if _location_identity(path) != identity:
        raise SourceChangedError('source_changed: JSON replaced during physical snapshot')
    sid = data.get('session_id') or data.get('id') or None
    records = [_physical_record(raw) for raw in data.get('messages', [])]
    value = {'path': str(path), 'identity': identity, 'session_id': sid,
             'identity_hash': _artifact_binding(data), 'raw_hash': raw_hash,
             'prefix_hash': digest([r['record_hash'] for r in records]), 'records': records}
    return value, data


def _feed_epoch(profile):
    return 'physical:' + digest({'profile': profile['profile'], 'state_db': profile['state_db'],
        'sqlite_epoch': profile['sqlite']['epoch'] if profile['sqlite'] else None,
        'sessions_dir': profile['sessions_dir'], 'directory_identity': profile['directory_identity']})


def capture_physical_baseline(hermes_home: Path, *, bank: str, profile_mode: str = 'all') -> dict:
    """Observe finite legacy locations; this proves exclusion, NOT completion."""
    home = checked_path(Path(hermes_home))
    profiles = []
    for source in _discover_sources(home, profile_mode):
        directory = checked_path(Path(source['sessions_dir']), root=home)
        db = checked_path(Path(source['state_db']), root=home)
        profile = {'profile': source['profile'], 'sessions_dir': str(directory), 'state_db': str(db),
                   'directory_identity': _location_identity(directory, directory=True) if directory.exists() else None,
                   'sqlite': _read_physical_db(db)[0] if db.exists() else None, 'artifacts': []}
        if directory.exists():
            for path in sorted(directory.glob('session_*.json')):
                checked_path(path, root=home)
                profile['artifacts'].append(_physical_artifact(path)[0])
        profile['feed_epoch'] = _feed_epoch(profile)
        profiles.append(profile)
    value = {'schema_version': PHYSICAL_BASELINE_SCHEMA, 'bank': bank, 'hermes_home': str(home),
             'status': 'legacy_unknown', 'completion_proven': False,
             'profile_mode': profile_mode, 'profiles': profiles}
    validate_physical_baseline(value, bank=bank, hermes_home=home)
    return value


def _valid_native(value, *, optional=False):
    return (optional and value is None) or (isinstance(value, str) and bool(value)
        and not any(ord(c) < 32 or ord(c) == 127 for c in value))


def _require_shape(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('invalid physical baseline shape')


def _require_hash(value, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid physical baseline hash')


def _require_identity(value, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, list) or len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
        raise ValueError('invalid physical baseline epoch identity')


def validate_physical_baseline(baseline: dict, *, bank: str, hermes_home: Path) -> None:
    """Validate content-free schema and exact bank/home/path/policy binding."""
    _require_shape(baseline, ('schema_version', 'bank', 'hermes_home', 'status',
                             'completion_proven', 'profile_mode', 'profiles'))
    home = checked_path(Path(hermes_home))
    if (not _valid_native(bank) or baseline['schema_version'] != PHYSICAL_BASELINE_SCHEMA
            or baseline['bank'] != bank or baseline['hermes_home'] != str(home)
            or baseline['status'] != 'legacy_unknown' or baseline['completion_proven'] is not False
            or not isinstance(baseline['profile_mode'], str) or baseline['profile_mode'] not in {'all', 'hindsight', 'none'}
            or not isinstance(baseline['profiles'], list)):
        raise ValueError('physical baseline bank/home/policy mismatch')
    names = set()
    for profile in baseline['profiles']:
        _require_shape(profile, ('profile', 'sessions_dir', 'state_db', 'directory_identity', 'sqlite', 'artifacts', 'feed_epoch'))
        name = profile['profile']
        if not _valid_native(name) or name in names or '/' in name or name in {'.', '..'}:
            raise ValueError('invalid physical baseline profile')
        names.add(name)
        base = home if name == 'default' else home / 'profiles' / name
        for key, expected in (('sessions_dir', base / 'sessions'), ('state_db', base / 'state.db')):
            if not isinstance(profile[key], str) or profile[key] != str(checked_path(expected, root=home)):
                raise ValueError('physical baseline source path mismatch')
        _require_identity(profile['directory_identity'], optional=True)
        db = profile['sqlite']
        if db is not None:
            _require_shape(db, ('path', 'identity', 'epoch', 'application_id', 'schema_version', 'message_columns', 'sessions', 'rows'))
            _require_identity(db['identity'])
            if (db['path'] != profile['state_db'] or type(db['application_id']) is not int or type(db['schema_version']) is not int
                    or db['epoch'] != 'sqlite:' + digest({'inode': db['identity'], 'application_id': db['application_id']})
                    or not isinstance(db['message_columns'], list) or not all(_valid_native(c) for c in db['message_columns'])
                    or len(set(db['message_columns'])) != len(db['message_columns'])
                    or not {'id', 'session_id', 'role', 'content'} <= set(db['message_columns'])
                    or not isinstance(db['sessions'], list) or not isinstance(db['rows'], list)):
                raise ValueError('invalid physical SQLite descriptor')
            session_ids, row_ids = set(), set()
            for session in db['sessions']:
                _require_shape(session, ('session_id', 'identity_hash'))
                if not _valid_native(session['session_id']) or session['session_id'] in session_ids:
                    raise ValueError('invalid physical session association')
                session_ids.add(session['session_id'])
                _require_hash(session['identity_hash'])
            for row in db['rows']:
                _require_shape(row, ('id', 'session_id', 'row_hash', 'copy_hash', 'export_hash', 'origin_hash'))
                if type(row['id']) is not int or row['id'] in row_ids or not _valid_native(row['session_id'], optional=True):
                    raise ValueError('invalid physical row location')
                row_ids.add(row['id'])
                for key in ('row_hash', 'copy_hash', 'export_hash', 'origin_hash'):
                    _require_hash(row[key], optional=key in {'export_hash', 'origin_hash'})
        if not isinstance(profile['artifacts'], list):
            raise ValueError('invalid physical artifact list')
        paths = set()
        for artifact in profile['artifacts']:
            _require_shape(artifact, ('path', 'identity', 'session_id', 'identity_hash', 'raw_hash', 'prefix_hash', 'records'))
            if not isinstance(artifact['path'], str):
                raise ValueError('invalid physical artifact path')
            path = checked_path(Path(artifact['path']), root=Path(profile['sessions_dir']))
            if (str(path) != artifact['path'] or path.parent != Path(profile['sessions_dir'])
                    or not path.match('session_*.json') or str(path) in paths
                    or not _valid_native(artifact['session_id'], optional=True)
                    or not isinstance(artifact['records'], list)):
                raise ValueError('invalid physical artifact location')
            paths.add(str(path))
            _require_identity(artifact['identity'])
            for key in ('identity_hash', 'raw_hash', 'prefix_hash'):
                _require_hash(artifact[key])
            for record in artifact['records']:
                _require_shape(record, ('record_hash', 'export_hash', 'origin_hash'))
                for key in record:
                    _require_hash(record[key], optional=key != 'record_hash')
            if artifact['prefix_hash'] != digest([r['record_hash'] for r in artifact['records']]):
                raise ValueError('physical artifact prefix hash mismatch')
        if profile['feed_epoch'] != _feed_epoch(profile):
            raise ValueError('physical feed epoch mismatch')
    if 'default' not in names:
        raise ValueError('physical baseline default profile missing')


def validate_source_unknown_policy(policy, *, physical_baseline, bank, hermes_home):
    """Only literal finite SQL bindings; this proves authorization, not copying.

    Deployment chooses the approved exact rows explicitly. No site IDs, ranges,
    time filters, or inferred completion enter this generic artifact schema.
    Sessions created after the descriptor stay admissible as future sessions;
    the exact row binding, drift and coverage checks at scan time remain the
    only consumption path, and baseline rows stay forbidden.
    """
    from hindsight_incremental_merge import canonical_hash

    validate_physical_baseline(physical_baseline, bank=bank, hermes_home=hermes_home)
    _require_shape(policy, ('schema_version', 'bank', 'hermes_home', 'physical_baseline_sha256',
                           'authorization', 'status', 'completion_proven', 'scopes'))
    if (policy['schema_version'] != SOURCE_UNKNOWN_POLICY_SCHEMA or policy['bank'] != bank
            or policy['hermes_home'] != physical_baseline['hermes_home']
            or policy['physical_baseline_sha256'] != canonical_hash(physical_baseline)
            or policy['authorization'] != 'explicit-finite-source-row-exclusion'
            or policy['status'] != 'unknown' or policy['completion_proven'] is not False
            or not isinstance(policy['scopes'], list) or not policy['scopes']):
        raise ValueError('source unknown policy baseline/bank/home/authorization mismatch')
    profiles = {p['profile']: p for p in physical_baseline['profiles']}
    scopes, row_ids = set(), set()
    for scope in policy['scopes']:
        _require_shape(scope, ('profile', 'feed_epoch', 'source_epoch', 'session_id', 'rows'))
        name, sid = scope['profile'], scope['session_id']
        if not _valid_native(name) or not _valid_native(sid) or (name, sid) in scopes:
            raise ValueError('invalid source unknown policy scope')
        scopes.add((name, sid))
        profile = profiles.get(name)
        if (profile is None or profile['sqlite'] is None
                or scope['feed_epoch'] != profile['feed_epoch']
                or scope['source_epoch'] != profile['sqlite']['epoch']
                or not isinstance(scope['rows'], list) or not scope['rows']):
            raise ValueError('source unknown policy feed/epoch/session mismatch')
        old_ids = {r['id'] for r in profile['sqlite']['rows']}
        ids = []
        for row in scope['rows']:
            _require_shape(row, ('id', 'session_id', 'row_hash', 'copy_hash', 'export_hash', 'origin_hash'))
            if (type(row['id']) is not int or row['id'] <= 0 or row['id'] in old_ids
                    or (name, row['id']) in row_ids or row['session_id'] != sid):
                raise ValueError('invalid source unknown policy row location')
            row_ids.add((name, row['id']))
            ids.append(row['id'])
            for key in ('row_hash', 'copy_hash', 'export_hash', 'origin_hash'):
                _require_hash(row[key], optional=key in {'export_hash', 'origin_hash'})
        if ids != sorted(ids):
            raise ValueError('source unknown policy rows must have canonical order')


def validate_source_transition_certificate(certificate, *, physical_baseline,
        source_unknown_policy, approved_transition_sha256):
    """The caller must pin a separately approved digest, not trust an approval flag.

    Each certificate aliases one exact original baseline header to one exact
    new header (a replacement or an in-place schema rebuild). Baseline rows,
    policy rows, lineage, receipts and ledger epochs stay intact.
    """
    from hindsight_incremental_merge import canonical_hash

    _require_hash(approved_transition_sha256)
    if canonical_hash(certificate) != approved_transition_sha256:
        raise ValueError('source transition approved digest mismatch')
    _require_shape(certificate, ('schema_version', 'bank', 'hermes_home', 'profile', 'state_db',
        'physical_baseline_sha256', 'source_unknown_policy_sha256', 'authorization',
        'approval_reference', 'continuity_proof_sha256', 'native_copy_receipt_precondition', 'old', 'new'))
    profiles = {p['profile']: p for p in physical_baseline['profiles']}
    profile = profiles.get(certificate['profile'])
    old = profile['sqlite'] if profile is not None else None
    expected_policy = (None if source_unknown_policy is _NO_SOURCE_UNKNOWN_POLICY
                       else canonical_hash(source_unknown_policy))
    fields = ('identity', 'epoch', 'schema_version', 'application_id', 'message_columns')
    if (certificate['schema_version'] != SOURCE_TRANSITION_SCHEMA
            or certificate['bank'] != physical_baseline['bank']
            or certificate['hermes_home'] != physical_baseline['hermes_home']
            or profile is None or certificate['state_db'] != profile['state_db']
            or certificate['physical_baseline_sha256'] != canonical_hash(physical_baseline)
            or certificate['source_unknown_policy_sha256'] != expected_policy
            or certificate['authorization'] != 'explicit-exact-sqlite-header-transition'
            or not _valid_native(certificate['approval_reference']) or old is None):
        raise ValueError('source transition scope/authorization mismatch')
    _require_hash(certificate['continuity_proof_sha256'])
    receipt = certificate['native_copy_receipt_precondition']
    if receipt.get('policy') == 'current-epoch-superset-inventory':
        _require_shape(receipt, ('policy', 'sessions', 'sha256', 'operations'))
        if (type(receipt['sessions']) is not int or receipt['sessions'] < 0
                or (receipt['sessions'] == 0) != (receipt['sha256'] is None)
                or type(receipt['operations']) is not dict
                or len(receipt['operations']) > receipt['sessions']):
            raise ValueError('source transition native receipt precondition mismatch')
        _require_hash(receipt['sha256'], optional=True)
        for sid, operations in receipt['operations'].items():
            if not _valid_native(sid) or type(operations) is not dict or not operations:
                raise ValueError('source transition native receipt precondition mismatch')
            for op_id, op_digest in operations.items():
                try:
                    parsed = uuid.UUID(op_id) if type(op_id) is str else None
                except (ValueError, AttributeError):
                    parsed = None
                if parsed is None or not parsed.int or str(parsed) != op_id:
                    raise ValueError('source transition native receipt precondition mismatch')
                _require_hash(op_digest)
    else:
        _require_shape(receipt, ('policy', 'sessions', 'sha256'))
        if (receipt['policy'] != 'current-epoch-exact-inventory'
                or type(receipt['sessions']) is not int or receipt['sessions'] < 0
                or (receipt['sessions'] == 0) != (receipt['sha256'] is None)):
            raise ValueError('source transition native receipt precondition mismatch')
        _require_hash(receipt['sha256'], optional=True)
    _require_shape(certificate['old'], fields)
    _require_shape(certificate['new'], (*fields, 'structure_sha256'))
    if certificate['old'] != {k: old[k] for k in fields}:
        raise ValueError('source transition original header mismatch')
    new = certificate['new']
    _require_identity(new['identity'])
    _require_hash(new['structure_sha256'])
    if (new['application_id'] != old['application_id']
            or new['epoch'] != 'sqlite:' + digest({'inode': new['identity'], 'application_id': new['application_id']})
            or type(new['schema_version']) is not int or new['schema_version'] < 0
            or new['message_columns'] != old['message_columns']):
        raise ValueError('source transition replacement header mismatch')
    return certificate


def validate_source_transition_certificate_set(certificates, *, physical_baseline,
        source_unknown_policy, approved_transition_sha256s):
    """Match every reviewed certificate to its own independently pinned digest.

    The caller supplies both lists explicitly; nothing is looked up here.
    Each certificate must match exactly one approved digest, every approved
    digest must be consumed by exactly one certificate, and each profile may
    appear at most once. Any profile without a certificate stays strict.
    """
    if certificates is None or approved_transition_sha256s is None:
        raise ValueError('source transition requires certificates and approved digests together')
    if not isinstance(certificates, (list, tuple)) or not isinstance(approved_transition_sha256s, (list, tuple)):
        raise ValueError('source transition certificates and digests must be explicit lists')
    if not certificates or not approved_transition_sha256s:
        raise ValueError('source transition requires certificates and approved digests together')
    from hindsight_incremental_merge import canonical_hash
    approved = list(approved_transition_sha256s)
    for value in approved:
        _require_hash(value)
    if len(set(approved)) != len(approved):
        raise ValueError('duplicate approved source transition digest')
    matched = {}
    for certificate in certificates:
        certificate_hash = canonical_hash(certificate)
        if certificate_hash not in approved:
            raise ValueError('source transition approved digest mismatch')
        if certificate_hash in matched:
            raise ValueError('duplicate source transition certificate')
        matched[certificate_hash] = certificate
    if set(matched) != set(approved):
        raise ValueError('source transition approved digest unused')
    by_profile = {}
    for certificate in certificates:
        validate_source_transition_certificate(certificate, physical_baseline=physical_baseline,
            source_unknown_policy=source_unknown_policy,
            approved_transition_sha256=canonical_hash(certificate))
        profile = certificate['profile']
        if profile in by_profile:
            raise ValueError('duplicate source transition profile')
        by_profile[profile] = certificate
    return by_profile


def validate_source_compaction_successors(policy, *, physical_baseline, bank, hermes_home):
    """Finite explicit approvals for compressor archive/successor pairs.

    Each pinned pair names one archived original row and the exact surviving
    successor row. The scanner admits only pairs whose complete row hashes
    and visibility flags match the pin; anything else stays ambiguous.
    Nothing is looked up here; the trusted caller supplies it explicitly.
    """
    from hindsight_incremental_merge import canonical_hash

    _require_shape(policy, ('schema_version', 'bank', 'hermes_home', 'physical_baseline_sha256',
        'authorization', 'status', 'completion_proven', 'pairs'))
    if (policy['schema_version'] != COMPACTION_SUCCESSOR_SCHEMA
            or policy['bank'] != bank or policy['hermes_home'] != str(hermes_home)
            or policy['physical_baseline_sha256'] != canonical_hash(physical_baseline)
            or policy['authorization'] != 'explicit-finite-compaction-successor-admission'
            or policy['status'] != 'approved' or policy['completion_proven'] is not False
            or type(policy['pairs']) is not list):
        raise ValueError('compaction successor policy binding mismatch')
    profiles = {p['profile'] for p in physical_baseline['profiles']}
    seen = set()
    for pair in policy['pairs']:
        _require_shape(pair, ('profile', 'session_id', 'event_id', 'message_hash',
            'original', 'successor', 'decision'))
        if (pair['profile'] not in profiles or not _valid_native(pair['session_id'])
                or not _valid_native(pair['event_id']) or not _valid_native(pair['message_hash'])
                or pair['decision'] != 'admit-successor-as-event'):
            raise ValueError('compaction successor policy pair mismatch')
        for side in ('original', 'successor'):
            _require_shape(pair[side], ('row_id', 'row_hash', 'copy_hash', 'active', 'compacted'))
            if (type(pair[side]['row_id']) is not int or pair[side]['row_id'] <= 0
                    or not _valid_native(pair[side]['row_hash'])
                    or not _valid_native(pair[side]['copy_hash'])
                    or pair[side]['active'] not in (0, 1)
                    or pair[side]['compacted'] not in (0, 1)):
                raise ValueError('compaction successor policy pair mismatch')
        if pair['original']['row_id'] >= pair['successor']['row_id']:
            raise ValueError('compaction successor policy pair mismatch')
        key = (pair['profile'], pair['session_id'], pair['event_id'])
        if key in seen:
            raise ValueError('compaction successor policy duplicate pair')
        seen.add(key)
    return policy


def _approved_compaction_successor(previous, binding, approved_successors, session_id, event_id):
    """Admit exactly the pinned successor of an archived original row."""
    if not approved_successors:
        return False
    pair = approved_successors.get((session_id, event_id))
    if pair is None:
        return False
    for side, seen_row in (('original', previous), ('successor', binding)):
        expected = pair[side]
        if (seen_row.get('id') != expected['row_id'] or seen_row.get('row_hash') != expected['row_hash']
                or seen_row.get('copy_hash') != expected['copy_hash']
                or seen_row.get('active') != expected['active']
                or seen_row.get('compacted') != expected['compacted']):
            return False
    return True


def _profile_anchors(profile):
    sids = {a['session_id'] for a in profile['artifacts'] if a['session_id']}
    if profile['sqlite']:
        sids.update(s['session_id'] for s in profile['sqlite']['sessions'])
        sids.update(r['session_id'] for r in profile['sqlite']['rows'] if r['session_id'])
    return sids


def _physical_snapshots(profile, views=None, *, fence_hash=None):
    views = {} if views is None else views
    anchors = _profile_anchors(profile)
    keys = {sid: f"{profile['profile']}:cutover-session:{sid}" for sid in anchors}
    keys.update({sid: f"{profile['profile']}:hermes:{sid}" for sid in views if sid not in anchors})
    if not anchors:
        keys[None] = f"{profile['profile']}:physical:empty"
    scope = sorted(keys.values())
    # The content-free descriptor fence is sufficient for empty anchors; future
    # payloads bind their own cleaned logical identities in snapshot_hash.
    fence = digest(profile) if fence_hash is None else fence_hash
    snapshots = []
    for sid, key in sorted(keys.items(), key=lambda item: item[1]):
        snapshot = {'source_key': key, 'source_epoch': profile['feed_epoch'],
            'epoch_scope': f"{profile['profile']}:physical-feed", 'scope_source_keys': scope,
            'profile': profile['profile'], 'kind': 'source', 'cleaning_version': CLEANING_VERSION,
            'messages': list(views.get(sid, {}).values()),
            'source_hash_before': fence, 'source_hash_after': fence}
        snapshot['snapshot_hash'] = snapshot_hash(snapshot)
        validate_snapshot(snapshot)
        snapshots.append(snapshot)
    return snapshots


def physical_baseline_snapshots(baseline: dict) -> list[dict]:
    if not isinstance(baseline, dict) or not isinstance(baseline.get('hermes_home'), str):
        raise ValueError('invalid physical baseline header')
    validate_physical_baseline(baseline, bank=baseline.get('bank', ''), hermes_home=Path(baseline['hermes_home']))
    return [snapshot for profile in baseline['profiles'] for snapshot in _physical_snapshots(profile)]


def _physical_lineage(sessions, artifacts, anchors):
    # Historical exports can disagree with native parents. Exact old exclusion
    # must not require reconstructing that history. Surface conflicts only when
    # they are used as forward compression proof; new SID evidence stays strict.
    evidence = {sid: dict(row) for sid, row in sessions.items()}
    for sid, data, raw_hash in artifacts:
        if sid not in anchors:
            evidence = _merge_lineage(evidence, [(sid, data, raw_hash)])
            continue
        current = evidence.setdefault(sid, {})
        for field in ('parent_session_id', 'end_reason'):
            value = data.get(field)
            if value in (None, ''):
                continue
            if current.get(field) not in (None, '', value):
                if field == 'end_reason':
                    current['_forward_conflict'] = True
            else:
                current[field] = value
    return evidence


def _cutover_root(sid, evidence, anchors):
    """Old native SID is a policy anchor, never a guessed historical root."""
    current, seen = sid, set()
    while current not in anchors and evidence.get(current, {}).get('parent_session_id'):
        if current in seen:
            raise SourceChangedError('source_changed: cyclic forward lineage')
        seen.add(current)
        parent = str(evidence[current]['parent_session_id'])
        if parent == current or parent in seen:
            raise SourceChangedError('source_changed: cyclic forward lineage')
        if parent in evidence and _independent_parent_edge(current, evidence):
            break
        if evidence[current].get('_forward_conflict') or evidence.get(parent, {}).get('_forward_conflict'):
            raise SourceChangedError('source_changed: conflicting forward lineage proof')
        if parent not in evidence or not evidence[parent].get('end_reason'):
            raise SourceChangedError('source_changed: missing forward lineage parent/end_reason proof')
        if evidence[parent]['end_reason'] != 'compression':
            break
        current = parent
    return current


def _compression_copy(old_sid, new_sid, evidence, *, cutover_anchors):
    if old_sid == new_sid:
        return False
    current, seen = new_sid, set()
    while current not in seen and current not in cutover_anchors:
        seen.add(current)
        if _independent_parent_edge(current, evidence):
            return False
        parent = evidence.get(current, {}).get('parent_session_id')
        if not parent or parent not in evidence or evidence[parent].get('end_reason') != 'compression':
            return False
        if parent == old_sid:
            return True
        current = parent
    return False


def _exclude_old_copy(record, sid, old_index, evidence, *, cutover_anchors,
                      sqlite_row=False, copy_index=None):
    candidates = ((copy_index or {}).get(record['copy_hash'], []) if sqlite_row
                  else old_index.get(record['export_hash'], []) if record['export_hash'] else [])
    for old_sid, old in candidates:
        if not sqlite_row and not record['origin_hash'] and 'record_hash' in old and old['record_hash'] != record['record_hash']:
            continue
        if ((record['origin_hash'] and old['origin_hash'] == record['origin_hash'] and old_sid == sid)
                or _compression_copy(old_sid, sid, evidence, cutover_anchors=cutover_anchors)):
            return True
    if candidates or (sqlite_row and record['export_hash'] and old_index.get(record['export_hash'])):
        raise SourceChangedError('source_changed: ambiguous legacy copy without native binding')
    return False


def _scan_physical_profile(profile, home, *, unknown_rows=(), policy_hash=None, header_transition=None,
                           approved_successors=None):
    directory, db = Path(profile['sessions_dir']), Path(profile['state_db'])
    if profile['directory_identity'] is not None:
        if _location_identity(directory, directory=True) != profile['directory_identity']:
            raise SourceChangedError('source_changed: JSON directory epoch changed')
    frozen_files = {a['path']: a for a in profile['artifacts']}
    present = {str(p) for p in directory.glob('session_*.json')} if directory.exists() else set()
    if not set(frozen_files) <= present:
        raise SourceChangedError('source_changed: legacy JSON membership lost')
    artifacts, old_index = [], defaultdict(list)
    copy_index, changes, current_db_exports = defaultdict(list), [], set()
    if header_transition is not None:
        changes.append({'approved_header_transition_sha256': digest(header_transition)})
    old_origins = defaultdict(list)
    for artifact in profile['artifacts']:
        for record in artifact['records']:
            old_index[record['export_hash']].append((artifact['session_id'], record))
            if record['origin_hash']:
                old_origins[record['origin_hash']].append((artifact['session_id'], record))
    old_db = profile['sqlite']
    expected_rows = {r['id']: r for r in old_db['rows']} if old_db else {}
    for record in expected_rows.values():
        copy_index[record['copy_hash']].append((record['session_id'], record))
        old_index[record['export_hash']].append((record['session_id'], record))
        if record['origin_hash']:
            old_origins[record['origin_hash']].append((record['session_id'], record))
    for path_text in sorted(present):
        path = checked_path(Path(path_text), root=home)
        current, data = _physical_artifact(path)
        old = frozen_files.get(path_text)
        if old is None or current['raw_hash'] != old['raw_hash']:
            changes.append({'path': path_text, 'raw_hash': current['raw_hash']})
        count = 0
        if old:
            count = len(old['records'])
            # Writers refresh overwritable session views via same-directory
            # tmp+rename, so a replacement (new inode) alone is not content
            # corruption; the stability contract is metadata identity, no
            # shrink, and an exact unchanged prefix. Replacement cannot bypass
            # any of those checks.
            if (current['identity_hash'] != old['identity_hash']
                    or len(current['records']) < count
                    or digest([r['record_hash'] for r in current['records'][:count]]) != old['prefix_hash']):
                raise SourceChangedError('source_changed: legacy JSON prefix/location rewritten')
        elif any(current['raw_hash'] == a['raw_hash'] and current['session_id'] == a['session_id'] for a in profile['artifacts']):
            count = len(current['records'])
        # Excluded old roles need no logical schema parsing. Future projection
        # still validates every message before discarding unretained roles.
        projected = _project_artifact({**data, 'messages': data.get('messages', [])[count:]})
        records = [record for raw, record in zip(data.get('messages', [])[count:], current['records'][count:])
                   if isinstance(raw, dict) and raw.get('role') in {'user', 'assistant'} and not raw.get('_compressed_summary')]
        del current['records']
        artifacts.append((current['session_id'], projected, current, records))
    lineage_artifacts = [(sid, data, current['raw_hash']) for sid, data, current, _ in artifacts if sid]
    anchors, views, physical = _profile_anchors(profile), defaultdict(dict), defaultdict(dict)
    seen_old, seen_unknown = set(), set()
    expected_unknown = {r['id']: r for r in unknown_rows}
    copy_roots, copy_sources = {}, set()
    evidence = _physical_lineage({}, lineage_artifacts, anchors)

    def check_origin(record, sid):
        for old_sid, old in old_origins.get(record['origin_hash'], []) if record['origin_hash'] else ():
            if (old_sid == sid or _compression_copy(old_sid, sid, evidence,
                    cutover_anchors=anchors)) and old['export_hash'] != record['export_hash']:
                raise SourceChangedError('source_changed: conflicting legacy stable origin')

    def consume(row, sessions, header):
        nonlocal evidence, copy_roots, copy_sources
        if row is None:
            if old_db:
                if header_transition is not None:
                    expected_header = header_transition['new']
                    if {k: header.get(k) for k in expected_header} != expected_header:
                        raise SourceChangedError('source_changed: approved SQLite header transition drift')
                    receipt = header_transition['native_copy_receipt_precondition']
                    if receipt['policy'] == 'current-epoch-superset-inventory':
                        for sid, operations in receipt['operations'].items():
                            container = sessions.get(sid, {}).get('model_config', {}).get(_COPY_RECEIPTS_KEY)
                            live = ({op['operation_id']: digest(op) for op in container['operations']}
                                    if type(container) is dict else {})
                            for op_id, op_digest in operations.items():
                                if live.get(op_id) != op_digest:
                                    raise SourceChangedError(
                                        'source_changed: approved native receipt operation lost')
                    elif (header['native_copy_receipt_sessions'] != receipt['sessions']
                            or header['copy_receipts_sha256'] != receipt['sha256']):
                        raise SourceChangedError('source_changed: approved native receipt inventory drift')
                elif header['epoch'] != old_db['epoch'] or header['schema_version'] != old_db['schema_version']:
                    raise SourceChangedError('source_changed: SQLite physical epoch/schema/session loss')
                if (header['message_columns'] != old_db['message_columns']
                        or not set(s['session_id'] for s in old_db['sessions']) <= set(sessions)):
                    raise SourceChangedError('source_changed: SQLite physical epoch/schema/session loss')
                for old in old_db['sessions']:
                    if _native_session_binding(sessions[old['session_id']]) != old['identity_hash']:
                        raise SourceChangedError('source_changed: legacy native session rewritten')
            evidence = _physical_lineage(sessions, lineage_artifacts, anchors)
            copy_roots, copy_sources = header['copy_roots'], header['copy_sources']
            if header['copy_receipts_sha256'] is not None:
                changes.append({'native_copy_receipts_sha256': header['copy_receipts_sha256']})
            return
        record = _physical_record(row, sqlite_row=True)
        old = expected_rows.get(row['id'])
        if old:
            if record['row_hash'] != old['row_hash']:
                raise SourceChangedError('source_changed: immutable legacy SQLite row rewritten')
            seen_old.add(row['id'])
            return
        changes.append({'id': row['id'], 'row_hash': record['row_hash']})
        sid = row['session_id']
        if sid not in sessions:
            raise SourceChangedError('source_changed: missing session for future message')
        if row['id'] in expected_unknown:
            if record != expected_unknown[row['id']]:
                raise SourceChangedError('source_changed: approved unknown SQLite binding drift')
            seen_unknown.add(row['id'])
            # Noncompletion exclusion only: never decode/capture its payload or
            # add it to clone evidence/export matching for other records.
            return
        if row['id'] in copy_roots:
            # Exact native ancestry was checked against both complete current
            # rows before streaming. Keep the original future event, or the
            # finite legacy/unknown exclusion, never a new copied occurrence.
            return
        if sessions[sid].get('source') == 'cron':
            return
        if row.get('role') not in {'user', 'assistant'} or row.get('_compressed_summary'):
            return
        root = _cutover_root(sid, evidence, anchors)
        check_origin(record, sid)
        if _exclude_old_copy(record, sid, old_index, evidence, cutover_anchors=anchors,
                             sqlite_row=True, copy_index=copy_index):
            return
        msg = _message({**row, 'content': _decode_db_content(row.get('content'))})
        if msg:
            binding = _clone_binding(record, row)
            previous = physical[root].get(msg['event_id'])
            if previous is not None and not msg['origin'] and not _proved_db_clone(
                    previous, binding, evidence, cutover_anchors=anchors):
                if not _approved_compaction_successor(previous, binding, approved_successors, sid,
                                                      msg['event_id']):
                    raise SourceChangedError('source_changed: ambiguous identical future SQLite rows')
                changes.append({'approved_compaction_successor_row_id': record['id']})
            physical[root][msg['event_id']] = binding
            if (row['id'] not in copy_sources and 'active' in row and 'compacted' in row
                    and not (row['active'] or row['compacted'])):
                # Hidden future rows still falsify a receipt-delete downgrade;
                # they are not independently admitted or stored as bodies.
                return
            current_db_exports.add((sid, record['export_hash']))
            _add_event(views[root], msg)

    if old_db and not db.exists():
        raise SourceChangedError('source_changed: legacy SQLite missing')
    if db.exists():
        if not old_db:
            raise SourceChangedError('source_changed: new SQLite feed needs explicit epoch binding')
        if header_transition is None:
            _read_physical_db(db, consume=consume)
        else:
            _read_physical_db(db, consume=consume, include_structure=True)
    if seen_old != set(expected_rows):
        raise SourceChangedError('source_changed: legacy SQLite row coverage lost')
    if seen_unknown != set(expected_unknown):
        raise SourceChangedError('source_changed: approved unknown SQLite row coverage lost')
    for sid, data, current, records in artifacts:
        messages = data['messages']
        if not messages:
            continue
        if not _valid_native(sid):
            raise SourceChangedError('source_changed: missing future native session identity')
        if data.get('platform') == 'cron' or Path(current['path']).name.startswith('session_cron_'):
            continue
        root = _cutover_root(sid, evidence, anchors)
        if data.get('lineage_root') and sid not in anchors and data['lineage_root'] != root:
            raise SourceChangedError('source_changed: unproved future lineage root')
        parent = data.get('parent_session_id')
        if parent and data.get('parent_end_reason') and evidence.get(str(parent), {}).get('end_reason') != data['parent_end_reason']:
            raise SourceChangedError('source_changed: conflicting forward parent proof')
        events = set()
        for raw, record in zip(messages, records):
            check_origin(record, sid)
            if _exclude_old_copy(record, sid, old_index, evidence, cutover_anchors=anchors):
                continue
            if (current['path'] not in frozen_files and sid in anchors and not record['origin_hash']
                    and (sid, record['export_hash']) not in current_db_exports):
                raise SourceChangedError('source_changed: new late export lacks native/exact copy binding')
            msg = _message(raw, codex=data.get('platform') == 'backfill-codex' or sid.startswith('codex-'))
            if msg:
                if msg['event_id'] in events:
                    raise SourceChangedError('source_changed: ambiguous identical future JSON events')
                events.add(msg['event_id'])
                _add_event(views[root], msg)
    fence = digest({'baseline': digest(profile), 'changes': changes}) if changes else digest(profile)
    if policy_hash is not None:
        fence = digest({'source': fence, 'source_unknown_policy': policy_hash,
                        'unknown_rows': list(unknown_rows)})
    return _physical_snapshots(profile, views, fence_hash=fence)


def _scan_physical_sources(hermes_home, profile_mode, baseline, *,
                           source_unknown_policy=_NO_SOURCE_UNKNOWN_POLICY,
                           source_transition_certificates=None, approved_transition_sha256s=None,
                           source_compaction_successors=None):
    home = checked_path(Path(hermes_home))
    if not isinstance(baseline, dict):
        raise ValueError('invalid physical baseline header')
    validate_physical_baseline(baseline, bank=baseline.get('bank', ''), hermes_home=home)
    if profile_mode != baseline['profile_mode']:
        raise ValueError('physical baseline profile coverage mode mismatch')
    unknown_rows, policy_hash = defaultdict(list), None
    if source_unknown_policy is not _NO_SOURCE_UNKNOWN_POLICY:
        validate_source_unknown_policy(source_unknown_policy, physical_baseline=baseline,
                                       bank=baseline['bank'], hermes_home=home)
        policy_hash = digest(source_unknown_policy)
        for scope in source_unknown_policy['scopes']:
            unknown_rows[scope['profile']].extend(scope['rows'])
    certs_by_profile = {}
    if source_transition_certificates is not None or approved_transition_sha256s is not None:
        certs_by_profile = validate_source_transition_certificate_set(source_transition_certificates,
            physical_baseline=baseline, source_unknown_policy=source_unknown_policy,
            approved_transition_sha256s=approved_transition_sha256s)
    successors_by_profile = {}
    if source_compaction_successors is not None:
        validate_source_compaction_successors(source_compaction_successors, physical_baseline=baseline,
            bank=baseline['bank'], hermes_home=home)
        for pair in source_compaction_successors['pairs']:
            successors_by_profile.setdefault(pair['profile'], {})[(pair['session_id'], pair['event_id'])] = pair
    current = {s['profile']: s for s in _discover_sources(home, profile_mode)}
    old_profiles = {p['profile']: p for p in baseline['profiles']}
    if not set(old_profiles) <= set(current):
        raise SourceChangedError('source_changed: legacy profile coverage lost')
    result = [snapshot for profile in baseline['profiles'] for snapshot in _scan_physical_profile(
        profile, home, unknown_rows=unknown_rows[profile['profile']], policy_hash=policy_hash,
        header_transition=certs_by_profile.get(profile['profile']),
        approved_successors=successors_by_profile.get(profile['profile'], {}))]
    for name in sorted(set(current) - set(old_profiles)):
        # A genuinely new profile is strict, not silently turned into legacy.
        profile_home = Path(current[name]['state_db']).parent
        for snapshot in scan_source_snapshots(profile_home, profile_mode='none'):
            for key in ('source_key', 'epoch_scope'):
                snapshot[key] = name + snapshot[key][len('default'):]
            snapshot['scope_source_keys'] = [name + k[len('default'):] for k in snapshot['scope_source_keys']]
            snapshot['profile'] = name
            snapshot['snapshot_hash'] = snapshot_hash(snapshot)
            result.append(snapshot)
    return result
