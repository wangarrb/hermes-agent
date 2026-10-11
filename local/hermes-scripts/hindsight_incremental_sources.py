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
import sqlite3
import stat
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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


def _lineage_root(sid, evidence):
    current, seen = sid, set()
    while evidence[current].get('parent_session_id'):
        parent = str(evidence[current]['parent_session_id'])
        if parent not in evidence or not evidence[parent].get('end_reason'):
            raise SourceChangedError('source_changed: missing lineage parent/end_reason proof')
        if parent in seen or parent == current:
            raise SourceChangedError('source_changed: cyclic compression lineage')
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


def _proved_db_clone(left, right, evidence):
    physical = {'id', 'session_id', 'active', 'compacted', 'display_order'}
    if {k: v for k, v in left.items() if k not in physical} != {k: v for k, v in right.items() if k not in physical}:
        return False
    a, b = left['session_id'], right['session_id']
    if a == b:
        # Archive/live flags alone do not prove copying: a later independent
        # insert can have the same body/time. Core carried-tail originals are
        # rewind-only and excluded by the active/compacted visibility query.
        return False
    current = b
    while evidence[current].get('parent_session_id'):
        parent = str(evidence[current]['parent_session_id'])
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
    try:
        con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        try:
            con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN')
            sessions = {r['id']: dict(r) for r in con.execute('SELECT * FROM sessions')}
            columns = {r[1] for r in con.execute('PRAGMA table_info(messages)')}
            active = ' WHERE (active = 1 OR compacted = 1)' if {'active', 'compacted'} <= columns else ''
            rows = [dict(r) for r in con.execute('SELECT * FROM messages' + active + ' ORDER BY id')]
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
    evidence = _merge_lineage(sessions, artifacts)
    roots, grouped, physical_rows = {}, defaultdict(dict), defaultdict(dict)
    for row in rows:
        sid = row['session_id']
        if sid not in sessions:
            raise SourceChangedError('source_changed: missing session for message')
        if sessions[sid].get('source') == 'cron':
            continue
        if row.get('role') not in {'user', 'assistant'} or row.get('_compressed_summary'):
            continue
        # Integer row IDs are not message origins: protected tails are cloned.
        msg = _message({**row, 'content': _decode_db_content(row.get('content'))})
        if msg:
            root = roots.setdefault(sid, _lineage_root(sid, evidence))
            previous = physical_rows[root].get(msg['event_id'])
            if previous is not None and not msg['origin'] and not _proved_db_clone(previous, row, evidence):
                raise SourceChangedError('source_changed: ambiguous identical SQLite rows without clone proof')
            _add_event(grouped[root], msg)
            physical_rows[root][msg['event_id']] = row
    epoch = 'sqlite:' + digest({'inode': identity, 'application_id': application_id})
    return evidence, {'roots': roots, 'messages': {r: list(view.values()) for r, view in grouped.items()}}, epoch


def scan_source_snapshots(hermes_home: Path, *, profile_mode: str = 'all') -> list[dict]:
    """Observe JSON/Codex current feeds + unsnapshotted persisted SQLite messages.

    This is a local read-only history *identity scan*, never paid processing.
    Logical views merge duplicate artifacts and compression lineage before
    capture. Timestamp-less snapshots fail closed rather than using array order.
    """
    if profile_mode not in {'all', 'hindsight', 'none'}:
        raise ValueError('invalid profile_mode')
    home = checked_path(Path(hermes_home))
    if not home.exists():
        return []
    profile_root = home / 'profiles'
    checked_path(profile_root)
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
                artifacts.append((sid, data, raw_hash))
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
