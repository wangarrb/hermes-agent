#!/usr/bin/env python3
"""Read-only plan for frozen App exports using existing physical identities.

No new namespace, capture, baseline, authorization or delivery is introduced.
Missing historical knowledge stays explicitly unknown.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

from hindsight_incremental_sources import (
    CLEANING_VERSION, _message, _physical_record, _profile_anchors, _read_bytes, checked_path,
    message_hash, snapshot_hash, validate_snapshot,
)
from hindsight_incremental_delivery import _read


class AppSourceAmbiguity(ValueError):
    """An exact frozen input or existing source boundary cannot be proved."""


def _uuid(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value and uuid.UUID(value).int != 0
    except (ValueError, AttributeError):
        return False


def _opaque(value):
    return isinstance(value, str) and bool(value) and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _app_messages(session):
    sid = session.get('source_session_id')
    if (not _uuid(sid) or session.get('session_id') != 'codex-app-' + sid
            or session.get('source') != 'codex-app' or session.get('platform') != 'backfill-codex-app'
            or not isinstance(session.get('messages'), list) or not session['messages']):
        raise AppSourceAmbiguity('invalid frozen App session')
    messages, identities = [], set()
    for raw in session['messages']:
        if not isinstance(raw, dict):
            raise AppSourceAmbiguity('invalid frozen message')
        turn, item = raw.get('source_turn_id'), raw.get('source_item_id')
        identity = (sid, turn, item)
        if (not _opaque(turn) or not _opaque(item) or raw.get('truncated')
                or raw.get('origin_message_id') != f'codex-app:{sid}:{turn}:{item}'
                or identity in identities):
            raise AppSourceAmbiguity('complete stable composite native identity required')
        message = _message(raw, codex=True)
        if message is None:
            raise AppSourceAmbiguity('frozen message cannot be silently filtered')
        identities.add(identity)
        messages.append(message)
    return messages


def _inputs(sessions, *, hermes_home, approved_session_ids, approved_source_sha256):
    allowed = set(approved_session_ids)
    if (not allowed or any(not _uuid(sid) for sid in allowed)
            or not isinstance(approved_source_sha256, dict)
            or set(approved_source_sha256) != allowed or not all(_sha(v) for v in approved_source_sha256.values())
            or not isinstance(sessions, list) or len(sessions) != len(allowed)
            or any(not isinstance(s, dict) for s in sessions)
            or {s.get('source_session_id') for s in sessions} != allowed):
        raise AppSourceAmbiguity('frozen session scope or independent raw pins changed')
    home = checked_path(Path(hermes_home))
    result = []
    for session in sorted(sessions, key=lambda s: s['source_session_id']):
        sid = session['source_session_id']
        path = checked_path(home / 'sessions' / f'session_codex-app-{sid}.json', root=home)
        if any(p.name != path.name for p in path.parent.glob(f'session_codex-*{sid}.json')):
            raise AppSourceAmbiguity('known_cli_session_overlap')
        before = _read_bytes(path)
        if hashlib.sha256(before).hexdigest() != approved_source_sha256[sid]:
            raise AppSourceAmbiguity('production export differs from approved raw pin')
        if json.loads(before) != session or _read_bytes(path) != before:
            raise AppSourceAmbiguity('frozen parsed input or production export changed')
        result.append((session, _app_messages(session)))
    return result


def _baseline(ledger, con, home):
    manifest = ledger._read(con, 'meta', 'legacy')
    reference = manifest.get('physical_baseline')
    if not isinstance(reference, dict) or reference.get('hermes_home') != str(home):
        raise AppSourceAmbiguity('original physical baseline and home binding required')
    ledger._physical_reference(reference)
    baseline = _read(ledger.root, reference['artifact_path'], reference['artifact_hash'])
    defaults = [p for p in baseline['profiles'] if p['profile'] == 'default']
    if len(defaults) != 1:
        raise AppSourceAmbiguity('original default feed boundary missing')
    epoch = defaults[0]['feed_epoch']
    feed = ledger._read(con, 'meta', 'epoch:default:physical-feed')
    if not isinstance(feed, dict) or feed.get('epoch') != epoch or not isinstance(feed.get('source_keys'), list):
        raise AppSourceAmbiguity('existing default feed epoch does not match original baseline')
    return baseline, reference, feed


def _snapshots(inputs, profile, source_sha256):
    result = []
    anchors = _profile_anchors(profile)
    epoch = profile['feed_epoch']
    for session, messages in inputs:
        sid = session['source_session_id']
        if session['session_id'] in anchors:
            raise AppSourceAmbiguity('approved_app_session_is_original_cutover_anchor')
        key = 'default:hermes:codex-app-' + sid
        snapshot = {'source_key': key, 'source_epoch': epoch, 'epoch_scope': key,
                    'scope_source_keys': [key], 'profile': 'default', 'kind': 'source',
                    'cleaning_version': CLEANING_VERSION, 'messages': messages,
                    'source_hash_before': source_sha256[sid], 'source_hash_after': source_sha256[sid]}
        snapshot['snapshot_hash'] = snapshot_hash(snapshot)
        validate_snapshot(snapshot)
        result.append(snapshot)
    return result


def build_app_snapshots(sessions, *, ledger, hermes_home, approved_session_ids, approved_source_sha256):
    """Fixture/internal projection; public inspection is plan_app_input."""
    home = checked_path(Path(hermes_home))
    inputs = _inputs(sessions, hermes_home=home, approved_session_ids=approved_session_ids,
                     approved_source_sha256=approved_source_sha256)
    with ledger.locked(), ledger._connection() as con:
        baseline, _, _ = _baseline(ledger, con, home)
        profile = next(p for p in baseline['profiles'] if p['profile'] == 'default')
        return _snapshots(inputs, profile, approved_source_sha256)


def plan_app_input(sessions, *, ledger, hermes_home, approved_session_ids, approved_source_sha256):
    """Inspect exact inputs and related payloads in one locked read transaction.

    This is no capture grant or global coverage claim. Pending batches are
    reported with their existing operation UUID and are never resumed here.
    """
    home = checked_path(Path(hermes_home))
    inputs = _inputs(sessions, hermes_home=home, approved_session_ids=approved_session_ids,
                     approved_source_sha256=approved_source_sha256)
    conflicts, related = [], []
    with ledger.locked(), ledger._connection() as con:
        baseline, reference, feed = _baseline(ledger, con, home)
        profile = next(p for p in baseline['profiles'] if p['profile'] == 'default')
        epoch = profile['feed_epoch']
        snapshots = _snapshots(inputs, profile, approved_source_sha256)
        expected = {(s['source_key'], m['event_id']): message_hash(m) for s in snapshots for m in s['messages']}
        target_ids = {event for _, event in expected}
        target_keys = {s['source_key'] for s in snapshots}
        known = set()
        def conflict(code, **fields):
            conflicts.append({'code': code, **fields})
        for (key,) in con.execute('SELECT source_key FROM sources'):
            state = ledger._read(con, 'sources', key)
            overlap = target_ids.intersection(state['events'])
            if key in target_keys:
                if state.get('bank') != ledger.bank:
                    conflict('source_bank_changed', source_key=key)
                if state['source_epoch'] != epoch:
                    conflict('source_epoch_changed', source_key=key)
                for event, old_hash in state['events'].items():
                    if (key, event) not in expected:
                        conflict('source_event_missing', source_key=key, event_id=event)
                    elif expected[(key, event)] != old_hash:
                        conflict('source_event_changed', source_key=key, event_id=event,
                                 existing_hash=old_hash, input_hash=expected[(key, event)])
                    else:
                        known.add((key, event))
                    if event in state.get('legacy_event_ids', []):
                        conflict('legacy_excluded_event', source_key=key, event_id=event)
                scope = ledger._read(con, 'meta', 'epoch:' + key)
                if scope is not None and (scope.get('epoch') != epoch or scope.get('source_keys') != [key]):
                    conflict('singleton_epoch_changed', source_key=key)
            elif overlap:
                for event in sorted(overlap):
                    conflict('native_event_in_other_source', source_key=key, event_id=event,
                             existing_hash=state['events'][event])
        # Per-input native/export hashes: unrelated finite exclusions do not
        # become a blanket ban and cannot become a completion watermark.
        origins, exports = {}, {}
        for session, _ in inputs:
            key = 'default:hermes:' + session['session_id']
            for raw in session['messages']:
                record = _physical_record(raw)
                event = _message(raw, codex=True)['event_id']
                origins.setdefault(record['origin_hash'], []).append((key, event))
                exports.setdefault(record['export_hash'], []).append((key, event))
        def check_record(record, code, profile, sid):
            matches = set(origins.get(record.get('origin_hash'), ())) | set(exports.get(record.get('export_hash'), ()))
            for key, event in sorted(matches):
                conflict(code, source_key=key, event_id=event, profile=profile, session_id=sid)
        for profile in baseline['profiles']:
            for artifact in profile['artifacts']:
                for record in artifact['records']:
                    check_record(record, 'physical_baseline_collision', profile['profile'], artifact['session_id'])
            for record in (profile['sqlite'] or {}).get('rows', []):
                check_record(record, 'physical_baseline_collision', profile['profile'], record['session_id'])
        policy_ref = ledger._read(con, 'meta', 'source-unknown-policy')
        policy_required = ledger._read(con, 'meta', 'source-unknown-policy-required')
        if (policy_ref is None) != (policy_required is None):
            raise AppSourceAmbiguity('source unknown policy required binding missing')
        if policy_ref is not None:
            ledger._source_unknown_artifact(policy_ref, reference)
            if policy_required != {'bank': ledger.bank, 'artifact_hash': policy_ref['artifact_hash'],
                                    'physical_baseline_sha256': reference['artifact_hash']}:
                raise AppSourceAmbiguity('source unknown policy required binding changed')
            policy = _read(ledger.root, policy_ref['artifact_path'], policy_ref['artifact_hash'])
            for scope in policy['scopes']:
                for record in scope['rows']:
                    check_record(record, 'finite_unknown_collision', scope['profile'], scope['session_id'])
        # Read metadata first; only native-event/source/parent-related batches
        # open immutable payloads. Unrelated pending work is not resumed.
        batches = {bid: ledger._read(con, 'batches', bid)
                   for (bid,) in con.execute('SELECT batch_id FROM batches')}
        relevant = {bid for bid, b in batches.items() if b['source_key'] in target_keys
                    or target_ids.intersection(b['source_event_ids'])}
        while True:
            closure = set(relevant)
            for bid, batch in batches.items():
                if bid in relevant or relevant.intersection(batch['parent_batch_ids']):
                    closure.add(bid)
                    closure.update(batch['parent_batch_ids'])
            if closure == relevant:
                break
            relevant = closure
        authorized, seen_owners = {}, {}
        for bid in sorted(relevant):
            if bid not in batches:
                conflict('related_parent_missing', batch_id=bid)
                continue
            batch = ledger._batch(con, bid, verify_payload=True)
            payload = _read(ledger.root, batch['payload_path'], batch['payload_hash'])
            related.append({k: batch[k] for k in ('batch_id', 'source_key', 'kind', 'phase',
                                                   'operation_id', 'source_event_ids')})
            if batch['kind'] != 'source':
                continue
            valid = batch['source_epoch'] == epoch and bool(payload['messages'])
            for message in payload['messages']:
                binding = (batch['source_key'], message['event_id'])
                if expected.get(binding) != message_hash(message):
                    valid = False
                elif binding in seen_owners:
                    conflict('multiple_source_batches', source_key=binding[0], event_id=binding[1])
                else:
                    seen_owners[binding] = bid
            authorized[bid] = valid
            if not valid:
                conflict('mixed_unauthorized_source_batch', batch_id=bid)
        for bid in sorted(relevant.intersection(batches)):
            batch = batches[bid]
            if batch['kind'] != 'source' and (not batch['parent_batch_ids']
                    or any(not authorized.get(parent, False) for parent in batch['parent_batch_ids'])):
                conflict('mixed_unauthorized_derived_parent', batch_id=bid)
        for key, event in sorted(known):
            if (key, event) not in seen_owners:
                conflict('known_event_without_batch', source_key=key, event_id=event)
        for (key, event), bid in sorted(seen_owners.items()):
            if authorized.get(bid) and (key, event) not in known:
                conflict('batch_event_without_source_coverage', source_key=key, event_id=event, batch_id=bid)
        unrelated_pending = [{k: batch[k] for k in ('batch_id', 'source_key', 'phase', 'operation_id')}
                             for bid, batch in batches.items() if bid not in relevant and batch['phase'] != 'published']
        for batch in unrelated_pending:
            conflict('unrelated_pending_batch', batch_id=batch['batch_id'], phase=batch['phase'])
        sources = [{k: s[k] for k in ('source_key', 'source_epoch', 'snapshot_hash')}
                   | {'messages': len(s['messages']),
                      'known_messages': sum((s['source_key'], m['event_id']) in known for m in s['messages']),
                      'new_messages': sum((s['source_key'], m['event_id']) not in known for m in s['messages'])}
                   for s in snapshots]
    return {'status': 'blocked' if conflicts else 'planning_only', 'admission_disabled': True,
            'input_status': 'blocked' if conflicts else 'no_relevant_conflict_observed',
            'sources': sources, 'new_messages': sum(s['new_messages'] for s in sources),
            'known_messages': len(known), 'source_sha256': dict(approved_source_sha256),
            'baseline_sha256': reference['artifact_hash'], 'physical_feed_scope_size': len(feed['source_keys']),
            'related_batches': related, 'unrelated_pending_batches': unrelated_pending,
            'conflicts': conflicts, 'historical_coverage': 'unknown',
            'app_history_coverage': 'supported_summary_view_only', 'app_history_complete': None,
            'ledger_capture': 'not_attempted', 'runtime_preflight': 'not_attempted', 'model_calls': 0}
