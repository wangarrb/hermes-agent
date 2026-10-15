#!/usr/bin/env python3
"""Finite App snapshot candidate; contract validation never grants execution.

The caller supplies an independently pinned contract and an explicit frozen
session allowlist. No discovery, source publication, ledger registration, model
call, or physical-reader fallback occurs here.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from hindsight_incremental_sources import (
    CLEANING_VERSION, _message, digest, snapshot_hash, validate_snapshot,
)


class AppSourceAmbiguity(ValueError):
    """The entire finite input remains inadmissible; no known subset is returned."""


def _uuid(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value and uuid.UUID(value).int != 0
    except (ValueError, AttributeError):
        return False


def _pin(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise AppSourceAmbiguity('ambiguous App source: independent contract pin required')


def load_contract(path: Path, *, approved_contract_sha256):
    """Read one contract byte view; the supplied approval is never self-derived."""
    _pin(approved_contract_sha256)
    contract = json.loads(Path(path).read_bytes())
    if digest(contract) != approved_contract_sha256:
        raise AppSourceAmbiguity('ambiguous App source: contract approval pin mismatch')
    return contract


def _app_messages(session):
    sid = session.get('source_session_id')
    if (not _uuid(sid) or session.get('session_id') != 'codex-app-' + sid
            or session.get('source') != 'codex-app'
            or session.get('platform') != 'backfill-codex-app'
            or not isinstance(session.get('messages'), list) or not session['messages']):
        raise AppSourceAmbiguity('ambiguous App source: invalid frozen App session')
    messages, identities = [], []
    for raw in session['messages']:
        if not isinstance(raw, dict):
            raise AppSourceAmbiguity('ambiguous App source: invalid frozen message')
        turn, item = raw.get('source_turn_id'), raw.get('source_item_id')
        if (not _uuid(turn) or not _uuid(item) or raw.get('truncated')
                or raw.get('origin_message_id') != f'codex-app:{sid}:{turn}:{item}'):
            raise AppSourceAmbiguity('ambiguous App source: complete stable native identity required')
        message = _message(raw, codex=True)
        if message is None:
            raise AppSourceAmbiguity('ambiguous App source: frozen message cannot be silently filtered')
        messages.append(message)
        identities.append((sid, turn, item))
    if len(set(identities)) != len(identities):
        raise AppSourceAmbiguity('ambiguous App source: repeated native identity')
    return messages, identities


def build_app_snapshots(sessions, *, contract, approved_contract_sha256,
                        bank, ledger_root, approved_session_ids):
    """Validate complete App-side closure against the dual-view finite contract.

    The ownership module separately validates the contract's native mapping,
    dual payload proofs and ledger scope. These snapshots are candidates only;
    the shared ledger gate still owns registration, capture and delivery state.
    """
    _pin(approved_contract_sha256)
    if contract is None:
        raise AppSourceAmbiguity('ambiguous App source: finite dual-view contract required')
    from hindsight_app_event_ownership import validate_contract

    validate_contract(contract, approved_contract_sha256=approved_contract_sha256,
                      bank=bank, ledger_root=ledger_root)
    allowed = set(approved_session_ids)
    if (not allowed or any(not _uuid(sid) for sid in allowed) or not isinstance(sessions, list)
            or len(sessions) != len(allowed) or any(not isinstance(s, dict) for s in sessions)
            or {s.get('source_session_id') for s in sessions} != allowed):
        raise AppSourceAmbiguity('ambiguous App source: frozen session scope changed')
    account = contract['account']
    account_hash = digest(account)
    keys = {sid: f'app-contract::{account_hash}::{sid}' for sid in allowed}
    epoch = f'app-contract-v1::{account_hash}'
    source_keys = sorted(keys.values())
    events = {(e['identity']['session'], e['identity']['turn'], e['identity']['item']): e
              for e in contract['events']}
    if len(events) != len(contract['events']):
        raise AppSourceAmbiguity('ambiguous App source: duplicate contract native identity')
    app_scopes = {s['source_key']: s for s in contract['scopes'] if s['view'] == 'app'}
    if set(app_scopes) != set(source_keys):
        raise AppSourceAmbiguity('ambiguous App source: App scope exceeds frozen allowlist')
    snapshots, seen, item_bindings = [], set(), {}
    for session in sorted(sessions, key=lambda s: s['source_session_id']):
        sid = session['source_session_id']
        messages, identities = _app_messages(session)
        for message, identity in zip(messages, identities):
            # This finite schema proves one App view per event; it has no native
            # multi-session/turn alias proof. Copies remain ambiguous.
            prior_binding = item_bindings.setdefault(identity[2], identity[:2])
            if prior_binding != identity[:2]:
                raise AppSourceAmbiguity('ambiguous App source: native item alias unproved')
            event = events.get(identity)
            if event is None or event['identity']['account'] != account:
                raise AppSourceAmbiguity('ambiguous App source: incomplete account/native mapping')
            matches = [v for v in event['views'] if v['view'] == 'app'
                       and v['source_key'] == keys[sid] and v['source_epoch'] == epoch
                       and v['event_id'] == message['event_id'] and v['message'] == message]
            if len(matches) != 1:
                raise AppSourceAmbiguity('ambiguous App source: App payload/identity proof mismatch')
            seen.add(identity)
        fence = digest({'contract_sha256': approved_contract_sha256, 'session': session})
        snapshot = {'source_key': keys[sid], 'source_epoch': epoch, 'epoch_scope': epoch,
            'scope_source_keys': source_keys, 'profile': 'default', 'kind': 'source',
            'cleaning_version': CLEANING_VERSION, 'messages': messages,
            'source_hash_before': fence, 'source_hash_after': fence}
        snapshot['snapshot_hash'] = snapshot_hash(snapshot)
        validate_snapshot(snapshot)
        scope = app_scopes[keys[sid]]
        if (scope['source_epoch'] != epoch or scope['snapshot_hash'] != snapshot['snapshot_hash']
                or scope['event_ids'] != [m['event_id'] for m in messages]):
            raise AppSourceAmbiguity('ambiguous App source: complete frozen snapshot proof mismatch')
        snapshots.append(snapshot)
    if seen != set(events):
        raise AppSourceAmbiguity('ambiguous App source: contract contains events outside frozen input')
    return snapshots


def describe_plan(snapshots):
    """Metadata only: planning is neither registration nor production admission."""
    return {'status': 'planning_only', 'admission_disabled': True,
        'contract_covers_frozen_event_set': True,
        'app_history_complete': None, 'app_history_coverage': 'supported_summary_view_only',
        'ownership_registration': 'not_attempted', 'ledger_capture': 'not_attempted',
        'runtime_preflight': 'not_attempted', 'model_calls': 0,
        'sources': [{k: s[k] for k in ('source_key', 'source_epoch', 'snapshot_hash')}
                    | {'messages': len(s['messages'])} for s in snapshots]}
