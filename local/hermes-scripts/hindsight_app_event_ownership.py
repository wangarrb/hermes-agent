"""Finite, independently pinned cross-view ownership. No network or source reads.

This is deliberately closed-world: while any contract is active, a source not
fully covered by its immutable scopes is ambiguous. This is not a global alias
registry, normalization recipe, permission grant, or a legacy success watermark.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from hindsight_incremental_sources import (SourceChangedError, digest, message_hash,
                                           snapshot_hash, validate_snapshot)

SCHEMA = 'hindsight-app-event-contract/1'
CONTRACT_PREFIX = 'app-contract:'
REQUIRED_PREFIX = 'app-contract-required:'
OWNER_PREFIX = 'app-owner:'
FIELDS = ('event_id', 'role', 'timestamp', 'origin', 'content')


class OwnershipAmbiguousError(SourceChangedError):
    """No unique, complete and authorized canonical owner can be proven."""


def _reject(reason):
    raise OwnershipAmbiguousError('app ownership ambiguous: ' + reason)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _text(value):
    return isinstance(value, str) and bool(value) and not any(ord(c) < 32 for c in value)


def canonical_event_key(identity):
    if (not isinstance(identity, dict) or set(identity) != {'account', 'session', 'turn', 'item'}
            or not all(_text(v) for v in identity.values())):
        _reject('invalid native identity')
    return digest(identity)


def validate_contract(contract, *, approved_contract_sha256, bank, ledger_root):
    """Validate a caller-supplied independent canonical-JSON SHA pin.

    Each scope binds the *entire* snapshot. Optional profile/epoch_scope/
    scope_source_keys use precisely the defaults in sources.snapshot_hash.
    identity_proof.sha256 names independently reviewed native identity evidence;
    payload equality alone never establishes that two events are aliases.
    """
    if not _sha(approved_contract_sha256) or digest(contract) != approved_contract_sha256:
        _reject('independent contract pin mismatch')
    if (not isinstance(contract, dict) or contract.get('schema') != SCHEMA
            or contract.get('bank') != bank
            or contract.get('ledger_root') != str(Path(ledger_root).absolute())
            or not _text(contract.get('account'))):
        _reject('contract context mismatch')
    scopes, events = contract.get('scopes'), contract.get('events')
    if not isinstance(scopes, list) or not scopes or not isinstance(events, list) or not events:
        _reject('empty finite contract')
    by_scope, by_binding, canonical = {}, {}, {}
    native_items, native_origins = {}, {}
    for scope in scopes:
        if (not isinstance(scope, dict) or scope.get('view') not in {'app', 'physical'}
                or not _text(scope.get('source_key')) or not _text(scope.get('source_epoch'))
                or not _sha(scope.get('snapshot_hash'))
                or not isinstance(scope.get('event_ids'), list) or not scope['event_ids']
                or not all(_text(v) for v in scope['event_ids'])
                or len(set(scope['event_ids'])) != len(scope['event_ids'])):
            _reject('invalid finite scope')
        if scope['source_key'] in by_scope:
            _reject('duplicate scope key')
        by_scope[scope['source_key']] = scope
    for event in events:
        key = canonical_event_key(event.get('identity'))
        if key in canonical or event['identity']['account'] != contract['account']:
            _reject('duplicate canonical identity or account mismatch')
        identity = event['identity']
        item_binding = (identity['account'], identity['item'])
        occurrence = (identity['session'], identity['turn'])
        if item_binding in native_items and native_items[item_binding] != occurrence:
            _reject('shared native item requires unavailable multi-occurrence alias proof')
        native_items[item_binding] = occurrence
        proof = event.get('identity_proof', {})
        if proof.get('kind') != 'native-account-session-turn-item' or not _sha(proof.get('sha256')):
            _reject('native identity evidence missing')
        payload = event.get('canonical_payload')
        if (not isinstance(payload, dict) or set(payload) != {'role', 'timestamp', 'content'}
                or payload.get('role') not in {'user', 'assistant'}
                or not isinstance(payload.get('content'), str)
                or event.get('canonical_payload_sha256') != digest(payload)):
            _reject('canonical payload proof mismatch')
        views = event.get('views')
        if not isinstance(views, list) or len(views) != 2 or {v.get('view') for v in views} != {'app', 'physical'}:
            _reject('exactly two views required')
        for view in views:
            scope = by_scope.get(view.get('source_key'))
            msg = view.get('message')
            prefix = view.get('content_prefix')
            if (scope is None or scope['view'] != view.get('view')
                    or scope['source_epoch'] != view.get('source_epoch')
                    or view.get('event_id') not in scope['event_ids']
                    or not isinstance(msg, dict) or set(msg) != set(FIELDS)
                    or msg.get('event_id') != view.get('event_id')
                    or view.get('message_hash') != message_hash(msg)
                    or msg.get('role') != payload['role'] or msg.get('timestamp') != payload['timestamp']
                    or not isinstance(prefix, str) or msg.get('content') != prefix + payload['content']
                    or not _text(msg.get('origin'))):
                _reject('view payload or native origin proof mismatch')
            binding = (scope['source_key'], view['event_id'])
            if binding in by_binding:
                _reject('view bound to multiple canonical identities')
            origin_key = (identity['account'], msg['origin'])
            if origin_key in native_origins and native_origins[origin_key] != key:
                _reject('shared view origin requires unavailable native alias proof')
            native_origins[origin_key] = key
            by_binding[binding] = (key, event, view)
        expected_proof_bindings = sorted([
            {'source_key': v['source_key'], 'event_id': v['event_id'],
             'origin': v['message']['origin'], 'message_hash': v['message_hash']}
            for v in views], key=lambda v: v['source_key'])
        if (proof.get('identity') != event['identity']
                or proof.get('bindings') != expected_proof_bindings):
            _reject('native identity evidence is not bound to both original views')
        canonical[key] = event
    for source_key, scope in by_scope.items():
        messages = []
        for event_id in scope['event_ids']:
            binding = by_binding.get((source_key, event_id))
            if binding is None:
                _reject('scope is not fully covered')
            messages.append(binding[2]['message'])
        snapshot = dict(scope, messages=messages)
        if snapshot_hash(snapshot) != scope['snapshot_hash']:
            _reject('finite scope hash does not bind all view payloads')
    return {'pin': approved_contract_sha256, 'contract': contract}


def _inventory(ledger, con, extra=None):
    contract_keys = [r[0] for r in con.execute(
        "SELECT key FROM meta WHERE key LIKE 'app-contract:%' ORDER BY key")]
    required_keys = [r[0] for r in con.execute(
        "SELECT key FROM meta WHERE key LIKE 'app-contract-required:%' ORDER BY key")]
    contract_pins = {k[len(CONTRACT_PREFIX):] for k in contract_keys}
    required_pins = {k[len(REQUIRED_PREFIX):] for k in required_keys}
    if contract_pins != required_pins:
        _reject('contract and required marker coverage differ')
    if not contract_keys and con.execute(
            "SELECT 1 FROM meta WHERE key LIKE 'app-owner:%' LIMIT 1").fetchone():
        _reject('owner exists without its required finite contract')
    contracts = []
    for key in contract_keys:
        stored = ledger._read(con, 'meta', key)
        pin = key[len(CONTRACT_PREFIX):]
        if not isinstance(stored, dict) or stored.get('pin') != pin:
            _reject('stored contract identity mismatch')
        if ledger._read(con, 'meta', REQUIRED_PREFIX + pin) != _required_marker(ledger, pin):
            _reject('contract required marker identity mismatch')
        contracts.append(validate_contract(stored.get('contract'), approved_contract_sha256=pin,
                         bank=ledger.bank, ledger_root=ledger.storage_root))
    if extra is not None and all(c['pin'] != extra['pin'] for c in contracts):
        contracts.append(extra)
    scopes, bindings, events = {}, {}, {}
    native_items, native_origins = {}, {}
    for registered in contracts:
        contract, pin = registered['contract'], registered['pin']
        for scope in contract['scopes']:
            old = scopes.get(scope['source_key'])
            if old is not None and old != scope:
                _reject('overlapping contracts disagree on immutable scope')
            scopes[scope['source_key']] = scope
        for event in contract['events']:
            key = canonical_event_key(event['identity'])
            identity = event['identity']
            item_binding = (identity['account'], identity['item'])
            occurrence = (identity['session'], identity['turn'])
            if item_binding in native_items and native_items[item_binding] != occurrence:
                _reject('cross-contract shared native item')
            native_items[item_binding] = occurrence
            old = events.get(key)
            if old is not None and old[0] != event:
                _reject('overlapping canonical evidence differs')
            events[key] = (event, pin)
            for view in event['views']:
                binding = (view['source_key'], view['event_id'])
                origin_key = (identity['account'], view['message']['origin'])
                if origin_key in native_origins and native_origins[origin_key] != key:
                    _reject('cross-contract shared view origin')
                native_origins[origin_key] = key
                if binding in bindings and bindings[binding][0] != key:
                    _reject('cross-contract view identity collision')
                bindings[binding] = (key, event, view, pin)
    by_pin = {c['pin']: c['contract'] for c in contracts}
    for (owner_key,) in con.execute("SELECT key FROM meta WHERE key LIKE 'app-owner:%'"):
        owner = ledger._read(con, 'meta', owner_key)
        canonical_key = owner_key[len(OWNER_PREFIX):]
        if (not isinstance(owner, dict) or owner.get('canonical_key') != canonical_key
                or canonical_key not in events or owner.get('contract_sha256') not in contract_pins
                or events[canonical_key][0] not in by_pin[owner['contract_sha256']]['events']
                or not _sha(owner.get('batch_id'))
                or ledger._read(con, 'batches', owner['batch_id']) is None):
            _reject('orphan owner outside registered contract closure')
    return scopes, bindings, events


def _required_marker(ledger, pin):
    return {'schema': 'hindsight-app-contract-required/1', 'bank': ledger.bank,
            'ledger_root': str(ledger.storage_root), 'contract_sha256': pin}


def _is_app(snapshot):
    # Existing local archival banks retain their original non-external workflow.
    # The new explicit source dispatch uses this namespace, with no bypass flag.
    return snapshot['source_key'].startswith('app-contract::')


def _check_scope(snapshot, scopes):
    scope = scopes.get(snapshot['source_key'])
    if (scope is None or scope['source_epoch'] != snapshot['source_epoch']
            or scope['snapshot_hash'] != snapshot['snapshot_hash']
            or scope['event_ids'] != [m['event_id'] for m in snapshot['messages']]):
        _reject('source outside finite approved scope or snapshot changed')


def _batch_evidence(ledger, con, batch_id, bindings, derived=None, cache=None, visiting=None):
    derived = {} if derived is None else derived
    cache = {} if cache is None else cache
    visiting = set() if visiting is None else visiting
    if batch_id in cache:
        return cache[batch_id]
    if batch_id in visiting:
        _reject('cyclic derived parent chain')
    visiting.add(batch_id)
    # Explicit True: published batches otherwise skip immutable payload validation.
    batch = ledger._batch(con, batch_id, verify_payload=True)
    fd = os.open(batch['payload_path'], os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != batch['payload_hash']:
        _reject('payload changed during owner lookup')
    payload = json.loads(raw)
    if batch['operation_id'] != str(uuid.uuid5(uuid.NAMESPACE_URL,
            f"hindsight-arrival:{ledger.bank}:{batch_id}")):
        _reject('operation UUID does not match immutable batch')
    if batch['kind'] != 'source':
        parents = batch['parent_batch_ids']
        if not parents or parents != sorted(set(parents)):
            _reject('derived parent coverage missing')
        identity = []
        for parent_id in parents:
            parent, _ = _batch_evidence(ledger, con, parent_id, bindings, derived, cache, visiting)
            if parent['kind'] != 'source' or not parent['receipts'].get('retained'):
                _reject('derived parent is not a confirmed retained source')
            ledger._validate_retained_receipt(parent, parent['receipts']['retained'])
            proof = {'batch_id': parent_id, 'payload_hash': parent['payload_hash'],
                     'operation_id': parent['operation_id'],
                     'retained_receipt_hash': digest(parent['receipts']['retained'])}
            identity.append(proof)
            parent_map = ledger._read(con, 'meta', 'derived-parent:' + parent_id)
            if (not isinstance(parent_map, dict)
                    or any(parent_map.get(k) != v for k, v in {'bank': ledger.bank, **proof,
                           'derived_source_key': batch['source_key']}.items())
                    or batch_id not in parent_map.get('derived_batch_ids', [])):
                _reject('derived parent binding incomplete')
            sibling_ids = parent_map['derived_batch_ids']
            actual_ids = [r[0] for r in con.execute('SELECT batch_id FROM batches')
                          if ledger._read(con, 'batches', r[0]).get('source_key') == batch['source_key']]
            if len(sibling_ids) != len(set(sibling_ids)) or set(sibling_ids) != set(actual_ids):
                _reject('derived sibling coverage incomplete')
        epoch = digest(identity)
        if (batch['source_key'] != 'derived:' + batch['kind'] + ':' + epoch
                or batch['source_epoch'] != epoch):
            _reject('derived source identity mismatch')
        state = derived.setdefault(batch['source_key'], {'source_epoch': epoch, 'events': {}})
        for message in payload['messages']:
            if message['event_id'] in state['events']:
                _reject('duplicate derived source event')
            state['events'][message['event_id']] = message_hash(message)
        visiting.remove(batch_id)
        cache[batch_id] = (batch, [])
        return cache[batch_id]
    keys = []
    for message in payload['messages']:
        binding = bindings.get((batch['source_key'], message['event_id']))
        if binding is None or binding[2]['message_hash'] != message_hash(message):
            _reject('mixed or unauthorized historical batch')
        if binding[2]['source_epoch'] != batch['source_epoch']:
            _reject('historical owner epoch mismatch')
        keys.append(binding[0])
    if len(set(keys)) != len(keys):
        _reject('batch contains duplicate canonical events')
    visiting.remove(batch_id)
    cache[batch_id] = (batch, keys)
    return cache[batch_id]


def _state_checks(ledger, con, scopes, bindings, derived):
    for (source_key,) in con.execute('SELECT source_key FROM sources'):
        state = ledger._read(con, 'sources', source_key)
        if source_key not in scopes:
            if (source_key not in derived or state['source_epoch'] != derived[source_key]['source_epoch']
                    or state['events'] != derived[source_key]['events'] or state.get('legacy_event_ids')):
                _reject('historical source outside finite closure')
            continue
        scope = scopes[source_key]
        if state['source_epoch'] != scope['source_epoch']:
            _reject('historical source epoch mismatch')
        for event_id, value in state['events'].items():
            binding = bindings.get((source_key, event_id))
            if binding is None or value != binding[2]['message_hash']:
                _reject('historical event outside finite closure')
            if event_id in state.get('legacy_event_ids', []):
                _reject('legacy excluded event is not a completed owner')
    # Existing unknown-policy inputs have not acquired logical source ownership.
    # They cannot be admitted under a finite alias contract by inventing an owner.
    if ledger._read(con, 'meta', 'source-unknown-policy') is not None:
        _reject('source unknown policy requires separate reviewed disjointness proof')


def register_contract(ledger, contract, *, approved_contract_sha256):
    """Store only an independently approved finite contract, atomically.

    Does not reserve owners, capture messages, change baselines or mark success.
    Real source-unknown-policy ledgers deliberately remain blocked in this v1.
    """
    value = validate_contract(contract, approved_contract_sha256=approved_contract_sha256,
                              bank=ledger.bank, ledger_root=ledger.storage_root)
    with ledger.locked(), ledger._connection(write=True) as con:
        scopes, bindings, events = _inventory(ledger, con, value)
        derived, cache = {}, {}
        historical = {}
        for (batch_id,) in con.execute('SELECT batch_id FROM batches'):
            batch, keys = _batch_evidence(ledger, con, batch_id, bindings, derived, cache)
            if set(historical).intersection(keys):
                _reject('multiple existing batches own the same canonical event')
            for canonical_key in keys:
                historical[canonical_key] = batch
                owner = ledger._read(con, 'meta', OWNER_PREFIX + canonical_key)
                if owner is not None and owner != _owner_record(canonical_key, events[canonical_key][0],
                                                                owner.get('contract_sha256'), batch):
                    _reject('owner record does not match complete immutable batch evidence')
        _state_checks(ledger, con, scopes, bindings, derived)
        for source_key in scopes:
            state = ledger._read(con, 'sources', source_key)
            for event_id in (state or {}).get('events', {}):
                if bindings[(source_key, event_id)][0] not in historical:
                    _reject('captured source event lacks durable batch owner')
        key = CONTRACT_PREFIX + approved_contract_sha256
        prior = ledger._read(con, 'meta', key)
        if prior is not None and prior != value:
            _reject('registered contract changed')
        if prior is None:
            ledger._put(con, 'meta', key, value)
            ledger._put(con, 'meta', REQUIRED_PREFIX + approved_contract_sha256,
                        _required_marker(ledger, approved_contract_sha256))
    return {'contract_sha256': approved_contract_sha256, 'registered': prior is None,
            'scopes': len(contract['scopes']), 'canonical_events': len(contract['events'])}


def prepare_capture(ledger, con, snapshot):
    """Common gate: called inside the bank lock AND the capture transaction."""
    scopes, bindings, events = _inventory(ledger, con)
    if not scopes:
        if _is_app(snapshot):
            _reject('App source has no independently approved finite contract')
        return None
    _check_scope(snapshot, scopes)
    owners, historical = {}, {}
    derived, cache = {}, {}
    # Inspect complete existing source batches, not an event-ID-only inventory.
    for (batch_id,) in con.execute('SELECT batch_id FROM batches'):
        batch, keys = _batch_evidence(ledger, con, batch_id, bindings, derived, cache)
        for key in keys:
            if key in historical:
                _reject('multiple existing batches own the same canonical event')
            historical[key] = batch
    _state_checks(ledger, con, scopes, bindings, derived)
    for key, (event, pin) in events.items():
        owner = ledger._read(con, 'meta', OWNER_PREFIX + key)
        batch = historical.get(key)
        if owner is not None:
            owner_contract = ledger._read(con, 'meta', CONTRACT_PREFIX + str(owner.get('contract_sha256')))
            if (batch is None or owner != _owner_record(key, event, owner.get('contract_sha256'), batch)
                    or owner_contract is None or event not in owner_contract['contract']['events']):
                _reject('owner record does not match complete immutable batch evidence')
        elif batch is not None:
            owner = _owner_record(key, event, pin, batch)
        if owner is not None:
            owners[key] = (owner, batch)
    for source_key, scope in scopes.items():
        state = ledger._read(con, 'sources', source_key)
        for event_id in (state or {}).get('events', {}):
            if bindings[(source_key, event_id)][0] not in owners:
                _reject('captured source event lacks durable batch owner')
    return {'bindings': bindings, 'events': events, 'owners': owners}


def _owner_record(key, event, pin, batch):
    return {'canonical_key': key, 'identity': event['identity'],
            'canonical_payload_sha256': event['canonical_payload_sha256'],
            'contract_sha256': pin, 'batch_id': batch['batch_id'],
            'payload_hash': batch['payload_hash'], 'source_key': batch['source_key'],
            'source_epoch': batch['source_epoch'], 'operation_id': batch['operation_id'],
            'document_id': batch['document_id']}


def select_unowned(plan, snapshot, unseen):
    if plan is None:
        return unseen
    return [m for m in unseen
            if plan['bindings'][(snapshot['source_key'], m['event_id'])][0] not in plan['owners']]


def commit_capture(ledger, con, plan, snapshot, batches):
    if plan is None:
        return
    for key, (owner, _) in plan['owners'].items():
        prior = ledger._read(con, 'meta', OWNER_PREFIX + key)
        if prior is None:
            ledger._put(con, 'meta', OWNER_PREFIX + key, owner)
        elif prior != owner:
            _reject('owner changed within capture transaction')
    for batch in batches:
        if batch is None:
            _reject('new canonical owner has no newly committed batch')
        for event_id in batch['source_event_ids']:
            key, event, _, pin = plan['bindings'][(snapshot['source_key'], event_id)]
            if key in plan['owners'] or ledger._read(con, 'meta', OWNER_PREFIX + key) is not None:
                _reject('canonical owner already exists')
            ledger._put(con, 'meta', OWNER_PREFIX + key, _owner_record(key, event, pin, batch))


def plan_snapshot(ledger, snapshot):
    """Read-only phase-aware plan. Never claims a reservation is completion."""
    validate_snapshot(snapshot)
    with ledger.locked(), ledger._connection() as con:
        plan = prepare_capture(ledger, con, snapshot)
        if plan is None:
            return {'status': 'ordinary-physical', 'events': []}
        rows = []
        for message in snapshot['messages']:
            key = plan['bindings'][(snapshot['source_key'], message['event_id'])][0]
            owned = plan['owners'].get(key)
            batch = owned[1] if owned else None
            rows.append({'canonical_key': key, 'event_id': message['event_id'],
                         'status': 'owned' if batch else 'unowned',
                         'phase': batch['phase'] if batch else None,
                         'batch_id': batch['batch_id'] if batch else None,
                         'operation_id': batch['operation_id'] if batch else None,
                         'completed': bool(batch and batch['phase'] == 'published')})
        return {'status': 'finite-proven', 'events': rows}
