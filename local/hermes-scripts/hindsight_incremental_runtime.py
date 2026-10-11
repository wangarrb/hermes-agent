#!/usr/bin/env python3
"""Concrete arrival-only REST/LLM adapter; publication remains a separate edge.

Never call the native whole-bank consolidation helper. A durable private lease
precedes every producer and records exact permitted scopes and operation IDs.
Unknown work leaves observations open with automatic consolidation OFF, rather
than quietly processing inherited pending facts or closing beneath live jobs.
"""
from __future__ import annotations

import copy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
import uuid
from urllib.parse import quote

from hindsight_incremental_delivery import _json_bytes, _secure_path, operation_receipt
from hindsight_incremental_ledger import PHASES
from hindsight_native_client import HindsightNativeClient
from hindsight_offline_retry import RemoteCallError, RetryDeferred, RetryPolicy, budget_for, classify_error

FACT_FIELDS = ('id', 'text', 'type', 'document_id', 'created_at')
OBS_TYPES = frozenset({'user_preference', 'project_decision', 'technical_lesson', 'risk',
                      'method_comparison', 'open_question', 'system_rule'})
OPEN_CONFIG = {'enable_auto_consolidation': False, 'enable_observations': True}


def _name(value):
    return isinstance(value, str) and bool(value.strip()) and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def _timestamp(value):
    if not _name(value):
        raise RemoteCallError('invalid_timestamp')
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise RemoteCallError('invalid_timestamp') from None
    return value


def _process_identity(pid):
    """PID reuse is not ownership. Unknown remote-host ownership is not stale."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except FileNotFoundError:
        return None
    return {'pid': pid, 'start_ticks': fields[19], 'boot_id': boot, 'host': socket.gethostname()}


def _owner_alive(owner):
    if owner.get('host') != socket.gethostname():
        return True
    current = _process_identity(owner['pid'])
    if current is None:
        return False
    return all(current[k] == owner.get(k) for k in current)


def _private_payload(payload, scope):
    if not isinstance(payload, dict):
        return False
    scopes, chosen = payload.get('observation_scopes'), payload.get('chosen_scope')
    # One precise field cannot authorize a contradictory shared/multiple scope.
    return ((scopes == [scope] or chosen == scope)
            and (scopes is None or scopes == [scope]) and (chosen is None or chosen == scope))


def _enabled_operation(operation):
    if not isinstance(operation, dict):
        raise RemoteCallError('operation_unknown')
    metadata = operation.get('result_metadata') or {}
    if not isinstance(metadata, dict):
        raise RemoteCallError('operation_unknown')
    if (metadata.get('status') in {'disabled', 'skipped'} or metadata.get('disabled') is True
            or metadata.get('skipped') is True or metadata.get('enable_observations') is False
            or metadata.get('reason') in {'observations_disabled', 'disabled'}):
        raise RemoteCallError('native_observations_disabled')
    children = operation.get('child_operations') or []
    if not isinstance(children, list):
        raise RemoteCallError('operation_unknown')
    for child in children:
        _enabled_operation(child)


class ArrivalRuntime:
    def __init__(self, config):
        self.config = config
        self.bank = config['bank']
        self.native = HindsightNativeClient(api=config['api_url'], bank=self.bank)
        self.ledger = None
        self._window = None
        self._owner_token = str(uuid.uuid4())
        self._publication_adapter = None
        self._native_isolation_adapter = None
        self._run_policy = None
        self.llm_calls = 0
        self.native_calls = 0
        # Native extraction/consolidation LLM costs are server-side and unknown.
        self.native_paid_calls = None

    @property
    def network_calls(self):
        return self.native_calls

    def begin_run(self, retry_policy):
        """Bind pre-delivery reads to the engine's actual shared call cap."""
        if not isinstance(retry_policy, RetryPolicy):
            raise TypeError('run retry policy must be RetryPolicy')
        self._run_policy = retry_policy
        self.llm_calls = self.native_calls = 0

    def _budget(self, *, cleanup=False):
        if not cleanup and self._run_policy is not None:
            return budget_for(self._run_policy)
        return budget_for(RetryPolicy(**((self.config.get('incremental') or {}).get('retry') or {})))

    def _request(self, method, suffix, *, budget, payload=None, params=None, reserved_timeout=None):
        attempt = 0
        while True:
            attempt += 1
            timeout = budget.charge() if reserved_timeout is None else reserved_timeout
            self.native_calls += 1
            try:
                return self.native.request(method, self.native.bank_path(suffix), payload=payload,
                                           params=params, timeout=timeout)
            except RetryDeferred:
                raise
            except Exception as error:
                category, transient = classify_error(error)
                if not transient or method != 'GET':
                    raise
                budget.wait(category, attempt, retry_after=getattr(error, 'retry_after', None))

    def _flags(self, budget):
        response = self._request('GET', 'config', budget=budget)
        if not isinstance(response, dict) or response.get('bank_id') != self.bank:
            raise RemoteCallError('bank_config_mismatch')
        config = response.get('config')
        keys = ('enable_auto_consolidation', 'enable_observations')
        if not isinstance(config, dict) or any(type(config.get(k)) is not bool for k in keys):
            raise RemoteCallError('bank_config_unknown')
        return {k: config[k] for k in keys}

    def _patch_flags(self, values, budget):
        self._request('PATCH', 'config', budget=budget, payload={'updates': values})
        if self._flags(budget) != values:
            raise RemoteCallError('bank_config_readback_mismatch')

    def _lease_path(self, ledger):
        if ledger.bank != self.bank:
            raise RemoteCallError('bank_mismatch')
        return _secure_path(ledger.root, ledger.root / 'runtime-window.json')

    def _load_lease(self, ledger):
        path = self._lease_path(ledger)
        if not path.exists():
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077
                    or info.st_uid != os.geteuid()):
                raise RemoteCallError('unsafe_window_lease')
            try:
                lease = json.load(handle)
            except (ValueError, UnicodeError):
                raise RemoteCallError('invalid_window_lease') from None
        if (not isinstance(lease, dict) or lease.get('bank') != self.bank
                or lease.get('api') != self.native.api or not _name(lease.get('window_id'))
                or lease.get('status') not in {'opening', 'active', 'cleanup_pending', 'restoring', 'restored'}):
            raise RemoteCallError('invalid_window_lease')
        owner = lease.get('owner')
        if (not isinstance(owner, dict) or type(owner.get('pid')) is not int
                or any(not _name(owner.get(k)) for k in ('start_ticks', 'boot_id', 'host', 'token'))):
            raise RemoteCallError('invalid_window_owner')
        scopes = lease.get('scopes')
        if not isinstance(scopes, dict) or lease.get('allowed_scopes') != [[tag] for tag in sorted(scopes)]:
            raise RemoteCallError('invalid_window_scope')
        for tag, record in scopes.items():
            current = self._batch(ledger.get_batch(record['batch_id']), confirmed=False)
            if tag != self._scope(current)[0] or record.get('retain_operation_id') != current['operation_id']:
                raise RemoteCallError('invalid_window_scope')
        before = lease.get('before_config')
        if (not isinstance(before, dict) or set(before) != set(OPEN_CONFIG)
                or any(type(v) is not bool for v in before.values()) or before['enable_observations']):
            raise RemoteCallError('invalid_window_before_config')
        return lease

    def _save_lease(self, ledger, lease):
        path = self._lease_path(ledger)
        raw = _json_bytes(lease)
        fd, name = tempfile.mkstemp(prefix='.runtime-window-', dir=ledger.root)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(ledger.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._window = lease

    def _owned(self):
        if self.ledger is None:
            raise RemoteCallError('window_ledger_missing')
        lease = self._load_lease(self.ledger)
        if (lease is None or lease['status'] == 'restored'
                or lease['owner']['token'] != self._owner_token
                or lease['owner']['pid'] != os.getpid() or not _owner_alive(lease['owner'])):
            raise RemoteCallError('window_owner_mismatch')
        return lease

    def _page(self, suffix, params, budget, preferred='items'):
        """Require exact pagination totals, stable IDs and complete enumeration."""
        rows, seen, total = [], set(), None
        limit = 100
        while True:
            page = self._request('GET', suffix, budget=budget, params={**params, 'limit': limit, 'offset': len(rows)})
            if not isinstance(page, dict):
                raise RemoteCallError('pagination_unknown')
            items, count = page.get(preferred), page.get('total')
            if (not isinstance(items, list) or type(count) is not int or count < 0
                    or (total is not None and count != total) or len(rows) + len(items) > count
                    or len(items) > limit or page.get('offset', len(rows)) != len(rows)):
                raise RemoteCallError('pagination_mismatch')
            total = count
            for row in items:
                if not isinstance(row, dict):
                    raise RemoteCallError('pagination_mismatch')
                identity = row.get('id') or row.get('operation_id')
                if not _name(identity) or identity in seen:
                    raise RemoteCallError('pagination_duplicate')
                seen.add(identity); rows.append(row)
            if len(rows) == total:
                return rows
            if not items:
                raise RemoteCallError('pagination_incomplete')

    def _active_operations(self, budget):
        return [op for status in ('pending', 'processing') for op in
                self._page('operations', {'status': status}, budget, preferred='operations')]

    def _count(self, state, budget, scope=None):
        params = {'consolidation_state': state, 'limit': 0, 'offset': 0}
        if scope is not None:
            if len(scope) != 1 or not scope[0].startswith('arrival-batch:'):
                raise RemoteCallError('unsafe_consolidation_scope')
            params.update(tags=scope, tags_match='all_strict')
        result = self._request('GET', 'memories/list', budget=budget, params=params)
        if (not isinstance(result, dict) or type(result.get('total')) is not int
                or result['total'] < 0 or result.get('items') != []):
            raise RemoteCallError('consolidation_count_unknown')
        return result['total']

    def _batch(self, batch, *, confirmed=True):
        if (not isinstance(batch, dict) or batch.get('bank') != self.bank
                or not re.fullmatch(r'[0-9a-f]{64}', batch.get('batch_id', ''))
                or batch.get('document_id') != f"hindsight-arrival::{self.bank}::{batch['batch_id']}"):
            raise RemoteCallError('batch_identity_mismatch')
        if confirmed:
            if batch.get('phase') not in PHASES[PHASES.index('retained'):]:
                raise RemoteCallError('batch_not_confirmed')
            receipt = batch.get('receipts', {}).get('retained')
            operation_receipt(receipt, batch['operation_id'], self.bank)
        return batch

    def _scope(self, batch):
        return [f"arrival-batch:{batch['batch_id']}"]

    def _admit(self, batch):
        lease = self._owned()
        tag = self._scope(batch)[0]
        if tag not in lease['scopes']:
            lease['scopes'][tag] = {'batch_id': batch['batch_id'], 'retain_operation_id': batch['operation_id'],
                                    'consolidation_operation_ids': [], 'trigger_unknown': False}
            lease['allowed_scopes'] = [[t] for t in sorted(lease['scopes'])]
            self._save_lease(self.ledger, lease)
        return lease, lease['scopes'][tag]

    def resume_window(self, ledger):
        """Claim only a dead/own persisted lease, without opening a producer.

        Call before no-op/publish-only returns too: cleanup from a previous
        crash is independent of whether new source work was admitted this run.
        """
        with ledger.locked():
            self.ledger = ledger
            lease = self._load_lease(ledger)
            if lease is None or lease['status'] == 'restored':
                return None
            if _owner_alive(lease['owner']) and lease['owner']['token'] != self._owner_token:
                raise RemoteCallError('window_live_owner')
            lease['owner'] = {**_process_identity(os.getpid()), 'token': self._owner_token}
            self._save_lease(ledger, lease)
            if self._flags(self._budget()) not in (OPEN_CONFIG, lease['before_config']):
                raise RemoteCallError('window_config_drift')
            return lease

    def open_window(self, ledger):
        with ledger.locked():
            self.ledger = ledger
            backend = self._isolation().attest()
            for current in ledger.list_batches():
                if current.get('phase') in {'captured', 'generated', 'submitted', 'unknown', 'retained'}:
                    self._isolation().verify(self._batch(current, confirmed=False))
            budget = self._budget()
            lease = self._load_lease(ledger)
            if lease is not None and lease['status'] != 'restored':
                if _owner_alive(lease['owner']) and lease['owner']['token'] != self._owner_token:
                    raise RemoteCallError('window_live_owner')
                lease['owner'] = {**_process_identity(os.getpid()), 'token': self._owner_token}
                self._save_lease(ledger, lease)
                flags = self._flags(budget)
                if flags not in (OPEN_CONFIG, lease['before_config']):
                    raise RemoteCallError('window_config_drift')
            else:
                flags = self._flags(budget)
                if flags['enable_observations']:
                    raise RemoteCallError('unowned_active_window')
                if self._active_operations(budget):
                    raise RemoteCallError('unowned_active_native_work')
                legacy_pending = self._count('pending', budget)
                # A restarted run may already have new retained facts before
                # opening this lease. Exclude those exact ledger-owned scopes
                # from the inherited inventory, without enumerating old rows.
                for current in ledger.list_batches():
                    if current.get('phase') in {'captured', 'generated', 'submitted', 'unknown', 'retained'}:
                        self._batch(current, confirmed=False)
                        legacy_pending -= self._count('pending', budget, self._scope(current))
                if legacy_pending < 0:
                    raise RemoteCallError('legacy_pending_count_mismatch')
                lease = {'schema': 'arrival-runtime-window/1', 'bank': self.bank, 'api': self.native.api,
                         'window_id': str(uuid.uuid4()), 'status': 'opening',
                         'owner': {**_process_identity(os.getpid()), 'token': self._owner_token},
                         'before_config': flags, 'allowed_scopes': [], 'scopes': {},
                         'legacy_pending': legacy_pending}
                self._save_lease(ledger, lease)
            if lease.get('native_backend') not in (None, backend):
                raise RemoteCallError('native_backend_changed')
            lease['native_backend'] = backend
            self._save_lease(ledger, lease)
            # The single PATCH closes automatic triggering BEFORE opening the
            # producer path; a two-request toggle could admit legacy backlog.
            if flags != OPEN_CONFIG:
                self._patch_flags(OPEN_CONFIG, budget)
            lease['status'] = 'active'
            self._save_lease(ledger, lease)
            return lease

    def retain_items(self, items, *, async_mode=True, operation_id=None, timeout=None):
        with self.ledger.locked():
            self._owned()
            if not isinstance(items, list) or not items:
                raise RemoteCallError('invalid_output')
            document_ids = {item.get('document_id') for item in items}
            if len(document_ids) != 1:
                raise RemoteCallError('retain_document_mismatch')
            document_id = next(iter(document_ids))
            if not isinstance(document_id, str):
                raise RemoteCallError('retain_document_mismatch')
            batch = self._batch(self.ledger.get_batch(document_id.split('::')[-1]), confirmed=False)
            if document_id != batch['document_id'] or operation_id != batch['operation_id'] or async_mode is not True:
                raise RemoteCallError('retain_document_mismatch')
            scope = self._scope(batch)
            if any(item.get('observation_scopes') != [scope] or item.get('tags') != scope
                   for item in items):
                raise RemoteCallError('retain_private_scope_missing')
            self._isolation().verify(batch, items=items)
            # Check only; mutating the delivery's sealed items invalidates UUID
            # payload binding and therefore is never done by this adapter.
            self._admit(batch)
            if self._flags(self._budget()) != OPEN_CONFIG:
                raise RemoteCallError('window_config_drift')
            self.native_calls += 1
            return self.native.retain_items(items, async_mode=async_mode, operation_id=operation_id, timeout=timeout)

    def get_operation(self, operation_id, *, timeout=None):
        self.native_calls += 1
        return self.native.get_operation(operation_id, timeout=timeout)

    def facts_for_batch(self, batch, *, _budget=None):
        batch = self._batch(batch)
        self._isolation().verify(batch)
        budget = _budget or self._budget()
        document_id = batch['document_id']
        document = self._request('GET', 'documents/' + quote(document_id, safe=''), budget=budget)
        if (not isinstance(document, dict) or document.get('id') != document_id
                or document.get('bank_id') != self.bank or type(document.get('memory_unit_count')) is not int):
            raise RemoteCallError('document_identity_mismatch')
        created = _timestamp(document.get('created_at'))
        facts, seen, counts = [], set(), {}
        for typ in ('world', 'experience'):
            records = self._page('memories/list', {'document_id': document_id, 'type': typ}, budget)
            counts[typ] = len(records)
            for row in records:
                # Native 0.9 list returns fact_type; single-memory GET uses
                # type. Do not let desired mock fields redefine the list wire.
                if (row.get('document_id') != document_id or row.get('fact_type') != typ
                        or ('type' in row and row['type'] != typ)
                        or not _name(row.get('text')) or row['id'] in seen):
                    raise RemoteCallError('fact_identity_mismatch')
                seen.add(row['id'])
                facts.append({'id': row['id'], 'text': row['text'], 'type': typ, 'document_id': document_id,
                              'created_at': _timestamp(row.get('created_at') or created)})
        types = document.get('nodes_by_fact_type')
        if types is None:
            valid = len(facts) == document['memory_unit_count']
        else:
            valid = (isinstance(types, dict) and all(type(v) is int and v >= 0 for v in types.values())
                     and sum(types.values()) == document['memory_unit_count']
                     and all(counts[t] == types.get(t, 0) for t in counts))
        if not valid:
            raise RemoteCallError('document_fact_count_mismatch')
        proof = self._isolation().verify(batch, expected_ids=[f['id'] for f in facts])
        if self.ledger is not None and self._window is not None and self._window['status'] == 'active':
            with self.ledger.locked():
                lease, state = self._admit(batch)
                state['native_isolation'] = proof
                self._save_lease(self.ledger, lease)
        return sorted(facts, key=lambda f: f['id'])

    def _operation(self, operation_id, budget, scope=None):
        response = self._request('GET', 'operations/' + quote(operation_id, safe=''), budget=budget,
                                 params={'include_payload': scope is not None})
        if scope is not None:
            payload = response.get('task_payload') if isinstance(response, dict) else None
            if not _private_payload(payload, scope):
                raise RemoteCallError('native_operation_scope_mismatch')
        # Fetch exact child detail when only child IDs are returned.
        if isinstance(response, dict) and response.get('child_operation_ids') and not response.get('child_operations'):
            response = dict(response)
            response['child_operations'] = [self._request('GET', 'operations/' + quote(oid, safe=''), budget=budget)
                                           for oid in response['child_operation_ids']]
        _enabled_operation(response)
        return operation_receipt(response, operation_id, self.bank)

    def _recover_trigger(self, lease, state, scope, budget):
        matched = []
        for entry in self._active_operations(budget):
            if entry.get('operation_type') not in {None, 'consolidate', 'consolidation'}:
                continue
            oid = entry.get('operation_id') or entry.get('id')
            operation = self._request('GET', 'operations/' + quote(oid, safe=''), budget=budget,
                                      params={'include_payload': True})
            payload = operation.get('task_payload') or {}
            if (operation.get('operation_id', operation.get('id')) != oid
                    or operation.get('bank_id', operation.get('bank', self.bank)) != self.bank):
                raise RemoteCallError('native_operation_mismatch')
            if _private_payload(payload, scope):
                matched.append(oid)
        if not matched:
            # An already-completed ACK-lost trigger without an ID cannot be
            # proved from zero queue counts; keep the uncertainty for review.
            raise RemoteCallError('native_trigger_unknown')
        state['consolidation_operation_ids'] = sorted(set(state['consolidation_operation_ids'] + matched))
        state['trigger_unknown'] = False
        self._save_lease(self.ledger, lease)

    def consolidate_batch(self, batch, retry_policy):
        batch = self._batch(batch)
        budget = budget_for(retry_policy)
        with self.ledger.locked():
            lease, state = self._admit(batch)
            scope = self._scope(batch)
            previous = state.get('native_isolation')
            if previous is None:
                self.facts_for_batch(batch, _budget=budget)
                lease = self._owned()
                state = lease['scopes'][scope[0]]
                previous = state['native_isolation']
            proof = self._isolation().verify(batch, expected_ids=previous['source_ids'])
            state['native_isolation'] = proof
            self._save_lease(self.ledger, lease)
            if state['trigger_unknown']:
                self._recover_trigger(lease, state, scope, budget)
            if self._flags(budget) != OPEN_CONFIG:
                raise RemoteCallError('native_observations_disabled')
            pending = self._count('pending', budget, scope)
            if self._count('failed', budget, scope):
                raise RemoteCallError('native_scoped_facts_failed')
            if pending and not state['consolidation_operation_ids']:
                if state['trigger_unknown']:
                    raise RemoteCallError('native_trigger_unknown')
                # A pre-I/O cap/deadline refusal is NOT ACK loss. Reserve the
                # bounded request before recording uncertainty, then consume
                # that same reservation exactly once in the one-shot POST.
                timeout = budget.charge()
                state['trigger_unknown'] = True
                self._save_lease(self.ledger, lease)
                # POST is never blindly retried after timeout/ACK loss: native
                # consolidation has no caller operation UUID in this schema.
                ack = self._request('POST', 'consolidate', budget=budget,
                    payload={'observation_scopes': [scope]}, reserved_timeout=timeout)
                if not isinstance(ack, dict) or not _name(ack.get('operation_id')):
                    raise RemoteCallError('native_trigger_unknown')
                state['consolidation_operation_ids'].append(ack['operation_id'])
                state['trigger_unknown'] = False
                self._save_lease(self.ledger, lease)
            attempt = 0
            while True:
                attempt += 1
                try:
                    for oid in state['consolidation_operation_ids']:
                        self._operation(oid, budget, scope)
                    if self._flags(budget) != OPEN_CONFIG:
                        raise RemoteCallError('native_observations_disabled')
                    pending = self._count('pending', budget, scope)
                    failed = self._count('failed', budget, scope)
                    if failed:
                        raise RemoteCallError('native_scoped_facts_failed')
                    if pending or self._active_operations(budget):
                        raise RemoteCallError('operation_pending')
                    proof = self._isolation().verify(batch, expected_ids=previous['source_ids'])
                    state['native_isolation'] = proof
                    self._save_lease(self.ledger, lease)
                    return {'bank': self.bank, 'scope': scope, 'pending_in_scope': 0,
                            'failed_in_scope': 0, 'ready': True, 'legacy_pending': lease['legacy_pending'],
                            'operation_ids': list(state['consolidation_operation_ids']), 'native_isolation': proof}
                except RetryDeferred:
                    raise
                except Exception as error:
                    category, transient = classify_error(error)
                    if not transient:
                        raise
                    budget.wait(category, attempt, retry_after=budget.policy.poll_s)
                    if category == 'operation_pending':
                        attempt = 0

    def cleanup_window(self, ledger, window):
        with ledger.locked():
            self.ledger = ledger
            lease = self._owned()
            if not isinstance(window, dict) or window.get('window_id') != lease['window_id']:
                raise RemoteCallError('window_owner_mismatch')
            # Exhausting the producer's budget must not prevent bounded safe
            # cleanup. Report these extra native reads separately as well.
            budget = self._budget(cleanup=True)
            try:
                self._isolation().attest()
                # Listing active work covers native continuations/requeue IDs,
                # not just the trigger's root operation. Nothing is drained here.
                if self._active_operations(budget):
                    raise RemoteCallError('operation_pending')
                for tag, state in lease['scopes'].items():
                    if state['trigger_unknown']:
                        raise RemoteCallError('native_trigger_unknown')
                    scope = [tag]
                    current = ledger.get_batch(state['batch_id'])
                    submitted = current.get('receipts', {}).get('submitted') or current.get('receipts', {}).get('unknown')
                    if submitted or current.get('phase') in PHASES[PHASES.index('retained'):]:
                        self._operation(state['retain_operation_id'], budget)
                    for oid in state['consolidation_operation_ids']:
                        self._operation(oid, budget, scope)
                    if self._count('pending', budget, scope) or self._count('failed', budget, scope):
                        raise RemoteCallError('operation_pending')
                flags = self._flags(budget)
                if flags not in (OPEN_CONFIG, lease['before_config']):
                    raise RemoteCallError('window_config_drift')
                lease['status'] = 'restoring'
                self._save_lease(ledger, lease)
                if flags != lease['before_config']:
                    self._patch_flags(lease['before_config'], budget)
                lease['status'] = 'restored'
                self._save_lease(ledger, lease)
                return 'restored'
            except Exception as error:
                lease['status'] = 'cleanup_pending'
                lease['cleanup_error_class'] = classify_error(error)[0]
                self._save_lease(ledger, lease)
                return 'pending'

    def _frozen_facts(self, frozen):
        if (not isinstance(frozen, dict) or frozen.get('bank') != self.bank
                or frozen.get('kind') != 'daily_summary' or not frozen.get('parent_batch_ids')):
            raise RemoteCallError('invalid_summary_input')
        messages = frozen.get('messages')
        max_chars = int((self.config.get('incremental') or {}).get('max_summary_chars', 60000))
        if not 1 <= max_chars <= 60000 or not isinstance(messages, list) or not messages:
            raise RemoteCallError('summary_input_budget')
        if sum(len(m.get('content', '')) for m in messages) > max_chars:
            raise RemoteCallError('summary_input_budget')
        facts, ids = [], set()
        for message in messages:
            try:
                fact = json.loads(message['content'])
            except (ValueError, TypeError, KeyError):
                raise RemoteCallError('invalid_summary_input') from None
            if (not isinstance(fact, dict) or set(fact) != set(FACT_FIELDS)
                    or any(not _name(fact.get(k)) for k in FACT_FIELDS)
                    or fact['type'] not in {'world', 'experience'} or fact['id'] in ids
                    or message.get('origin') != fact['id'] or message.get('timestamp') != fact['created_at']
                    or fact['document_id'] not in {f'hindsight-arrival::{self.bank}::{bid}' for bid in frozen['parent_batch_ids']}):
                raise RemoteCallError('invalid_summary_input')
            _timestamp(fact['created_at']); ids.add(fact['id']); facts.append(fact)
        return facts

    def _summary_profile(self):
        from hindsight_minimax_import import get_llm_profile
        overrides = (self.config.get('incremental') or {}).get('llm_override')
        try:
            if overrides is None:
                profile = get_llm_profile(self.config.get('llm_profile'))
            else:
                if not isinstance(overrides, dict) or not set(overrides) <= {'model', 'base_url', 'api_key_env', 'provider'}:
                    raise RemoteCallError('llm_configuration')
                explicit = {('hindsight_provider' if k == 'provider' else k): v for k, v in overrides.items()}
                profile = get_llm_profile(self.config.get('llm_profile'), overrides=explicit)
        except SystemExit:
            raise RemoteCallError('llm_configuration') from None
        if profile.get('hindsight_provider', 'openai') not in {'openai', 'minimax'}:
            # reflect.call_llm uses an OpenAI-compatible chat JSON wire, not
            # Anthropic's protocol. Never accept a flag and silently ignore it.
            raise RemoteCallError('unsupported_llm_provider')
        return profile

    def preflight(self, *, require_llm):
        """Verify this interpreter/configured SQL edge before paid producers.

        No-op never calls this. The SQL probe uses an empty exact ID set, so it
        checks the real schema/driver without exporting old memory bodies.
        """
        publication = self._publication()
        backend = self._isolation().attest()
        from hindsight_incremental_merge import _absolute, _read
        from hindsight_incremental_publication import _read_remote
        for label in ('generic', 'local'):
            value = publication.config.get(label + '_benchmark')
            if not _name(value):
                raise RemoteCallError('publication_configuration')
            try:
                if not _read(_absolute(value)).strip():
                    raise ValueError('empty benchmark')
            except (ValueError, OSError):
                raise RemoteCallError('publication_configuration') from None
        if require_llm:
            self._summary_profile()
        try:
            with publication._readonly() as conn:
                documents, units = _read_remote(conn, self.bank, [])
                if documents or units:
                    raise ValueError('empty-ID probe returned data')
        except Exception:
            raise RemoteCallError('publication_sql_preflight') from None
        return {'bank': self.bank, 'status': 'ready', 'read_only': True,
                'sql_driver_and_schema': True, 'benchmark_paths': True,
                'llm_profile_checked': bool(require_llm), 'native_backend': backend}

    def generate_summary(self, frozeninput, *, retry_policy=None):
        facts = self._frozen_facts(frozeninput)
        if self.ledger is None:
            raise RemoteCallError('window_ledger_missing')
        self._isolation().attest()
        for bid in frozeninput['parent_batch_ids']:
            parent = self._batch(self.ledger.get_batch(bid))
            self._isolation().verify(parent, expected_ids=[f['id'] for f in facts if f['document_id'] == parent['document_id']])
        items = frozeninput.get('items')
        if not isinstance(items, list) or len(items) != 1 or not _name(items[0].get('document_id')):
            raise RemoteCallError('invalid_summary_document')
        import offline_hindsight_reflect_consolidate as reflect
        profile = self._summary_profile()
        content = ('Only the following frozen new-document facts are evidence. Do not retrieve or infer history. '
                   'Each canonical_observations row MUST include nonempty id, topic, insight, type, applicability, '
                   'evidence_ids (exact input fact/document IDs), confidence, valid_from and valid_until. '
                   'Keep source/evidence identifiers exactly; do not replace them with source indexes.\n' +
                   json.dumps(facts, ensure_ascii=False, sort_keys=True))
        unit = reflect.ReflectUnit(scope='daily', period='arrival-incremental', topic='new-arrival-facts',
            index=0, content=content, source_count=len(facts), source_ids=[f['id'] for f in facts],
            date_range_start=min(f['created_at'] for f in facts), date_range_end=max(f['created_at'] for f in facts))
        budget = budget_for(retry_policy)
        before = budget.calls
        try:
            obj, _raw = reflect.call_llm(unit, model=profile['model'], base_url=profile['base_url'],
                api_key_env=profile['api_key_env'], label=profile['label'],
                response_format=profile.get('response_format', True), emit_observations=True,
                output_language=self.config.get('output_language', 'auto'), retry_policy=budget)
        except SystemExit:
            raise RemoteCallError('llm_configuration') from None
        finally:
            self.llm_calls += budget.calls - before
        if not isinstance(obj, dict) or not isinstance(obj.get('canonical_observations'), list):
            raise RemoteCallError('invalid_output')
        observations = copy.deepcopy(obj['canonical_observations'])
        fact_ids = {f['id'] for f in facts}
        docs = {f['document_id'] for f in facts}
        for row in observations:
            if not isinstance(row, dict) or row.get('bank', self.bank) != self.bank:
                raise RemoteCallError('invalid_output')
            row['bank'] = self.bank
            evidence = row.get('evidence_ids')
            if not isinstance(evidence, list) or not evidence or any(e not in fact_ids | docs for e in evidence):
                raise RemoteCallError('invalid_output')
            used = [f for f in facts if f['id'] in evidence or f['document_id'] in evidence]
            row.setdefault('source_documents', sorted({f['document_id'] for f in used}))
            row.setdefault('source_ids', list(evidence))
            row.setdefault('source_facts', copy.deepcopy(used))
        artifact = {'bank': self.bank, 'canonical_observations': observations,
                    'source_facts': facts, 'source_ids': [f['id'] for f in facts],
                    'source_documents': sorted(docs), 'llm_json': obj}
        self.validate_summary(artifact)
        return {'items': [{'document_id': items[0]['document_id'],
                           'content': json.dumps(artifact, ensure_ascii=False, sort_keys=True),
                           'timestamp': facts[0]['created_at'], 'metadata': {'bank': self.bank, 'kind': 'daily_summary'}}],
                'artifact': artifact}

    def validate_summary(self, artifact):
        if not isinstance(artifact, dict) or artifact.get('bank') != self.bank:
            raise RemoteCallError('invalid_output')
        facts = artifact.get('source_facts')
        rows = artifact.get('canonical_observations')
        if not isinstance(facts, list) or not facts or not isinstance(rows, list) or not rows:
            raise RemoteCallError('invalid_output')
        ids = {f['id'] for f in facts}
        docs = {f['document_id'] for f in facts}
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or row.get('bank') != self.bank
                    or any(not _name(row.get(k)) for k in ('id', 'topic', 'insight', 'type', 'applicability'))
                    or row['type'] not in OBS_TYPES or row['id'] in seen
                    or row.get('confidence') not in {'high', 'medium', 'low'}
                    or 'valid_from' not in row or 'valid_until' not in row):
                raise RemoteCallError('invalid_output')
            seen.add(row['id'])
            for field, allowed in (('evidence_ids', ids | docs), ('source_ids', ids | docs), ('source_documents', docs)):
                values = row.get(field)
                if not isinstance(values, list) or not values or any(not isinstance(v, str) or v not in allowed for v in values):
                    raise RemoteCallError('invalid_output')
        _json_bytes(artifact)
        return True

    def _isolation(self):
        if self._native_isolation_adapter is None:
            from hindsight_incremental_native_isolation import NativeIsolation
            self._native_isolation_adapter = NativeIsolation(self)
        return self._native_isolation_adapter

    def _publication(self):
        if self._publication_adapter is None:
            from hindsight_incremental_publication import PublicationAdapter
            self._publication_adapter = PublicationAdapter(self.config, self.bank)
        return self._publication_adapter

    def _scoped_proof(self, manifest):
        if self.ledger is None or manifest.get('bank') != self.bank:
            raise RemoteCallError('publication_bank_mismatch')
        proofs = []
        for bid in manifest['consumed_batch_ids']:
            batch = self.ledger.get_batch(bid)
            proof = batch.get('receipts', {}).get('consolidated', {}).get('scoped_proof')
            if (not isinstance(proof, dict) or proof.get('bank') != self.bank
                    or proof.get('scope') != self._scope(batch) or proof.get('ready') is not True
                    or proof.get('pending_in_scope') != 0):
                raise RemoteCallError('scoped_drain_unverified')
            proofs.append(proof)
        if not proofs:
            raise RemoteCallError('scoped_drain_unverified')
        return {'bank': self.bank, 'ready': True, 'pending_in_scope': 0,
                'scopes': [p['scope'] for p in proofs], 'batches': proofs,
                'legacy_pending': max((p.get('legacy_pending', 0) for p in proofs))}

    def gate(self, manifest):
        return self._publication().gate(manifest, ledger=self.ledger, scoped_proof=self._scoped_proof(manifest))

    def publish_generation(self, manifest, gate):
        return self._publication().publish_generation(manifest, gate, ledger=self.ledger)

    def readback_generation(self, manifest):
        return self._publication().readback_generation(manifest, ledger=self.ledger)

    def export_seed(self, path):
        return self._publication().export_seed(path)
