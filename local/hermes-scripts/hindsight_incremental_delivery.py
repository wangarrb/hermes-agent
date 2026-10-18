"""Frozen per-unit delivery, exact async receipts and bounded resumption.

API: generate(frozen_input, *, retry_policy=RetryBudget) returns
{'items': [MemoryItem, ...], 'artifact': schema_valid_dict}. validate(artifact)
returns True (or None after raising on invalid data). Raw-source delivery omits
both callbacks and uses the frozen {'items': [...]} payload. Generated receipts
never label raw retain data as a daily LLM artifact. Call under ledger.locked().
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path

import hindsight_incremental_merge as merge
from hindsight_offline_retry import RemoteCallError, RetryDeferred, budget_for, classify_error

TERMINAL_FAILURE = frozenset({'failed', 'cancelled', 'canceled', 'error'})
SUCCESS_PHASES = frozenset({'retained', 'consolidated', 'locally_merged', 'gated', 'published', 'readback_verified'})


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _secure_path(root, path):
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(path)) or '..' in Path(path).parts:
        raise RemoteCallError('unsafe_payload_path')
    root, path = Path(root).absolute(), Path(path).absolute()
    if not path.is_relative_to(root):
        raise RemoteCallError('unsafe_payload_path')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise RemoteCallError('unsafe_payload_path')
    return path


def _read(root, path, expected_hash):
    path = _secure_path(root, path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077:
                raise RemoteCallError('unsafe_payload_path')
            if not isinstance(expected_hash, str) or merge._payload_info(merge._file_chunks(stream))['sha256'] != expected_hash:
                raise RemoteCallError('payload_mismatch')
            stream.seek(0)
            digest = hashlib.sha256()
            try:
                value = merge._load_json(stream, digest=digest)
            except (ValueError, UnicodeError):
                raise RemoteCallError('invalid_output') from None
            if digest.hexdigest() != expected_hash:
                raise RemoteCallError('payload_mismatch')
            return value
    except OSError:
        raise RemoteCallError('payload_missing') from None


def _freeze(root, value, suffix):
    digest = merge.canonical_hash(value)
    directory = _secure_path(root, Path(root) / 'delivery-artifacts')
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        raise RemoteCallError('unsafe_payload_path')
    path = _secure_path(root, directory / f'{digest}.{suffix}.json')
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        _read(root, path, digest)
    else:
        with os.fdopen(fd, 'wb') as stream:
            written = hashlib.sha256()
            for chunk in merge._json_chunks(value):
                stream.write(chunk)
                written.update(chunk)
            if written.hexdigest() != digest:
                raise RemoteCallError('payload_mismatch')
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400)
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    return str(path), digest


def _items(value):
    items = value.get('items') if isinstance(value, dict) else None
    if not isinstance(items, list) or not items:
        raise RemoteCallError('invalid_output')
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('content'), str) or not item['content'].strip():
            raise RemoteCallError('invalid_output')
        if not isinstance(item.get('document_id'), str) or not item['document_id'].strip():
            raise RemoteCallError('invalid_output')
    return items


def _wire_items(output, source, batch_id):
    """Freeze the API representation separately from the original input hash."""
    from datetime import datetime
    from hindsight_session_retain_runner import normalize_metadata_for_hindsight

    messages = source.get('messages', []) if isinstance(source, dict) else []
    timestamp = next((m.get('timestamp') for m in messages if m.get('timestamp')), None)
    items = []
    for original in _items(output):
        item = dict(original)
        metadata = item.get('metadata')
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise RemoteCallError('invalid_output')
        try:
            item['metadata'] = normalize_metadata_for_hindsight(metadata)
            scope_tag = f'arrival-batch:{batch_id}'
            tags = item.get('tags') or []
            if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags):
                raise ValueError('invalid tags')
            item['tags'] = sorted(set(tags) | {scope_tag})
            # Never inherit a model-supplied global/shared consolidation scope.
            item['observation_scopes'] = [[scope_tag]]
            # MemoryItem uses timestamp; event_date alone is ignored by the API.
            event_date = item.pop('event_date', None)
            item_timestamp = item.get('timestamp') or event_date or timestamp
            if item_timestamp is not None:
                if not isinstance(item_timestamp, str):
                    raise ValueError('invalid timestamp')
                if item_timestamp.lower() != 'unset':
                    datetime.fromisoformat(item_timestamp.replace('Z', '+00:00'))
                item['timestamp'] = item_timestamp
            _json_bytes(item)  # Nonfinite/unserializable metadata must never reach POST.
        except (ValueError, TypeError, OverflowError):
            raise RemoteCallError('invalid_output') from None
        items.append(item)
    return items


def _validate_artifact(artifact, validate):
    if not isinstance(artifact, dict) or not artifact:
        raise RemoteCallError('invalid_output')
    if validate is None:
        raise RemoteCallError('invalid_output')
    if validate is not None:
        try:
            result = validate(artifact)
        except (ValueError, TypeError, KeyError):
            raise RemoteCallError('invalid_output') from None
        if result is not None and result is not True:
            raise RemoteCallError('invalid_output')


def operation_receipt(operation, operation_id, bank):
    """Reduce server response to non-sensitive proof; verify child completeness."""
    if (not isinstance(operation_id, str) or not operation_id.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in operation_id)):
        raise RemoteCallError('operation_unknown')
    if not isinstance(operation, dict):
        raise RemoteCallError('operation_unknown')
    remote_id = operation.get('operation_id') or operation.get('id')
    if remote_id != operation_id:
        raise RemoteCallError('operation_mismatch')
    if operation.get('bank_id', operation.get('bank', bank)) != bank:
        raise RemoteCallError('bank_mismatch')
    status = operation.get('status')
    if status in TERMINAL_FAILURE:
        raise RemoteCallError('operation_failed')
    children = operation.get('child_operations', operation.get('children', []))
    # Installed 0.9.0 represents a leaf's absent children as JSON null.
    if children is None:
        children = []
    if not isinstance(children, list):
        raise RemoteCallError('operation_unknown')
    child_ids = operation.get('child_operation_ids') or []
    if child_ids:
        actual_ids = [(child.get('operation_id') or child.get('id')) if isinstance(child, dict) else None
                      for child in children]
        if (not isinstance(child_ids, list) or any(not isinstance(i, str) for i in actual_ids + child_ids)
                or len(set(actual_ids)) != len(actual_ids) or len(actual_ids) != len(child_ids)
                or set(actual_ids) != set(child_ids)):
            raise RemoteCallError('operation_pending')
    proofs = []
    for child in children:
        if not isinstance(child, dict):
            raise RemoteCallError('operation_unknown')
        child_id = child.get('operation_id') or child.get('id')
        if not child_id:
            raise RemoteCallError('operation_unknown')
        proofs.append(operation_receipt(child, child_id, bank))
    counts = operation.get('child_operation_counts') or operation.get('children_counts') or {}
    total = operation.get('total_children', counts.get('total', len(children)))
    completed = operation.get('completed_children', counts.get('completed', len(children)))
    failed = operation.get('failed_children', counts.get('failed', 0))
    if failed:
        raise RemoteCallError('operation_failed')
    if status != 'completed' or total != completed or total != len(children):
        raise RemoteCallError('operation_pending')
    return {'operation_id': operation_id, 'bank': bank, 'status': 'completed', 'children': proofs}


def _exact_operation(client, operation_id, bank, budget):
    attempt = 0
    while True:
        attempt += 1
        timeout = budget.charge()
        try:
            try:
                op = client.get_operation(operation_id, timeout=timeout)
            except Exception as error:
                if classify_error(error)[0] == 'http_404':
                    return None
                raise
            # A 200 not_found is explicit absence, but only for this exact bank/ID.
            if isinstance(op, dict) and op.get('status') == 'not_found':
                if (op.get('operation_id') or op.get('id')) != operation_id:
                    raise RemoteCallError('operation_mismatch')
                if op.get('bank_id', op.get('bank', bank)) != bank:
                    raise RemoteCallError('bank_mismatch')
                return None
            # Child IDs without details require exact child GET, not queue totals.
            if isinstance(op, dict) and op.get('child_operation_ids') and not op.get('child_operations'):
                op = dict(op)
                op['child_operations'] = [client.get_operation(child_id, timeout=budget.charge())
                                          for child_id in op['child_operation_ids']]
            return operation_receipt(op, operation_id, bank)
        except RetryDeferred:
            raise
        except Exception as error:
            category, transient = classify_error(error)
            if not transient:
                raise RemoteCallError(category) from None
            budget.wait(category, attempt, retry_after=budget.policy.poll_s if category == 'operation_pending'
                        else getattr(error, 'retry_after', None))
            if category == 'operation_pending':
                attempt = 0


_PENDING = object()


def _exact_operation_step(client, operation_id, bank, budget):
    """One poll: the receipt when complete, None when absent, ``_PENDING``
    while the operation still runs.  Never sleeps; the run scheduler paces
    polls so in-flight operations overlap instead of serializing wall time."""
    timeout = budget.charge()
    try:
        op = client.get_operation(operation_id, timeout=timeout)
    except Exception as error:
        if classify_error(error)[0] == 'http_404':
            return None
        raise
    if isinstance(op, dict) and op.get('status') == 'not_found':
        if (op.get('operation_id') or op.get('id')) != operation_id:
            raise RemoteCallError('operation_mismatch')
        if op.get('bank_id', op.get('bank', bank)) != bank:
            raise RemoteCallError('bank_mismatch')
        return None
    if isinstance(op, dict) and op.get('child_operation_ids') and not op.get('child_operations'):
        op = dict(op)
        op['child_operations'] = [client.get_operation(child_id, timeout=budget.charge())
                                  for child_id in op['child_operation_ids']]
    try:
        return operation_receipt(op, operation_id, bank)
    except RemoteCallError as error:
        if error.error_class == 'operation_pending':
            return _PENDING
        raise


def deliver_batch(ledger, batch: dict, *, client, generate=None, validate=None,
                  retry_policy=None, single_poll=False) -> dict:
    """Return status/phase/receipt/attempts/calls/error_class; never content.

    ``single_poll`` advances one poll step at most and returns status
    ``pending`` instead of waiting or deferring; the caller's rotation
    scheduler owns pacing and per-attempt deadlines."""
    budget = budget_for(retry_policy)
    batch_id = batch['batch_id']
    current = ledger.get_batch(batch_id)
    starting_attempts = int(current.get('attempts') or 0)
    result = {'batch_id': batch_id, 'phase': current['phase'], 'status': 'no_op',
              'receipt': None, 'operation_id': None, 'attempts': starting_attempts,
              'calls': 0, 'error_class': None, 'next_retry_at': None}
    already_retained = current['phase'] in SUCCESS_PHASES
    if not already_retained and current.get('next_retry_at', 0) > budget.policy.wall_clock():
        result.update(status='deferred', next_retry_at=current['next_retry_at'], error_class=current.get('last_error_class'))
        return result
    try:
        if client.bank != current['bank']:
            raise RemoteCallError('bank_mismatch')
        root = ledger.root
        source_hash = current.get('payload_hash') or current.get('payload_sha256')
        source = _read(root, current['payload_path'], source_hash)
        generated = current.get('receipts', {}).get('generated')
        if generated is None:
            if current['phase'] != 'captured':
                raise RemoteCallError('generated_receipt_missing')
            if generate is not None and not callable(validate):
                raise RemoteCallError('invalid_output')
            output = generate(source, retry_policy=budget) if generate else source
            items = _wire_items(output, source, batch_id)
            artifact_path = artifact_hash = None
            if generate is not None:
                artifact = output.get('artifact')
                _validate_artifact(artifact, validate)
                artifact_path, artifact_hash = _freeze(root, artifact, 'artifact')
            operation_id = current.get('operation_id') or str(uuid.uuid5(uuid.NAMESPACE_URL, f"hindsight:{current['bank']}:{batch_id}"))
            if str(uuid.UUID(operation_id)) != operation_id:
                raise RemoteCallError('operation_mismatch')
            payload = {'async': True, 'items': items, 'operation_id': operation_id}
            payload_path, payload_hash = _freeze(root, payload, 'retain')
            generated = {'bank': current['bank'], 'batch_id': batch_id, 'input_hash': source_hash,
                         'operation_id': operation_id, 'payload_hash': payload_hash, 'payload_path': payload_path,
                         'artifact_path': artifact_path, 'artifact_hash': artifact_hash,
                         'artifact_kind': 'summary' if generate else 'raw_source'}
            current = ledger.set_phase(batch_id, 'generated', receipt=generated)
        if (generated.get('input_hash') != source_hash or generated.get('bank') != current['bank']
                or generated.get('batch_id') != batch_id):
            raise RemoteCallError('payload_mismatch')
        payload = _read(root, generated['payload_path'], generated['payload_hash'])
        items = _items(payload)
        operation_id = generated['operation_id']
        result['operation_id'] = operation_id
        if str(uuid.UUID(operation_id)) != operation_id:
            raise RemoteCallError('operation_mismatch')
        if current.get('operation_payload_hash') not in (None, generated['payload_hash']):
            raise RemoteCallError('payload_mismatch')
        if _wire_items(payload, source, batch_id) != items:
            raise RemoteCallError('payload_mismatch')
        if payload.get('operation_id') != operation_id or payload.get('async') is not True:
            raise RemoteCallError('payload_mismatch')
        if current.get('operation_id') and current['operation_id'] != operation_id:
            raise RemoteCallError('operation_mismatch')
        if already_retained:
            receipt = current.get('receipts', {}).get('retained')
            expected = {'bank': current['bank'], 'batch_id': batch_id, 'operation_id': operation_id,
                        'input_hash': source_hash, 'payload_hash': generated['payload_hash'], 'status': 'completed'}
            if (not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items())
                    or not isinstance(receipt.get('children'), list)):
                raise RemoteCallError('retained_receipt_mismatch')
            operation_receipt(receipt, operation_id, current['bank'])
            if generated.get('artifact_kind') == 'summary':
                _read(root, generated['artifact_path'], generated['artifact_hash'])
            result['receipt'] = receipt
            return result
        if generated.get('artifact_kind') == 'summary':
            _validate_artifact(_read(root, generated['artifact_path'], generated['artifact_hash']), validate)
        for phase in ('submitted', 'unknown'):
            binding = current.get('receipts', {}).get(phase)
            if binding and (binding.get('operation_id') != operation_id or binding.get('payload_hash') != generated['payload_hash']):
                raise RemoteCallError('payload_mismatch')
        receipt = None
        if current['phase'] in {'submitted', 'unknown'}:
            if single_poll:
                receipt = _exact_operation_step(client, operation_id, current['bank'], budget)
                if receipt is _PENDING:
                    result.update(status='pending', phase=current['phase'],
                                  calls=budget.calls, attempts=starting_attempts + budget.calls)
                    return result
            else:
                receipt = _exact_operation(client, operation_id, current['bank'], budget)
        post_attempts = 0
        while receipt is None:
            post_attempts += 1
            timeout = budget.charge()
            # Persist uncertainty BEFORE POST so a crash is reconciled by GET.
            binding = {'bank': current['bank'], 'batch_id': batch_id, 'operation_id': operation_id,
                       'payload_hash': generated['payload_hash']}
            current = ledger.set_phase(batch_id, 'submitted', receipt=binding)
            try:
                ack = client.retain_items(items, async_mode=True, operation_id=operation_id, timeout=timeout)
                if not isinstance(ack, dict) or ack.get('operation_id') != operation_id:
                    raise RemoteCallError('operation_mismatch')
            except RetryDeferred:
                # A local pre-I/O refusal must preserve its exact retry fence;
                # converting it into RemoteCallError would block it forever.
                raise
            except Exception as error:
                category, transient = classify_error(error)
                if not transient:
                    # A definite rejection is not a remote success. Preserve
                    # generated artifact; manual intervention may repair auth.
                    raise RemoteCallError(category) from None
                current = ledger.set_phase(batch_id, 'unknown', receipt=binding)
                if single_poll:
                    receipt = _exact_operation_step(client, operation_id, current['bank'], budget)
                    if receipt is _PENDING:
                        result.update(status='pending', phase=current['phase'],
                                      calls=budget.calls, attempts=starting_attempts + budget.calls)
                        return result
                    if receipt is None:
                        budget.defer('operation_unknown')
                    continue
                receipt = _exact_operation(client, operation_id, current['bank'], budget)
                if receipt is None:
                    budget.wait(category, post_attempts, retry_after=getattr(error, 'retry_after', None))
                continue
            if single_poll:
                receipt = _exact_operation_step(client, operation_id, current['bank'], budget)
                if receipt is _PENDING:
                    result.update(status='pending', phase=current['phase'],
                                  calls=budget.calls, attempts=starting_attempts + budget.calls)
                    return result
            else:
                receipt = _exact_operation(client, operation_id, current['bank'], budget)
            if receipt is None:
                budget.defer('operation_unknown')
        receipt.update(batch_id=batch_id, input_hash=source_hash, payload_hash=generated['payload_hash'],
                       attempts=starting_attempts + budget.calls, calls=budget.calls)
        current = ledger.set_phase(batch_id, 'retained', receipt=receipt)
        result.update(status='completed', receipt=receipt)
    except RetryDeferred as error:
        current = ledger.defer(batch_id, error_class=error.error_class, next_retry_at=error.next_retry_at,
                               attempts=starting_attempts + budget.calls)
        result.update(status='deferred', error_class=error.error_class, next_retry_at=error.next_retry_at)
    except Exception as error:
        category, transient = classify_error(error)
        next_retry_at = budget.policy.wall_clock() + budget.policy.defer_delay if transient else 1e30
        if not already_retained:
            current = ledger.defer(batch_id, error_class=category, next_retry_at=next_retry_at,
                                   attempts=starting_attempts + budget.calls)
        result.update(status='deferred' if transient else 'blocked', error_class=category, next_retry_at=next_retry_at)
    result.update(phase=current['phase'], calls=budget.calls, attempts=starting_attempts + budget.calls)
    return result
