#!/usr/bin/env python3
"""Arrival-only orchestration: frozen units, scoped work and exact generation receipts.

The concrete REST/DB adapter is a sibling module. No legacy day reconstruction or
paid all-history weekly branch is reachable from this entrypoint.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

from hindsight_incremental_delivery import deliver_batch, _read, _freeze, _secure_path
from hindsight_incremental_ledger import IncrementalLedger, PHASES, _private
from hindsight_incremental_merge import (canonical_hash, load_manifest_artifacts,
    merge_observations, validate_generation, write_generation, _read_json)
from hindsight_incremental_sources import scan_source_snapshots, digest
from hindsight_incremental_progress import (active_reporter, record_stage,
    reporter_context, reporting_run, span, timed)
from hindsight_offline_retry import RetryDeferred, RetryPolicy, classify_error
from hindsight_pipeline_common import load_config

CONFIRM_RUN = 'run-hindsight-pipeline'
# Safety ceilings, not small budgets: heavy loads must drain in a single run
# (user-directed). Per-unit deadlines, attempt caps and the call budget remain
# the real guards; the deferral machinery stays as the fallback net.
MAX_PENDING_UNITS_CEILING = 200
MAX_PENDING_CHARS_CEILING = 20_000_000


def _binding(batch, **values):
    return {'bank': batch['bank'], 'batch_id': batch['batch_id'], **values}


def _advance(ledger, batch_id, phase, receipt):
    current = ledger.get_batch(batch_id)
    if PHASES.index(current['phase']) >= PHASES.index(phase):
        return current
    return ledger.set_phase(batch_id, phase, receipt=_binding(current, **receipt))


@timed('seed_state')
def _seed_state(ledger):
    seed = ledger.initialization_manifest()['seed']
    if seed.get('bank') != ledger.bank:
        raise ValueError('seed_bank_mismatch')
    with span('seed_read') as counts:
        rows = _read(ledger.root, seed['artifact_path'], seed['artifact_hash'])
        if isinstance(rows, list):
            counts.update(row_count=len(rows))
    if not isinstance(rows, list):
        raise ValueError('seed_state_invalid')
    parent = seed['provenance']
    pointers = []
    for batch in ledger.list_batches(phases=('published',)):
        receipt = batch['receipts'].get('published', {})
        if receipt.get('manifest'):
            pointers.append(receipt)
    if pointers:
        latest = max(pointers, key=lambda r: int(r['sequence']))
        parent, _cards, rows = load_manifest_artifacts(Path(latest['manifest']['root']), bank=ledger.bank)
    if not isinstance(rows, list):
        raise ValueError('published_state_index_missing')
    return parent, rows, max((int(r['sequence']) for r in pointers), default=0)


def _typed_delta(batches, ledger):
    rows = []
    for batch in batches:
        generated = batch['receipts']['generated']
        artifact = _read(ledger.root, generated['artifact_path'], generated['artifact_hash'])
        observations = artifact.get('canonical_observations')
        if not isinstance(observations, list):
            raise ValueError('invalid_summary_artifact')
        for row in observations:
            if not isinstance(row, dict) or row.get('bank') != ledger.bank:
                raise ValueError('summary_bank_mismatch')
            rows.append(row)
    return rows


def _gate_valid(receipt, manifest, bank):
    if not isinstance(receipt, dict) or receipt.get('bank') != bank:
        return False
    if receipt.get('manifest_sha256') != manifest['manifest_sha256']:
        return False
    if receipt.get('decision') != 'eligible_for_local_proposal':
        return False
    required = ('schema', 'coverage', 'conflict', 'generic_eval', 'local_eval', 'scoped_drain')
    checks = receipt.get('checks') or {}
    return all(checks.get(name) is True for name in required)


def _gate_reference(gate):
    """Per-batch receipts reference the frozen gate; never duplicate its bytes.

    A gate without frozen bytes (test doubles) keeps the legacy inline copy.
    """
    if not isinstance(gate, dict):
        raise ValueError('gate_reference_requires_gate')
    if not isinstance(gate.get('gate_path'), str) or not isinstance(gate.get('gate_sha256'), str):
        return {'gate': gate}
    return {'gate_reference': {key: gate.get(key) for key in
            ('bank', 'manifest_sha256', 'decision', 'checks', 'gate_path', 'gate_sha256')}}


def _stored_gate(receipt):
    """Full gate for a stored receipt: legacy inline copy or frozen reference."""
    if not isinstance(receipt, dict):
        return None
    if isinstance(receipt.get('gate'), dict):
        return receipt['gate']
    reference = receipt.get('gate_reference')
    if not isinstance(reference, dict):
        return None
    path, sha = reference.get('gate_path'), reference.get('gate_sha256')
    if not isinstance(path, str) or not isinstance(sha, str):
        return None
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('gate_sha256') != sha
            or canonical_hash({k: v for k, v in value.items() if k != 'gate_sha256'}) != sha):
        raise ValueError('gate_receipt_hash_mismatch')
    for key in ('bank', 'manifest_sha256', 'decision', 'checks'):
        if value.get(key) != reference.get(key):
            raise ValueError('gate_receipt_field_mismatch')
    return value


def _published_valid(receipt, manifest, bank):
    return (isinstance(receipt, dict) and receipt.get('bank') == bank
            and receipt.get('manifest_sha256') == manifest['manifest_sha256']
            and receipt.get('published') is True and receipt.get('readback_verified') is True
            and receipt.get('legacy_seed_preserved') is True)


def _record(report, stage, **values):
    report['stages'].append({'stage': stage, **values})
    print(json.dumps({'stage': stage, **{k: v for k, v in values.items()
          if k not in {'receipt', 'artifact', 'facts'}}}, ensure_ascii=False), flush=True)
    record_stage(stage, values)


def run_incremental(config, *, ledger, client, llm, clock=time.time, execute=False,
                    source_reader=scan_source_snapshots, retry_policy=None):
    """Execute genuine component paths; injected adapters make isolation explicit.

    client implements exact REST delivery plus facts_for_batch/consolidate_batch,
    open_window/cleanup_window and gate/publish/readback methods. The production
    adapter implements these, not a success stub or an alternate provider.
    """
    knobs = config.get('incremental') or {}
    max_units = int(knobs.get('max_pending_units', 12))
    mode = knobs.get('mode', 'daily')
    if mode not in {'daily', 'weekly', 'full'}:
        raise ValueError('invalid_incremental_mode')
    max_chars = int(knobs.get('max_pending_chars', 500000))
    if not 1 <= max_units <= MAX_PENDING_UNITS_CEILING or not 1 <= max_chars <= MAX_PENDING_CHARS_CEILING:
        raise ValueError('normal_incremental_budget_exceeded')
    report = {'schema_version': 'hindsight-arrival-run/1', 'bank': ledger.bank,
              'started_at': clock(), 'execute': execute, 'status': 'running',
              'history_replay': False, 'stages': [], 'errors': [], 'cleanup': 'not_opened',
              'calls': 0, 'llm_calls': 0, 'published': False, 'mode': mode,
              'optional_tails': {'wiki': {'requested': bool(knobs.get('include_wiki')),
                                         'status': 'pending' if knobs.get('include_wiki') else 'not_requested'}}}
    policy = retry_policy or RetryPolicy(**(knobs.get('retry') or {}))
    window = None
    producer_ready = False
    active_batch_id = None
    deferred = False
    used_units = used_chars = 0
    progress_scope = ExitStack()
    reporter = None
    inherited_reporter = active_reporter()
    progress_scope.enter_context(reporter_context(None))
    try:
        with ledger.locked():
            if hasattr(client, 'begin_run'):
                client.begin_run(policy)
            if execute and hasattr(client, 'resume_window'):
                window = client.resume_window(ledger)
                if window is not None:
                    report['cleanup'] = 'pending'
            snapshots = []
            if not knobs.get('resume_only'):
                reader_options = {'profile_mode': 'all'}
                legacy = ledger.initialization_manifest()
                unknown_reference = ledger.source_unknown_policy_reference()
                if 'physical_baseline' in legacy:
                    reference = ledger.physical_baseline_reference()
                    if not isinstance(reference, dict):
                        raise ValueError('physical_baseline_reference_invalid')
                    value = _read(ledger.root, reference['artifact_path'], reference['artifact_hash'])
                    from hindsight_incremental_sources import validate_physical_baseline
                    validate_physical_baseline(value, bank=ledger.bank,
                                               hermes_home=Path(config['hermes_home']))
                    reader_options['physical_baseline'] = value
                if unknown_reference is not None:
                    if 'physical_baseline' not in reader_options:
                        raise ValueError('source_unknown_policy_requires_physical_baseline')
                    unknown = _read(ledger.root, unknown_reference['artifact_path'], unknown_reference['artifact_hash'])
                    from hindsight_incremental_sources import validate_source_unknown_policy
                    validate_source_unknown_policy(unknown, physical_baseline=reader_options['physical_baseline'],
                                                   bank=ledger.bank, hermes_home=Path(config['hermes_home']))
                    reader_options['source_unknown_policy'] = unknown
                    report['source_unknown_policy'] = {'artifact_hash': unknown_reference['artifact_hash'],
                        'status': 'unknown', 'completion_proven': False,
                        'excluded_sql_rows': sum(len(scope['rows']) for scope in unknown['scopes'])}
                snapshots = source_reader(Path(config['hermes_home']), **reader_options)
            if not execute:
                pending = ledger.pending_batches(now=clock(), max_units=max_units, max_chars=max_chars)
                report.update(status='plan', captured_pending_units=len(pending),
                              captured_pending_chars=sum(b['input_chars'] for b in pending),
                              observed_sources=len(snapshots), unfrozen_arrivals='capture_required')
                return report
            captured = []
            for snapshot in snapshots:
                captured.extend(ledger.capture_snapshot(snapshot))
            _record(report, 'capture', new_batches=len(captured), sources=len(snapshots))
            initial = ledger.pending_batches(now=clock(), max_units=max_units, max_chars=max_chars)
            if not initial:
                report['status'] = 'no_op'
                return report
            if hasattr(client, 'preflight'):
                # Never require current credentials for publish-only recovery.
                parents_with_children = {p for b in ledger.list_batches()
                    if b['kind'] == 'daily_summary' for p in b['parent_batch_ids']}
                require_llm = any((b['kind'] == 'daily_summary' and b['phase'] == 'captured')
                    or (b['kind'] == 'source' and b['phase'] in
                        {'captured', 'generated', 'submitted', 'unknown', 'retained'}
                        and b['batch_id'] not in parents_with_children) for b in initial)
                client.preflight(require_llm=require_llm)
                _record(report, 'preflight', status='ready')
            if inherited_reporter is not None and inherited_reporter.owner is ledger:
                progress_scope.enter_context(reporter_context(inherited_reporter))
            reporter = progress_scope.enter_context(reporting_run(ledger))
            record_stage('capture', {'new_batches': len(captured), 'sources': len(snapshots)})
            if hasattr(client, 'preflight'):
                record_stage('preflight', {})
            # Open lazily: a frozen publish retry must not open a retain window.
            # Bounded-concurrency rotation (user-directed): up to
            # max_inflight_units batches hold in-flight native operations and
            # each cycle advances every in-flight batch by one single-poll
            # step, so native work overlaps instead of serializing run wall
            # time. Receipts, phases and deferral stay unchanged; deferred
            # batches are retried in-run (bounded) so a heavy load can drain
            # in one run instead of spilling to the next night.
            max_inflight = int(knobs.get('max_inflight_units', 4))
            if not 1 <= max_inflight <= 32:
                raise ValueError('invalid max_inflight_units')
            retryable = {'captured', 'generated', 'submitted', 'unknown', 'retained'}
            in_run_cap = max(1, policy.max_attempts)
            in_flight = {}
            seen = set()
            in_run_attempts = {}
            while True:
                # Fill: enter batches until the in-flight budget is full.
                while len(in_flight) < max_inflight:
                    selected = ledger.pending_batches(now=clock(), max_units=max_units, max_chars=max_chars)
                    candidate = None
                    for b in selected:
                        bid = b['batch_id']
                        if bid in in_flight or b['phase'] not in retryable:
                            continue
                        if bid in seen:
                            if (in_run_attempts.get(bid, 0) >= in_run_cap
                                    or (b.get('last_error_class') or '') in {'call_cap', 'llm_call_cap'}):
                                # Run-level caps cannot recover inside this run.
                                continue
                        elif used_units >= max_units or used_chars + b['input_chars'] > max_chars:
                            continue
                        candidate = b
                        break
                    if candidate is None:
                        break
                    bid = candidate['batch_id']
                    if bid not in seen:
                        seen.add(bid)
                        used_units += 1
                        used_chars += candidate['input_chars']
                    active_batch_id = bid
                    # A recovered lease owns cleanup but may have crashed before
                    # PATCH. Reopen once, only when an actual producer is selected.
                    if not producer_ready:
                        window = client.open_window(ledger)
                        producer_ready = True
                        report['cleanup'] = 'pending'
                        _record(report, 'window', status='opened_scoped_manual_only')
                    current = ledger.get_batch(bid)
                    stage = 'deliver'
                    if PHASES.index(current['phase']) >= PHASES.index('retained'):
                        stage = 'facts' if current['kind'] == 'source' else 'consolidate'
                    in_flight[bid] = {'stage': stage, 'entered': policy.clock(), 'calls': 0}
                if not in_flight:
                    # Wait toward the soonest in-run retry rather than spilling
                    # it to the next run; blocked batches (far-future) never wait.
                    soonest = None
                    horizon = clock() + policy.defer_delay + policy.poll_s
                    for b in ledger.list_batches():
                        bid = b['batch_id']
                        if b['phase'] not in retryable or bid in in_flight:
                            continue
                        if bid in seen and (in_run_attempts.get(bid, 0) >= in_run_cap
                                or (b.get('last_error_class') or '') in {'call_cap', 'llm_call_cap'}):
                            continue
                        next_at = b.get('next_retry_at')
                        if next_at is None or next_at > 1e29 or next_at <= clock() or next_at > horizon:
                            continue
                        soonest = next_at if soonest is None else min(soonest, next_at)
                    if soonest is not None:
                        policy.sleep(min(policy.poll_s, max(0.0, soonest - clock())))
                        continue
                    break
                progressed = False
                for bid in list(in_flight):
                    item = in_flight[bid]
                    batch = ledger.get_batch(bid)
                    attempts_so_far = int(batch.get('attempts') or 0) + item['calls']
                    if policy.clock() - item['entered'] >= policy.unit_deadline_s:
                        next_at = policy.wall_clock() + policy.defer_delay
                        ledger.defer(bid, error_class='operation_pending', next_retry_at=next_at,
                                     attempts=attempts_so_far)
                        in_run_attempts[bid] = in_run_attempts.get(bid, 0) + 1
                        del in_flight[bid]
                        deferred = True
                        _record(report, 'native_stage', batch_id=bid, status='deferred',
                                error_class='operation_pending', next_retry_at=next_at)
                        continue
                    try:
                        if item['stage'] == 'deliver':
                            generator = llm if batch['kind'] == 'daily_summary' else None
                            outcome = deliver_batch(ledger, batch, client=client, generate=generator,
                                validate=client.validate_summary if generator else None,
                                retry_policy=policy, single_poll=True)
                            item['calls'] += outcome['calls']
                            if outcome['status'] == 'pending':
                                continue
                            _record(report, 'delivery', kind=batch['kind'], batch_id=bid,
                                    status=outcome['status'], phase=outcome['phase'],
                                    calls=item['calls'], error_class=outcome.get('error_class'))
                            current = ledger.get_batch(bid)
                            if outcome['status'] in {'deferred', 'blocked'}:
                                del in_flight[bid]
                                deferred = True
                                progressed = True
                                continue
                            if PHASES.index(current['phase']) < PHASES.index('retained'):
                                del in_flight[bid]
                                deferred = True
                                progressed = True
                                continue
                            item['stage'] = 'facts' if current['kind'] == 'source' else 'consolidate'
                            item['entered'] = policy.clock()
                            progressed = True
                            continue
                        if item['stage'] == 'facts':
                            current = ledger.get_batch(bid)
                            children = [b for b in ledger.list_batches()
                                        if b['kind'] == 'daily_summary' and bid in b['parent_batch_ids']]
                            if not children:
                                facts = client.facts_for_batch(current)
                                messages = [{'event_id': digest({'fact': f}), 'role': 'user',
                                    'timestamp': f.get('created_at'), 'origin': f['id'],
                                    'content': json.dumps(f, ensure_ascii=False, sort_keys=True)} for f in facts]
                                if messages:
                                    children = ledger.capture_derived([bid], messages)
                                else:
                                    _record(report, 'facts', batch_id=bid, status='zero_facts')
                            item['stage'] = 'consolidate'
                            item['entered'] = policy.clock()
                            progressed = True
                            continue
                        current = ledger.get_batch(bid)
                        proof = client.consolidate_batch(current, policy, single_poll=True)
                        if proof is None:
                            continue
                        expected_scope = [f"arrival-batch:{bid}"]
                        if (not isinstance(proof, dict) or proof.get('bank') != ledger.bank
                                or proof.get('scope') != expected_scope or proof.get('pending_in_scope') != 0
                                or proof.get('ready') is not True):
                            raise ValueError('scoped_drain_unverified')
                        current = _advance(ledger, bid, 'consolidated', {'scoped_proof': proof})
                        _record(report, 'scoped_drain', batch_id=bid, status='completed')
                        if current['kind'] == 'source':
                            children = [b for b in ledger.list_batches()
                                        if bid in b['parent_batch_ids']]
                            if not children:
                                # A genuinely completed retain can extract zero facts.
                                # Record no-canonical-delta, not a fictitious publish.
                                receipt = {'no_canonical_delta': True, 'retained_receipt_hash':
                                           canonical_hash(current['receipts']['retained'])}
                                for phase in ('locally_merged', 'gated', 'published'):
                                    current = _advance(ledger, bid, phase, receipt)
                                _record(report, 'canonical', batch_id=bid, status='no_canonical_delta')
                        del in_flight[bid]
                        progressed = True
                    except RetryDeferred as error:
                        ledger.defer(bid, error_class=error.error_class,
                                     next_retry_at=error.next_retry_at,
                                     attempts=attempts_so_far + error.attempts)
                        in_run_attempts[bid] = in_run_attempts.get(bid, 0) + 1
                        del in_flight[bid]
                        deferred = True
                        _record(report, 'native_stage', batch_id=bid, status='deferred',
                                error_class=error.error_class, next_retry_at=error.next_retry_at)
                    except Exception as error:
                        category, transient = classify_error(error)
                        if transient:
                            next_at = policy.wall_clock() + policy.defer_delay
                            ledger.defer(bid, error_class=category, next_retry_at=next_at,
                                         attempts=attempts_so_far)
                            in_run_attempts[bid] = in_run_attempts.get(bid, 0) + 1
                            del in_flight[bid]
                            deferred = True
                            _record(report, 'native_stage', batch_id=bid, status='deferred',
                                    error_class=category, next_retry_at=next_at)
                        else:
                            ledger.defer(bid, error_class=category, next_retry_at=1e30,
                                         attempts=attempts_so_far)
                            del in_flight[bid]
                            deferred = True
                            report['errors'].append({'class': category, 'type': type(error).__name__})
                            _record(report, 'native_stage', batch_id=bid, status='blocked',
                                    error_class=category)
                if not progressed and in_flight:
                    policy.sleep(policy.poll_s)
            all_batches = ledger.list_batches()
            summaries = [b for b in all_batches if b['kind'] == 'daily_summary'
                         and b['phase'] in {'consolidated', 'locally_merged', 'gated'}
                         and b['next_retry_at'] <= clock()]
            # Publish only complete parent cohorts, not part of a split parent.
            ready = []
            for batch in summaries:
                siblings = [b for b in all_batches if b['kind'] == 'daily_summary'
                            and set(b['parent_batch_ids']) & set(batch['parent_batch_ids'])]
                parents = [ledger.get_batch(bid) for bid in batch['parent_batch_ids']]
                if all(PHASES.index(b['phase']) >= PHASES.index('consolidated') for b in siblings + parents):
                    ready.append(batch)
            saved = [b['receipts']['locally_merged']['manifest'] for b in ready
                     if b['receipts'].get('locally_merged')]
            if saved:
                with span('saved_manifest_validate'):
                    manifest = validate_generation(Path(saved[0]['manifest_path']))
                ready = [b for b in ready if b['batch_id'] in manifest['consumed_batch_ids']]
            elif ready:
                parent, seed, _seq = _seed_state(ledger)
                with span('merge', row_count=len(seed)) as counts:
                    merged = merge_observations(seed, _typed_delta(ready, ledger))
                    from hindsight_incremental_merge import delta_rows
                    manifest = write_generation(ledger.root / 'canonical', bank=ledger.bank,
                        parent=parent, observations=merged, consumed_batch_ids=[b['batch_id'] for b in ready],
                        delta=delta_rows(seed, merged))
                    counts.update(row_count=len(merged))
                for batch in ready:
                    _advance(ledger, batch['batch_id'], 'locally_merged', {'manifest': manifest})
                _record(report, 'merge', status='completed', generation_id=manifest['generation_id'])
            else:
                manifest = None
            if manifest is not None:
                first = ledger.get_batch(ready[0]['batch_id'])
                gate = _stored_gate(first['receipts'].get('gated', {}))
                if gate is None:
                    with span('gate'):
                        gate = client.gate(manifest)
                if not _gate_valid(gate, manifest, ledger.bank):
                    for batch in ready:
                        ledger.defer(batch['batch_id'], error_class='quality_gate',
                                     next_retry_at=clock() + policy.defer_delay, attempts=batch['attempts'])
                    deferred = True
                    _record(report, 'gate', status='blocked')
                else:
                    for batch in ready:
                        _advance(ledger, batch['batch_id'], 'gated',
                                 {**_gate_reference(gate), 'manifest': manifest})
                    _record(report, 'gate', status='passed')
                    with span('readback_generation'):
                        publication = client.readback_generation(manifest)
                    if not _published_valid(publication, manifest, ledger.bank):
                        with span('publish_generation'):
                            client.publish_generation(manifest, gate)
                        with span('readback_generation'):
                            publication = client.readback_generation(manifest)
                    if not _published_valid(publication, manifest, ledger.bank):
                        raise ValueError('publish_readback_unverified')
                    _parent, _rows, sequence = _seed_state(ledger)
                    for batch in ready:
                        _advance(ledger, batch['batch_id'], 'published', {'manifest': manifest,
                            'publication': publication, 'sequence': sequence + 1})
                    # Source parents progress only once every derived child is published.
                    parents = sorted({p for b in ready for p in b['parent_batch_ids']})
                    for parent_id in parents:
                        siblings = [b for b in ledger.list_batches() if parent_id in b['parent_batch_ids']]
                        if all(b['phase'] == 'published' for b in siblings):
                            for phase, proof in [('locally_merged', {'manifest': manifest}),
                                                ('gated', {**_gate_reference(gate), 'manifest': manifest}),
                                                ('published', {'manifest': manifest, 'publication': publication,
                                                               'sequence': sequence + 1})]:
                                operation = {'locally_merged': 'source_parent_merge',
                                             'gated': 'source_parent_gate',
                                             'published': 'source_parent_publish'}[phase]
                                with span(operation, unit_count=1):
                                    _advance(ledger, parent_id, phase, proof)
                    report['published'] = True
                    _record(report, 'publish', status='readback_verified', generation_id=manifest['generation_id'])
            remaining = [b for b in ledger.list_batches() if b['phase'] != 'published']
            report.update(status='deferred' if remaining else 'completed',
                          remaining_batches=len(remaining), used_units=used_units, used_chars=used_chars)
    except RetryDeferred as error:
        if active_batch_id is not None and execute:
            if reporter is not None:
                reporter.fail(error)
            current = ledger.get_batch(active_batch_id)
            ledger.defer(active_batch_id, error_class=error.error_class,
                next_retry_at=error.next_retry_at, attempts=current['attempts'] + error.attempts)
        report['errors'].append({'class': error.error_class, 'type': type(error).__name__})
        report['status'] = 'deferred'
    except Exception as error:
        category, _transient = classify_error(error)
        if reporter is not None:
            reporter.fail(error)
        report['errors'].append({'class': category, 'type': type(error).__name__})
        report['status'] = 'blocked'
    finally:
        try:
            report['core_status'] = report['status']
            report['calls'] = policy.calls_used
            report['offline_llm_calls'] = policy.llm_calls_used
            report['native_paid_calls'] = getattr(client, 'native_paid_calls', None)
            report['llm_calls'] = getattr(client, 'llm_calls', 0)
            # PATCH can succeed before its readback raises; the assignment to
            # window never executes then. The adapter persists ownership first.
            # Do not reclaim a restored previous run or mutate a read-only plan.
            opened = getattr(client, '_window', None)
            if (execute and window is None and isinstance(opened, dict)
                    and opened.get('status') in {'opening', 'active', 'cleanup_pending', 'restoring'}):
                window = opened
            if window is not None:
                try:
                    with ledger.locked(), span('cleanup'):
                        report['cleanup'] = client.cleanup_window(ledger, window)
                    if report['cleanup'] != 'restored':
                        report['status'] = 'cleanup_pending'
                except Exception as error:
                    if reporter is not None:
                        reporter.fail(error)
                    report['cleanup'] = 'pending'
                    report['status'] = 'cleanup_pending'
                    report['errors'].append({'class': 'cleanup', 'type': type(error).__name__})
            report['finished_at'] = clock()
            report['network_calls'] = getattr(client, 'network_calls', policy.calls_used)
            report['network_calls_scope'] = 'native_only_excludes_eval_and_embeddings'
            report['end_to_end_network_calls'] = None
            if report['optional_tails']['wiki']['requested']:
                # The old tail scans/re-extracts history. Until a bounded replacement
                # is verified, report it explicitly rather than silently ignore the
                # flag or claim a wiki completion from core success.
                tail = report['optional_tails']['wiki']
                tail.update(status='degraded' if report['status'] in {'completed', 'no_op', 'plan'}
                            else 'not_run_core_incomplete', reason='bounded_tail_not_implemented')
                _record(report, 'optional_tail', name='wiki', status=tail['status'], reason=tail['reason'])
            if execute:
                path, checksum = _freeze(ledger.root, report, 'run-report')
                report['report_path'], report['report_sha256'] = path, checksum
        finally:
            progress_scope.__exit__(*sys.exc_info())
    return report


INITIALIZATION_PLAN_SCHEMA = 'hindsight-physical-initialization-plan/1'


def _initialization_artifacts(ledger):
    """Inspect only typed preparation files in this bank's private directory."""
    directory = _secure_path(ledger.root, ledger.root / 'delivery-artifacts')
    if not directory.exists():
        return [], []
    _private(directory, directory=True)
    entries = list(directory.iterdir())
    # Include unfinished/malformed typed names: ignoring them would recapture H.
    return ([p for p in entries if re.search(r'(?:^|\.)initialization-plan(?:\.|$)', p.name)],
            [p for p in entries if re.search(r'(?:^|\.)physical-baseline(?:\.|$)', p.name)])


def _prepared_physical_baseline(ledger, plans, physical, *, home, seed_hash):
    """Recover exactly one immutable seed/home/bank scope, never the latest H."""
    from hindsight_incremental_sources import validate_physical_baseline
    if not plans:
        if physical:
            raise ValueError('physical_initialization_unprepared_orphan')
        return None
    if len(plans) != 1:
        raise ValueError('physical_initialization_ambiguous_preparation')
    path = _secure_path(ledger.root, plans[0])
    match = re.fullmatch(r'([0-9a-f]{64})\.initialization-plan\.json', path.name)
    if match is None:
        raise ValueError('physical_initialization_invalid_preparation_path')
    before = _private(path)
    if stat.S_IMODE(before.st_mode) != 0o400:
        raise ValueError('physical_initialization_unfinished_preparation')
    checksum = match.group(1)
    plan = _read(ledger.root, path, checksum)
    fields = {'schema_version', 'bank', 'hermes_home', 'seed_export_sha256', 'physical_baseline'}
    if (not isinstance(plan, dict) or set(plan) != fields or
            plan.get('schema_version') != INITIALIZATION_PLAN_SCHEMA or
            plan.get('bank') != ledger.bank or plan.get('hermes_home') != str(home) or
            plan.get('seed_export_sha256') != seed_hash or canonical_hash(plan) != checksum):
        raise ValueError('physical_initialization_preparation_binding_mismatch')
    baseline = plan['physical_baseline']
    validate_physical_baseline(baseline, bank=ledger.bank, hermes_home=home)
    expected = ledger.root / 'delivery-artifacts' / (canonical_hash(baseline) + '.physical-baseline.json')
    if any(p != expected for p in physical):
        raise ValueError('physical_initialization_ambiguous_physical_artifact')
    # A killed freeze may have completed chmod but not the directory fsync.
    # Revalidate immutability first, then complete durability before any commit.
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if ledger._artifact_fence(before) != ledger._artifact_fence(_private(path)):
        raise ValueError('physical_initialization_preparation_changed')
    return baseline


def initialize(config, *, seed_file, confirm, source_reader=scan_source_snapshots,
               physical_baseline=False):
    if confirm != 'initialize-hindsight-incremental':
        raise ValueError('explicit_initialize_confirmation_required')
    knobs = config.get('incremental') or {}
    root = Path(knobs.get('root') or Path(config['hermes_home']) / 'hindsight' / 'incremental')
    seed = _read_json(Path(seed_file))
    if not isinstance(seed, dict) or seed.get('bank') != config['bank'] or not isinstance(seed.get('observations'), list):
        raise ValueError('seed_export_invalid')
    if seed.get('export_sha256') != canonical_hash({k: v for k, v in seed.items() if k != 'export_sha256'}):
        raise ValueError('seed_export_hash_mismatch')
    ledger = IncrementalLedger(root, config['bank'], create=True)
    with ledger.locked():
        with ledger._connection(initialized=False) as con:
            if ledger._read(con, 'meta', 'legacy') is not None:
                raise ValueError('ledger already initialized')
        plans, physical = _initialization_artifacts(ledger)
        baseline = None
        if physical_baseline:
            from hindsight_incremental_sources import (capture_physical_baseline, checked_path,
                validate_physical_baseline, physical_baseline_snapshots)
            home = checked_path(Path(config['hermes_home']))
            baseline = _prepared_physical_baseline(ledger, plans, physical, home=home,
                                                  seed_hash=seed['export_sha256'])
            if baseline is None:
                baseline = capture_physical_baseline(home, bank=ledger.bank, profile_mode='all')
                validate_physical_baseline(baseline, bank=ledger.bank, hermes_home=home)
                # The FIRST frozen artifact is independently recoverable. Freezing
                # a descriptor and writing a pointer later leaves a data-loss gap.
                plan = {'schema_version': INITIALIZATION_PLAN_SCHEMA, 'bank': ledger.bank,
                        'hermes_home': str(home), 'seed_export_sha256': seed['export_sha256'],
                        'physical_baseline': baseline}
                _freeze(ledger.root, plan, 'initialization-plan')
                plans, physical = _initialization_artifacts(ledger)
                baseline = _prepared_physical_baseline(ledger, plans, physical, home=home,
                                                      seed_hash=seed['export_sha256'])
            if baseline is None:
                raise ValueError('physical_initialization_missing_preparation')
            snapshots = physical_baseline_snapshots(baseline)
        else:
            if plans or physical:
                raise ValueError('physical_initialization_preparation_requires_physical_mode')
            snapshots = source_reader(Path(config['hermes_home']), profile_mode='all')
        path, checksum = _freeze(ledger.root, seed['observations'], 'seed')
        state = {'bank': ledger.bank, 'artifact_path': path, 'artifact_hash': checksum,
                 'provenance': seed['provenance'], 'remote_seed': seed.get('remote_seed')}
        if baseline is not None:
            path, checksum = _freeze(ledger.root, baseline, 'physical-baseline')
            reference = {key: baseline[key] for key in ('schema_version', 'bank', 'hermes_home')}
            reference.update(artifact_path=path, artifact_hash=checksum)
            return ledger.initialize_legacy(snapshots, state, seed.get('legacy_unknown', []),
                                            confirm=confirm, physical_baseline=reference)
        return ledger.initialize_legacy(snapshots, state, seed.get('legacy_unknown', []), confirm=confirm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('initialize', 'plan', 'run', 'status', 'export-seed'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed-file', type=Path)
    parser.add_argument('--confirm')
    parser.add_argument('--mode', choices=('daily', 'weekly', 'full'), default='daily')
    parser.add_argument('--include-wiki', action='store_true',
        help='Report the requested optional wiki lane separately; currently degraded, never invokes legacy history writers')
    parser.add_argument('--resume-only', action='store_true')
    parser.add_argument('--bank')
    parser.add_argument('--llm-profile')
    parser.add_argument('--llm-model')
    parser.add_argument('--llm-base-url')
    parser.add_argument('--llm-api-key-env')
    parser.add_argument('--llm-provider')
    parser.add_argument('--max-units', type=int)
    parser.add_argument('--max-chars', type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config, require_exists=True)
    if args.bank is not None:
        config['bank'] = args.bank
    if args.llm_profile is not None:
        config['llm_profile'] = args.llm_profile
    config['incremental'] = dict(config.get('incremental') or {})
    config['incremental']['mode'] = args.mode
    if args.include_wiki:
        config['incremental']['include_wiki'] = True
    if args.resume_only:
        config['incremental']['resume_only'] = True
    for value, key, ceiling in [(args.max_units, 'max_pending_units', MAX_PENDING_UNITS_CEILING),
                                 (args.max_chars, 'max_pending_chars', MAX_PENDING_CHARS_CEILING)]:
        if value is not None:
            if not 1 <= value <= ceiling:
                parser.error('normal incremental budget cannot be unlimited or exceed safety ceilings')
            config['incremental'][key] = value
    overrides = {key: value for key, value in {'model': args.llm_model,
        'base_url': args.llm_base_url, 'api_key_env': args.llm_api_key_env,
        'provider': args.llm_provider}.items() if value is not None}
    if overrides:
        config['incremental']['llm_override'] = overrides
    knobs = config.get('incremental') or {}
    root = Path(knobs.get('root') or Path(config['hermes_home']) / 'hindsight' / 'incremental')
    if args.action == 'initialize':
        if args.seed_file is None:
            parser.error('--seed-file is required for explicit migration')
        result = initialize(config, seed_file=args.seed_file, confirm=args.confirm,
                            physical_baseline=True)
    elif args.action == 'status':
        ledger = IncrementalLedger(root, config['bank'])
        result = {'bank': ledger.bank, 'legacy': ledger.initialization_manifest(),
                  'batch_counts': {p: len(ledger.list_batches(phases=(p,))) for p in PHASES}}
    else:
        from hindsight_incremental_runtime import ArrivalRuntime
        runtime = ArrivalRuntime(config)
        if args.action == 'export-seed':
            if args.seed_file is None:
                parser.error('--seed-file is required for read-only seed export')
            result = runtime.export_seed(args.seed_file)
        else:
            if args.action == 'run' and args.confirm != CONFIRM_RUN:
                parser.error('--confirm run-hindsight-pipeline is required')
            ledger = IncrementalLedger(root, config['bank'])
            runtime.ledger = ledger
            result = run_incremental(config, ledger=ledger, client=runtime,
                llm=runtime.generate_summary, execute=args.action == 'run')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result.get('status') in {'plan', 'completed', 'no_op', None} else 2


if __name__ == '__main__':
    raise SystemExit(main())
