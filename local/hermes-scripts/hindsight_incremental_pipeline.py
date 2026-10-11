#!/usr/bin/env python3
"""Arrival-only orchestration: frozen units, scoped work and exact generation receipts.

The concrete REST/DB adapter is a sibling module. No legacy day reconstruction or
paid all-history weekly branch is reachable from this entrypoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hindsight_incremental_delivery import deliver_batch, _read, _freeze
from hindsight_incremental_ledger import IncrementalLedger, PHASES
from hindsight_incremental_merge import (canonical_hash, load_manifest_artifacts,
    merge_observations, validate_generation, write_generation)
from hindsight_incremental_sources import scan_source_snapshots, digest
from hindsight_offline_retry import RetryDeferred, RetryPolicy, classify_error
from hindsight_pipeline_common import load_config

CONFIRM_RUN = 'run-hindsight-pipeline'


def _binding(batch, **values):
    return {'bank': batch['bank'], 'batch_id': batch['batch_id'], **values}


def _advance(ledger, batch_id, phase, receipt):
    current = ledger.get_batch(batch_id)
    if PHASES.index(current['phase']) >= PHASES.index(phase):
        return current
    return ledger.set_phase(batch_id, phase, receipt=_binding(current, **receipt))


def _seed_state(ledger):
    seed = ledger.initialization_manifest()['seed']
    if seed.get('bank') != ledger.bank:
        raise ValueError('seed_bank_mismatch')
    rows = _read(ledger.root, seed['artifact_path'], seed['artifact_hash'])
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


def _published_valid(receipt, manifest, bank):
    return (isinstance(receipt, dict) and receipt.get('bank') == bank
            and receipt.get('manifest_sha256') == manifest['manifest_sha256']
            and receipt.get('published') is True and receipt.get('readback_verified') is True
            and receipt.get('legacy_seed_preserved') is True)


def _record(report, stage, **values):
    report['stages'].append({'stage': stage, **values})
    print(json.dumps({'stage': stage, **{k: v for k, v in values.items()
          if k not in {'receipt', 'artifact', 'facts'}}}, ensure_ascii=False), flush=True)


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
    if not 1 <= max_units <= 12 or not 1 <= max_chars <= 500000:
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
    try:
        with ledger.locked():
            if hasattr(client, 'begin_run'):
                client.begin_run(policy)
            if execute and hasattr(client, 'resume_window'):
                window = client.resume_window(ledger)
                if window is not None:
                    report['cleanup'] = 'pending'
            snapshots = [] if knobs.get('resume_only') else source_reader(Path(config['hermes_home']), profile_mode='all')
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
            # Open lazily: a frozen publish retry must not open a retain window.
            seen = set()
            while used_units < max_units:
                selected = ledger.pending_batches(now=clock(), max_units=max_units, max_chars=max_chars)
                selected = [b for b in selected if b['batch_id'] not in seen
                            and b['phase'] in {'captured', 'generated', 'submitted', 'unknown', 'retained'}]
                if not selected:
                    break
                batch = selected[0]
                active_batch_id = batch['batch_id']
                seen.add(batch['batch_id'])
                if used_chars + batch['input_chars'] > max_chars:
                    deferred = True
                    continue
                used_units += 1
                used_chars += batch['input_chars']
                # A recovered lease owns cleanup but may have crashed before
                # PATCH. Reopen once, only when an actual producer is selected.
                if not producer_ready:
                    window = client.open_window(ledger)
                    producer_ready = True
                    report['cleanup'] = 'pending'
                    _record(report, 'window', status='opened_scoped_manual_only')
                generator = llm if batch['kind'] == 'daily_summary' else None
                outcome = deliver_batch(ledger, batch, client=client, generate=generator,
                    validate=client.validate_summary if generator else None, retry_policy=policy)
                _record(report, 'delivery', kind=batch['kind'], batch_id=batch['batch_id'],
                        status=outcome['status'], phase=outcome['phase'], calls=outcome['calls'],
                        error_class=outcome.get('error_class'))
                current = ledger.get_batch(batch['batch_id'])
                if PHASES.index(current['phase']) < PHASES.index('retained'):
                    deferred = True
                    continue
                stage_start = policy.calls_used
                try:
                    if current['phase'] == 'retained':
                        if current['kind'] == 'source':
                            children = [b for b in ledger.list_batches()
                                        if b['kind'] == 'daily_summary' and current['batch_id'] in b['parent_batch_ids']]
                            if not children:
                                facts = client.facts_for_batch(current)
                                messages = [{'event_id': digest({'fact': f}), 'role': 'user',
                                    'timestamp': f.get('created_at'), 'origin': f['id'],
                                    'content': json.dumps(f, ensure_ascii=False, sort_keys=True)} for f in facts]
                                if messages:
                                    children = ledger.capture_derived([current['batch_id']], messages)
                                else:
                                    _record(report, 'facts', batch_id=current['batch_id'], status='zero_facts')
                        proof = client.consolidate_batch(current, policy)
                        expected_scope = [f"arrival-batch:{current['batch_id']}"]
                        if (not isinstance(proof, dict) or proof.get('bank') != ledger.bank
                                or proof.get('scope') != expected_scope or proof.get('pending_in_scope') != 0
                                or proof.get('ready') is not True):
                            raise ValueError('scoped_drain_unverified')
                        current = _advance(ledger, current['batch_id'], 'consolidated', {'scoped_proof': proof})
                        _record(report, 'scoped_drain', batch_id=current['batch_id'], status='completed')
                        if current['kind'] == 'source':
                            children = [b for b in ledger.list_batches()
                                        if current['batch_id'] in b['parent_batch_ids']]
                            if not children:
                                # A genuinely completed retain can extract zero facts.
                                # Record no-canonical-delta, not a fictitious publish.
                                receipt = {'no_canonical_delta': True, 'retained_receipt_hash':
                                           canonical_hash(current['receipts']['retained'])}
                                for phase in ('locally_merged', 'gated', 'published'):
                                    current = _advance(ledger, current['batch_id'], phase, receipt)
                                _record(report, 'canonical', batch_id=current['batch_id'], status='no_canonical_delta')
                except RetryDeferred as error:
                    ledger.defer(current['batch_id'], error_class=error.error_class,
                        next_retry_at=error.next_retry_at,
                        attempts=outcome['attempts'] + policy.calls_used - stage_start)
                    deferred = True
                    _record(report, 'native_stage', batch_id=current['batch_id'], status='deferred',
                            error_class=error.error_class, next_retry_at=error.next_retry_at)
                    continue
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
                manifest = validate_generation(Path(saved[0]['manifest_path']))
                ready = [b for b in ready if b['batch_id'] in manifest['consumed_batch_ids']]
            elif ready:
                parent, seed, _seq = _seed_state(ledger)
                merged = merge_observations(seed, _typed_delta(ready, ledger))
                manifest = write_generation(ledger.root / 'canonical', bank=ledger.bank,
                    parent=parent, observations=merged, consumed_batch_ids=[b['batch_id'] for b in ready])
                for batch in ready:
                    _advance(ledger, batch['batch_id'], 'locally_merged', {'manifest': manifest})
                _record(report, 'merge', status='completed', generation_id=manifest['generation_id'])
            else:
                manifest = None
            if manifest is not None:
                first = ledger.get_batch(ready[0]['batch_id'])
                gate = first['receipts'].get('gated', {}).get('gate')
                if gate is None:
                    gate = client.gate(manifest)
                if not _gate_valid(gate, manifest, ledger.bank):
                    for batch in ready:
                        ledger.defer(batch['batch_id'], error_class='quality_gate',
                                     next_retry_at=clock() + policy.defer_delay, attempts=batch['attempts'])
                    deferred = True
                    _record(report, 'gate', status='blocked')
                else:
                    for batch in ready:
                        _advance(ledger, batch['batch_id'], 'gated', {'gate': gate, 'manifest': manifest})
                    _record(report, 'gate', status='passed')
                    publication = client.readback_generation(manifest)
                    if not _published_valid(publication, manifest, ledger.bank):
                        client.publish_generation(manifest, gate)
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
                                                ('gated', {'gate': gate, 'manifest': manifest}),
                                                ('published', {'manifest': manifest, 'publication': publication,
                                                               'sequence': sequence + 1})]:
                                _advance(ledger, parent_id, phase, proof)
                    report['published'] = True
                    _record(report, 'publish', status='readback_verified', generation_id=manifest['generation_id'])
            remaining = [b for b in ledger.list_batches() if b['phase'] != 'published']
            report.update(status='deferred' if deferred or remaining else 'completed',
                          remaining_batches=len(remaining), used_units=used_units, used_chars=used_chars)
    except RetryDeferred as error:
        if active_batch_id is not None and execute:
            current = ledger.get_batch(active_batch_id)
            ledger.defer(active_batch_id, error_class=error.error_class,
                next_retry_at=error.next_retry_at, attempts=current['attempts'] + error.attempts)
        report['errors'].append({'class': error.error_class, 'type': type(error).__name__})
        report['status'] = 'deferred'
    except Exception as error:
        category, _transient = classify_error(error)
        report['errors'].append({'class': category, 'type': type(error).__name__})
        report['status'] = 'blocked'
    finally:
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
                with ledger.locked():
                    report['cleanup'] = client.cleanup_window(ledger, window)
                if report['cleanup'] != 'restored':
                    report['status'] = 'cleanup_pending'
            except Exception as error:
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
    return report


def initialize(config, *, seed_file, confirm, source_reader=scan_source_snapshots):
    if confirm != 'initialize-hindsight-incremental':
        raise ValueError('explicit_initialize_confirmation_required')
    knobs = config.get('incremental') or {}
    root = Path(knobs.get('root') or Path(config['hermes_home']) / 'hindsight' / 'incremental')
    seed = json.loads(Path(seed_file).read_text())
    if seed.get('bank') != config['bank'] or not isinstance(seed.get('observations'), list):
        raise ValueError('seed_export_invalid')
    if seed.get('export_sha256') != canonical_hash({k: v for k, v in seed.items() if k != 'export_sha256'}):
        raise ValueError('seed_export_hash_mismatch')
    ledger = IncrementalLedger(root, config['bank'], create=True)
    observations = seed['observations']
    path, checksum = _freeze(ledger.root, observations, 'seed')
    state = {'bank': config['bank'], 'artifact_path': path, 'artifact_hash': checksum,
             'provenance': seed['provenance'], 'remote_seed': seed.get('remote_seed')}
    snapshots = source_reader(Path(config['hermes_home']), profile_mode='all')
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
    for value, key, ceiling in [(args.max_units, 'max_pending_units', 12),
                                 (args.max_chars, 'max_pending_chars', 500000)]:
        if value is not None:
            if not 1 <= value <= ceiling:
                parser.error('normal incremental budget cannot be unlimited or exceed regular caps')
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
        result = initialize(config, seed_file=args.seed_file, confirm=args.confirm)
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
