import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def base_row(doc_id, event_date, *, route='cluster_revisit', status='zero_units', topic='domain:hindsight', retry=False, units=0):
    return {
        'document_id': doc_id,
        'topic_key': topic,
        'review': {'recommended_route': route},
        'current_retain_outcome': {'status': status, 'memory_unit_count': units},
        'retry_evidence': [{'bank': 'tmp'}] if retry else [],
        'deterministic_anomalies': [],
        'value_class_guess': ['tool_lesson'],
        'content_chars': 100,
        'event_date': event_date,
    }


def test_cleanup_time_window_archives_old_rows_without_count_cap():
    cleanup = load_module('hindsight_review_backlog_cleanup')
    rows = [
        base_row('recent-zero', '2026-04-20T00:00:00'),
        base_row('old-zero', '2026-01-01T00:00:00'),
        base_row('old-has', '2026-01-02T00:00:00', route='monitor', status='has_units', units=2),
    ]
    active, archive, summary = cleanup.compact_backlog(
        rows,
        cleanup.parse_quotas(None),
        max_records=0,
        retention_months=3,
        reference_date='2026-05-09T00:00:00+00:00',
    )
    assert {r['document_id'] for r in active} == {'recent-zero'}
    assert {r['document_id'] for r in archive} == {'old-zero', 'old-has'}
    assert summary['count_cap_enabled'] is False
    assert summary['cutoff_event_date'].startswith('2026-02-09')
    assert summary['by_archive_reason']['older_than_retention_window'] == 2


def test_cleanup_pin_bucket_keeps_old_manual_review_active():
    cleanup = load_module('hindsight_review_backlog_cleanup')
    rows = [
        base_row('old-manual', '2026-01-01T00:00:00', route='manual_review'),
        base_row('old-zero', '2026-01-01T00:00:00'),
    ]
    active, archive, summary = cleanup.compact_backlog(
        rows,
        cleanup.parse_quotas(None),
        max_records=0,
        retention_months=3,
        reference_date='2026-05-09T00:00:00+00:00',
        pin_buckets={'manual_review'},
    )
    assert {r['document_id'] for r in active} == {'old-manual'}
    assert {r['document_id'] for r in archive} == {'old-zero'}
    assert active[0]['cleanup']['reason'] == 'pinned_bucket_over_age'


def test_cleanup_optional_count_cap_archives_hot_overflow():
    cleanup = load_module('hindsight_review_backlog_cleanup')
    rows = [base_row(f'zero-{i}', '2026-05-01T00:00:00', topic='domain:hindsight' if i < 4 else 'domain:autodrive', retry=i < 2) for i in range(8)]
    rows += [base_row(f'has-{i}', '2026-05-01T00:00:00', route='monitor', status='has_units', topic='domain:paper', units=2) for i in range(4)]
    active, archive, summary = cleanup.compact_backlog(
        rows,
        cleanup.parse_quotas(None),
        max_records=8,
        retention_months=3,
        reference_date='2026-05-09T00:00:00+00:00',
    )
    assert len(active) == 8
    assert len(archive) == 4
    assert summary['count_cap_enabled'] is True
    assert summary['by_archive_reason']['quantity_overflow_safety_cap'] == 4
    assert sum(r['current_retain_outcome']['status'] == 'zero_units' for r in active) >= 4


def test_cleanup_dedupes_by_document_id_preferring_retry_evidence():
    cleanup = load_module('hindsight_review_backlog_cleanup')
    rows = [
        dict(base_row('dup', '2026-05-01T00:00:00'), hardening_overlay={'semantic_score': 10}),
        dict(base_row('dup', '2026-05-01T00:00:00', retry=True), hardening_overlay={'semantic_score': 20}, deterministic_anomalies=['context_compaction_or_recall_context']),
    ]
    active, archive, summary = cleanup.compact_backlog(rows, cleanup.parse_quotas('zero_with_retry_evidence=1'), max_records=1, retention_months=3, reference_date='2026-05-09T00:00:00+00:00')
    assert len(active) == 1
    assert len(archive) == 0
    assert active[0]['retry_evidence']
    assert summary['by_active_bucket']['zero_with_retry_evidence'] == 1
