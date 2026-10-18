import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        'hindsight_manual_review_sampler', ROOT / 'hindsight_manual_review_sampler.py'
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_sampler_ignores_old_manual_review_records():
    module = load_module()
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    records = [
        {
            'action': 'manual_review',
            'document_id': 'old-egomotion4d',
            'event_date': '2026-08-13T00:00:00+00:00',
            'content_chars': 120000,
            'tags': ['project:egomotion4d', 'topic:memory-management'],
            'reason': 'multi_scope_or_overbroad_tags',
        },
        {
            'action': 'manual_review',
            'document_id': 'recent-egomotion4d',
            'event_date': '2026-09-27T00:00:00+00:00',
            'content_chars': 1200,
            'tags': ['project:egomotion4d', 'topic:algorithm'],
            'reason': 'multi_scope_or_overbroad_tags',
        },
    ]

    sampled = module.sample_records(records, 5, {}, now=now, max_age_days=14)

    assert [r['document_id'] for r in sampled] == ['recent-egomotion4d']


def test_sampler_ignores_recent_memory_maintenance_only_records():
    module = load_module()
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    record = {
        'action': 'manual_review',
        'document_id': 'recent-bootstrap',
        'event_date': '2026-09-27T00:00:00+00:00',
        'content_chars': 120000,
        'tags': [
            'project:egomotion4d',
            'topic:memory-management',
            'topic:native-consolidation',
        ],
        'reason': 'multi_scope_or_overbroad_tags',
    }

    assert module.sample_records([record], 5, {}, now=now, max_age_days=14) == []
