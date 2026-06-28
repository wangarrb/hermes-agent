import argparse
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


def reflect_args(**overrides):
    defaults = {
        'llm_model': 'MiniMax-M2.7',
        'llm_label': 'minimax',
        'emit_observations': True,
        'output_language': 'zh',
        'budget_max_pending_units': -1,
        'budget_max_pending_chars': -1,
        'force_repost': False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_unit_progress_key_ignores_period_but_tracks_versions_and_sources():
    mod = load_module('offline_hindsight_reflect_consolidate')
    unit_w19 = mod.ReflectUnit(
        scope='weekly',
        period='history-through-2026-W19',
        topic='egomotion4d',
        index=0,
        content='scope: weekly\nperiod: history-through-2026-W19\ntopic: egomotion4d\nbody unchanged\n',
        source_count=2,
        source_ids=['daily/a.md', 'daily/b.md'],
        date_range_start='2026-01-01T00:00:00',
        date_range_end='2026-05-07T00:00:00',
    )
    unit_w20 = mod.ReflectUnit(
        scope='weekly',
        period='history-through-2026-W20',
        topic='egomotion4d',
        index=0,
        content='scope: weekly\nperiod: history-through-2026-W20\ntopic: egomotion4d\nbody unchanged\n',
        source_count=2,
        source_ids=['daily/a.md', 'daily/b.md'],
        date_range_start='2026-01-01T00:00:00',
        date_range_end='2026-05-14T00:00:00',
    )

    key_w19 = mod.unit_progress_key(unit_w19, args=reflect_args())
    key_w20 = mod.unit_progress_key(unit_w20, args=reflect_args())
    assert key_w19 == key_w20

    different_model = mod.unit_progress_key(unit_w20, args=reflect_args(llm_model='Different-Model'))
    assert different_model != key_w19

    source_changed = mod.ReflectUnit(
        scope='weekly',
        period='history-through-2026-W20',
        topic='egomotion4d',
        index=0,
        content=unit_w20.content,
        source_count=2,
        source_ids=['daily/a.md', 'daily/c.md'],
        date_range_start=unit_w20.date_range_start,
        date_range_end=unit_w20.date_range_end,
    )
    assert mod.unit_progress_key(source_changed, args=reflect_args()) != key_w19


def test_parse_doc_day_topic_supports_legacy_and_native_session_ids():
    mod = load_module('offline_hindsight_reflect_consolidate')

    assert mod.parse_doc_day_topic('hermes-sqlite::day-topic::2026-04-20__egomotion4d::abc') == ('2026-04-20', 'egomotion4d')
    assert mod.parse_doc_day_topic('hermes-session::20260420_182547_6feedd') == ('2026-04-20', None)
    assert mod.parse_doc_day_topic('hermes-session::session_20260420_182547_6feedd::part-000') == ('2026-04-20', None)
    assert mod.parse_doc_day_topic('other') == (None, None)


def test_query_facts_for_days_includes_legacy_and_native_session_ids(monkeypatch):
    mod = load_module('offline_hindsight_reflect_consolidate')
    captured = {}

    def fake_psql_json(sql):
        captured['sql'] = sql
        return [
            {
                'fact_id': 'f1',
                'document_id': 'hermes-session::20260420_182547_6feedd',
                'text': 'Egomotion4D 结论',
                'fact_type': 'technical_lesson',
                'event_date': '2026-04-20T18:25:47+08:00',
                'created_at': '2026-05-10T10:00:00+08:00',
            }
        ]

    monkeypatch.setattr(mod, 'psql_json', fake_psql_json)
    facts = mod.query_facts_for_days('hermes', ['2026-04-20'])

    assert "hermes-sqlite::day-topic::2026-04-20__%" in captured['sql']
    assert "hermes-session::20260420%" in captured['sql']
    assert "hermes-session::session_20260420%" in captured['sql']
    assert len(facts) == 1
    assert facts[0].document_id == 'hermes-session::20260420_182547_6feedd'
    assert facts[0].topic


def test_budget_report_counts_cached_pending_and_blocks_when_over_threshold():
    mod = load_module('offline_hindsight_reflect_consolidate')
    units = [
        mod.ReflectUnit('weekly', 'history-through-2026-W20', 'topic-a', 0, 'period: history-through-2026-W20\nA', 1, ['daily/a.md'], 's', 'e'),
        mod.ReflectUnit('weekly', 'history-through-2026-W20', 'topic-a', 1, 'period: history-through-2026-W20\nB', 1, ['daily/b.md'], 's', 'e'),
    ]
    cached_key = mod.unit_progress_key(units[0], args=reflect_args())
    progress = {
        'processed_units_v2': {
            cached_key: {'document_id': 'doc-a', 'output_markdown': 'a.md'}
        },
        'processed_unit_keys': [],
        'processed_document_ids': [],
    }

    report = mod.build_budget_report(
        units,
        progress,
        args=reflect_args(budget_max_pending_units=0, budget_max_pending_chars=100000),
    )

    assert report['total_units'] == 2
    assert report['cached_units'] == 1
    assert report['pending_units'] == 1
    assert report['budget_decision'] == 'blocked_budget_exceeded'
    assert report['block_reasons'] == ['pending_units 1 > max 0']
    assert report['cache']['reused_v2'] == 1
    assert report['cache']['new'] == 1


def test_cron_weekly_budget_command_uses_direct_offline_script_not_paid_wrapper():
    cron = load_module('hindsight_offline_cron_runner')
    args = argparse.Namespace(
        prefilter='safe',
        weekly_budget_max_pending_units=12,
        weekly_budget_max_pending_chars=500000,
        _offline_reflect_llm_args=['--llm-model', 'glm-5', '--llm-label', 'glm', '--no-response-format'],
    )
    cmd = cron.weekly_budget_cmd(args, '2026-W20')
    joined = ' '.join(map(str, cmd))

    assert 'offline_hindsight_reflect_consolidate.py' in joined
    assert 'hindsight_minimax_import.py' not in joined
    assert '--mode dry-run' in joined
    assert '--budget-json' in joined
    assert '--budget-max-pending-units 12' in joined
    assert '--budget-max-pending-chars 500000' in joined
    assert '--llm-model glm-5' in joined
    assert '--llm-label glm' in joined
    assert '--no-response-format' in joined


def test_budget_report_blocks_when_weekly_backfill_is_missing():
    mod = load_module('offline_hindsight_reflect_consolidate')
    args = reflect_args(budget_max_pending_units=12, budget_max_pending_chars=500000)
    args._budget_missing_daily = ['2026-05-07']
    report = mod.build_budget_report([], {'processed_units_v2': {}}, args=args)

    assert report['budget_decision'] == 'blocked_budget_exceeded'
    assert report['missing_daily_outputs'] == ['2026-05-07']
    assert any('missing_daily_outputs 1' in reason for reason in report['block_reasons'])


def test_dry_run_budget_only_skips_daily_for_both_task():
    cron = load_module('hindsight_offline_cron_runner')
    args = argparse.Namespace(task='both', dry_run_budget_only=True)
    assert cron.should_run_daily(args) is False
    assert cron.should_run_weekly(args) is True


def test_publish_safety_preflight_blocks_missing_confirm_and_skip_flags():
    mod = load_module('hindsight_offline_v2_rebuild')
    base = {
        'mode': 'publish',
        'confirm_publish': None,
        'skip_conflict_audit': False,
        'skip_eval_gate': False,
    }
    assert mod.publish_safety_errors(argparse.Namespace(**base)) == [
        'publish requires --confirm-publish publish-hindsight-v2-canonical'
    ]

    args = argparse.Namespace(**{**base, 'confirm_publish': 'publish-hindsight-v2-canonical', 'skip_conflict_audit': True})
    assert mod.publish_safety_errors(args) == ['publish with --skip-conflict-audit is unsafe and blocked']

    args = argparse.Namespace(**{**base, 'confirm_publish': 'publish-hindsight-v2-canonical', 'skip_eval_gate': True})
    assert mod.publish_safety_errors(args) == ['publish with --skip-eval-gate is unsafe and blocked']

    safe = argparse.Namespace(**{**base, 'confirm_publish': 'publish-hindsight-v2-canonical'})
    assert mod.publish_safety_errors(safe) == []


def test_daily_completion_report_detects_partial_existing_day(tmp_path, monkeypatch):
    mod = load_module('offline_hindsight_reflect_consolidate')
    expected_units = [
        mod.ReflectUnit('daily', '2026-05-07', 'topic-a', 0, 'a', 1, ['src-a'], 's', 'e'),
        mod.ReflectUnit('daily', '2026-05-07', 'topic-b', 0, 'b', 1, ['src-b'], 's', 'e'),
    ]

    def fake_build_daily_for_args(args):
        assert args.scope == 'daily'
        assert args.date == '2026-05-07'
        return expected_units, None, None

    monkeypatch.setattr(mod, 'build_daily_for_args', fake_build_daily_for_args)
    day_dir = tmp_path / 'daily' / '2026-05-07'
    day_dir.mkdir(parents=True)
    (day_dir / 'topic-a__00__hash.md').write_text(
        '# Daily\n\n'
        'scope: daily\n'
        'period: 2026-05-07\n'
        'topic: topic-a\n\n'
        '## Source IDs\n'
        '- src-a\n',
        encoding='utf-8',
    )
    args = argparse.Namespace(output_dir=str(tmp_path), scope='weekly', date=None, daily_source='facts', group_by='topic')

    report = mod.daily_completion_report(args, ['2026-05-07'])

    assert report['done_days'] == []
    assert report['missing_days'] == ['2026-05-07']
    assert report['details']['2026-05-07']['expected_units'] == 2
    assert report['details']['2026-05-07']['matched_units'] == 1
    assert report['details']['2026-05-07']['missing_units'] == 1
