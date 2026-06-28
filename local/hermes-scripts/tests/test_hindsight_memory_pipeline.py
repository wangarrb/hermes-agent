import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name='hindsight_memory_pipeline'):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def ns(**kw):
    defaults = dict(
        mode='daily',
        history='incremental',
        bank='hermes',
        llm_profile='minimax',
        prefilter='safe',
        date_mode='auto',
        week_mode='current',
        poll=60,
        timeout=0,
        batch_size=5,
        session_retain_wait_timeout=86400,
        native_consolidation_wait_timeout=86400,
        native_consolidation_poll=None,
        native_consolidation_max_pending=0,
        allow_active_native_operations=False,
        no_wait_native_consolidation=False,
        include_wiki=False,
        skip_daily=False,
        skip_repair_zone=False,
        skip_proposal_review=False,
        execute_proposal_review_llm=False,
        confirm_proposal_review=None,
        notify_proposal_review=False,
        max_proposal_review_llm_calls=None,
        top_proposals=80,
        weekly_budget_max_pending_units=12,
        weekly_budget_max_pending_chars=500000,
        session_limit=None,
        session_profile_mode='hindsight',
        execute=False,
        confirm=None,
        output_root=Path('/tmp/out'),
    )
    defaults.update(kw)
    return Namespace(**defaults)


def names(plan):
    return [s['name'] for s in plan['steps']]


def test_daily_plan_defaults_to_incremental_session_and_no_write():
    mod = load_module()
    plan = mod.build_plan(ns(mode='daily'))
    assert plan['mode'] == 'daily'
    assert 'build_session_manifest' in names(plan)
    retain = next(s for s in plan['steps'] if s['name'] == 'retain_session_manifest')
    assert 'session-manifest-retain-llm' in retain['command']
    assert '--wait-timeout-s' in retain['command']
    assert '86400' in retain['command']
    manifest = next(s for s in plan['steps'] if s['name'] == 'build_session_manifest')
    assert '--profile-mode' in manifest['command']
    assert manifest['command'][manifest['command'].index('--profile-mode') + 1] == 'hindsight'
    assert '--ignore-submit-state' not in retain['command']
    assert '--execute' not in retain['command']
    assert retain['mutating'] is False
    assert 'daily_reflect' in names(plan)
    assert 'native_consolidation_drain_after_daily' in names(plan)
    daily = next(s for s in plan['steps'] if s['name'] == 'daily_reflect')
    bank_idx = daily['command'].index('--bank')
    assert daily['command'][bank_idx + 1] == 'hermes'


def test_full_history_plan_forces_retain_state_ignore_and_includes_weekly_wiki():
    mod = load_module()
    plan = mod.build_plan(ns(mode='full', history='all', include_wiki=True))
    assert plan['mode'] == 'full'
    retain = next(s for s in plan['steps'] if s['name'] == 'retain_session_manifest')
    assert '--ignore-submit-state' in retain['command']
    assert 'weekly_reflect' in names(plan)
    weekly = next(s for s in plan['steps'] if s['name'] == 'weekly_reflect')
    bank_idx = weekly['command'].index('--bank')
    assert weekly['command'][bank_idx + 1] == 'hermes'
    assert 'wiki_auto_maintenance' in names(plan)
    assert 'conflict_audit' in names(plan)


def test_session_profile_mode_can_be_overridden_for_manifest_build():
    mod = load_module()
    plan = mod.build_plan(ns(mode='daily', session_profile_mode='all'))
    manifest = next(s for s in plan['steps'] if s['name'] == 'build_session_manifest')
    assert manifest['command'][manifest['command'].index('--profile-mode') + 1] == 'all'


def test_execute_requires_pipeline_confirm_and_passes_underlying_confirm():
    mod = load_module()
    try:
        mod.build_plan(ns(mode='daily', execute=True, confirm='wrong'))
    except SystemExit as e:
        assert 'run-hindsight-pipeline' in str(e)
    else:
        raise AssertionError('expected confirm gate')
    plan = mod.build_plan(ns(mode='daily', execute=True, confirm='run-hindsight-pipeline'))
    retain = next(s for s in plan['steps'] if s['name'] == 'retain_session_manifest')
    assert '--execute' in retain['command']
    assert 'retain-hindsight-session-manifest' in retain['command']
    assert retain['mutating'] is True


def test_weekly_plan_includes_repair_zone_proposal_sweep_unless_disabled():
    mod = load_module()
    plan = mod.build_plan(ns(mode='weekly'))
    assert 'repair_zone_proposals' in names(plan)
    assert 'proposal_review' in names(plan)
    review = next(s for s in plan['steps'] if s['name'] == 'proposal_review')
    assert '{{proposal_jsons}}' in review['command']
    assert '--execute-llm' not in review['command']
    assert review['mutating'] is False
    plan2 = mod.build_plan(ns(mode='weekly', skip_repair_zone=True))
    assert 'repair_zone_proposals' not in names(plan2)
    assert 'proposal_review' not in names(plan2)
    plan3 = mod.build_plan(ns(mode='weekly', skip_proposal_review=True))
    assert 'repair_zone_proposals' in names(plan3)
    assert 'proposal_review' not in names(plan3)


def test_proposal_review_llm_requires_separate_advisory_confirm_in_command():
    mod = load_module()
    plan = mod.build_plan(ns(
        mode='weekly',
        execute=True,
        confirm='run-hindsight-pipeline',
        execute_proposal_review_llm=True,
        confirm_proposal_review='review-hindsight-proposals',
        notify_proposal_review=True,
    ))
    review = next(s for s in plan['steps'] if s['name'] == 'proposal_review')
    assert '--execute-llm' in review['command']
    assert 'review-hindsight-proposals' in review['command']
    assert '--notify' in review['command']
    assert review['mutating'] is False


def test_full_skip_daily_starts_from_weekly_without_daily_retain_or_daily_v2():
    mod = load_module()
    plan = mod.build_plan(ns(
        mode='full',
        skip_daily=True,
        execute=True,
        confirm='run-hindsight-pipeline',
        execute_proposal_review_llm=True,
        confirm_proposal_review='review-hindsight-proposals',
    ))
    step_names = names(plan)
    assert 'build_session_manifest' not in step_names
    assert 'retain_session_manifest' not in step_names
    assert 'daily_reflect' not in step_names
    assert step_names.count('v2_rebuild_gate') == 1
    assert step_names[:2] == ['preflight', 'runtime_status']
    assert step_names[2] == 'native_consolidation_drain_before_weekly'
    assert step_names[3] == 'weekly_reflect'
    assert 'native_consolidation_drain_after_weekly' in step_names
    assert 'conflict_audit' in step_names
    assert 'proposal_review' in step_names
    assert plan['production_writes_possible'] is True


def test_no_wait_native_consolidation_escape_hatch_removes_gates():
    mod = load_module()
    plan = mod.build_plan(ns(mode='full', skip_daily=True, no_wait_native_consolidation=True))
    step_names = names(plan)
    assert 'native_consolidation_drain_before_weekly' not in step_names
    assert 'native_consolidation_drain_after_weekly' not in step_names
    assert step_names[2] == 'weekly_reflect'
    assert plan['wait_native_consolidation'] is False


def test_skip_daily_is_full_mode_only():
    mod = load_module()
    for mode in ['daily', 'weekly', 'preflight']:
        try:
            mod.build_plan(ns(mode=mode, skip_daily=True))
        except SystemExit as e:
            assert '--skip-daily is only valid with full mode' in str(e)
        else:
            raise AssertionError(f'expected skip-daily gate for {mode}')


def test_custom_bank_propagates_to_retain_and_reflect_steps():
    mod = load_module()
    plan = mod.build_plan(ns(mode='full', bank='hermes_custom'))
    retain = next(s for s in plan['steps'] if s['name'] == 'retain_session_manifest')
    daily = next(s for s in plan['steps'] if s['name'] == 'daily_reflect')
    weekly = next(s for s in plan['steps'] if s['name'] == 'weekly_reflect')
    for step in (retain, daily, weekly):
        idx = step['command'].index('--bank')
        assert step['command'][idx + 1] == 'hermes_custom'
