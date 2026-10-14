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


def test_wrapper_budget_clamp_matches_safety_ceilings(monkeypatch):
    """Wrapper must accept explicit budgets up to the safety ceilings and reject
    anything beyond them, matching the incremental pipeline constants."""
    import types
    mp = load_module()
    calls = {}
    stub = types.ModuleType('hindsight_incremental_entrypoint')
    def fake_alias(**kw):
        calls.update(kw)
        return 0
    stub.run_alias = fake_alias
    monkeypatch.setitem(sys.modules, 'hindsight_incremental_entrypoint', stub)
    assert mp.main(['daily', '--weekly-budget-max-pending-units', '200',
                    '--weekly-budget-max-pending-chars', '20000000']) == 0
    assert calls['max_units'] == 200 and calls['max_chars'] == 20000000
    import pytest as _pytest
    for bad in (['--weekly-budget-max-pending-units', '201'],
                ['--weekly-budget-max-pending-chars', '20000001'],
                ['--weekly-budget-max-pending-units', '0']):
        with _pytest.raises(SystemExit) as err:
            mp.main(['daily'] + bad)
        assert err.value.code == 2


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
        backfill_missing_daily=False,
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
        manual_review_enabled=False,
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


def test_daily_plan_does_not_auto_submit_manual_review_backlog():
    mod = load_module()
    plan = mod.build_plan(ns(mode='daily', execute=True, confirm='run-hindsight-pipeline'))
    review = next(s for s in plan['steps'] if s['name'] == 'manual_review_sampling')
    assert '--execute' not in review['command']
    assert '--submit' not in review['command']



def test_full_history_plan_forces_retain_state_ignore_and_includes_weekly_wiki():
    mod = load_module()
    plan = mod.build_plan(ns(mode='full', history='all', include_wiki=True))
    assert plan['mode'] == 'full'
    retain = next(s for s in plan['steps'] if s['name'] == 'retain_session_manifest')
    assert '--ignore-submit-state' in retain['command']
    assert 'weekly_reflect' in names(plan)
    weekly = next(s for s in plan['steps'] if s['name'] == 'weekly_reflect')
    assert '--backfill-missing-daily' not in weekly['command']
    bank_idx = weekly['command'].index('--bank')
    assert weekly['command'][bank_idx + 1] == 'hermes'
    assert 'wiki_auto_maintenance' in names(plan)
    assert 'conflict_audit' in names(plan)
    step_names = names(plan)
    assert step_names.count('v2_rebuild_gate') == 1
    assert step_names.index('weekly_reflect') < step_names.index('v2_rebuild_gate')
    assert step_names.index('native_consolidation_drain_after_weekly') < step_names.index('v2_rebuild_gate')


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


def test_preflight_plan_does_not_include_observation_restore():
    mod = load_module()
    plan = mod.build_plan(ns(mode='preflight'))
    assert names(plan) == ['preflight']
    assert not plan['production_writes_possible']


def test_pipeline_env_defers_wrapper_restore_and_uses_configured_api(monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, 'get_cfg', lambda _args: {'api_url': 'http://hindsight.test:8888'})
    monkeypatch.setattr(mod, 'consolidation_tuning', lambda _cfg: {
        'consolidation_batch_size': 4,
        'consolidation_llm_batch_size': 8,
        'max_memories_per_round': 64,
        'parallel_batches': 1,
        'recall_budget': 8,
        'source_facts_max_tokens': 1000,
        'source_facts_max_tokens_per_observation': 500,
    })
    env = mod.pipeline_env(ns(mode='daily', execute=True))
    assert env['HINDSIGHT_DEFER_OBSERVATIONS_RESTORE'] == '1'
    assert env['HINDSIGHT_API_URL'] == 'http://hindsight.test:8888'
    preflight_env = mod.pipeline_env(ns(mode='preflight', execute=False))
    assert preflight_env.get('HINDSIGHT_DEFER_OBSERVATIONS_RESTORE') != '1'


def test_execute_plan_restores_observations_after_later_step_failure(monkeypatch):
    mod = load_module()
    args = ns(mode='daily', execute=True, confirm='run-hindsight-pipeline')
    monkeypatch.setattr(mod, 'bank_observations_state', lambda *_args: (False, False))
    calls = []

    def fake_run_command(cmd, env=None):
        calls.append(list(cmd))
        if cmd == ['daily']:
            raise SystemExit(7)
        return mod.subprocess.CompletedProcess(cmd, 0, stdout='{}', stderr='')

    monkeypatch.setattr(mod, 'run_command', fake_run_command)
    plan = {'steps': [
        {'name': 'preflight', 'command': ['preflight']},
        {'name': 'retain_session_manifest', 'command': ['retain']},
        {'name': 'daily_reflect', 'command': ['daily']},
        {'name': 'restore_observations_off', 'command': ['restore']},
    ]}
    try:
        mod.execute_plan(plan, args)
    except SystemExit as exc:
        assert exc.code == 7
    else:
        raise AssertionError('expected the original daily-reflect failure')
    assert ['restore'] in calls


def test_restore_observations_off_patches_then_verifies_both_values(monkeypatch):
    mod = load_module()
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def read(self):
            import json
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        calls.append(request)
        if request.method == 'PATCH':
            return Response({'config': {'enable_observations': False}})
        return Response({'config': {'enable_observations': False}, 'overrides': {'enable_observations': False}})

    monkeypatch.setattr(mod.urllib.request, 'urlopen', fake_urlopen)
    result = mod.restore_observations_off('http://hindsight.test:8888', 'hermes')
    assert [request.method for request in calls] == ['PATCH', 'GET']
    assert result == {'config_enable_observations': False, 'override_enable_observations': False}


def test_restore_observations_off_fails_if_readback_is_inconsistent(monkeypatch):
    mod = load_module()
    responses = [
        {'config': {'enable_observations': False}},
        {'config': {'enable_observations': False}, 'overrides': {'enable_observations': True}},
    ]

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def read(self):
            import json
            return json.dumps(self.payload).encode()

    monkeypatch.setattr(mod.urllib.request, 'urlopen', lambda *_args, **_kwargs: Response(responses.pop(0)))
    try:
        mod.restore_observations_off('http://hindsight.test:8888', 'hermes')
    except RuntimeError as exc:
        assert 'readback' in str(exc).lower()
    else:
        raise AssertionError('inconsistent bank settings must fail closed')


def test_weekly_plan_does_not_opt_into_pre_checkpoint_history_repair_by_default():
    mod = load_module()
    plan = mod.build_plan(ns(mode='weekly'))
    weekly = next(step for step in plan['steps'] if step['name'] == 'weekly_reflect')
    assert '--backfill-missing-daily' not in weekly['command']


def test_weekly_plan_requires_opt_in_for_pre_checkpoint_history_repair():
    mod = load_module()
    plan = mod.build_plan(ns(mode='weekly', backfill_missing_daily=True))
    weekly = next(step for step in plan['steps'] if step['name'] == 'weekly_reflect')
    assert '--backfill-missing-daily' in weekly['command']


def test_weekly_execute_plan_enforces_budget_on_the_submit_path():
    mod = load_module()
    step = mod.weekly_reflect_step(ns(mode='weekly', execute=True))
    command = step.command

    assert '--enforce-budget' in command
    unit_flag = command.index('--budget-max-pending-units')
    char_flag = command.index('--budget-max-pending-chars')
    assert command[unit_flag + 1] == '12'
    assert command[char_flag + 1] == '500000'


def test_backfill_opt_in_is_rejected_for_daily_mode():
    mod = load_module()
    try:
        mod.build_plan(ns(mode='daily', backfill_missing_daily=True))
    except SystemExit as exc:
        assert 'weekly/full' in str(exc)
    else:
        raise AssertionError('daily mode must not accept weekly backfill')
