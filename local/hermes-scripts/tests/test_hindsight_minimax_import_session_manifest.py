import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_base_env_keeps_embeddings_local_for_hindsight_container():
    mod = load_module('hindsight_minimax_import')
    env = mod.base_env()
    assert env['HINDSIGHT_API_EMBEDDINGS_PROVIDER'] == 'local'
    assert env['HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL'] == 'BAAI/bge-m3'
    assert env['HF_HUB_OFFLINE'] == '1'
    assert env['TRANSFORMERS_OFFLINE'] == '1'
    assert 'HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL' not in env
    assert 'HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL' not in env


def test_operation_count_from_response_prefers_total_metadata_and_has_list_fallback():
    mod = load_module('hindsight_minimax_import')
    assert mod._operation_count_from_response({'total': 42, 'operations': [{}]}) == 42
    assert mod._operation_count_from_response({'total_operations': 7}) == 7
    assert mod._operation_count_from_response({'count': 3}) == 3
    assert mod._operation_count_from_response({'operations': [{'id': 'a'}]}) == 1
    assert mod._operation_count_from_response({}) == 0


def test_switch_mode_import_non_default_bank_checks_and_protects_default_bank(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    calls = []
    profile = {
        'label': 'minimax',
        'hindsight_provider': 'minimax',
        'model': 'MiniMax-M2.7',
        'base_url': 'https://api.minimaxi.com/v1',
        'api_key': 'test-key',
        'api_key_env': 'MINIMAX_API_KEY',
        'response_format': True,
    }

    monkeypatch.setattr(mod, 'ensure_hermes_hindsight_idle_config', lambda: calls.append(('idle_config',)))
    monkeypatch.setattr(mod, 'queue_counts', lambda bank: calls.append(('queue_counts', bank)) or (0, 0, {'queue_counts_source': 'test'}))
    monkeypatch.setattr(mod, 'try_disable_observations_before_restart', lambda reason, **kwargs: calls.append(('pre_disable', reason, kwargs)))
    monkeypatch.setattr(mod, 'recreate_container', lambda env: calls.append(('recreate_container', env.get('HINDSIGHT_API_ENABLE_OBSERVATIONS'))))
    monkeypatch.setattr(mod, 'wait_health', lambda timeout_s=300: calls.append(('wait_health', timeout_s)))
    monkeypatch.setattr(mod, 'patch_json_parser_and_restart', lambda: calls.append(('patch_container',)))
    monkeypatch.setattr(mod, 'patch_bank_config', lambda **kwargs: calls.append(('patch_bank_config', kwargs)))

    mod.switch_mode(
        'import-minimax',
        allow_existing_queue=False,
        enable_observations=True,
        llm_profile=profile,
        health_timeout_s=321,
        queue_bank='hermes_v3_minimax_smoke_20260508',
    )

    assert ('queue_counts', 'hermes_v3_minimax_smoke_20260508') in calls
    assert ('queue_counts', 'hermes') in calls
    assert ('pre_disable', 'import-minimax scoped restart', {'bank': 'hermes'}) in calls
    assert ('pre_disable', 'import-minimax restart', {'bank': 'hermes_v3_minimax_smoke_20260508'}) not in calls
    assert ('patch_bank_config', {'enable_observations': True, 'bank': 'hermes_v3_minimax_smoke_20260508'}) in calls


def test_session_manifest_retain_paid_llm_switches_provider_runs_runner_and_restores(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    calls = []

    monkeypatch.setattr(mod, 'get_llm_profile', lambda name=None: {
        'label': 'minimax',
        'hindsight_provider': 'minimax',
        'model': 'MiniMax-M2.7',
        'base_url': 'https://api.minimaxi.com/v1',
        'api_key': 'test-key',
        'api_key_env': 'MINIMAX_API_KEY',
        'response_format': True,
    })

    def fake_switch_mode(mode, **kwargs):
        calls.append(('switch_mode', mode, kwargs))

    monkeypatch.setattr(mod, 'switch_mode', fake_switch_mode)
    monkeypatch.setattr(mod, 'ensure_bank_exists', lambda bank: calls.append(('ensure_bank_exists', bank)))
    monkeypatch.setattr(mod, 'patch_bank_config', lambda **kwargs: calls.append(('patch_bank_config', kwargs)))
    monkeypatch.setattr(mod, 'wait_queue_drained', lambda poll, timeout, **kwargs: calls.append(('wait_queue_drained', poll, timeout, kwargs)))

    class FakeProc:
        returncode = 0

    def fake_run(cmd):
        calls.append(('subprocess.run', cmd))
        return FakeProc()

    monkeypatch.setattr(mod.subprocess, 'run', fake_run)

    args = Namespace(
        llm_profile='minimax',
        allow_existing_queue=False,
        enable_observations=True,
        no_wait=False,
        poll=7,
        timeout=123,
        health_timeout_s=321,
        manifest=Path('/tmp/session-manifest.jsonl'),
        bank='hermes_v3_minimax_smoke_20260508',
        limit=3,
        batch_size=2,
        submit_state=Path('/tmp/submit-state.json'),
        wait_timeout_s=600,
        poll_s=5.0,
        ignore_submit_state=False,
        execute=True,
        confirm='retain-hindsight-session-manifest',
        runner_args=[],
    )

    rc = mod.run_session_manifest_retain_llm(args)

    assert rc == 0
    assert calls[0][0] == 'switch_mode'
    assert calls[0][1] == 'import-minimax'
    assert calls[0][2]['enable_observations'] is True
    assert calls[0][2]['health_timeout_s'] == 321
    assert calls[0][2]['queue_bank'] == 'hermes_v3_minimax_smoke_20260508'
    assert ('ensure_bank_exists', 'hermes_v3_minimax_smoke_20260508') in calls
    assert ('patch_bank_config', {'enable_observations': True, 'bank': 'hermes_v3_minimax_smoke_20260508'}) in calls
    runner_cmd = next(c for c in calls if c[0] == 'subprocess.run')[1]
    joined = ' '.join(map(str, runner_cmd))
    assert 'hindsight_session_retain_runner.py' in joined
    assert '--manifest /tmp/session-manifest.jsonl' in joined
    assert '--bank hermes_v3_minimax_smoke_20260508' in joined
    assert '--limit 3' in joined
    assert '--batch-size 2' in joined
    assert '--execute' in runner_cmd
    assert '--confirm' in runner_cmd
    assert 'retain-hindsight-session-manifest' in runner_cmd
    waits = [c for c in calls if c[0] == 'wait_queue_drained']
    assert waits == [
        ('wait_queue_drained', 7, 123, {'bank': 'hermes_v3_minimax_smoke_20260508'}),
    ]
    # The runner itself waits for submitted retain operation ids. The wrapper
    # must not drain the whole bank queue before restore, because native
    # consolidation can requeue until the entire unconsolidated backlog is empty.
    run_idx = next(i for i, c in enumerate(calls) if c[0] == 'subprocess.run')
    first_disable_after_run = next(
        i for i, c in enumerate(calls)
        if i > run_idx and c == ('patch_bank_config', {'enable_observations': False, 'bank': 'hermes_v3_minimax_smoke_20260508'})
    )
    wait_idx = next(i for i, c in enumerate(calls) if c[0] == 'wait_queue_drained')
    assert run_idx < first_disable_after_run < wait_idx
    normal_restore = [c for c in calls if c[0] == 'switch_mode' and c[1] == 'normal-local']
    assert normal_restore
    # The target bank must be explicitly protected after restore; otherwise
    # non-default smoke banks can retain enable_observations=true in config.
    patch_calls = [c for c in calls if c == ('patch_bank_config', {'enable_observations': False, 'bank': 'hermes_v3_minimax_smoke_20260508'})]
    assert len(patch_calls) >= 2


def test_session_manifest_retain_paid_llm_post_restore_wait_failure_is_nonzero(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    calls = []

    monkeypatch.setattr(mod, 'get_llm_profile', lambda name=None: {
        'label': 'minimax',
        'hindsight_provider': 'minimax',
        'model': 'MiniMax-M2.7',
        'base_url': 'https://api.minimaxi.com/v1',
        'api_key': 'test-key',
        'api_key_env': 'MINIMAX_API_KEY',
        'response_format': True,
    })
    monkeypatch.setattr(mod, 'switch_mode', lambda mode, **kwargs: calls.append(('switch_mode', mode, kwargs)))
    monkeypatch.setattr(mod, 'ensure_bank_exists', lambda bank: calls.append(('ensure_bank_exists', bank)))
    monkeypatch.setattr(mod, 'patch_bank_config', lambda **kwargs: calls.append(('patch_bank_config', kwargs)))

    def fail_wait(poll, timeout, **kwargs):
        calls.append(('wait_queue_drained', poll, timeout, kwargs))
        raise TimeoutError('stuck queue')

    monkeypatch.setattr(mod, 'wait_queue_drained', fail_wait)

    class FakeProc:
        returncode = 0

    monkeypatch.setattr(mod.subprocess, 'run', lambda cmd: FakeProc())

    args = Namespace(
        llm_profile='minimax',
        allow_existing_queue=False,
        enable_observations=True,
        no_wait=False,
        poll=7,
        timeout=0,
        health_timeout_s=321,
        manifest=Path('/tmp/session-manifest.jsonl'),
        bank='hermes_v3_minimax_smoke_20260508',
        limit=None,
        batch_size=2,
        submit_state=Path('/tmp/submit-state.json'),
        wait_timeout_s=600,
        poll_s=5.0,
        ignore_submit_state=False,
        execute=True,
        confirm='retain-hindsight-session-manifest',
        runner_args=[],
    )

    assert mod.run_session_manifest_retain_llm(args) == 1
    assert ('switch_mode', 'normal-local', {'allow_existing_queue': True, 'health_timeout_s': 321}) in calls
    assert ('wait_queue_drained', 7, 600, {'bank': 'hermes_v3_minimax_smoke_20260508'}) in calls


def test_session_manifest_retain_paid_llm_dry_run_does_not_switch_provider(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    calls = []

    monkeypatch.setattr(mod, 'switch_mode', lambda *a, **k: calls.append(('switch_mode', a, k)))

    class FakeProc:
        returncode = 0

    def fake_run(cmd):
        calls.append(('subprocess.run', cmd))
        return FakeProc()

    monkeypatch.setattr(mod.subprocess, 'run', fake_run)
    args = Namespace(
        llm_profile='minimax',
        allow_existing_queue=False,
        enable_observations=False,
        no_wait=True,
        poll=7,
        timeout=123,
        health_timeout_s=321,
        manifest=Path('/tmp/session-manifest.jsonl'),
        bank='hermes_v3_minimax_smoke_20260508',
        limit=3,
        batch_size=2,
        submit_state=Path('/tmp/submit-state.json'),
        wait_timeout_s=600,
        poll_s=5.0,
        ignore_submit_state=False,
        execute=False,
        confirm=None,
        runner_args=[],
    )

    rc = mod.run_session_manifest_retain_llm(args)

    assert rc == 0
    assert not [c for c in calls if c[0] == 'switch_mode']
    runner_cmd = calls[0][1]
    assert '--execute' not in runner_cmd
    assert '--json' in runner_cmd


def test_half_consolidation_overrides_only_reduce_native_concurrency():
    mod = load_module('hindsight_minimax_import')
    runtime = {
        'HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT': '8',
        'HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES': '8',
        'HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT': '60',
    }
    overrides = mod.build_half_consolidation_overrides(runtime)
    assert overrides == {
        'HINDSIGHT_NATIVE_CONSOLIDATION_LLM_MAX_CONCURRENT': '4',
        'HINDSIGHT_NATIVE_CONSOLIDATION_PARALLEL_BATCHES': '4',
        'HINDSIGHT_NATIVE_CONSOLIDATION_RECALL_MAX_CONCURRENT': '30',
    }


def test_paid_llm_env_allows_consolidation_only_concurrency_override(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    monkeypatch.setenv('HINDSIGHT_OFFLINE_LLM_CONCURRENCY', '8')
    monkeypatch.setenv('HINDSIGHT_NATIVE_CONSOLIDATION_LLM_MAX_CONCURRENT', '4')
    profile = {
        'label': 'minimax',
        'hindsight_provider': 'minimax',
        'model': 'MiniMax-M2.7',
        'base_url': 'https://api.minimaxi.com/v1',
        'api_key': 'test-key',
        'api_key_env': 'MINIMAX_API_KEY',
        'response_format': True,
    }
    env = mod.paid_llm_env(profile, enable_observations=True)
    assert env['HINDSIGHT_API_LLM_MAX_CONCURRENT'] == '8'
    assert env['HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT'] == '8'
    assert env['HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT'] == '8'
    assert env['HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT'] == '4'


def test_session_manifest_retain_paid_llm_execute_requires_confirm():
    mod = load_module('hindsight_minimax_import')
    args = Namespace(
        execute=True,
        confirm=None,
        llm_profile='minimax',
        allow_existing_queue=False,
        enable_observations=False,
        no_wait=True,
        poll=7,
        timeout=123,
        health_timeout_s=321,
        manifest=Path('/tmp/session-manifest.jsonl'),
        bank='hermes_v3_minimax_smoke_20260508',
        limit=3,
        batch_size=2,
        submit_state=Path('/tmp/submit-state.json'),
        wait_timeout_s=600,
        poll_s=5.0,
        ignore_submit_state=False,
        runner_args=[],
    )

    try:
        mod.run_session_manifest_retain_llm(args)
    except SystemExit as exc:
        assert 'retain-hindsight-session-manifest' in str(exc)
    else:
        raise AssertionError('execute must require retain confirm token')
