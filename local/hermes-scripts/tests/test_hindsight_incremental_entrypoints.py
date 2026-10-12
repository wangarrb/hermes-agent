from pathlib import Path
import importlib.util
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('entry_test_' + name, ROOT / (name + '.py'))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stub_arrival(monkeypatch):
    calls = []
    module = types.ModuleType('hindsight_incremental_pipeline')
    setattr(module, 'main', lambda argv: calls.append(argv) or 2)
    monkeypatch.setitem(sys.modules, 'hindsight_incremental_pipeline', module)
    return calls


def test_normal_memory_entry_never_builds_legacy_all_history_plan(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    calls = stub_arrival(monkeypatch)
    module = load('hindsight_memory_pipeline')
    monkeypatch.setattr(module, 'build_plan', lambda *_a: pytest.fail('legacy plan reached'))
    rc = module.main(['full', '--execute', '--confirm', 'run-hindsight-pipeline', '--include-wiki'])
    assert rc == 2 and len(calls) == 1
    assert calls[0][:3] == ['run', '--mode', 'full']
    assert '--include-wiki' in calls[0]
    assert '--backfill-missing-daily' not in calls[0]


def test_memory_normal_entry_rejects_history_replay_before_arrival(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    calls = stub_arrival(monkeypatch)
    module = load('hindsight_memory_pipeline')
    monkeypatch.setattr(module, 'build_plan', lambda *_a: pytest.fail('legacy plan reached'))
    with pytest.raises(SystemExit):
        module.main(['full', '--history', 'all', '--execute', '--confirm', 'run-hindsight-pipeline'])
    assert calls == []


def test_daily_wrapper_normal_path_does_not_preclean_or_generate_digest(monkeypatch, tmp_path):
    monkeypatch.setenv('HINDSIGHT_DAILY_HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    calls = stub_arrival(monkeypatch)
    module = load('hindsight_daily_noagent')
    monkeypatch.setattr(module, 'check_hindsight', lambda: pytest.fail('legacy health path reached'))
    monkeypatch.setattr(module, 'run_research_summary', lambda *_a, **_k: pytest.fail('digest blocks core'))
    monkeypatch.setattr(module.subprocess, 'run', lambda *_a, **_k: pytest.fail('legacy subprocess reached'))
    monkeypatch.setattr(sys, 'argv', ['hindsight_daily_noagent.py', '--mode', 'full', '--skip-daily'])
    assert module.main() == 2
    assert len(calls) == 1 and '--resume-only' in calls[0]


def test_configured_supported_interpreter_is_explicit_not_import_fallback(monkeypatch, tmp_path):
    calls = stub_arrival(monkeypatch)
    module = load('hindsight_incremental_entrypoint')
    import hindsight_pipeline_common as common
    target = tmp_path / 'explicit-python'
    target.write_text('fixture interpreter path')
    monkeypatch.setattr(common, 'load_config', lambda *_a, **_k: {'publication': {'publisher_python': str(target)}})
    seen = []
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kwargs: seen.append((argv, kwargs)) or types.SimpleNamespace(returncode=2))
    rc = module.run_alias(mode='full', execute=True, confirm='run-hindsight-pipeline', config=tmp_path / 'explicit.json')
    assert rc == 2 and calls == []
    assert seen[0][0][:3] == [str(target), '-u', str(ROOT / 'hindsight_incremental_pipeline.py')]
    assert seen[0][0][3:6] == ['run', '--mode', 'full']
    assert seen[0][1] == {'check': False}


@pytest.mark.parametrize('invalid', ['', None, 'relative-python', 'bad\npython'])
def test_explicit_bad_interpreter_blocks_without_hidden_reexec(monkeypatch, invalid):
    calls = stub_arrival(monkeypatch)
    module = load('hindsight_incremental_entrypoint')
    import hindsight_pipeline_common as common
    monkeypatch.setattr(common, 'load_config', lambda *_a, **_k: {'publication': {'publisher_python': invalid}})
    with pytest.raises(ValueError, match='runtime_interpreter'):
        module.run_alias(mode='daily', execute=True, confirm='run-hindsight-pipeline')
    assert calls == []


@pytest.mark.parametrize('scope', ['publication', 'incremental.publication'])
@pytest.mark.parametrize('invalid', [None, '', False, 0, [], [['publisher_python', sys.executable]], 'private-config-value'])
def test_explicit_invalid_publication_scopes_block_both_paths_with_real_config(monkeypatch, tmp_path, scope, invalid):
    import json
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    calls = stub_arrival(monkeypatch)
    module = load('hindsight_incremental_entrypoint')
    cfg = {'publication': {'publisher_python': sys.executable}, 'incremental': {'publication': {}}}
    if scope == 'publication':
        cfg['publication'] = invalid
    else:
        cfg['incremental']['publication'] = invalid
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(module.subprocess, 'run', lambda *_a, **_k: pytest.fail('invalid configuration reached subprocess'))
    with pytest.raises(ValueError, match='^invalid_publication_configuration$'):
        module.run_alias(mode='full', execute=True, confirm='run-hindsight-pipeline', config=path)
    assert calls == []


@pytest.mark.parametrize('invalid', [None, '', False, 0, []])
def test_invalid_incremental_scope_cannot_hide_publication(monkeypatch, tmp_path, invalid):
    import json
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    calls = stub_arrival(monkeypatch)
    module = load('hindsight_incremental_entrypoint')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'incremental': invalid}))
    with pytest.raises(ValueError, match='^invalid_incremental_configuration$'):
        module.run_alias(mode='daily', execute=False, config=path)
    assert calls == []


def test_weekly_wrapper_uses_its_own_verified_sibling_not_live_main_path():
    module = load('hindsight_weekly_wrapper')
    command = module.build_command('/existing/python')
    assert command[0] == '/existing/python'
    assert Path(command[1]) == ROOT / 'hindsight_daily_noagent.py'
    assert '--mode' in command and 'full' in command and '--include-wiki' in command
