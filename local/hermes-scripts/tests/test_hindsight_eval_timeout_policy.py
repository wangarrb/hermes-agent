"""Invariant: incremental eval subprocess inherits the established legacy policy."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('provided', [None, {'ONLY_THIS_SCOPE': 'fixture', 'HINDSIGHT_EVAL_REQUEST_TIMEOUT': '120'}])
def test_incremental_eval_uses_the_actual_legacy_request_policy_only_for_eval(monkeypatch, tmp_path, provided):
    publication = load('hindsight_incremental_publication', monkeypatch)
    rebuild = sys.modules['hindsight_offline_v2_rebuild']
    legacy = []
    monkeypatch.setenv('HINDSIGHT_EVAL_REQUEST_TIMEOUT', '120')
    monkeypatch.setattr(rebuild, 'run_json', lambda cmd, **kw: legacy.append((cmd, kw)) or {})
    rebuild.eval_pair('generic', tmp_path/'bench.jsonl', tmp_path/'cards', tmp_path/'out',
        api='http://unused.invalid', bank='fixture', raw_limit=40, limit=10,
        top_k=5, include_hindsight_observations=True)
    actual_legacy_timeout = legacy[0][1]['env']['HINDSIGHT_EVAL_REQUEST_TIMEOUT']
    captured = []
    monkeypatch.setattr(publication, 'default_run_json', lambda cmd, **kw: captured.append((cmd, kw)) or {})
    adapter = publication.PublicationAdapter({'script_dir': str(ROOT)}, 'fixture')
    command = [sys.executable, str(ROOT/'hindsight_eval.py'), '--benchmark', str(tmp_path/'bench.jsonl')]
    extra = {'env': provided} if provided is not None else {}
    original = dict(provided) if provided is not None else None
    adapter._run_json(command, timeout=legacy[0][1]['timeout'], **extra)
    assert captured[0][1].get('env', {}).get('HINDSIGHT_EVAL_REQUEST_TIMEOUT') == actual_legacy_timeout
    assert captured[0][1]['timeout'] == legacy[0][1]['timeout']
    assert captured[0][0][0:2] == [sys.executable, '-c']
    assert captured[0][0][3:] == command[1:]
    assert provided == original
    assert __import__('os').environ['HINDSIGHT_EVAL_REQUEST_TIMEOUT'] == '120'
    if provided is not None:
        assert captured[0][1]['env'] == {**provided, 'HINDSIGHT_EVAL_REQUEST_TIMEOUT': actual_legacy_timeout}
    audit = [sys.executable, str(ROOT/'hindsight_conflict_audit.py'), '--json']
    adapter._run_json(audit, timeout=91, **extra)
    assert captured[1] == (audit, {'timeout': 91, **extra})


def test_actual_incremental_bootstrap_receives_legacy_timeout_in_isolated_child(monkeypatch, tmp_path):
    publication = load('hindsight_incremental_publication', monkeypatch)
    rebuild = sys.modules['hindsight_offline_v2_rebuild']
    captured = []
    with monkeypatch.context() as scoped:
        scoped.setattr(rebuild, 'run_json', lambda cmd, **kw: captured.append(kw) or {})
        rebuild.eval_pair('generic', tmp_path/'bench.jsonl', tmp_path/'cards', tmp_path/'out',
            api='http://unused.invalid', bank='fixture', raw_limit=40, limit=10,
            top_k=5, include_hindsight_observations=True)
    expected = captured[0]['env']['HINDSIGHT_EVAL_REQUEST_TIMEOUT']
    script = tmp_path/'hindsight_eval.py'
    script.write_text('''import json,os

def run_eval(*,bank,api,cards_root):
    return {"observed_request_timeout":os.environ.get("HINDSIGHT_EVAL_REQUEST_TIMEOUT","120"),
            "child_scope":os.environ.get("ONLY_THIS_SCOPE")}

def main():
    print(json.dumps(run_eval(bank="fixture",api="http://unused.invalid",cards_root=None)))
''')
    # Synthetic child validates real subprocess/env/bootstrap I/O only; it is
    # not an eval report accepted by a quality gate or a production API response.
    adapter = publication.PublicationAdapter({'script_dir': str(tmp_path), 'publisher_python': sys.executable}, 'fixture')
    environment = {'ONLY_THIS_SCOPE': 'isolated-child', 'HINDSIGHT_EVAL_REQUEST_TIMEOUT': '120', 'PYTHONPATH': str(ROOT)}
    report = adapter._run_json([sys.executable, str(script), '--benchmark', str(tmp_path/'bench.jsonl')],
                              timeout=19, env=environment)
    assert report['observed_request_timeout'] == expected
    assert report['child_scope'] == 'isolated-child'
    assert report['candidate_parse_proof'] is None
    assert report['invocation'] == {'bank': 'fixture', 'api': 'http://unused.invalid', 'benchmark': str(tmp_path/'bench.jsonl')}
    assert environment['HINDSIGHT_EVAL_REQUEST_TIMEOUT'] == '120'
