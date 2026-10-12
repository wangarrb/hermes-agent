"""Aggregation fidelity only: actual evaluator, synthetic per-case metrics, no network."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def evaluator_report(monkeypatch):
    evaluator = load('hindsight_eval', monkeypatch)
    # These numeric fixtures expose the rounding boundary observed in a real
    # report. Per-case scoring and RPC are outside this aggregation-only test.
    scores = [120.0, 120.0, 120.0, 113.89, 100.0, 82.5, 103.33, 20.0, 48.33, 120.0]
    cases = [{'id': f'case-{i}', 'query': f'fixture-{i}', 'metric': {
        'score': value, 'term_recall': 0.6, 'expected_layer_hits_top_k': 5}}
        for i, value in enumerate(scores)]
    transport = SimpleNamespace(recall=lambda *a, **kw: [], layered_recall=lambda *a, **kw: [])
    monkeypatch.setattr(evaluator, 'load_layered_module', lambda: transport)
    monkeypatch.setattr(evaluator, 'evaluate_results', lambda case, rows, top_k: dict(case['metric']))
    report = evaluator.run_eval(cases, api='http://unused.invalid', bank='fixture',
                                top_k=5, raw_limit=40, limit=10, cards_root=None)
    return json.loads(json.dumps(report)), sorted(case['id'] for case in cases)


def test_validator_accepts_original_evaluator_ordered_float_accumulation(monkeypatch):
    report, ids = evaluator_report(monkeypatch)
    publication = load('hindsight_incremental_publication', monkeypatch)
    frozen = copy.deepcopy(report)
    publication._validate_eval(report, ids)
    assert report == frozen


@pytest.mark.parametrize('metric,delta', [('layered_avg_score', 0.01), ('layered_avg_term_recall', 0.001), ('layered_expected_layer_hits', 1)])
def test_validator_still_rejects_modified_summary_without_tolerance(monkeypatch, metric, delta):
    report, ids = evaluator_report(monkeypatch)
    publication = load('hindsight_incremental_publication', monkeypatch)
    report['summary'][metric] += delta
    with pytest.raises(ValueError, match='aggregate differs'):
        publication._validate_eval(report, ids)
