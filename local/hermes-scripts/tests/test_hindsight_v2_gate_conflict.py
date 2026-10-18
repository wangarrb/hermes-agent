import importlib.util
import json
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')
SPEC = importlib.util.spec_from_file_location('hindsight_offline_v2_gate', ROOT / 'hindsight_offline_v2_gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_evaluate_conflict_audit_passes_clean_report(tmp_path):
    p = tmp_path / 'audit.json'
    p.write_text(json.dumps({'decision': 'pass', 'summary': {'blocking_cases': 0}, 'cases': []}), encoding='utf-8')
    check = gate.evaluate_conflict_audit(p, block_severity='P1')
    assert check['passed'] is True
    assert check['name'] == 'conflict_audit_passed'


def test_evaluate_conflict_audit_blocks_p1_case(tmp_path):
    p = tmp_path / 'audit.json'
    p.write_text(json.dumps({
        'decision': 'blocked_conflict_review_required',
        'cases': [{'case_id': 'c1', 'severity': 'P1', 'type': 'missing_lineage'}],
    }), encoding='utf-8')
    check = gate.evaluate_conflict_audit(p, block_severity='P1')
    assert check['passed'] is False
    assert check['blocking_case_count'] == 1
    assert check['blocking_examples'][0]['case_id'] == 'c1'
