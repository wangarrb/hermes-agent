import importlib.util
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')
SPEC = importlib.util.spec_from_file_location('hindsight_conflict_core', ROOT / 'hindsight_conflict_core.py')
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def test_detect_contamination_flags_heartbeat_and_handoff():
    text = 'CONTEXT COMPACTION — REFERENCE ONLY\nHEARTBEAT: still running\nnormal fact'
    hits = core.detect_contamination(text)
    names = {h['name'] for h in hits}
    assert 'context_compaction' in names
    assert 'heartbeat' in names
    assert max(h['severity'] for h in hits) in {'P0', 'P1', 'P2'}


def test_extract_source_sessions_from_import_bundle_header():
    raw = '''# bundle\nsource_sessions:\n  - id=abc session_start=2026-05-01T01:00:00 first_msg=2026-05-01T01:01:00 last_msg=2026-05-01T01:02:00 topic=hermes model=unknown chars=123\n  - id=def first_msg=2026-05-02T02:01:00 last_msg=2026-05-02T02:02:00 topic=egomotion4d chars=456\n\n--- conversations ---\n'''
    sessions = core.extract_source_sessions(raw)
    assert sessions[0]['id'] == 'abc'
    assert sessions[0]['first_msg'] == '2026-05-01T01:01:00'
    assert sessions[0]['chars'] == 123
    assert sessions[1]['topic'] == 'egomotion4d'


def test_build_conflict_cases_detects_missing_lineage_and_contamination():
    observations = [
        {'id': 'obs:bad', 'topic': 'Hindsight', 'type': 'risk', 'insight': 'HEARTBEAT: loop entered high-level memory', 'evidence_ids': [], 'source_documents': []},
        {'id': 'obs:orphan', 'topic': 'Hindsight', 'type': 'project_decision', 'insight': '结论存在但没有来源', 'evidence_ids': [], 'source_documents': []},
    ]
    cases = core.build_conflict_cases(observations, known_document_ids=set(), known_memory_ids=set())
    by_type = {c['type']: c for c in cases}
    assert 'contamination' in by_type
    assert 'missing_lineage' in by_type
    assert by_type['missing_lineage']['repair_class'] == 'lineage_required'


def test_manual_case_enters_same_case_schema():
    case = core.manual_conflict_case(claim='这条结论和之前冲突', target_id='obs:123', severity='P1')
    assert case['source'] == 'manual'
    assert case['type'] == 'manual_conflict'
    assert case['target']['id'] == 'obs:123'
    assert 'provenance_trace' in case['required_flow']


def test_repair_proposal_maps_case_to_safe_actions():
    case = core.manual_conflict_case(claim='这条结论不对', target_id='obs:123', severity='P1')
    proposal = core.repair_proposal_for_case(case)
    actions = [a['action'] for a in proposal['recommended_actions']]
    assert actions[0] == 'trace_lineage'
    assert 'human_confirmation_required' in proposal
    assert proposal['human_confirmation_required'] is True
