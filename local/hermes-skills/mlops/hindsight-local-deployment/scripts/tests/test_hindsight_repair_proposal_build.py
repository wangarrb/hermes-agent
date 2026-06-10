import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_jsonl(path, rows):
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')


def base_row(text, *, rid='repair-sidecar::a', topic='hindsight', docs=None, facts=None, tags=None):
    return {
        'id': rid,
        'status': 'approved',
        'layer': 'approved_repair_sidecar',
        'topic': topic,
        'type': 'technical_lesson',
        'text': text,
        'insight': text,
        'source_documents': docs or ['hermes-session::1'],
        'source_fact_ids': facts or ['fact-1'],
        'evidence_ids': facts or ['fact-1'],
        'tags': tags or ['domain:hindsight'],
        'provenance': {'candidate_id': 'a-candidate-001'},
    }


def test_build_proposals_dedupes_and_rejects_artifacts(tmp_path):
    mod = load_module('hindsight_repair_proposal_build')
    src = tmp_path / 'approved.jsonl'
    rows = [
        base_row('Hindsight offline repair proposals must stay local until user approval.', rid='repair-sidecar::1', facts=['f1', 'f2']),
        base_row('Hindsight offline repair proposals must stay local until user approval.', rid='repair-sidecar::2', facts=['f3']),
        base_row('CONTEXT COMPACTION reference-only block should not be retained.', rid='repair-sidecar::3'),
        base_row('Contains API key ' + 'sk-' + 'a' * 32 + ' and must be manual review.', rid='repair-sidecar::4'),
    ]
    write_jsonl(src, rows)
    result = mod.build_proposals(src)
    assert result['quality']['rows_seen'] == 4
    assert result['quality']['accepted_unique'] == 1
    assert result['quality']['rejected'] == 2
    assert result['quality']['deduped_rows'] == 1
    prop = result['proposals'][0]
    assert prop['canonical_text'] == 'Hindsight offline repair proposals must stay local until user approval.'
    assert prop['evidence_count'] == 3
    assert sorted(prop['sidecar_ids']) == ['repair-sidecar::1', 'repair-sidecar::2']
    assert prop['merge_gate'] == 'user_approval_required'


def test_cli_writes_json_markdown_and_report(tmp_path):
    mod = load_module('hindsight_repair_proposal_build')
    src = tmp_path / 'approved.jsonl'
    out = tmp_path / 'out'
    write_jsonl(src, [base_row('Use 20x3 consolidation tuning for balanced fanout.', topic='hindsight')])
    rc = mod.main(['--approved-index', str(src), '--output-root', str(out), '--stem', 'demo', '--top', '10'])
    assert rc == 0
    proposal_json = out / 'demo-canonical-proposals.json'
    proposal_md = out / 'demo-canonical-proposals.md'
    quality_json = out / 'demo-quality-report.json'
    assert proposal_json.exists()
    assert proposal_md.exists()
    assert quality_json.exists()
    data = json.loads(proposal_json.read_text(encoding='utf-8'))
    assert data['proposals'][0]['topic'] == 'hindsight'
    assert 'Use 20x3 consolidation tuning' in proposal_md.read_text(encoding='utf-8')
