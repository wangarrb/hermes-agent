import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name='hindsight_proposal_review'):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def proposal(**kw):
    base = {
        'proposal_id': 'proposal::abc',
        'canonical_text': 'Use proposal-only review packets before any Hindsight production retain merge.',
        'topic': 'hindsight',
        'type': 'technical_lesson',
        'tags': ['domain:hindsight'],
        'evidence_count': 2,
        'source_document_count': 1,
        'source_documents': ['doc-1'],
        'source_fact_ids': ['fact-1', 'fact-2'],
        'quality_flags': [],
        'priority_score': 42,
        'merge_gate': 'user_approval_required',
        'production_action': 'proposal_only_no_write',
    }
    base.update(kw)
    return base


def write_bundle(path, proposals):
    path.write_text(json.dumps({
        'schema_version': 'hindsight-repair-canonical-proposals-v1',
        'generated_at': '2026-05-13T00:00:00+00:00',
        'quality': {'accepted_unique': len(proposals)},
        'proposals': proposals,
    }, ensure_ascii=False), encoding='utf-8')


def test_review_packet_written_without_llm_blocks_on_pending_llm(tmp_path):
    mod = load_module()
    src = tmp_path / 'demo-canonical-proposals.json'
    review_root = tmp_path / 'reviews'
    write_bundle(src, [proposal()])

    rc = mod.main(['--proposal-json', str(src), '--review-root', str(review_root), '--stem', 'demo', '--json'])
    assert rc == 0
    packet_path = review_root / 'demo-review-packet.json'
    assert packet_path.exists()
    packet = json.loads(packet_path.read_text(encoding='utf-8'))
    assert packet['safety']['production_mutation_allowed'] is False
    assert packet['summary']['llm_required'] is True
    review = packet['reviews'][0]
    assert review['llm_judgement']['status'] == 'pending_not_executed'
    assert review['go_no_go']['go_no_go'] == 'no_go'
    assert 'LLM review is required' in review['go_no_go']['reason']


def test_go_no_go_blocks_proposals_with_quality_flags(tmp_path):
    mod = load_module()
    reviews, summary = mod.review_proposals(
        [proposal(evidence_count=0, source_fact_ids=[], quality_flags=['no_evidence_ids'])],
        review_root=tmp_path,
        require_llm_review=False,
        require_human_approval=True,
    )
    assert summary['production_mutation_allowed'] is False
    assert reviews[0]['go_no_go']['go_no_go'] == 'no_go'
    assert 'missing_evidence' in reviews[0]['go_no_go']['deterministic_block_codes']
    assert 'quality_flag:no_evidence_ids' in reviews[0]['go_no_go']['deterministic_block_codes']


def test_llm_merge_ready_still_requires_human_conditional_go(tmp_path):
    mod = load_module()

    def fake_llm(_messages):
        return {
            'proposal_id': 'proposal::abc',
            'decision': 'merge_ready',
            'risk': 'low',
            'reason_brief': 'Evidence is specific and durable.',
            'required_human_checks': ['verify source doc ids'],
            'rollback_or_quarantine_notes': ['retain to temp bank first'],
        }

    reviews, summary = mod.review_proposals(
        [proposal()],
        review_root=tmp_path,
        require_llm_review=True,
        require_human_approval=True,
        execute_llm=True,
        confirm_review='review-hindsight-proposals',
        llm_fn=fake_llm,
    )
    assert summary['llm_calls_made'] == 1
    assert reviews[0]['llm_judgement']['decision'] == 'merge_ready'
    assert reviews[0]['go_no_go']['go_no_go'] == 'conditional_go'
    assert reviews[0]['go_no_go']['human_final_decision'] == 'pending'
    assert reviews[0]['go_no_go']['production_merge_allowed_by_this_packet'] is False


def test_deterministic_blocked_secret_proposal_is_not_sent_to_llm(tmp_path):
    mod = load_module()
    calls = []

    def fake_llm(messages):
        calls.append(messages)
        return {'decision': 'merge_ready', 'risk': 'low'}

    reviews, summary = mod.review_proposals(
        [proposal(canonical_text='leaked token ' + 'sk-' + 'a' * 32)],
        review_root=tmp_path,
        require_llm_review=True,
        require_human_approval=True,
        execute_llm=True,
        confirm_review='review-hindsight-proposals',
        llm_fn=fake_llm,
    )
    assert calls == []
    assert summary['llm_calls_made'] == 0
    assert reviews[0]['llm_judgement']['status'] == 'skipped_deterministic_block'
    assert reviews[0]['llm_judgement']['decision'] == 'quarantine'
    assert reviews[0]['go_no_go']['go_no_go'] == 'no_go'


def test_notification_block_is_optional_and_manual_path_always_exists(tmp_path):
    mod = load_module()
    src = tmp_path / 'demo-canonical-proposals.json'
    review_root = tmp_path / 'reviews'
    write_bundle(src, [proposal()])
    rc = mod.main(['--proposal-json', str(src), '--review-root', str(review_root), '--stem', 'demo', '--notify', '--json'])
    assert rc == 0
    packet = json.loads((review_root / 'demo-review-packet.json').read_text(encoding='utf-8'))
    assert packet['notification']['human_action_required'] is True
    assert packet['manual_review']['status'] == 'pending'
    assert (review_root / 'demo-review-packet.md').exists()


def test_review_script_source_has_no_production_hindsight_mutation_calls():
    text = (ROOT / 'hindsight_proposal_review.py').read_text(encoding='utf-8')
    forbidden = ['retain_items(', 'retain_batch(', 'patch_document_tags(', 'delete_document(', 'delete_operation(', 'trigger_consolidation(']
    for token in forbidden:
        assert token not in text
