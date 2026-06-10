import importlib.util
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_blocks_when_summary_disagrees_with_cases(tmp_path):
    gate = load_module('hindsight_offline_v2_gate')
    p = tmp_path / 'audit.json'
    p.write_text(json.dumps({
        'decision': 'pass',
        'summary': {'blocking_cases': 0},
        'cases': [{'case_id': 'c1', 'severity': 'P1', 'type': 'missing_lineage'}],
    }), encoding='utf-8')
    check = gate.evaluate_conflict_audit(p, block_severity='P1')
    assert check['passed'] is False
    assert check['blocking_case_count'] == 1
    assert check.get('summary_mismatch') is True


def test_core_uuid_matching_is_case_insensitive():
    core = load_module('hindsight_conflict_core')
    uid_lower = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    uid_upper = uid_lower.upper()
    cases = core.build_conflict_cases([
        {'id': 'obs:uuid', 'topic': 't', 'type': 'risk', 'insight': 'traceable', 'evidence_ids': [uid_upper]},
    ], known_memory_ids={uid_lower})
    assert not [c for c in cases if c['type'] == 'dangling_evidence_id']


def test_core_resolves_relative_source_paths_against_known_files():
    core = load_module('hindsight_conflict_core')
    cases = core.build_conflict_cases([
        {'id': 'obs:path', 'topic': 't', 'type': 'risk', 'insight': 'traceable', 'source_documents': ['daily/foo.json']},
    ], known_file_paths={'/tmp/offline/daily/foo.json', 'daily/foo.json'})
    assert not [c for c in cases if c['type'] == 'dangling_source_document']


def test_core_does_not_treat_negated_default_as_positive_and_negative():
    core = load_module('hindsight_conflict_core')
    p = core._polarity('not default / opt-in only')
    assert 'default' not in p['positive']
    assert 'default' in p['negative']


def test_numeric_divergence_ignores_mixed_units_and_unlabeled_context():
    core = load_module('hindsight_conflict_core')
    obs = [
        {'id': 'o1', 'topic': 't', 'type': 'technical_lesson', 'tags': ['alpha'], 'insight': 'scene0/4/15 has 100 frames and 20251021 source date', 'evidence_ids': ['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee']},
        {'id': 'o2', 'topic': 't', 'type': 'technical_lesson', 'tags': ['alpha'], 'insight': 'scene0/4/15 has 300 samples and 68m trajectory', 'evidence_ids': ['bbbbbbbb-cccc-dddd-eeee-ffffffffffff']},
    ]
    cases = core.build_conflict_cases(obs, known_memory_ids={'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee','bbbbbbbb-cccc-dddd-eeee-ffffffffffff'})
    assert not [c for c in cases if c['type'] == 'numeric_divergence_candidate']


def test_numeric_divergence_flags_same_metric_different_values():
    core = load_module('hindsight_conflict_core')
    obs = [
        {'id': 'o1', 'topic': 't', 'type': 'technical_lesson', 'tags': ['alpha'], 'insight': 'method alpha ATE=0.5m on same benchmark', 'evidence_ids': ['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee']},
        {'id': 'o2', 'topic': 't', 'type': 'technical_lesson', 'tags': ['alpha'], 'insight': 'method alpha ATE=2.0m on same benchmark', 'evidence_ids': ['bbbbbbbb-cccc-dddd-eeee-ffffffffffff']},
    ]
    cases = core.build_conflict_cases(obs, known_memory_ids={'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee','bbbbbbbb-cccc-dddd-eeee-ffffffffffff'})
    numeric = [c for c in cases if c['type'] == 'numeric_divergence_candidate']
    assert numeric
    assert numeric[0]['evidence']['numeric_key'] == 'ate|m'


def test_lineage_scans_all_raw_messages_for_contamination(tmp_path):
    lineage = load_module('hindsight_lineage_trace')
    db = tmp_path / 'state.db'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, started_at TEXT, ended_at TEXT, title TEXT, message_count INTEGER, tool_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER)')
    con.execute('CREATE TABLE messages (session_id TEXT, role TEXT, timestamp TEXT, content TEXT)')
    con.execute("INSERT INTO sessions VALUES ('s1','cli','model','2026-01-01','2026-01-01','title',25,0,0,0)")
    for i in range(25):
        content = 'normal message'
        if i == 24:
            content = 'HEARTBEAT: leaked after first twenty messages'
        con.execute('INSERT INTO messages VALUES (?,?,?,?)', ('s1', 'assistant', f'2026-01-01T00:{i:02d}:00', content))
    con.commit(); con.close()
    raw = lineage.raw_session_summary('s1', db)
    assert raw['message_count'] == 25
    assert raw['contamination_summary']['scanned_messages'] == 25
    assert raw['contamination_summary']['hit_messages'] == 1
    assert raw['contamination']


def test_lineage_document_and_memory_rows_use_official_api_client():
    lineage = load_module('hindsight_lineage_trace')

    class FakeClient:
        def get_document(self, document_id):
            return {
                'id': document_id,
                'original_text': 'source_sessions:\n  - id=s1 model=m chars=10\nATE=0.5 normal',
                'document_metadata': {'topic': 't'},
                'created_at': '2026-01-01',
                'updated_at': '2026-01-02',
            }

        def get_memory(self, memory_id):
            return {
                'id': memory_id,
                'document_id': 'doc1',
                'type': 'world',
                'text': 'Traceback (most recent call last)',
                'metadata': {'k': 'v'},
                'created_at': '2026-01-03',
                'source_memory_ids': ['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'],
            }

    doc = lineage.document_row('doc1', 'hermes', client=FakeClient())
    mem = lineage.memory_row('mem1', 'hermes', client=FakeClient())
    assert doc['id'] == 'doc1'
    assert doc['layer'] == 'other'
    assert doc['source_sessions'][0]['id'] == 's1'
    assert mem['id'] == 'mem1'
    assert mem['document_id'] == 'doc1'
    assert mem['fact_type'] == 'world'
    assert mem['contamination'][0]['name'] == 'raw_stack_trace'


def test_repair_proposal_missing_case_id_fails(tmp_path):
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'cases': [{'case_id': 'exists', 'type': 'manual_conflict', 'severity': 'P1', 'target': {'id': 'obs:1'}, 'evidence': {'claim': 'x'}}]}), encoding='utf-8')
    out = tmp_path / 'out'
    proc = subprocess.run([
        sys.executable,
        str(ROOT / 'hindsight_repair_proposal.py'),
        '--audit-json', str(audit),
        '--case-id', 'missing',
        '--output-dir', str(out),
        '--json',
    ], text=True, capture_output=True)
    assert proc.returncode != 0
    assert 'case-id not found' in (proc.stderr + proc.stdout)


def test_repair_proposal_preserves_case_evidence():
    core = load_module('hindsight_conflict_core')
    case = {
        'case_id': 'c1',
        'type': 'numeric_divergence_candidate',
        'severity': 'P2',
        'title': 'numbers disagree',
        'target': {'id': 'obs:g'},
        'evidence': {'range': [1, 9]},
        'repair_class': 'scope_trace_required',
        'required_flow': ['conflict_intake', 'provenance_trace'],
    }
    proposal = core.repair_proposal_for_case(case)
    assert proposal['case_type'] == 'numeric_divergence_candidate'
    assert proposal['severity'] == 'P2'
    assert proposal['evidence'] == {'range': [1, 9]}
    assert proposal['required_flow'] == ['conflict_intake', 'provenance_trace']


def test_conflict_audit_source_scan_unavailable_creates_p1_case(monkeypatch):
    audit = load_module('hindsight_conflict_audit')
    audit.API_SCAN_ERRORS.clear()
    audit.API_SCAN_ERRORS.append('official API unavailable')
    cases = audit.db_unavailable_cases()
    assert cases
    assert cases[0]['type'] == 'source_scan_unavailable'
    assert cases[0]['severity'] == 'P1'


def test_api_contamination_case_not_dropped_when_memory_text_has_hit():
    audit = load_module('hindsight_conflict_audit')

    class FakeClient:
        def iter_memories(self, types=None, max_items=None):
            yield {'id': 'mem1', 'document_id': 'doc1', 'type': 'world', 'text': 'Traceback (most recent call last) leaked'}

    cases = audit.db_contamination_cases('hermes', client=FakeClient())
    assert cases
    assert cases[0]['type'] == 'source_fact_contamination'
    assert cases[0]['evidence']['contamination_hits'][0]['name'] == 'raw_stack_trace'


def test_gate_proposal_uses_detailed_observations_index(tmp_path):
    gate = load_module('hindsight_offline_v2_gate')
    cards_root = tmp_path / 'cards'
    (cards_root / 'topics').mkdir(parents=True)
    (cards_root / 'global').mkdir()
    (cards_root / 'manifest.json').write_text('{}', encoding='utf-8')
    card = {
        'scope': 'topic',
        'topic': 'topic-a',
        'card_id': 'card-abcdef123456',
        'schema_version': '2',
        'generated_at': '2026-01-01',
        'executive_summary': ['summary'],
        'canonical_observations': [{'id': 'obs:compact', 'type': 'risk', 'confidence': 0.8, 'insight': 'compact only'}],
        'evidence_index': [],
    }
    (cards_root / 'topics' / 'topic-a.json').write_text(json.dumps(card), encoding='utf-8')
    detailed = [
        {'id': 'obs:1', 'topic': 'topic-a', 'type': 'risk', 'confidence': 0.9, 'insight': 'detail one', 'evidence_ids': ['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee']},
        {'id': 'obs:2', 'topic': 'topic-a', 'type': 'risk', 'confidence': 0.7, 'insight': 'detail two', 'evidence_ids': ['bbbbbbbb-cccc-dddd-eeee-ffffffffffff']},
    ]
    (cards_root / 'observations_index.jsonl').write_text('\n'.join(json.dumps(x) for x in detailed) + '\n', encoding='utf-8')
    out = tmp_path / 'proposal'
    out.mkdir()
    proposal = gate.write_proposal(cards_root, out)
    rows = [json.loads(line) for line in Path(proposal['jsonl_path']).read_text(encoding='utf-8').splitlines() if line.strip()]
    assert rows[0]['metadata']['observation_count'] == 2
    assert 'id: obs:2' in rows[0]['content']
    assert 'evidence_ids:' in rows[0]['content']


def test_publish_filters_semantic_links_to_existing_memory_units(monkeypatch):
    class FakeExtras(types.SimpleNamespace):
        RealDictCursor = object

        def execute_values(self, *args, **kwargs):
            raise AssertionError('not used by this unit test')

    fake_psycopg2 = types.SimpleNamespace(connect=lambda *args, **kwargs: None, extras=FakeExtras())
    monkeypatch.setitem(sys.modules, 'psycopg2', fake_psycopg2)
    monkeypatch.setitem(sys.modules, 'psycopg2.extras', fake_psycopg2.extras)
    publish = load_module('hindsight_offline_v2_publish')

    rows = publish.filter_semantic_rows_to_existing_targets([
        ('new-unit', 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'semantic', 0.85, 'hermes'),
        ('new-unit', 'bbbbbbbb-cccc-dddd-eeee-ffffffffffff', 'semantic', 0.85, 'hermes'),
    ], existing_target_ids={'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'})

    assert rows['kept'] == [('new-unit', 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'semantic', 0.85, 'hermes')]
    assert rows['skipped_missing_targets'] == 1


def test_hindsight_container_patch_adds_json_fence_and_rate_limit_guards():
    patcher = load_module('patch_hindsight_minimax_json_parser')
    source = '''def _strip_code_fences(content: str) -> str:
    if "```" not in content:
        return content
    return content


class OpenAICompatibleLLM:
    async def call(self):
        try:
            pass
        except APIStatusError as e:
            if e.status_code in (401, 403):
                raise
            last_exception = e
            if attempt < max_retries:
                backoff = min(initial_backoff * (2**attempt), max_backoff)
                jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                sleep_time = backoff + jitter
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"API error after {max_retries + 1} attempts: {str(e)}")
                raise
'''

    patched, changed = patcher.patch_file_text(source)
    assert changed is True
    assert 'HERMES_MINIMAX_JSON_FENCE_FIX_V4' in patched
    assert 'HERMES_RATE_LIMIT_BACKOFF_FIX_V1' in patched
    assert 'HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS' in patched

    patched_again, changed_again = patcher.patch_file_text(patched)
    assert changed_again is False
    assert patched_again == patched


def test_paid_llm_env_caps_retries_and_uses_long_rate_limit_backoff(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    monkeypatch.setenv('HINDSIGHT_OFFLINE_LLM_CONCURRENCY', '4')
    profile = {
        'api_key': 'test-key',
        'hindsight_provider': 'minimax',
        'model': 'MiniMax-M2.7',
        'base_url': 'https://api.minimaxi.com/v1',
    }

    env = mod.paid_llm_env(profile, enable_observations=True)

    assert env['HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS'] == '300'
    assert env['HINDSIGHT_API_LLM_MAX_RETRIES'] == '2'
    assert env['HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES'] == '2'
    assert env['HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES'] == '2'
    assert env['HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES'] == '1'
    assert env['HINDSIGHT_API_CONSOLIDATION_MAX_ATTEMPTS'] == '1'
    assert env['HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB'] == '60'
    assert env['HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND'] == '60'
    assert env['HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET'] == 'low'
    assert env['HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE'] == '20'
    assert env['HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE'] == '20'
    assert env['HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS'] == '4096'
    assert env['HINDSIGHT_API_WORKER_MAX_SLOTS'] == '5'
    assert env['HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS'] == '1'


def test_minimax_import_defaults_to_pinned_hindsight_v061_image(monkeypatch):
    monkeypatch.delenv('HINDSIGHT_IMAGE', raising=False)
    mod = load_module('hindsight_minimax_import')
    assert mod.IMAGE == 'ghcr.io/vectorize-io/hindsight:0.6.1'


def test_patch_bank_config_does_not_send_unsupported_recall_budget(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    patch_payloads = []

    class Resp:
        status_code = 200
        text = ''
        def __init__(self, payload):
            self.payload = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self.payload

    supported = {
        'config': {
            'retain_chunk_size': 8000,
            'retain_extraction_mode': 'custom',
            'retain_custom_instructions': '',
            'enable_observations': False,
            'recall_max_tokens': 4096,
            'recall_chunks_max_tokens': 4096,
            'consolidation_llm_batch_size': 20,
            'consolidation_max_memories_per_round': 60,
            'consolidation_source_facts_max_tokens': 4096,
            'consolidation_source_facts_max_tokens_per_observation': 256,
        }
    }

    monkeypatch.setattr(mod.requests, 'get', lambda *a, **k: Resp(supported))
    def fake_patch(url, json, **kwargs):
        patch_payloads.append(json)
        return Resp(supported)
    monkeypatch.setattr(mod.requests, 'patch', fake_patch)

    mod.patch_bank_config(enable_observations=True, bank='hermes')

    updates = patch_payloads[0]['updates']
    assert 'consolidation_recall_budget' not in updates
    assert updates['consolidation_max_memories_per_round'] == 60


def test_queue_counts_prefers_operations_api_excluding_parent_rows(monkeypatch):
    mod = load_module('hindsight_minimax_import')
    calls = []

    class Resp:
        def __init__(self, payload):
            self.payload = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith('/stats'):
            # Stats can include pending parent rows; queue_counts must not trust
            # this when the v0.6.1 operations API is available.
            return Resp({'pending_operations': 99, 'operations_by_status': {'processing': 88}})
        status = kwargs['params']['status']
        return Resp({'total': 2 if status == 'pending' else 3})

    monkeypatch.setattr(mod.requests, 'get', fake_get)

    pending, processing, data = mod.queue_counts('hermes_v3')

    assert (pending, processing) == (2, 3)
    assert data['queue_counts_source'] == 'operations_api_exclude_parents'
    assert all(c[1].get('params', {}).get('exclude_parents') == 'true' for c in calls if c[0].endswith('/operations'))


def test_ollama_env_can_enable_worker_consolidation_slots_without_auto_observations():
    mod = load_module('hindsight_minimax_import')

    env = mod.ollama_env(model='qwen2:7b-instruct', consolidation_slots=1)

    assert env['HINDSIGHT_API_LLM_PROVIDER'] == 'ollama'
    assert env['HINDSIGHT_API_LLM_MODEL'] == 'qwen2:7b-instruct'
    assert env['HINDSIGHT_API_RETAIN_LLM_MODEL'] == 'qwen2:7b-instruct'
    assert env['HINDSIGHT_API_CONSOLIDATION_LLM_MODEL'] == 'qwen2:7b-instruct'
    assert env['HINDSIGHT_API_ENABLE_OBSERVATIONS'] == 'false'
    assert env['HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS'] == '1'
    assert env['HINDSIGHT_API_WORKER_MAX_SLOTS'] == '2'


def test_consolidation_budget_patch_caps_native_job_loop():
    patcher = load_module('patch_hindsight_native_consolidation_budget')
    source = '''"""doc"""
import json
import logging

async def run_consolidation_job(memory_engine, bank_id, request_context, operation_id=None):
    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)
    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    llm_batch_size = max(1, config.consolidation_llm_batch_size)
    stats = {"memories_processed": 0}
    while True:
        # Fetch next batch of unconsolidated memories
        async with pool.acquire() as conn:
            t0 = time.time()
            memories = await conn.fetch(
                f"""
                SELECT id, text, fact_type, occurred_start, occurred_end, event_date, tags, mentioned_at,
                       observation_scopes
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND consolidated_at IS NULL
                  AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience', 'world')
                ORDER BY created_at ASC
                LIMIT $2
                """,
                bank_id,
                max_memories_per_batch,
            )
        for llm_batch in llm_batches:
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} llm_batch #{llm_batch_num}"
                f" ({len(llm_batch)} memories, {llm_calls_made} llm calls)"
                f" | {stats['memories_processed']}/{total_count} processed"
                f" | {', '.join(timing_parts)}"
                f" | created={batch_created} updated={batch_updated} skipped={batch_skipped}"
                + (f" failed={batch_failed}" if batch_failed else "")
                + f" | input_tokens=~{input_tokens}"
                f" | avg={llm_batch_time / len(llm_batch):.3f}s/memory"
            )

    # Build summary
'''

    patched, changed = patcher.patch_file_text(source)

    assert changed is True
    assert 'import os' in patched
    assert 'HERMES_CONSOLIDATION_MAX_MEMORIES_PER_JOB_V1' in patched
    assert 'HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB' in patched
    assert 'fetch_limit' in patched
    assert 'max_memories_per_job - stats["memories_processed"]' in patched

    patched_again, changed_again = patcher.patch_file_text(patched)
    assert changed_again is False
    assert patched_again == patched


def test_native_consolidation_estimate_sql_uses_sorted_tag_groups_and_limit():
    guard = load_module('hindsight_native_workflow_guard')

    sql = guard.build_native_consolidation_estimate_sql("hermes'bank", fetch_size=50, llm_batch_size=50, max_facts=50)

    assert "bank_id = 'hermes''bank'" in sql
    assert "row_number() OVER (ORDER BY created_at ASC)" in sql
    assert "ARRAY(SELECT unnest" in sql
    assert "ORDER BY 1" in sql
    assert "WHERE rn <= 50" in sql
    assert "fact_type IN ('experience', 'world')" in sql
    assert "ceil(n::numeric / 50)" in sql


def test_native_paid_runner_dry_run_does_not_switch_provider(monkeypatch, capsys):
    guard = load_module('hindsight_native_workflow_guard')
    report = {
        'health': {'status': 'healthy', 'database': 'connected'},
        'stats': {'pending_operations': 0, 'failed_operations': 0},
        'config_focus': {'enable_observations': False},
        'provider_env': {'HINDSIGHT_API_LLM_PROVIDER': 'ollama', 'HINDSIGHT_API_ENABLE_OBSERVATIONS': 'false'},
        'active_payload_null': 0,
        'native_consolidation_unconsolidated_candidates': 5944,
    }
    monkeypatch.setattr(guard, 'collect_status', lambda bank: report)
    monkeypatch.setattr(guard, 'estimate_native_consolidation_calls', lambda *a, **k: {'facts': 50, 'fetch_rounds': 1, 'tag_groups': 4, 'llm_calls': 4})

    args = types.SimpleNamespace(
        bank='hermes', llm_profile='minimax', jobs=1, facts_per_job=50,
        fetch_size=None, llm_batch_size=None, source_facts_max_tokens=4096,
        max_unconsolidated=10000, allow_existing_queue=False, allow_failed=False,
        expect_local_provider=True, execute=False, confirm=None,
        api_timeout=30, operation_timeout=3600, poll=10,
    )

    assert guard.cmd_run_native_consolidation_paid(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['execute'] is False
    assert out['preflight_ok'] is True
    assert out['estimate_window']['llm_calls'] == 4
    assert 'Add --execute --confirm' in out['next_step']


def test_native_paid_runner_execute_requires_confirm(monkeypatch, capsys):
    guard = load_module('hindsight_native_workflow_guard')
    report = {
        'health': {'status': 'healthy', 'database': 'connected'},
        'stats': {'pending_operations': 0, 'failed_operations': 0},
        'config_focus': {'enable_observations': False},
        'provider_env': {'HINDSIGHT_API_LLM_PROVIDER': 'ollama', 'HINDSIGHT_API_ENABLE_OBSERVATIONS': 'false'},
        'active_payload_null': 0,
        'native_consolidation_unconsolidated_candidates': 50,
    }
    monkeypatch.setattr(guard, 'collect_status', lambda bank: report)
    monkeypatch.setattr(guard, 'estimate_native_consolidation_calls', lambda *a, **k: {'facts': 50, 'fetch_rounds': 1, 'tag_groups': 1, 'llm_calls': 1})

    args = types.SimpleNamespace(
        bank='hermes', llm_profile='minimax', jobs=1, facts_per_job=50,
        fetch_size=None, llm_batch_size=None, source_facts_max_tokens=4096,
        max_unconsolidated=10000, allow_existing_queue=False, allow_failed=False,
        expect_local_provider=True, execute=True, confirm='wrong-token',
        api_timeout=30, operation_timeout=3600, poll=10,
    )

    assert guard.cmd_run_native_consolidation_paid(args) == 2
    out = json.loads(capsys.readouterr().out)
    assert out['ok'] is False
    assert 'confirmation token mismatch' in out['error']


def test_rebuild_publish_requires_explicit_confirm_even_if_gates_skipped(monkeypatch, tmp_path):
    rebuild = load_module('hindsight_offline_v2_rebuild')
    calls = []

    def fake_run_json(cmd, **kwargs):
        calls.append(cmd)
        cmd_s = ' '.join(map(str, cmd))
        if 'hindsight_offline_v2_reduce.py' in cmd_s:
            return {'card_count': 1, 'collected_observations': 1}
        if 'hindsight_offline_v2_publish.py' in cmd_s:
            raise AssertionError('publish must not be called without --confirm-publish')
        if 'hindsight_offline_v2_audit.py' in cmd_s:
            return {'ok': True}
        return {'decision': 'pass', 'json_path': str(tmp_path / 'dummy.json')}

    monkeypatch.setattr(rebuild, 'run_json', fake_run_json)
    monkeypatch.setattr(rebuild, 'run_text', lambda *a, **k: 'status ok')
    monkeypatch.setattr(sys, 'argv', [
        'hindsight_offline_v2_rebuild.py',
        '--mode', 'publish',
        '--skip-eval-gate',
        '--skip-conflict-audit',
        '--output-dir', str(tmp_path / 'rebuild'),
        '--cards-root', str(tmp_path / 'cards'),
        '--json',
    ])
    import pytest
    with pytest.raises(SystemExit) as exc:
        rebuild.main()
    assert exc.value.code == 2
    assert not calls
