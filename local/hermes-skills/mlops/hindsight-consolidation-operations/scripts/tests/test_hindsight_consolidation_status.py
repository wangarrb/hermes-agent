import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location('hindsight_consolidation_status', ROOT / 'hindsight_consolidation_status.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_async_ops_tsv_and_summary():
    mod = load_module()
    ops = mod.parse_async_ops_tsv('completed\tconsolidation\t10\nprocessing\tconsolidation\t1\npending\tretain\t2\n')
    assert ops == {
        'completed': {'consolidation': 10},
        'pending': {'retain': 2},
        'processing': {'consolidation': 1},
    }
    summary = mod.summarize_async_ops(ops)
    assert summary['active_count'] == 1
    assert summary['pending_count'] == 2
    assert summary['completed_count'] == 10
    assert summary['has_active_work'] is True


def test_build_async_ops_sql_is_read_only_and_escapes_bank():
    mod = load_module()
    sql = mod.build_async_ops_sql("bank'o")
    lowered = sql.lower()
    assert lowered.startswith('select ')
    assert 'update ' not in lowered
    assert 'delete ' not in lowered
    assert "bank''o" in sql


def test_build_report_skip_psql_with_fake_api(monkeypatch):
    mod = load_module()

    def fake_api(url, timeout=10):
        if url.endswith('/health'):
            return {'status': 'healthy'}
        return {'total_documents': 2, 'total_nodes': 3, 'total_observations': 4, 'operations_by_status': {'completed': 1}}

    monkeypatch.setattr(mod, 'api_get_json', fake_api)
    args = mod.build_parser().parse_args(['--skip-psql', '--json'])
    report = mod.build_report(args)
    assert report['read_only'] is True
    assert report['overall']['ok'] is True
    assert report['checks']['async_operations_psql']['skipped'] is True
    assert report['bank_summary']['total_observations'] == 4


def test_render_human_summary_is_not_raw_json():
    mod = load_module()
    report = {
        'generated_at': '2026-01-01T00:00:00+00:00',
        'tenant': 'default',
        'bank': 'hermes',
        'checks': {
            'health': {'ok': True},
            'async_operations_psql': {'skipped': True},
        },
        'bank_summary': {'total_documents': 1, 'total_nodes': 2, 'total_observations': 3},
        'overall': {'ok': True, 'has_active_work': False, 'restart_guidance': 'wait for idle'},
    }
    text = mod.render_human_summary(report)
    assert text.startswith('Hindsight consolidation status')
    assert 'documents/nodes/observations: 1/2/3' in text
    assert not text.lstrip().startswith('{')


def test_discover_default_psql_falls_back_to_path_when_missing(monkeypatch, tmp_path):
    mod = load_module()
    monkeypatch.setattr(mod.Path, 'home', lambda: tmp_path)
    assert mod.discover_default_psql() == 'psql'
