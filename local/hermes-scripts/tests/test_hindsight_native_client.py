import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_operations_api_supports_exclude_parents_and_safe_retry_guard():
    native = load_module('hindsight_native_client')
    calls = []

    def transport(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        if path.endswith('/operations'):
            return {'total': 0, 'operations': []}
        return {'ok': True}

    c = native.HindsightNativeClient(api='http://example', bank='hermes', transport=transport)
    c.list_operations(status='failed', exclude_parents=True, limit=5)
    assert calls[-1] == ('GET', '/v1/default/banks/hermes/operations', None, {'status': 'failed', 'limit': 5, 'offset': 0, 'exclude_parents': True})

    preview = c.retry_operation('op-1')
    assert preview['dry_run'] is True
    assert preview['required_confirm'] == native.RETRY_OPERATION_CONFIRM

    try:
        c.retry_operation('op-1', dry_run=False, confirm='wrong')
    except native.HindsightUnsafeOperation:
        pass
    else:
        raise AssertionError('retry without confirm should fail')

    c.retry_operation('op-1', dry_run=False, confirm=native.RETRY_OPERATION_CONFIRM)
    assert calls[-1][0] == 'POST'
    assert calls[-1][1] == '/v1/default/banks/hermes/operations/op-1/retry'


def test_v061_observability_export_import_and_repair_paths_are_wrapped():
    native = load_module('hindsight_native_client')
    calls = []

    def transport(method, path, payload=None, params=None):
        calls.append((method, path, payload, params))
        return {'method': method, 'path': path, 'payload': payload, 'params': params}

    c = native.HindsightNativeClient(api='http://example', bank='bank a', transport=transport)
    c.memories_timeseries(period='7d', time_field='mentioned_at')
    assert calls[-1][1] == '/v1/default/banks/bank%20a/stats/memories-timeseries'
    assert calls[-1][3]['time_field'] == 'mentioned_at'

    c.audit_log_stats(period='1d')
    assert calls[-1][1] == '/v1/default/banks/bank%20a/audit-logs/stats'

    c.export_bank_template()
    assert calls[-1][1] == '/v1/default/banks/bank%20a/export'

    c.bank_template_schema()
    assert calls[-1][1] == '/v1/bank-template-schema'

    c.import_bank_template({'template': True}, dry_run=True)
    assert calls[-1][1] == '/v1/default/banks/bank%20a/import'
    assert calls[-1][3]['dry_run'] is True

    for method_name, arg, confirm in [
        ('reprocess_document', 'doc 1', native.REPROCESS_DOCUMENT_CONFIRM),
        ('regenerate_entity', 'entity 1', native.REGENERATE_ENTITY_CONFIRM),
        ('refresh_mental_model', 'mm 1', native.REFRESH_MENTAL_MODEL_CONFIRM),
    ]:
        preview = getattr(c, method_name)(arg)
        assert preview['dry_run'] is True
        assert preview['required_confirm'] == confirm

    preview = c.recover_consolidation()
    assert preview['dry_run'] is True
    assert preview['required_confirm'] == native.RECOVER_CONSOLIDATION_CONFIRM
