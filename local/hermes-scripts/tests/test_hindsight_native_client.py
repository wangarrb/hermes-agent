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


def test_retain_passes_caller_uuid_and_exact_request_timeout():
    native = load_module('hindsight_native_client')
    import uuid
    import pytest
    calls = []
    operation_id = str(uuid.uuid4())
    client = native.HindsightNativeClient(bank='isolated', transport=lambda *a, **kw: calls.append((a, kw)) or {})
    client.retain_items([{'content': 'fact'}], operation_id=operation_id, timeout=7)
    assert calls[0][1]['payload']['operation_id'] == operation_id
    assert calls[0][1]['payload']['async'] is True
    with pytest.raises(ValueError):
        client.retain_items([{'content': 'fact'}], operation_id='not-a-uuid')
    assert len(calls) == 1


import pytest


@pytest.mark.parametrize('iterator,field', [
    ('iter_memories', 'items'), ('iter_documents', 'items'),
    ('iter_operations', 'operations'), ('iter_entities', 'items'),
])
@pytest.mark.parametrize('invalid', [{}, None, 'invalid', False, [7], [{'id': 'safe'}, None]])
def test_inventory_page_rejects_malformed_arrays_without_filtering(iterator, field, invalid):
    native = load_module('hindsight_native_client')
    client = native.HindsightNativeClient(transport=lambda *a, **kw: {field: invalid, 'total': 1})
    with pytest.raises(native.HindsightApiError, match='paginated') as caught:
        list(getattr(client, iterator)())
    assert caught.value.error_class == 'invalid_output'


@pytest.mark.parametrize('iterator,field', [
    ('iter_memories', 'items'), ('iter_documents', 'items'),
    ('iter_operations', 'operations'), ('iter_entities', 'items'),
])
def test_declared_nonempty_inventory_cannot_end_on_empty_page(iterator, field):
    native = load_module('hindsight_native_client')
    client = native.HindsightNativeClient(transport=lambda *a, **kw: {field: [], 'total': 1})
    with pytest.raises(native.HindsightApiError, match='paginated'):
        list(getattr(client, iterator)())


@pytest.mark.parametrize('iterator,field', [
    ('iter_memories', 'items'), ('iter_documents', 'items'),
    ('iter_operations', 'operations'), ('iter_entities', 'items'),
])
def test_short_inventory_pages_continue_to_declared_total(iterator, field):
    native = load_module('hindsight_native_client')
    calls = []
    rows = [{'id': f'id-{i}'} for i in range(3)]
    def transport(method, path, payload=None, params=None):
        assert params is not None
        calls.append(params['offset'])
        offset = params['offset']
        return {field: rows[offset:offset+2], 'total': len(rows)}
    client = native.HindsightNativeClient(transport=transport)
    assert [r['id'] for r in getattr(client, iterator)(limit=100)] == ['id-0', 'id-1', 'id-2']
    assert calls == [0, 2]


@pytest.mark.parametrize('total', [True, -1, '1', None, 1.0])
def test_memory_inventory_rejects_unknown_or_invalid_declared_total(total):
    native = load_module('hindsight_native_client')
    client = native.HindsightNativeClient(transport=lambda *a, **kw: {'items': [], 'total': total})
    with pytest.raises(native.HindsightApiError, match='paginated'):
        list(client.iter_memories())


def test_malformed_tail_inventory_is_p1_in_real_single_scan_audit():
    native = load_module('hindsight_native_client')
    audit = load_module('hindsight_conflict_audit')
    first = [{'id': f'id-{i}', 'document_id': 'doc', 'text': 'ordinary fixture'} for i in range(100)]
    def transport(method, path, payload=None, params=None):
        assert params is not None
        if params['offset'] == 0:
            return {'items': first, 'total': 101}
        return {'items': {'id': 'hidden', 'text': '[CONTEXT COMPACTION] fixture'}, 'total': 101}
    client = native.HindsightNativeClient(transport=transport)
    audit.API_SCAN_ERRORS.clear()
    ids, cases, outliers = audit.scan_memory_inventory('hermes', client=client)
    assert ids == set() and cases == [] and outliers == []
    assert audit.API_SCAN_ERRORS
    unavailable = audit.source_scan_unavailable_cases()
    assert [(c['type'], c['severity']) for c in unavailable] == [('source_scan_unavailable', 'P1')]


def test_valid_empty_inventory_and_explicit_limit_keep_compatibility():
    native = load_module('hindsight_native_client')
    empty = native.HindsightNativeClient(transport=lambda *a, **kw: {'items': [], 'total': 0})
    assert list(empty.iter_memories()) == []
    rows = [{'id': f'id-{i}'} for i in range(3)]
    limited = native.HindsightNativeClient(transport=lambda *a, **kw: {'items': rows, 'total': 30})
    assert list(limited.iter_memories(max_items=2)) == rows[:2]


@pytest.mark.parametrize('field', ['operations', 'items'])
@pytest.mark.parametrize('current_total', [0, 1])
def test_status_filtered_operations_accept_finished_empty_page(field, current_total):
    native = load_module('hindsight_native_client')
    calls = []
    rows = [{'id': 'operation-a'}, {'id': 'operation-b'}]
    def transport(method, path, payload=None, params=None):
        assert params is not None
        assert params['status'] == 'pending' and params['type'] == 'retain'
        calls.append(params['offset'])
        if params['offset'] == 0:
            return {field: rows, 'total': 3, 'offset': 0}
        return {field: [], 'total': current_total, 'offset': 2}
    client = native.HindsightNativeClient(transport=transport)
    assert list(client.iter_operations(status='pending', type='retain', limit=2, exclude_parents=True)) == rows
    assert calls == [0, 2]


@pytest.mark.parametrize('iterator', ['iter_memories', 'iter_documents', 'iter_entities'])
def test_static_inventory_shrink_does_not_gain_operation_eof_exception(iterator):
    native = load_module('hindsight_native_client')
    rows = [{'id': 'fact-a'}, {'id': 'fact-b'}]
    def transport(method, path, payload=None, params=None):
        assert params is not None
        return {'items': rows, 'total': 3, 'offset': 0} if params['offset'] == 0 else {'items': [], 'total': 0, 'offset': 2}
    client = native.HindsightNativeClient(transport=transport)
    with pytest.raises(native.HindsightApiError, match='paginated'):
        list(getattr(client, iterator)(limit=2))
