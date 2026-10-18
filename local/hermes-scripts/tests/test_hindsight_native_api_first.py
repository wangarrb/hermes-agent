import importlib.util
import sys
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeTransport:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def __call__(self, method, path, *, payload=None, params=None):
        rec = {'method': method, 'path': path, 'payload': payload, 'params': params or {}}
        self.calls.append(rec)
        if not self.responses:
            return {}
        nxt = self.responses.pop(0)
        if callable(nxt):
            return nxt(rec)
        return nxt


def test_native_client_paginates_documents_via_official_api():
    native = load_module('hindsight_native_client')
    transport = FakeTransport([
        {'items': [{'id': 'doc-a'}, {'id': 'doc-b'}], 'total': 3, 'limit': 2, 'offset': 0},
        {'items': [{'id': 'doc-c'}], 'total': 3, 'limit': 2, 'offset': 2},
    ])
    client = native.HindsightNativeClient(api='http://api', bank='hermes', transport=transport)

    docs = client.list_all_documents(limit=2)

    assert [d['id'] for d in docs] == ['doc-a', 'doc-b', 'doc-c']
    assert [c['path'] for c in transport.calls] == [
        '/v1/default/banks/hermes/documents',
        '/v1/default/banks/hermes/documents',
    ]
    assert [c['params']['offset'] for c in transport.calls] == [0, 2]


def test_native_client_destructive_operation_requires_confirm_token():
    native = load_module('hindsight_native_client')
    transport = FakeTransport([{'ok': True}])
    client = native.HindsightNativeClient(api='http://api', bank='hermes', transport=transport)

    dry = client.delete_operation('op-1', dry_run=True)
    assert dry['dry_run'] is True
    assert transport.calls == []

    try:
        client.delete_operation('op-1', dry_run=False)
    except native.HindsightUnsafeOperation as exc:
        assert 'confirm' in str(exc).lower()
    else:
        raise AssertionError('delete_operation must require explicit confirm token')
    assert transport.calls == []

    res = client.delete_operation('op-1', dry_run=False, confirm='delete-hindsight-operation')
    assert res == {'ok': True}
    assert transport.calls[-1]['method'] == 'DELETE'
    assert transport.calls[-1]['path'] == '/v1/default/banks/hermes/operations/op-1'


def test_cancel_pending_defaults_to_dry_run_and_uses_official_operations_api(monkeypatch, tmp_path):
    native = load_module('hindsight_native_client')
    cancel = load_module('cancel_hindsight_bank_pending')
    transport = FakeTransport([
        {'total': 2, 'operations': [
            {'id': 'op-a', 'task_type': 'retain', 'created_at': 't1'},
            {'id': 'op-b', 'task_type': 'batch_retain', 'created_at': 't2'},
        ]},
    ])
    client = native.HindsightNativeClient(api='http://api', bank='hermes', transport=transport)

    total, ops = cancel.get_pending('hermes', client=client)
    assert total == 2
    assert [op['id'] for op in ops] == ['op-a', 'op-b']
    assert transport.calls[0]['path'] == '/v1/default/banks/hermes/operations'
    assert transport.calls[0]['params']['status'] == 'pending'

    ok, status, body = cancel.cancel_one('hermes', 'op-a', client=client, dry_run=True)
    assert ok is True
    assert status == 0
    assert 'dry-run' in body
    assert len(transport.calls) == 1  # no DELETE call in dry-run


def test_offline_audit_uses_official_api_for_layer_counts():
    audit = load_module('hindsight_offline_audit')

    class FakeClient:
        def list_all_documents(self, max_items=None):
            return [
                {'id': 'hermes-sqlite::day-topic::2026-05-06::a'},
                {'id': 'hermes-offline-consolidation::daily::2026-05-06::b'},
                {'id': 'hermes-offline-consolidation::weekly::2026-W19::c'},
            ]

        def iter_memories(self, types=None, max_items=None):
            yield {'id': 'm1', 'document_id': 'hermes-sqlite::day-topic::2026-05-06::a', 'type': 'world'}
            yield {'id': 'm2', 'document_id': 'hermes-offline-consolidation::daily::2026-05-06::b', 'type': 'observation'}

        def iter_operations(self, max_items=None):
            yield {'id': 'op1', 'task_type': 'retain', 'status': 'completed'}

    layers = audit.hindsight_layer_counts(bank='hermes', client=FakeClient())
    assert layers['source'] == 'official_api'
    assert layers['documents_by_bank'] == {'hermes': 3}
    assert layers['facts_by_bank'] == {'hermes': 2}
    assert layers['sqlite_import_days'] == {'2026-05-06': 1}
    assert layers['daily_consolidated_days'] == {'2026-05-06': 1}
    assert layers['weekly_consolidated_periods'] == {'2026-W19': 1}
    assert layers['operations'] == [['retain', 'completed', '1']]

def test_minimax_import_purge_sqlite_uses_official_api_and_defaults_dry_run():
    native = load_module('hindsight_native_client')
    mod = load_module('hindsight_minimax_import')

    class FakeClient:
        def __init__(self):
            self.deleted_docs = []
            self.deleted_ops = []

        def list_all_documents(self, max_items=None):
            return [{'id': 'hermes-sqlite::day-topic::2026-05-06::a'}, {'id': 'other::doc'}]

        def iter_operations(self, status=None, max_items=None):
            yield {'id': 'op1', 'task_type': 'retain', 'status': status or 'pending', 'task_payload': {'document_id': 'hermes-sqlite::day-topic::2026-05-06::a'}}
            yield {'id': 'op2', 'task_type': 'consolidation', 'status': status or 'failed', 'task_payload': {'document_id': 'other::doc'}}

        def delete_document(self, document_id, dry_run=True, confirm=None):
            if dry_run:
                return {'dry_run': True, 'document_id': document_id}
            if confirm != native.DELETE_DOCUMENT_CONFIRM:
                raise native.HindsightUnsafeOperation('confirm required')
            self.deleted_docs.append(document_id)
            return {'deleted': document_id}

        def delete_operation(self, operation_id, dry_run=True, confirm=None):
            if dry_run:
                return {'dry_run': True, 'operation_id': operation_id}
            if confirm != native.DELETE_OPERATION_CONFIRM:
                raise native.HindsightUnsafeOperation('confirm required')
            self.deleted_ops.append(operation_id)
            return {'deleted': operation_id}

    client = FakeClient()
    assert mod.existing_sqlite_doc_count('hermes', client=client) == 1
    dry = mod.purge_sqlite_documents('hermes', client=client, dry_run=True)
    assert dry['dry_run'] is True
    assert dry['documents_matched'] == 1
    assert dry['operations_matched'] == 1
    assert client.deleted_docs == []
    assert client.deleted_ops == []

    try:
        mod.purge_sqlite_documents('hermes', client=client, dry_run=False)
    except native.HindsightUnsafeOperation:
        pass
    else:
        raise AssertionError('purge must require explicit confirm token')

    done = mod.purge_sqlite_documents(
        'hermes',
        client=client,
        dry_run=False,
        confirm_documents=native.DELETE_DOCUMENT_CONFIRM,
        confirm_operations=native.DELETE_OPERATION_CONFIRM,
    )
    assert done['dry_run'] is False
    assert client.deleted_docs == ['hermes-sqlite::day-topic::2026-05-06::a']
    assert client.deleted_ops == ['op1']

def test_conflict_audit_source_scans_use_api_without_psql(monkeypatch):
    audit = load_module('hindsight_conflict_audit')

    class FakeClient:
        def list_all_documents(self, max_items=None):
            return [{'id': 'doc-1'}, {'id': 'hermes-offline-canonical::topic'}]

        def iter_memories(self, types=None, max_items=None):
            yield {'id': 'mem-1', 'document_id': 'doc-1', 'type': 'world', 'text': 'normal stable fact'}
            yield {'id': 'mem-2', 'document_id': 'doc-2', 'type': 'world', 'text': 'Traceback (most recent call last) leaked'}

    audit.API_SCAN_ERRORS.clear()
    assert audit.known_document_ids('hermes', client=FakeClient()) == {'doc-1', 'hermes-offline-canonical::topic'}
    assert audit.known_memory_ids('hermes', client=FakeClient()) == {'mem-1', 'mem-2'}
    cases = audit.db_contamination_cases('hermes', client=FakeClient())
    assert cases
    assert cases[0]['type'] == 'source_fact_contamination'
    assert cases[0]['target']['id'] == 'mem-2'
    assert audit.API_SCAN_ERRORS == []
