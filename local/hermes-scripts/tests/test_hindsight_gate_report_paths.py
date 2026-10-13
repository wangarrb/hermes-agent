"""Real audit writer -> completed-child recovery, on tiny isolated generations.

Inventories are explicit synthetic seams; the audit CLI, report serialization,
path metadata and normal gate/receipt validation are real. No native coverage,
production API/SQL/LLM, or production quality claim is made by these controls.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

from test_hindsight_incremental_publication import env
from test_hindsight_gate_attempt_recovery import recovery, invoke, reference, legacy_gate
import hindsight_incremental_gate_attempts as attempts
import hindsight_incremental_merge as merge
import hindsight_conflict_audit as audit


def real_audit_writer(recovery, monkeypatch, *, corrupt=None):
    """Extend the existing runner with the actual audit CLI's output writer."""
    _, adapter, ledger, _, manifest, _, runner = recovery
    rows = merge.load_manifest_artifacts(Path(manifest['root']))[2]
    assert rows is not None
    count = len(rows)
    # Avoid the fixture's disk-normalizing adapter: this exact I/O seam must
    # preserve the two real producers' distinct on-disk formats.
    adapter._test_runner = runner

    def emit(step, report):
        if step != 'conflict':
            return
        output = Path(report['json_path']).parent
        argv = ['hindsight_conflict_audit.py', '--cards-root', manifest['root'],
                '--offline-root', manifest['root'], '--output-dir', str(output),
                '--bank', ledger.bank, '--api', adapter.config['api'], '--json']
        captured = io.StringIO()
        with monkeypatch.context() as patcher:
            patcher.setattr(sys, 'argv', argv)
            patcher.setattr(audit, 'load_observations', lambda _root: [{} for _ in range(count)])
            patcher.setattr(audit, 'known_document_ids', lambda *_a, **_kw: set())
            patcher.setattr(audit, 'scan_memory_inventory', lambda *_a, **_kw: (set(), [], []))
            patcher.setattr(audit, 'known_local_lineage_ids', lambda _root: (set(), set()))
            patcher.setattr(audit, 'known_files', lambda _root: set())
            patcher.setattr(audit.core, 'build_conflict_cases', lambda *_a, **_kw: [])
            with contextlib.redirect_stdout(captured):
                audit.main()
        actual = json.loads(captured.getvalue())
        disk = Path(actual['json_path'])
        assert json.loads(disk.read_text()) == actual
        assert actual['markdown_path'] and Path(actual['markdown_path']).is_file()
        if corrupt is not None:
            payload = json.loads(disk.read_text())
            if corrupt == 'extra_semantic_field':
                payload['unreported_semantic_claim'] = 'must reject'
            else:
                payload[corrupt] = str(output / 'wrong-metadata-target')
            disk.write_text(json.dumps(payload))
        report.clear()
        report.update(actual)

    runner.callback = emit
    return runner


def test_real_audit_writer_completed_receipt_recovers_without_reaudit(recovery, monkeypatch):
    _, adapter, ledger, db, manifest, _, _ = recovery
    runner = real_audit_writer(recovery, monkeypatch)
    original = attempts.atomic_write

    def crash_stdout(path, payload):
        if path.name == 'conflict.json':
            raise RuntimeError('isolated completed-receipt stdout crash')
        return original(path, payload)

    monkeypatch.setattr(attempts, 'atomic_write', crash_stdout)
    with pytest.raises(RuntimeError, match='stdout crash'):
        invoke(recovery)
    saved_reference = json.loads(reference(ledger, manifest).read_text())
    directory = Path(saved_reference['attempt_root'])
    completed = directory / 'conflict.completion.json'
    receipt = json.loads(completed.read_text())
    disk = Path(receipt['disk_report']['path'])
    frozen = {str(p): p.read_bytes() for p in (completed, disk)}
    assert not (directory / 'conflict.json').exists()
    assert [step for step, _ in runner.calls] == ['conflict']

    monkeypatch.setattr(attempts, 'atomic_write', original)
    gate = invoke(recovery)
    assert [step for step, _ in runner.calls] == list(attempts.STEPS)
    assert gate['decision'] == 'eligible_for_local_proposal'
    adapter._validate_gate(manifest, gate, ledger)
    for token, before in frozen.items():
        assert Path(token).read_bytes() == before
    materialized = (directory / 'conflict.json').read_bytes()
    assert materialized == receipt['stdout_json'].encode()
    assert hashlib.sha256(materialized).hexdigest() == receipt['stdout']['sha256']
    assert json.loads(materialized) == json.loads(disk.read_text())
    assert db.commits == 0 and not db.embeds


def test_real_audit_writer_keeps_complete_legacy_gate_compatible(recovery, monkeypatch):
    _, adapter, ledger, db, manifest, _, _ = recovery
    runner = real_audit_writer(recovery, monkeypatch)
    gate = legacy_gate(recovery)
    before = len(runner.calls)
    directory = Path(gate['gate_path']).parent
    audit_copy = json.loads((directory / 'conflict.json').read_text())
    assert json.loads(Path(audit_copy['json_path']).read_text()) == audit_copy
    adapter._validate_gate(manifest, gate, ledger)
    assert len(runner.calls) == before
    assert not reference(ledger, manifest).exists()
    assert db.commits == 0 and not db.embeds


@pytest.mark.parametrize('corrupt', ['json_path', 'markdown_path', 'extra_semantic_field'])
def test_real_audit_writer_rejects_metadata_or_payload_contradiction(recovery, monkeypatch, corrupt):
    _, _, _, db, _, _, _ = recovery
    runner = real_audit_writer(recovery, monkeypatch, corrupt=corrupt)
    with pytest.raises(ValueError, match='stdout/report artifact mismatch'):
        invoke(recovery)
    assert [step for step, _ in runner.calls] == ['conflict']
    assert db.commits == 0 and not db.embeds
