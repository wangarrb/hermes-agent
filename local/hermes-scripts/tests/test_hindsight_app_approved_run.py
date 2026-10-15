"""The paid runner consumes exactly its independently pinned frozen bytes."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runner = importlib.import_module('hindsight_app_approved_run')


def frozen_sources(tmp_path, monkeypatch):
    source = tmp_path / 'sessions'
    source.mkdir()
    pins, original = {}, {}
    for sid, count in zip(sorted(runner.APPROVED), (165, 265, 17)):
        session = {'source_session_id': sid, 'session_id': 'codex-app-' + sid,
                   'messages': [{'role': 'user', 'content': 'Approved fixture evidence',
                                 'origin_message_id': f'{sid}:{i}', 'timestamp': 1792022400 + i}
                                for i in range(count)]}
        raw = (json.dumps(session, sort_keys=True) + '\n').encode()
        path = source / f'session_codex-app-{sid}.json'
        path.write_bytes(raw)
        pins[sid] = hashlib.sha256(raw).hexdigest()
        original[path] = raw
    # This substitutes the trusted code-side allowlist for isolated test data;
    # no external proof file or mutable source computes approval at run time.
    monkeypatch.setattr(runner, 'SOURCE', source)
    monkeypatch.setattr(runner, 'APPROVED_SOURCE_SHA256', pins, raising=False)
    return original


@pytest.mark.parametrize('changed_session', sorted(runner.APPROVED))
def test_same_count_content_replacement_is_rejected_before_json_parse(
        tmp_path, monkeypatch, changed_session):
    originals = frozen_sources(tmp_path, monkeypatch)
    target = runner.SOURCE / f'session_codex-app-{changed_session}.json'
    changed = json.loads(originals[target])
    changed['messages'][0]['content'] = 'Unapproved replacement evidence'
    raw = (json.dumps(changed, sort_keys=True) + '\n').encode()
    target.write_bytes(raw)
    assert len(changed['messages']) == len(json.loads(originals[target])['messages'])
    loads = json.loads

    def guarded_loads(value, *args, **kwargs):
        assert value != raw, 'unapproved bytes reached the JSON parser'
        return loads(value, *args, **kwargs)

    def unapproved_runtime():
        pytest.fail('unapproved source reached production configuration/runtime')

    monkeypatch.setattr(runner.json, 'loads', guarded_loads)
    monkeypatch.setattr(runner, 'load_config', unapproved_runtime)
    with pytest.raises(ValueError, match='approved frozen source bytes changed'):
        runner.main([])


def test_hash_and_parse_use_one_read_even_if_path_is_replaced(tmp_path, monkeypatch):
    originals = frozen_sources(tmp_path, monkeypatch)
    read_bytes = Path.read_bytes
    reads = {path: 0 for path in originals}

    def replace_after_read(path):
        raw = read_bytes(path)
        if path in originals:
            reads[path] += 1
            changed = json.loads(raw)
            changed['messages'][0]['content'] = 'Later unapproved replacement'
            path.write_text(json.dumps(changed))
        return raw

    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    sessions = runner._load_approved_sessions()
    assert all(count == 1 for count in reads.values())
    assert sum(len(s['messages']) for s in sessions) == 447
    assert all(s['messages'][0]['content'] == 'Approved fixture evidence' for s in sessions)
    assert all(b'Later unapproved replacement' in read_bytes(path) for path in originals)
