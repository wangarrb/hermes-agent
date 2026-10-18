import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CST = timezone(timedelta(hours=8))


def load_module(name='daily_report'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'daily_report.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _event(ts: str, last_in: int, last_cached: int, last_out: int, total_in: int) -> str:
    """One rollout token_count line (last_* = this call, total_* = cumulative)."""
    return json.dumps({
        'timestamp': ts,
        'type': 'event_msg',
        'payload': {
            'type': 'token_count',
            'info': {
                'total_token_usage': {'input_tokens': total_in, 'cached_input_tokens': 0, 'output_tokens': 0},
                'last_token_usage': {
                    'input_tokens': last_in,
                    'cached_input_tokens': last_cached,
                    'output_tokens': last_out,
                    'total_tokens': last_in + last_out,
                },
            },
        },
    })


def _write(tmp_path: Path, name: str, lines: list[str]) -> str:
    f = tmp_path / name
    f.write_text('\n'.join(lines) + '\n')
    return str(f)


# Window: 2026-09-12 08:30 CST -> 2026-09-13 08:30 CST
W_START = datetime(2026, 9, 12, 8, 30, tzinfo=CST)
W_END = datetime(2026, 9, 13, 8, 30, tzinfo=CST)


def test_windowed_events_count_calls_and_deltas(tmp_path):
    mod = load_module()
    # UTC timestamps: 00:30Z == 08:30 CST
    f1 = _write(tmp_path, 'rollout-a.jsonl', [
        _event('2026-09-12T00:29:59Z', 999, 999, 999, 1),        # before start -> skip
        _event('2026-09-12T00:30:00Z', 5000, 4000, 100, 500000),  # at start -> count
        _event('2026-09-12T04:00:00Z', 8000, 7000, 200, 1234567), # inside -> count
        _event('2026-09-13T00:29:59Z', 3000, 2000, 50, 2000),     # inside; cumulative RESET below first -> must not go negative
        _event('2026-09-13T00:30:00Z', 777, 777, 777, 3),         # at end -> skip
        '{"timestamp":"2026-09-12T05:00:00Z","type":"other"',      # broken json -> ignore
        json.dumps({'timestamp': '2026-09-12T05:00:00Z', 'type': 'event_msg',
                    'payload': {'type': 'token_count', 'info': {
                        'last_token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}}}}),  # zero usage -> skip
    ])
    f2 = _write(tmp_path, 'rollout-b.jsonl', [
        _event('2026-09-12T12:00:00Z', 100, 50, 10, 100),
    ])

    out = mod.parse_codex_rollout_events(
        [('Codex(kanban)', f1), ('Codex(kanban)', f2), ('Codex', '/nonexistent.jsonl')],
        W_START, W_END,
    )
    assert out is not None
    assert set(out['sources']) == {'Codex(kanban)'}
    s = out['sources']['Codex(kanban)']
    assert s['calls'] == 4                       # 3 from f1 + 1 from f2
    assert s['input'] == 5000 + 8000 + 3000 + 100
    assert s['cached'] == 4000 + 7000 + 2000 + 50
    assert s['output'] == 100 + 200 + 50 + 10
    assert out['total'] == s
    assert s['input'] > 0                        # reset scenario: never negative


def test_none_when_no_events(tmp_path):
    mod = load_module()
    f = _write(tmp_path, 'rollout-empty.jsonl', ['not json', '{"a":1}'])
    assert mod.parse_codex_rollout_events([('Codex', f)], W_START, W_END) is None
    assert mod.parse_codex_rollout_events([], W_START, W_END) is None


def test_timestamp_parsing_variants():
    mod = load_module()
    want = datetime(2026, 9, 13, 8, 30, tzinfo=CST).timestamp()
    for ts in ('2026-09-13T00:30:00Z', '2026-09-13T00:30:00+00:00', '2026-09-13T08:30:00+08:00'):
        dt = mod._rollout_event_dt(ts)
        assert dt is not None and abs(dt.timestamp() - want) < 1
    assert mod._rollout_event_dt('2026-09-13T00:30:00').timestamp() == want  # naive -> UTC
    assert mod._rollout_event_dt('garbage') is None
