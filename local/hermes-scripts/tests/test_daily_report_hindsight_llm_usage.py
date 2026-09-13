import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name='daily_report'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'daily_report.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Mixed formats: legacy slow-only line, legacy line with cached_tokens mid-field
# (previously swallowed by the strict regex), new full-logging formats
# (call() with trailing cache info, call_with_tools), and a consolidation batch
# line that must not be counted as an LLM call.
FIXTURE = '''\
2026-09-12 10:00:00,000 - INFO - slow llm call: scope=consolidation, model=openai/glm-5, input_tokens=50877, output_tokens=14188, total_tokens=65065, time=295.017s
2026-09-12 10:01:00,000 - INFO - slow llm call: scope=consolidation, model=openai/glm-5, input_tokens=1000, output_tokens=100, total_tokens=1100, cached_tokens=900, time=12.500s, ratio out/in=0.10
2026-09-13 09:44:06,387 - INFO - llm call: scope=reflect, model=openai/auto_cheep_model, input_tokens=5000, output_tokens=200, total_tokens=5200, time=3.500s, cached_tokens=4500, ratio out/in=0.04
2026-09-13 09:43:50,728 - INFO - llm call: scope=reflect_tool_call, model=openai/auto_cheep_model, input_tokens=2564, output_tokens=99, total_tokens=2663, time=1.250s, ratio out/in=0.04
2026-09-13 01:11:40,940 - INFO - [CONSOLIDATION] bank=hermes llm_batch #11 (3 memories, 5 llm calls) | processed=64/8 | recall=16.288s, llm=43.891s, created=41 updated=37 skipped=2 failed=0
'''


def test_every_call_line_counted_including_previously_swallowed():
    mod = load_module()
    rows, batch = mod.parse_hindsight_llm_logs(FIXTURE)
    by = {(r['scope'], r['model']): r for r in rows}
    assert len(rows) == 3

    cons = by[('consolidation', 'openai/glm-5')]
    assert cons['calls'] == 2  # both legacy lines, incl. the one cached mid-field
    assert cons['input_tokens'] == 51877
    assert cons['output_tokens'] == 14288
    assert cons['total_tokens'] == 66165
    assert abs(cons['seconds'] - 307.517) < 1e-6

    reflect = by[('reflect', 'openai/auto_cheep_model')]
    assert reflect['calls'] == 1
    assert reflect['input_tokens'] == 5000

    tools = by[('reflect_tool_call', 'openai/auto_cheep_model')]
    assert tools['calls'] == 1
    assert tools['total_tokens'] == 2663

    # batch line is NOT a call row, but is captured by batch_stats
    assert batch['batches'] == 1
    assert batch['llm_calls'] == 5


def test_line_with_missing_fields_still_counted():
    mod = load_module()
    rows, _ = mod.parse_hindsight_llm_logs('llm call: scope=odd_scope, model=x/y, note=no-tokens-here\n')
    assert len(rows) == 1
    assert rows[0]['calls'] == 1
    assert rows[0]['total_tokens'] == 0
    assert rows[0]['seconds'] == 0.0


def test_non_usage_lines_ignored():
    mod = load_module()
    noise = 'INFO - Log slow calls\nINFO - stripped reasoning tokens\nINFO - [WORKER_STATS] consolidation=0/2\n'
    rows, batch = mod.parse_hindsight_llm_logs(noise)
    assert rows == []
    assert batch['batches'] == 0


def test_oneapi_counts_filter_channel_type_and_window(tmp_path):
    mod = load_module()
    db = tmp_path / 'one-api.db'
    con = sqlite3.connect(db)
    con.execute(
        'CREATE TABLE logs (created_at INTEGER, channel_id INTEGER, type INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER)'
    )
    con.executemany(
        'INSERT INTO logs VALUES (?,?,?,?,?)',
        [
            (1000, 6, 2, 10, 5),    # auto_cheep channel, in window -> count
            (1001, 7, 2, 20, 8),    # count
            (1002, 5, 2, 30, 9),    # historical auto_cheep channel -> count
            (1003, 1, 2, 11, 1),    # other channel -> skip
            (1004, 6, 5, 0, 0),     # non-consumption row (error) -> skip
            (900, 6, 2, 99, 9),     # before window -> skip
            (1100, 6, 2, 77, 7),    # at end (exclusive) -> skip
        ],
    )
    con.commit()
    con.close()

    out = mod.parse_oneapi_auto_cheep_counts(str(db), datetime.fromtimestamp(999), datetime.fromtimestamp(1100))
    assert out == {'requests': 3, 'prompt_tokens': 60, 'completion_tokens': 22}
