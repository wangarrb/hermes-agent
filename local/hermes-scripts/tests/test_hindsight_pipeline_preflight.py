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


def test_expected_tuning_uses_v06_round_key_and_legacy_alias():
    common = load_module('hindsight_pipeline_common')
    preflight = load_module('hindsight_pipeline_preflight')
    cfg = common.default_config('/tmp/hermes')
    payload = preflight.expected_tuning_payload(cfg)
    assert payload['max_memories_per_round'] == 60
    assert payload['max_memories_per_job'] == 60
    assert payload['recall_budget'] == 'low'
    assert payload['source_facts_max_tokens'] == 4096
    assert payload['source_facts_max_tokens_per_observation'] == 256


def test_compare_tuning_accepts_legacy_job_alias_but_requires_source_caps():
    common = load_module('hindsight_pipeline_common')
    preflight = load_module('hindsight_pipeline_preflight')
    cfg = common.default_config('/tmp/hermes')
    legacy_complete = {
        'consolidation_batch_size': 20,
        'consolidation_llm_batch_size': 20,
        'max_memories_per_job': 60,
        'parallel_batches': 3,
        'source_facts_max_tokens': 4096,
        'source_facts_max_tokens_per_observation': 256,
    }
    assert preflight.compare_tuning(legacy_complete, cfg)['ok'] is True
    legacy_missing_caps = dict(legacy_complete)
    legacy_missing_caps.pop('source_facts_max_tokens')
    assert preflight.compare_tuning(legacy_missing_caps, cfg)['ok'] is False
