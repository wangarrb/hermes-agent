import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Minimal replica of the upstream (unpatched) code shapes the patcher anchors on:
# 16-indent "# Log slow calls" block, trailing "if return_usage:", and a
# 16-indent "return LLMToolCallResult(" line.
UNPATCHED = '''\
class FakeLLM:
    async def call(self):
        try:
            if ready:
                # Log slow calls
                if duration > 10.0 and usage:
                    ratio = max(1, output_tokens) / max(1, input_tokens)
                    cache_info = f", cached_tokens={cached_tokens}" if cached_tokens > 0 else ""
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"total_tokens={total_tokens}{cache_info}, time={duration:.3f}s, ratio out/in={ratio:.2f}"
                    )

                if return_usage:
                    return None
                return None
        except Exception:
            pass

    async def call_with_tools(self):
        try:
            if tool_calls:
                return LLMToolCallResult(
                    content="x",
                )
        except Exception:
            pass
'''


def load_module(name='patch_hindsight_llm_token_log'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'patch_hindsight_llm_token_log.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fixture(tmp_path: Path, text: str = UNPATCHED):
    target = tmp_path / 'openai_compatible_llm.py'
    target.write_text(text)
    mod = load_module()
    mod.TARGET = target
    return mod, target


def test_check_detects_unpatched(tmp_path):
    mod, _ = _fixture(tmp_path)
    assert mod.check() == (False, False)


def test_apply_patches_verifies_and_is_idempotent(tmp_path):
    mod, target = _fixture(tmp_path)

    assert mod.apply() == 0
    patched = target.read_text()
    assert '# HERMES_LLM_TOKEN_LOG_FIX_V1\n' in patched
    assert '# HERMES_LLM_TOKEN_LOG_FIX_V1_TOOLS' in patched
    assert 'llm call: scope=' in patched
    assert 'slow llm call:' not in patched
    assert mod.check() == (True, True)
    assert list(tmp_path.glob('openai_compatible_llm.py.bak-*'))  # backup written before edit

    assert mod.apply() == 0  # idempotent second run
    assert target.read_text() == patched


def test_apply_refuses_when_anchors_missing(tmp_path):
    mod, target = _fixture(tmp_path, 'def untouched():\n    return 1\n')
    assert mod.apply() == 2
    assert target.read_text() == 'def untouched():\n    return 1\n'
