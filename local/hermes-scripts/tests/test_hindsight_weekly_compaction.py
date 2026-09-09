import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    name = "offline_hindsight_reflect_consolidate_weekly_compaction"
    spec = importlib.util.spec_from_file_location(name, ROOT / "offline_hindsight_reflect_consolidate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weekly_projection_removes_only_terminal_json_block():
    module = load_module()
    markdown = """# Daily Result

## Knowledge Points
- Keep this conclusion.

```bash
python verify.py --json
```

## Source IDs
- fact-1
- fact-2

## JSON
```json
{"knowledge_points": ["duplicate"]}
```
"""

    projected = module.weekly_input_markdown(markdown)

    assert "Keep this conclusion." in projected
    assert "python verify.py --json" in projected
    assert "## Source IDs\n- fact-1\n- fact-2" in projected
    assert "## JSON" not in projected
    assert '"knowledge_points": ["duplicate"]' not in projected


def test_weekly_projection_preserves_earlier_json_section():
    module = load_module()
    markdown = """# Daily Result

## JSON
```json
{"evidence": "keep"}
```

## Analysis
- Keep this section too.

## JSON
```json
{"duplicate": true}
```
"""

    projected = module.weekly_input_markdown(markdown)

    assert '{"evidence": "keep"}' in projected
    assert "## Analysis" in projected
    assert "Keep this section too." in projected
    assert '{"duplicate": true}' not in projected


def test_weekly_cache_payload_has_projection_version_only_for_weekly():
    module = load_module()
    weekly = module.ReflectUnit("weekly", "p", "topic", 0, "body", 1, ["a"], "s", "e")
    daily = module.ReflectUnit("daily", "p", "topic", 0, "body", 1, ["a"], "s", "e")

    weekly_payload = module.unit_cache_payload(weekly)
    daily_payload = module.unit_cache_payload(daily)

    assert weekly_payload["weekly_input_projection_version"]
    assert "weekly_input_projection_version" not in daily_payload
