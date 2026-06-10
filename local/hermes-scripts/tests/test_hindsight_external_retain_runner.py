import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_external_runner_rehydrates_omitted_chatmemo_content_and_uses_external_confirm(tmp_path):
    manifest_mod = load_module("hindsight_external_manifest")
    runner = load_module("hindsight_external_retain_runner")
    memo = tmp_path / "chatgpt_20260310115739_aeb.txt"
    memo.write_text(
        "\n".join([
            "Title: AEB 单目测速回溯方案",
            "URL: https://chatgpt.com/c/69af968b-90a0-8320-9568-a02489f0ec42",
            "Platform: ChatGPT",
            "Created: 2026-03-10 11:57:39",
            "Messages: 2",
            "",
            "User: [2026-03-10 11:57:39]",
            "AEB 单目测速误差长尾，如何用 CUSUM 触发回溯？",
            "",
            "AI: [2026-03-10 11:58:00]",
            "用固定时滞 CA/RTS 平滑短窗口，控制极端误差。",
        ]),
        encoding="utf-8",
    )
    records, _ = manifest_mod.records_from_chat_memo_dir(tmp_path, bank_target="external_chatmemo", min_file_age_seconds=0)
    paths = manifest_mod.write_manifest(records, tmp_path / "manifests", include_content=False)

    loaded = runner.load_manifest(paths["manifest"])
    assert "content" not in loaded[0]
    item = runner.record_to_memory_item(loaded[0])

    assert runner.RETAIN_CONFIRM == "retain-hindsight-external-manifest"
    assert item["document_id"] == "external-chatmemo::chatgpt::69af968b-90a0-8320-9568-a02489f0ec42"
    assert item["context"] == "external_conversation"
    assert item["content"]
    assert item["update_mode"] == "replace"
    assert item["observation_scopes"]
    assert ["domain:autodrive"] in item["observation_scopes"]


def test_external_runner_dry_run_does_not_submit_or_write_state(tmp_path):
    manifest_mod = load_module("hindsight_external_manifest")
    runner = load_module("hindsight_external_retain_runner")
    memo = tmp_path / "chatgpt_20260310115739_aeb.txt"
    memo.write_text(
        "\n".join([
            "Title: AEB 单目测速回溯方案",
            "URL: https://chatgpt.com/c/69af968b-90a0-8320-9568-a02489f0ec42",
            "Platform: ChatGPT",
            "Created: 2026-03-10 11:57:39",
            "Messages: 2",
            "",
            "User: [2026-03-10 11:57:39]",
            "AEB 单目测速误差长尾，如何用 CUSUM 触发回溯？",
            "",
            "AI: [2026-03-10 11:58:00]",
            "用固定时滞 CA/RTS 平滑短窗口，控制极端误差。",
        ]),
        encoding="utf-8",
    )
    records, _ = manifest_mod.records_from_chat_memo_dir(tmp_path, bank_target="external_chatmemo", min_file_age_seconds=0)
    paths = manifest_mod.write_manifest(records, tmp_path / "manifests", include_content=False)
    state_path = tmp_path / "submit_state.json"

    result = runner.run_manifest(
        paths["manifest"],
        bank="external_chatmemo",
        dry_run=True,
        submit_state_path=state_path,
        limit=1,
    )

    assert result["dry_run"] is True
    assert result["manual_only"] is True
    assert result["daily_pipeline_integrated"] is False
    assert result["would_submit_items"] == 1
    assert result["submitted_items"] == 0
    assert not state_path.exists()


def make_openclaw_db(path: Path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE conversations (
            conversation_id INTEGER PRIMARY KEY,
            session_id TEXT,
            session_key TEXT,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            active INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            seq INTEGER,
            role TEXT,
            content TEXT,
            token_count INTEGER,
            created_at TEXT
        )
        """
    )
    cur.execute("INSERT INTO conversations VALUES (?,?,?,?,?,?,?)", (1, "s-main", "agent:main:main", None, "2026-04-01 10:00:00", "2026-04-01 10:20:00", 0))
    cur.executemany(
        "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
        [
            (101, 1, 1, "user", "AEB 单目测速里，固定时滞回溯对急刹长尾误差有帮助吗？", 10, "2026-04-01 10:00:02"),
            (102, 1, 2, "assistant", "可以，关键是用 CUSUM 触发后做 CA/RTS 平滑。", 20, "2026-04-01 10:00:04"),
        ],
    )
    con.commit()
    con.close()


def test_external_runner_rehydrates_omitted_openclaw_content(tmp_path):
    manifest_mod = load_module("hindsight_external_manifest")
    runner = load_module("hindsight_external_retain_runner")
    db = tmp_path / "lcm.db"
    make_openclaw_db(db)
    records, _ = manifest_mod.records_from_openclaw_lcm(db, bank_target="external_openclaw", min_age_seconds=0)
    paths = manifest_mod.write_manifest(records, tmp_path / "manifests", include_content=False)

    loaded = runner.load_manifest(paths["manifest"])
    assert "content" not in loaded[0]
    item = runner.record_to_memory_item(loaded[0])

    assert item["document_id"] == "external-openclaw::1::seg-001"
    assert item["context"] == "external_conversation"
    assert "CUSUM" in item["content"]


def test_external_runner_execute_creates_bank_and_enables_observations(tmp_path):
    manifest_mod = load_module("hindsight_external_manifest")
    runner = load_module("hindsight_external_retain_runner")
    memo = tmp_path / "chatgpt_20260310115739_aeb.txt"
    memo.write_text(
        "\n".join([
            "Title: AEB 单目测速回溯方案",
            "URL: https://chatgpt.com/c/69af968b-90a0-8320-9568-a02489f0ec42",
            "Platform: ChatGPT",
            "Created: 2026-03-10 11:57:39",
            "Messages: 2",
            "",
            "User: [2026-03-10 11:57:39]",
            "AEB 单目测速误差长尾，如何用 CUSUM 触发回溯？",
            "",
            "AI: [2026-03-10 11:58:00]",
            "用固定时滞 CA/RTS 平滑短窗口，控制极端误差。",
        ]),
        encoding="utf-8",
    )
    records, _ = manifest_mod.records_from_chat_memo_dir(tmp_path, bank_target="external_chatmemo", min_file_age_seconds=0)
    paths = manifest_mod.write_manifest(records, tmp_path / "manifests", include_content=False)

    class FakeClient:
        bank = "external_chatmemo_unit"

        def __init__(self):
            self.calls = []
            self.config = {key: None for key in runner.DEFAULT_EXTERNAL_BANK_CONFIG}

        def request(self, method, path, *, payload=None, params=None, timeout=None):
            self.calls.append((method, path, payload))
            if method == "GET" and path == "/v1/default/banks":
                return {"banks": []}
            if method == "PUT" and path == "/v1/default/banks/external_chatmemo_unit":
                return {"bank_id": "external_chatmemo_unit"}
            raise AssertionError((method, path, payload))

        def get_config(self):
            return {"config": dict(self.config)}

        def patch_config(self, updates):
            self.calls.append(("PATCH_CONFIG", updates))
            self.config.update(updates)
            return {"config": dict(self.config)}

        def retain_items(self, items, async_mode=True):
            self.calls.append(("RETAIN", items, async_mode))
            return {"operation_id": "op-1"}

    fake = FakeClient()
    result = runner.run_manifest(
        paths["manifest"],
        client=fake,
        bank="external_chatmemo_unit",
        dry_run=False,
        confirm=runner.RETAIN_CONFIRM,
        submit_state_path=None,
        wait=False,
    )

    assert result["bank_create"] == {"bank": "external_chatmemo_unit", "created": True}
    assert result["bank_config"]["effective"]["enable_observations"] is True
    assert ("PUT", "/v1/default/banks/external_chatmemo_unit", {"name": "external_chatmemo_unit"}) in fake.calls
    assert any(call[0] == "RETAIN" for call in fake.calls)


def test_retain_runner_reports_markdown_artifact_policy_and_stats(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    runner = load_module("hindsight_external_retain_runner")
    report = tmp_path / "周报_2026-05-16.md"
    report.write_text(
        "\n".join([
            "# 周报 - 2026-05-16",
            "",
            "## 本周工作总结",
            "",
            "### 算法交付",
            "",
            "1. 外厂主目标测距测速误差对比完成，融合版本整体最优。",
            "2. RAW夜间无光场景行人检测距离比H1-F RGB设备长10m。",
        ]),
        encoding="utf-8",
    )
    records = manifest.records_from_markdown_file(report, bank_target="hermes")
    manifest_path = manifest.write_manifest(records, tmp_path / "manifests", include_content=False)["manifest"]
    loaded = runner.load_manifest(manifest_path)

    selected, skipped = runner.prepare_retain_records(loaded, bank="hermes")
    assert selected
    assert skipped["action"] == 0
    assert skipped["unchanged"] == 0
    item = runner.record_to_memory_item(selected[0])
    assert item["context"] == "external_markdown_artifact"
    assert item["metadata"]["source_kind"] == "markdown_artifact_md"
    assert item["metadata"]["record_kind"] in {"document_outline", "item", "section"}

    summary = runner.summarize_records(loaded)
    assert summary["by_source"]["markdown_artifact_md"] >= 1
    assert summary["by_action"]["production"] >= 1
    assert summary["by_reason"]["production:markdown_artifact_structured"] >= 1
