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
    conversations = [
        (1, "s-main", "agent:main:main", None, "2026-04-01 10:00:00", "2026-04-01 10:20:00", 0),
        (2, "s-cron", "agent:main:cron:abc", None, "2026-04-01 11:00:00", "2026-04-01 11:10:00", 0),
        (3, "s-history", "agent:main:main", "历史: 当前会话", "2026-04-01 12:00:00", "2026-04-01 12:10:00", 0),
        (4, "s-dingtalk", "agent:main:dingtalk:direct:170941191029642680", None, "2026-04-01 13:00:00", "2026-04-01 13:10:00", 0),
    ]
    cur.executemany("INSERT INTO conversations VALUES (?,?,?,?,?,?,?)", conversations)
    messages = [
        (101, 1, 1, "user", "System (untrusted): should be dropped", 1, "2026-04-01 10:00:01"),
        (102, 1, 2, "user", "Conversation info (untrusted metadata): ```json\n{\"sender_id\":\"secret-ish\",\"sender\":\"王雅儒\"}\n```\nAEB 单目测速里，固定时滞回溯对急刹长尾误差有帮助吗？", 10, "2026-04-01 10:00:02"),
        (103, 1, 3, "tool", "tool output must never be imported", 10, "2026-04-01 10:00:03"),
        (104, 1, 4, "assistant", "可以，关键是用 CUSUM 触发后对短窗口做 CA/RTS 平滑，避免在线 IMM 被重尾噪声误导。", 20, "2026-04-01 10:00:04"),
        (201, 2, 1, "user", "cron content should be excluded", 1, "2026-04-01 11:00:01"),
        (202, 2, 2, "assistant", "cron assistant", 1, "2026-04-01 11:00:02"),
        (301, 3, 1, "user", "history aggregate should be excluded", 1, "2026-04-01 12:00:01"),
        (302, 3, 2, "assistant", "history assistant", 1, "2026-04-01 12:00:02"),
        (401, 4, 1, "user", "钉钉里讨论 OpenClaw gateway 和 Hindsight 外部导入规则。", 10, "2026-04-01 13:00:01"),
        (402, 4, 2, "assistant", "钉钉 direct 会话应按用户直接对话导入，但仍过滤工具和系统噪声。", 10, "2026-04-01 13:00:02"),
    ]
    cur.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?,?)", messages)
    con.commit()
    con.close()


def test_chat_memo_txt_records_parse_headers_messages_and_stable_document_id(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    memo = tmp_path / "chatgpt_20260310115739_aeb.txt"
    memo.write_text(
        "\n".join([
            "Title: 我在设计一个用于AEB的基于单目3D的测速算法",
            "URL: https://chatgpt.com/c/69af968b-90a0-8320-9568-a02489f0ec42",
            "Platform: ChatGPT",
            "Created: 2026-03-10 11:57:39",
            "Messages: 2",
            "",
            "User: [2026-03-10 11:57:39]",
            "AEB 单目测速误差长尾，如何用回溯降低极端误差？",
            "",
            "AI: [2026-03-10 11:58:00]",
            "固定时滞 smoothing 可以利用后验时序证据，适合急刹和 cutin。",
        ]),
        encoding="utf-8",
    )

    records, diagnostics = manifest.records_from_chat_memo_dir(tmp_path, bank_target="external_chatmemo_smoke", min_file_age_seconds=0)

    assert diagnostics["source_kind"] == "chat_memo_txt"
    assert len(records) == 1
    rec = records[0]
    assert rec["document_id"] == "external-chatmemo::chatgpt::69af968b-90a0-8320-9568-a02489f0ec42"
    assert rec["bank_target"] == "external_chatmemo_smoke"
    assert rec["context"] == "external_conversation"
    assert rec["event_date"] == "2026-03-10 11:57:39"
    assert "User: [2026-03-10 11:57:39]" in rec["content"]
    assert "Assistant: [2026-03-10 11:58:00]" in rec["content"]
    assert "source:external-chatmemo" in rec["tags"]
    assert "platform:chatgpt" in rec["tags"]
    assert rec["metadata"]["source_kind"] == "chat_memo_txt"
    assert rec["metadata"]["url"] == "https://chatgpt.com/c/69af968b-90a0-8320-9568-a02489f0ec42"
    assert rec["observation_scopes"]
    assert ["domain:autodrive"] in rec["observation_scopes"]


def test_chat_memo_txt_deduplicates_same_conversation_id_by_newer_source_file(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    old = tmp_path / "doubao_20260401200122_PDMS.txt"
    new = tmp_path / "doubao_20260518190313_PDMS.txt"
    body_old = "\n".join([
        "Title: PDMS是什么指标与Collision Rate有什么区别",
        "URL: https://www.doubao.com/chat/38419629838080770",
        "Platform: 豆包",
        "Created: 2026-04-01 20:01:22",
        "Messages: 2",
        "",
        "User: [2026-04-01 20:01:22]",
        "PDMS是什么指标，与自动驾驶Collision Rate有什么区别？",
        "",
        "AI: [2026-04-01 20:01:23]",
        "旧导出：PDMS 是自动驾驶规划评价指标。",
    ])
    body_new = body_old.replace("旧导出", "新导出")
    old.write_text(body_old, encoding="utf-8")
    new.write_text(body_new, encoding="utf-8")
    old_ts = 1_700_000_000
    new_ts = 1_800_000_000
    import os
    os.utime(old, (old_ts, old_ts))
    os.utime(new, (new_ts, new_ts))

    records, diagnostics = manifest.records_from_chat_memo_dir(tmp_path, bank_target="external_chatmemo_smoke", min_file_age_seconds=0)

    assert len(records) == 1
    assert records[0]["document_id"] == "external-chatmemo::doubao::38419629838080770"
    assert records[0]["metadata"]["source_path"] == str(new)
    assert diagnostics["duplicate_document_ids"] == {"external-chatmemo::doubao::38419629838080770": 2}


def test_openclaw_lcm_records_use_strict_rules_include_dingtalk_and_exclude_noise(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    db = tmp_path / "lcm.db"
    make_openclaw_db(db)

    records, diagnostics = manifest.records_from_openclaw_lcm(
        db,
        bank_target="external_openclaw_smoke",
        min_age_seconds=0,
        max_segment_turns=60,
        max_segment_chars=80000,
        gap_split_hours=6,
    )

    assert diagnostics["source_kind"] == "openclaw_lcm"
    assert diagnostics["excluded_conversations_by_reason"]["session_key_excluded"] == 1
    assert diagnostics["excluded_conversations_by_reason"]["history_aggregate"] == 1
    assert {r["metadata"]["conversation_id"] for r in records} == {"1", "4"}

    main = next(r for r in records if r["metadata"]["conversation_id"] == "1")
    assert main["document_id"] == "external-openclaw::1::seg-001"
    assert main["event_date"] == "2026-04-01 10:00:02"
    assert "System (untrusted)" not in main["content"]
    assert "Conversation info (untrusted metadata)" not in main["content"]
    assert "sender_id" not in main["content"]
    assert "tool output" not in main["content"]
    assert "AEB 单目测速" in main["content"]
    assert main["metadata"]["segment_started_at"] == "2026-04-01 10:00:02"
    assert main["metadata"]["segment_ended_at"] == "2026-04-01 10:00:04"
    assert main["metadata"]["seq_start"] == "2"
    assert main["metadata"]["seq_end"] == "4"
    assert "source:external-openclaw" in main["tags"]
    assert "platform:openclaw" in main["tags"]
    assert main["observation_scopes"]
    assert ["domain:autodrive"] in main["observation_scopes"]

    dingtalk = next(r for r in records if r["metadata"]["conversation_id"] == "4")
    assert dingtalk["metadata"]["session_key"] == "agent:main:dingtalk:direct:170941191029642680"
    assert "钉钉 direct" in dingtalk["content"]


def test_openclaw_lcm_splits_segments_on_time_gap(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    db = tmp_path / "lcm.db"
    make_openclaw_db(db)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.executemany(
        "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
        [
            (105, 1, 5, "user", "六小时后继续讨论 Hindsight 外部导入分段策略。", 10, "2026-04-01 18:30:00"),
            (106, 1, 6, "assistant", "时间间隔过长时应切成新的 segment，避免跨主题污染。", 10, "2026-04-01 18:30:01"),
        ],
    )
    con.commit()
    con.close()

    records, _ = manifest.records_from_openclaw_lcm(db, bank_target="external_openclaw_smoke", min_age_seconds=0, gap_split_hours=6)
    conversation_1 = [r for r in records if r["metadata"]["conversation_id"] == "1"]

    assert [r["document_id"] for r in conversation_1] == [
        "external-openclaw::1::seg-001",
        "external-openclaw::1::seg-002",
    ]
    assert conversation_1[1]["event_date"] == "2026-04-01 18:30:00"


def test_openclaw_cleaner_drops_untrusted_blocks_and_structured_tool_traces():
    manifest = load_module("hindsight_external_manifest")

    cleaned, reason = manifest.clean_openclaw_message_content(
        'Conversation info (untrusted metadata): ```json\n{"sender_id":"170"}\n```\n'
        '[{"type":"thinking","thinking":"secret reasoning"},{"type":"toolCall","name":"terminal"},{"type":"text","text":"最终给用户看的回答"}]'
    )

    assert reason is None
    assert cleaned == "最终给用户看的回答"
    assert "sender_id" not in cleaned
    assert "secret reasoning" not in cleaned
    assert "toolCall" not in cleaned


def test_external_tag_rules_are_conservative_for_long_autodrive_transcripts():
    manifest = load_module("hindsight_external_manifest")
    text = (
        "AEB 单目测速误差长尾，目标测距误差5%，自车航向角噪声0.005rad。"
        "这里只是顺带提到 paper、claim、recall 这些英文词，不是在讨论写文章、法律文本或知识库。"
    )

    tags = manifest.semantic_tags_for_text(text, "在L2级别智能驾驶里，目标测距误差怎么建模", "ChatGPT")

    assert "domain:autodrive" in tags
    assert "domain:patent" not in tags
    assert "domain:paper" not in tags
    assert "domain:hindsight" not in tags
    assert "topic:memory-management" not in tags



def test_markdown_artifact_parser_preserves_dynamic_sections_and_items(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    report = tmp_path / "周报_2026-05-16.md"
    report.write_text(
        "\n".join([
            "# 周报 - 2026-05-16",
            "",
            "## 本周工作总结",
            "",
            "### 算法交付",
            "",
            "1. 外厂主目标测距测速误差对比完成，融合版本整体最优，车辆中近距接近达标。",
            "2. RAW夜间无光场景行人检测距离比H1-F RGB设备长10m，采集处理暗光数据7000+例。",
            "",
            "### 风险与不足",
            "",
            "- Z1座舱开启AVM和仪表后CPU跑满，ADAS暂调至15帧。",
        ]),
        encoding="utf-8",
    )

    records = manifest.records_from_markdown_file(report, bank_target="hermes")

    assert records
    assert {r["metadata"]["source_kind"] for r in records} == {"markdown_artifact_md"}
    assert {r["metadata"]["artifact_type"] for r in records} == {"weekly_report"}
    assert {r["metadata"]["report_date"] for r in records} == {"2026-05-16"}
    item_records = [r for r in records if r["metadata"]["record_kind"] == "item"]
    assert len(item_records) == 3
    assert [r["metadata"]["section_title"] for r in item_records] == ["算法交付", "算法交付", "风险与不足"]
    assert any("7000+" in r["content"] and "Section-Path: 周报 - 2026-05-16 > 本周工作总结 > 算法交付" in r["content"] for r in item_records)
    assert any("source:external-markdown-artifact" in r["tags"] for r in records)
    assert any("topic:weekly-report" in r["tags"] for r in records)


def test_markdown_artifact_dir_records_and_rehydrate(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    runner = load_module("hindsight_external_retain_runner")
    doc = tmp_path / "方案设计.md"
    doc.write_text(
        "# 方案设计\n\n## 验收标准\n\n- recall smoke 必须命中具体 section 和 item。\n",
        encoding="utf-8",
    )

    records, diagnostics = manifest.records_from_markdown_artifacts([tmp_path], bank_target="hermes", min_file_age_seconds=0)
    assert diagnostics["source_kind"] == "markdown_artifact_md"
    assert diagnostics["files_seen"] == 1
    assert len(records) >= 2

    paths = manifest.write_manifest(records, tmp_path / "manifests", include_content=False)
    loaded = runner.load_manifest(paths["manifest"])
    target = next(r for r in loaded if (r.get("metadata") or {}).get("record_kind") == "item")
    item = runner.record_to_memory_item(target)
    assert item["context"] == "external_markdown_artifact"
    assert item["document_id"].startswith("external-md-artifact::")
    assert "recall smoke" in item["content"]
    assert item["metadata"]["source_kind"] == "markdown_artifact_md"


def test_conversation_markdown_discovery_uses_hermes_write_file_evidence_and_skips_cron_or_reads(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    produced = tmp_path / "produced.md"
    produced.write_text("# Produced Doc\n\n- durable item from write_file.\n", encoding="utf-8")
    read_only = tmp_path / "read-only.md"
    read_only.write_text("# Read Only\n", encoding="utf-8")
    cron_doc = tmp_path / "cron.md"
    cron_doc.write_text("# Cron Doc\n", encoding="utf-8")

    (sessions / "session_20260501_ok.json").write_text(json.dumps({
        "session_id": "20260501_ok",
        "platform": "cli",
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"path": str(produced), "content": "# Produced Doc"})}}]},
            {"role": "tool", "name": "search_files", "content": json.dumps({"files": [str(read_only)]})},
            {"role": "assistant", "content": f"直接看已有文件 `{read_only}`，不是新写入。"},
        ],
    }), encoding="utf-8")
    (sessions / "session_cron_skip.json").write_text(json.dumps({
        "session_id": "cron_skip",
        "platform": "cron",
        "messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"path": str(cron_doc), "content": "# Cron"})}}]}],
    }), encoding="utf-8")

    paths, diagnostics = manifest.discover_markdown_paths_from_hermes_sessions([sessions], min_file_age_seconds=0)

    assert paths == [produced.resolve()]
    assert diagnostics["sessions_seen"] == 2
    assert diagnostics["sessions_skipped_cron"] == 1


def test_conversation_markdown_discovery_uses_openclaw_write_outputs_and_skips_missing(tmp_path):
    manifest = load_module("hindsight_external_manifest")
    db = tmp_path / "lcm.db"
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("CREATE TABLE conversations (conversation_id INTEGER PRIMARY KEY, session_id TEXT, session_key TEXT, title TEXT, created_at TEXT, updated_at TEXT, active INTEGER)")
    cur.execute("CREATE TABLE messages (message_id INTEGER PRIMARY KEY, conversation_id INTEGER, seq INTEGER, role TEXT, content TEXT, token_count INTEGER, created_at TEXT)")
    cur.execute("INSERT INTO conversations VALUES (1,'s','agent:main:main',NULL,'2026-05-01','2026-05-01',0)")
    produced = tmp_path / "openclaw-produced.md"
    produced.write_text("# OpenClaw Produced\n\n- 周报条目。\n", encoding="utf-8")
    cur.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?,?)", [
        (1, 1, 1, "tool", f"Successfully wrote 42 bytes to {produced}", 1, "2026-05-01 10:00:00"),
        (2, 1, 2, "tool", "Successfully wrote 42 bytes to /does/not/exist.md", 1, "2026-05-01 10:00:01"),
        (3, 1, 3, "assistant", "读取 `/does/not/import.md` 作为参考", 1, "2026-05-01 10:00:02"),
    ])
    con.commit(); con.close()

    paths, diagnostics = manifest.discover_markdown_paths_from_openclaw_lcm(db, min_file_age_seconds=0, allowed_roots=[tmp_path])

    assert paths == [produced.resolve()]
    assert diagnostics["candidate_mentions"] >= 2
    assert diagnostics["paths_found"] == 1


def test_external_tag_rules_detect_real_hindsight_and_paper_topics():
    manifest = load_module("hindsight_external_manifest")

    hindsight_tags = manifest.semantic_tags_for_text(
        "Hermes 内置的长期记忆有几种？Hindsight memory provider、retain、consolidation 怎么选？",
        "hermes内置的长期记忆有好几种，各有什么特点",
        "ChatGPT",
    )
    paper_tags = manifest.semantic_tags_for_text(
        "我想写一篇文章投science或其子刊，主要内容是自动驾驶世界模型，请提投稿建议。",
        "我想写一篇文章投science或其子刊",
        "ChatGPT",
    )

    assert {"domain:hindsight", "topic:memory-management", "topic:native-consolidation"}.issubset(set(hindsight_tags))
    assert "domain:paper" in paper_tags
    assert "domain:autodrive" in paper_tags
    assert "domain:patent" not in paper_tags


def test_external_tag_rules_single_patent_mention_is_not_patent_domain():
    manifest = load_module("hindsight_external_manifest")

    tags = manifest.semantic_tags_for_text(
        "九识无人车技术路线，需查官网、专利及报道。",
        "新石器x3/x6使用毫米波雷达和超声波雷达吗",
        "豆包",
    )

    assert "domain:autodrive" not in tags  # no strong autodrive keyword in this short fixture
    assert "domain:patent" not in tags


def test_diverse_sample_records_collapses_repeated_titles_and_round_robins_topics():
    manifest = load_module("hindsight_external_manifest")
    records = [
        {"document_id": "a1", "action": "production", "tags": ["domain:autodrive"], "content_chars": 1000, "estimated_retain_chunks": 1, "metadata": {"title": "同一个AEB主题"}},
        {"document_id": "a2", "action": "production", "tags": ["domain:autodrive"], "content_chars": 500, "estimated_retain_chunks": 1, "metadata": {"title": "同一个AEB主题"}},
        {"document_id": "h1", "action": "production", "tags": ["domain:hindsight", "topic:memory-management"], "content_chars": 800, "estimated_retain_chunks": 1, "metadata": {"title": "Hindsight 本地部署"}},
        {"document_id": "p1", "action": "production", "tags": ["domain:paper"], "content_chars": 700, "estimated_retain_chunks": 1, "metadata": {"title": "Science 投稿"}},
        {"document_id": "m1", "action": "manual_review", "tags": [], "content_chars": 1, "estimated_retain_chunks": 1, "metadata": {"title": "人工审核"}},
    ]

    sample = manifest.diverse_sample_records(records, limit=3, action="production")

    assert [r["document_id"] for r in sample] == ["a2", "h1", "p1"]


def test_write_manifest_allocates_unique_paths_within_same_second(tmp_path, monkeypatch):
    manifest = load_module("hindsight_external_manifest")

    class FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            from datetime import datetime
            return datetime(2026, 5, 18, 21, 44, 39, tzinfo=tz)

    monkeypatch.setattr(manifest, "datetime", FixedDateTime)
    records = [{"document_id": "doc", "action": "production", "metadata": {}, "tags": [], "observation_scopes": []}]

    first = manifest.write_manifest(records, tmp_path, include_content=False)
    second = manifest.write_manifest(records, tmp_path, include_content=False)

    assert first["manifest"] != second["manifest"]
    assert first["manifest"].exists()
    assert second["manifest"].exists()
