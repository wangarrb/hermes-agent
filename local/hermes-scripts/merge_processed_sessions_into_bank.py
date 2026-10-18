#!/usr/bin/env python3
"""
将已成功导入到源 bank 的 Hermes session 文件，再导入到目标 bank。
默认用于把 `hermes-sessions` 已导入成功的历史会话合并进 `hermes`。

特性：
- 复用 import_sessions_to_hindsight.py 的清洗/分块/重试逻辑
- 独立 merge 进度文件，支持中断续跑
- fresh/retry/all/auto 四种模式
- 非破坏性：只复制，不删除源 bank

用法：
  python3 merge_processed_sessions_into_bank.py [target_bank] [mode]
示例：
  python3 merge_processed_sessions_into_bank.py hermes auto
  python3 merge_processed_sessions_into_bank.py hermes retry
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

MODULE_PATH = Path.home() / ".hermes" / "scripts" / "import_sessions_to_hindsight.py"
SOURCE_PROGRESS_FILE = Path.home() / ".hermes" / "hindsight" / "import_progress.json"
SESSIONS_DIR = Path.home() / ".hermes" / "sessions"
VALID_MODES = {"auto", "fresh", "retry", "all"}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("hindsight_importer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def unique_list(items):
    return list(dict.fromkeys(items or []))


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print("Usage: python3 merge_processed_sessions_into_bank.py [target_bank] [auto|fresh|retry|all]")
        raise SystemExit(0)

    target_bank = sys.argv[1] if len(sys.argv) > 1 else "hermes"
    requested_mode = sys.argv[2].lower() if len(sys.argv) > 2 else "auto"
    if requested_mode not in VALID_MODES:
        raise SystemExit(f"Unsupported mode: {requested_mode}. Valid: {', '.join(sorted(VALID_MODES))}")

    merge_progress_file = Path.home() / ".hermes" / "hindsight" / f"merge_into_{target_bank}.json"

    importer = load_module(MODULE_PATH)
    importer.BANK_ID = target_bank
    importer.PROGRESS_FILE = merge_progress_file

    source_progress = load_json(SOURCE_PROGRESS_FILE, {"processed": [], "failed": [], "total": 0, "last_run": None})
    source_processed = unique_list(source_progress.get("processed", []))

    if not source_processed:
        raise SystemExit(f"No source processed sessions found in {SOURCE_PROGRESS_FILE}")

    progress = importer.normalize_progress(importer.load_progress())
    save_json(merge_progress_file, progress)

    merged_set = set(progress.get("processed", []))
    failed_map = {x.get("file"): x for x in progress.get("failed", []) if x.get("file")}

    fresh_files = [f for f in source_processed if f not in merged_set and f not in failed_map]
    retry_files = [
        f for f in source_processed
        if f in failed_map and f not in merged_set and importer.is_retry_candidate(failed_map[f].get("error"))
    ]
    mode = importer.pick_mode(requested_mode, fresh_files, retry_files)

    if mode == "fresh":
        to_process = fresh_files
    elif mode == "retry":
        to_process = retry_files
    elif mode == "all":
        to_process = fresh_files + [f for f in retry_files if f not in set(fresh_files)]
    else:
        to_process = []

    print("=" * 60)
    print("Merge imported Hermes sessions into target bank")
    print("=" * 60)
    print(f"Target bank: {target_bank}")
    print(f"Source processed sessions: {len(source_processed)}")
    print(f"Already merged: {len(merged_set)}")
    print(f"Previous merge failures: {len(failed_map)}")
    print(f"Fresh remaining: {len(fresh_files)}")
    print(f"Retryable failed: {len(retry_files)}")

    if not to_process:
        print(f"\nNo sessions to process for mode={mode}.")
        progress["total"] = len(source_processed)
        progress["last_run"] = datetime.now().isoformat()
        save_json(merge_progress_file, importer.normalize_progress(progress))
        return

    print(f"Mode: {mode}")
    print(f"To process: {len(to_process)}")

    importer.ensure_bank()
    batch_count = 0
    success_count = 0
    fail_count = 0

    for i, filename in enumerate(to_process):
        filepath = SESSIONS_DIR / filename
        try:
            with open(filepath, encoding="utf-8") as f:
                session_data = json.load(f)
            success, error = importer.retain_session(filename, session_data)
            if success:
                if filename not in merged_set:
                    progress.setdefault("processed", []).append(filename)
                    merged_set.add(filename)
                importer.remove_failed_entry(progress, filename)
                success_count += 1
                if i % 5 == 0:
                    print(f"[{i+1}/{len(to_process)}] ✓ {filename}")
            else:
                importer.record_failed(progress, filename, error)
                fail_count += 1
                print(f"[{i+1}/{len(to_process)}] ✗ {filename}: {error}")
        except Exception as e:
            importer.record_failed(progress, filename, str(e))
            fail_count += 1
            print(f"[{i+1}/{len(to_process)}] ✗ {filename}: {e}")

        batch_count += 1
        if batch_count >= importer.BATCH_SIZE:
            progress = importer.normalize_progress(progress)
            progress["total"] = len(source_processed)
            progress["last_run"] = datetime.now().isoformat()
            save_json(merge_progress_file, progress)
            batch_count = 0
            if i < len(to_process) - 1:
                importer.time.sleep(importer.DELAY_SECONDS)

    progress = importer.normalize_progress(progress)
    progress["total"] = len(source_processed)
    progress["last_run"] = datetime.now().isoformat()
    save_json(merge_progress_file, progress)

    print("\n" + "=" * 60)
    print("Merge pass complete!")
    print(f"  Target bank: {target_bank}")
    print(f"  Mode: {mode}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Progress saved to: {merge_progress_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
