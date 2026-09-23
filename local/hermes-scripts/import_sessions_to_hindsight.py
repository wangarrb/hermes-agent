#!/usr/bin/env python3
"""
将 Hermes session 文件导入到 Hindsight memory bank
支持中断后继续处理
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import requests
import time

# 配置
HINDSIGHT_API = "http://localhost:8888"
BANK_ID = "hermes-sessions"
SESSIONS_DIR = Path.home() / ".hermes" / "sessions"
PROGRESS_FILE = Path.home() / ".hermes" / "hindsight" / "import_progress.json"
BATCH_SIZE = 10  # 每批处理数量
DELAY_SECONDS = 2  # 每批间隔（避免 API 过载）

def load_progress():
    """加载进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": [], "failed": [], "total": 0, "last_run": None}

def save_progress(progress):
    """保存进度"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_session_files():
    """获取所有 session 文件"""
    files = sorted(SESSIONS_DIR.glob("session_*.json"))
    return [f.name for f in files]

def extract_conversation(session_data):
    """提取对话内容"""
    messages = session_data.get("messages", [])
    conversation = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "user" and content:
            conversation.append(f"User: {content}")
        elif role == "assistant":
            text = content or ""
            reasoning = msg.get("reasoning", "")
            if reasoning:
                text = f"{text}\n[Reasoning: {reasoning}]"
            if text:
                conversation.append(f"Assistant: {text}")
    
    return "\n\n".join(conversation)

def ensure_bank():
    """确保 bank 存在（retain 会自动创建）"""
    try:
        resp = requests.get(
            f"{HINDSIGHT_API}/v1/default/banks",
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            banks = data.get("banks", data) if isinstance(data, dict) else data
            if any(b.get("bank_id") == BANK_ID for b in banks):
                print(f"Bank exists: {BANK_ID}")
            else:
                print(f"Bank will be auto-created on first retain: {BANK_ID}")
            return True
    except Exception as e:
        print(f"Check banks error: {e}")
    
    return True

def retain_session(session_file, session_data):
    """将 session 导入 Hindsight"""
    session_id = session_data.get("session_id", Path(session_file).stem)
    conversation = extract_conversation(session_data)
    
    if not conversation or len(conversation) < 50:
        return False, "Empty or too short"
    
    try:
        resp = requests.post(
            f"{HINDSIGHT_API}/v1/default/banks/{BANK_ID}/memories",
            json={
                "items": [{
                    "content": conversation,
                    "document_id": session_id,
                    "context": "hermes_conversation"
                }]
            },
            timeout=120
        )
        
        if resp.status_code in [200, 201]:
            return True, None
        else:
            return False, f"API error: {resp.status_code} {resp.text[:100]}"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Hermes Sessions → Hindsight Import Tool")
    print("=" * 60)
    
    ensure_bank()
    
    progress = load_progress()
    all_files = get_session_files()
    total = len(all_files)
    
    print(f"\nTotal sessions: {total}")
    print(f"Already processed: {len(progress['processed'])}")
    print(f"Previously failed: {len(progress['failed'])}")
    
    to_process = [f for f in all_files if f not in progress["processed"]]
    
    if not to_process:
        print("\nAll sessions already processed!")
        return
    
    print(f"To process: {len(to_process)}")
    
    batch_count = 0
    success_count = 0
    fail_count = 0
    
    for i, filename in enumerate(to_process):
        filepath = SESSIONS_DIR / filename
        
        try:
            with open(filepath) as f:
                session_data = json.load(f)
            
            success, error = retain_session(filename, session_data)
            
            if success:
                progress["processed"].append(filename)
                success_count += 1
                print(f"[{i+1}/{len(to_process)}] ✓ {filename}")
            else:
                progress["failed"].append({"file": filename, "error": error})
                fail_count += 1
                print(f"[{i+1}/{len(to_process)}] ✗ {filename}: {error}")
            
        except Exception as e:
            progress["failed"].append({"file": filename, "error": str(e)})
            fail_count += 1
            print(f"[{i+1}/{len(to_process)}] ✗ {filename}: {e}")
        
        batch_count += 1
        if batch_count >= BATCH_SIZE:
            progress["total"] = total
            progress["last_run"] = datetime.now().isoformat()
            save_progress(progress)
            batch_count = 0
            
            if i < len(to_process) - 1:
                time.sleep(DELAY_SECONDS)
    
    progress["total"] = total
    progress["last_run"] = datetime.now().isoformat()
    save_progress(progress)
    
    print("\n" + "=" * 60)
    print(f"Import complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()