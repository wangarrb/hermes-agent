#!/usr/bin/env python3
"""Agent Router: @codex / @claude 统一路由网关
支持流式输出、上下文注入、不区分大小写触发。
"""
import os
import sys
import argparse
import json
import httpx

def get_env_or_fail(var_name: str) -> str:
    val = os.environ.get(var_name, "").strip()
    if not val:
        print(f"\n[错误] 缺少环境变量: {var_name}", file=sys.stderr)
        sys.exit(1)
    return val

def run_codex(query: str, context: str):
    """路由到 CCH / OpenAI Responses API (gpt-5.4)"""
    api_key = get_env_or_fail("CCH_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://cch.jmadas.com/v1").rstrip("/")
    if "/v1" not in base_url:
        base_url += "/v1"

    system_prompt = "You are a helpful coding assistant running in a terminal. Be concise and direct."
    if context:
        system_prompt += f"\n\n## Conversation Context:\n{context}"

    payload = {
        "model": "gpt-5.4",
        "input": [{"role": "user", "content": f"[SYSTEM]: {system_prompt}\n\nUser: {query}"}],
        "stream": True,
        "max_output_tokens": 4096,
        "reasoning": {"effort": "medium"}
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    url = f"{base_url}/responses"

    try:
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=60.0) as response:
            if response.status_code != 200:
                print(f"\n[HTTP {response.status_code}] {response.text[:200]}", file=sys.stderr)
                sys.exit(1)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                        if evt.get("type") == "response.output_text.delta":
                            print(evt.get("delta", ""), end="", flush=True)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"\n[Codex Error] {e}", file=sys.stderr)
        sys.exit(1)

def run_claude(query: str, context: str):
    """路由到 Anthropic Claude API"""
    api_key = get_env_or_fail("ANTHROPIC_API_KEY")

    system_prompt = "You are Claude, a helpful AI assistant. Be concise and direct."
    if context:
        system_prompt += f"\n\n## Conversation Context:\n{context}"

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": query}],
        "stream": True
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    try:
        with httpx.stream("POST", "https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=60.0) as response:
            if response.status_code != 200:
                print(f"\n[HTTP {response.status_code}] {response.text[:200]}", file=sys.stderr)
                sys.exit(1)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        evt = json.loads(data)
                        if evt.get("type") == "content_block_delta":
                            print(evt["delta"].get("text", ""), end="", flush=True)
                    except:
                        continue
    except Exception as e:
        print(f"\n[Claude Error] {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Agent Router")
    parser.add_argument("target", choices=["codex", "claude"], help="Target agent")
    parser.add_argument("query", help="User query")
    parser.add_argument("--context", default="", help="Conversation context (optional)")
    args = parser.parse_args()

    print(f"\n>>> Routing to {args.target}...\n", flush=True)
    if args.target == "codex":
        run_codex(args.query, args.context)
    elif args.target == "claude":
        run_claude(args.query, args.context)
    print("\n\n>>> Done.", flush=True)

if __name__ == "__main__":
    main()
