#!/usr/bin/env python3
"""微信 @角色 消息 → zellij pane 注入 → 返回输出。

用法:
  python3 scripts/wechat_inject.py --role implementer --message "查一下任务状态"
  echo "@implementer 查一下任务状态" | python3 scripts/wechat_inject.py --from-stdin

依赖: zellij session (默认 kanban-egomotion4d)
"""

import subprocess, sys, time, re, os, argparse

ZELLIJ_SESSION = os.environ.get("ZELLIJ_SESSION", "kanban-egomotion4d")

ROLE_PANE_MAP = {
    "planner":     "terminal_0",
    "implementer": "terminal_1",
    "reviewer":    "terminal_2",
    "designer":    "terminal_3",
    "coordinator": "terminal_4",
}

def zellij(*args):
    subprocess.run(["zellij", "--session", ZELLIJ_SESSION, "action"] + list(args),
                   capture_output=True, timeout=10)

def dump(pane: str) -> str:
    r = subprocess.run(["zellij", "--session", ZELLIJ_SESSION, "action",
                        "dump-screen", "-p", pane, "--full"],
                       capture_output=True, text=True, timeout=5)
    return r.stdout

def is_idle(pane: str) -> bool:
    lines = [l for l in dump(pane).splitlines() if l.strip()]
    if not lines:
        return False
    tail = "\n".join(lines[-5:]).lower()
    busy = any(m in tail for m in
               ("⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏","thinking","running"))
    prompt = any(m in tail for m in ("› ", "> ", "❯", "hermes ❯", "implementer ❯",
                                       "planner ❯", "reviewer ❯", "critic ❯", "coordinator ❯"))
    return prompt and not busy

def inject(pane: str, text: str):
    zellij("write-chars", "-p", pane, text)
    time.sleep(0.3)
    zellij("send-keys", "-p", pane, "Enter")

def extract_response(before: str, after: str) -> str:
    before_set = set(before.splitlines())
    return "\n".join(l for l in after.splitlines()
                     if l not in before_set).strip()

def wait_response(pane: str, before: str, timeout: int = 180) -> str:
    start = time.time()
    last = before
    idle_count = 0
    while time.time() - start < timeout:
        time.sleep(2)
        cur = dump(pane)
        if is_idle(pane):
            idle_count += 1
            if idle_count >= 2:
                return extract_response(before, cur)
        else:
            idle_count = 0
            last = cur
    return extract_response(before, last) + "\n\n[超时]"

def main():
    p = argparse.ArgumentParser(description="微信消息注入 zellij pane")
    p.add_argument("--role", choices=list(ROLE_PANE_MAP))
    p.add_argument("--message", type=str)
    p.add_argument("--from-stdin", action="store_true",
                   help="从 stdin 读取 @角色 消息")
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    if args.from_stdin:
        raw = sys.stdin.read().strip()
        m = re.match(r'@(\w+)\s+(.+)', raw, re.DOTALL)
        if not m:
            print("格式错误，需要 @角色 消息", file=sys.stderr)
            sys.exit(1)
        role, message = m.group(1), m.group(2).strip()
    elif args.role and args.message:
        role, message = args.role, args.message
    else:
        p.print_help()
        sys.exit(1)

    if role not in ROLE_PANE_MAP:
        print(f"未知角色: {role}", file=sys.stderr)
        sys.exit(1)

    pane = ROLE_PANE_MAP[role]

    if not is_idle(pane):
        print(f"[{role}] 正忙，拒绝注入", file=sys.stderr)
        sys.exit(2)

    before = dump(pane)
    inject(pane, message)
    print(f"[{role}] 已注入，等待响应...", file=sys.stderr)
    response = wait_response(pane, before, args.timeout)
    print(response)

if __name__ == "__main__":
    main()