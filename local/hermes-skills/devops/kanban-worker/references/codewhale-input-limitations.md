# Codewhale (CodeWhale) Interactive Pane — Input Limitations

## Critical User Experience Issue

When a codewhale pane has an active tool call (`run running` / `shell running` / `read running` / `write running`), the input box is **completely locked** — it does not accept any stdin input. All keystrokes are silently discarded.

**UI deception**: The placeholder `编写任务或使用 /。` remains visible during tool call execution, making users believe they can type. But the input box is non-functional.

## Impact on Users

Users trying to type in a busy codewhale pane will lose **80%+ of their input** because 80%+ of the time the pane is executing tool calls. Only the brief gaps between tool calls (<1s typically) accept input.

## What This Is NOT Caused By

- fcitx5/IME — English input is also discarded identically
- Watcher inject — watcher only injects once per task
- `zellij dump-screen` — only 7ms per call, every 30s
- PTY buffer issues — this is codewhale's internal Ink TUI behavior

## Workarounds

1. **Wait for tool call completion** — status bar changes from `⏳ shell running` to `✓ done`
2. **Press Escape to interrupt** — cancels current tool call (may affect task progress)
3. **Cannot be fixed at watcher level** — requires codewhale upstream change

## Verification Tests (codewhale v0.8.55)

| Test | Pane State | Command | Result |
|------|------------|---------|--------|
| Idle pane | No tool call | `zellij action write-chars -p 3 'hello'` | ✅ Text appears |
| Busy pane | `shell running` | `zellij action write-chars -p 1 'hello'` | ❌ Text discarded |
| Busy + Tab | `shell running` | `zellij action write-chars -p 1 $'\t' 'test'` | ❌ Still discarded |
| Chinese input | Idle | `zellij action write-chars -p 3 '中文测试'` | ✅ Full text accepted |

## Related Documentation

- `upgrading-codewhale` skill, 陷阱 9g
- `upgrading-codewhale/references/codewhale-v0853-watcher-pitfalls.md` section 9

**Date discovered**: 2026-06-10
**Affected versions**: CodeWhale v0.8.55 (likely all v0.8.x)