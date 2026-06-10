# External Import — Document Granularity & Third-Party Design Notes

## The Granularity Problem

Hindsight's fact extraction from external imports (openclaw lcm.db, chat-memo txt files) operates at the **sub-document level** — it extracts individual knowledge points, preferences, decisions, tooling lessons from the content. This is good for recall of atomic facts, but **does NOT preserve whole-document integrity**.

Observed behavior (2026-05-19):
- 145 external documents were imported into `hermes` bank (92 chat_memo_txt + 53 openclaw_lcm)
- Consolidation completed, producing observations
- But a **weekly report** written and saved by openclaw could NOT be recalled as a complete document from hindsight
- The report's individual facts (project items, format preferences) WERE extractable individually

**Root cause:** Hindsight retain processes documents chunk-by-chunk via `retain_chunk_size=8000` with extraction instructions to produce 3-5 facts per chunk. It is optimized for memory-granularity extraction, not document-preservation.

## Implications

- If external content needs to be **searchable as whole documents** (full weekly reports, meeting minutes, research notes), hindsight is not the right retrieval path
- For that use case, keep the original files in a known directory (e.g. openclaw's `memory/others/` or a dedicated wiki) and use filesystem search or a separate indexing pipeline
- Hindsight excels at **cross-document synthesis** (e.g. "what did user say about AEB in the last month") but not at "show me the complete weekly report from May 16"

## Third-Party Content Import Design (unimplemented sketch)

The user requested (2026-05-19) a design for extending hindsight to cover:
1. Openclaw conversation records (lcm.db conversations)
2. Chat-memo files on desktop temp directory

**Current state:** Both are partially imported via the existing `hindsight_external_retain_runner.py` pipeline with:
- `chat_memo_txt` adapter — parses chatgpt/doubao memo txt files
- `openclaw_lcm` adapter — parses openclaw lcm.db segments via lossless-claw

**Design considerations for future expansion:**

1. **Source discovery** — should auto-scan known directories (`~/.openclaw/workspace/conversations/`, `~/桌面/temp/chat-memo_*/`) rather than requiring manual manifest building
2. **Format normalization** — each source type needs an adapter (txt memo, lcm.db segments, openclaw conversation files). Current adapters are hardcoded in `hindsight_external_manifest.py`
3. **Deduplication** — lcm.db and conversation txt files may overlap (same dialogue stored in both). Source dedup by content SHA256 already exists
4. **Import routing** — currently imports into `hermes` bank. Consider separate external bank vs single unified bank tradeoff:
   - Single bank (`hermes`): easier cross-reference, but external content dilutes hermes-native dialogue signal
   - Separate bank: cleaner separation, but no cross-bank recall in current hindsight
5. **Tagging** — external imports get semantic tags (currently `external-tag-rules-v7`). Source kind identifies origin (`openclaw_lcm`, `chat_memo_txt`)
6. **Granularity control** — if whole-document preservation is needed, consider a parallel document-store index alongside hindsight's fact graph

## Known Adapters

| Source Kind | Adapter | Data Source | Document ID Format |
|---|---|---|---|
| `chat_memo_txt` | `chat-memo-txt-v1` | `~/桌面/temp/chat-memo_*/` txt files | `external-chatmemo::chatgpt::{uuid}` |
| `openclaw_lcm` | `openclaw-lcm-v1` | `~/.openclaw/lcm.db` | `external-openclaw::{session_id}::seg-{n}` |

## Submit State Tracking

The file at `~/.hermes/hindsight/external_import/submit_state.json` tracks:
- per-document status (`production`, `smoke`, `failed`)
- content SHA256 (dedup key)
- last submit manifest path and timestamp
- bank assignment
- adapter and cleaning version

Two-tier submit states exist:
- `submit_state.json` — production run targeting `hermes` bank
- `submit_state_10pct_*.json` — 10% smoke test runs targeting separate test banks