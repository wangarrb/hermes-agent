# Hindsight Skill Bundle Export and Sharing

Use this reference when packaging the Hindsight skills and helper scripts for another Hermes user/machine.

## Scope

Export only the reusable skill library content:

- `hindsight-local-deployment/`
- `hindsight-consolidation-operations/`
- their `SKILL.md`, `references/`, `templates/`, `scripts/`, and packaged tests
- import README, manifest, checksum, and export report

Do not include runtime state:

- `.env` or credential files
- API keys, bearer tokens, passwords, `sk-*` strings
- Hindsight database dumps or bank exports
- session history, retain manifests, tool traces, pipeline output, logs, caches, `__pycache__`, `.pytest_cache`
- personal/home absolute paths, staff/chat IDs, or machine-specific service files unless explicitly templated

## Export Workflow

1. Stage a clean temporary bundle directory outside the live skill tree.
2. Copy only the two intended skill directories and prune caches/build artifacts.
3. Add `README_IMPORT.md` with exact import commands and post-import validation.
4. Add `EXPORT_REPORT.json` containing:
   - source skill names and versions/paths,
   - export timestamp,
   - included file count,
   - privacy scan counts,
   - test/compile status,
   - archive sha256 policy.
5. Generate `MANIFEST.sha256` over every regular file in the staged bundle before compression.
6. Create `tar.gz` and a sibling `.sha256` file.
7. Verify the final archive from a fresh temp extraction. If privacy or verification fails, do not delete the draft silently; move it to a clearly named quarantine such as `_drafts_privacy_blocked/` and produce a clean final archive.
8. Optionally update `*-latest.tar.gz` and `*-latest.tar.gz.sha256` symlinks for local convenience.

Do not try to store the final tarball's own SHA256 inside `EXPORT_REPORT.json` inside that same tarball: changing the report changes the archive hash. Put the exact final archive hash in the sibling `.tar.gz.sha256` file and, if useful, an outside `*.EXPORT_REPORT.json` sidecar that is not part of the archive. Inside-archive report should say `see sibling .tar.gz.sha256`.

## Privacy and Portability Scan

Before compression, scan the staged tree for:

```text
.env
api_key|apikey|token|password|secret|bearer
sk-[A-Za-z0-9_-]{20,}
ghp_[A-Za-z0-9_]{20,}
/home/<local-user>
staffId|ding|钉钉|account_id
__pycache__|.pytest_cache|.mypy_cache
```

A keyword hit is not automatically a blocker if it appears in security guidance, templates, or tests that intentionally assert redaction behavior. Inspect hits and record `credential_like=0` for real leaked credentials, not merely for the word `token` inside a warning.

Treat `/home/hindsight` as an allowed container-internal user path, not a leaked local home path. Treat `/home/wyr`, `/home/<real-user>`, or any host-specific absolute path as blocking unless explicitly templated. Path-level cache hits (`__pycache__`, `.pytest_cache`, `.mypy_cache`) are blocking for a shareable archive even if the source text itself is safe.

## Verification Workflow

Verify from a fresh temporary extraction directory, not only from the source tree:

```bash
sha256sum -c /path/to/archive.tar.gz.sha256
TMPDIR=$(mktemp -d)
tar -xzf /path/to/archive.tar.gz -C "$TMPDIR"
cd "$TMPDIR"/<bundle-root>/skills/mlops/hindsight-local-deployment

python3 - <<'PY'
from pathlib import Path
import re, yaml
p = Path('SKILL.md')
text = p.read_text(encoding='utf-8')
assert text.startswith('---')
m = re.search(r'\n---\s*\n', text[3:])
assert m
fm = yaml.safe_load(text[3:3+m.start()])
assert fm['name'] == 'hindsight-local-deployment'
assert len(text) < 100_000
print('skill ok', len(text))
PY

python3 -m py_compile scripts/*.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/tests -q
```

Repeat at least the frontmatter/compile checks for `hindsight-consolidation-operations` if its scripts/tests are included separately.

## Import README Shape

Give recipients commands that do not assume the sender's home path:

```bash
ARCHIVE=/path/to/hindsight-offline-skills-bundle.tar.gz
TMPDIR=$(mktemp -d)
tar -xzf "$ARCHIVE" -C "$TMPDIR"
mkdir -p "$HOME/.hermes/skills/mlops"
cp -a "$TMPDIR"/<bundle-root>/skills/mlops/hindsight-local-deployment "$HOME/.hermes/skills/mlops/"
cp -a "$TMPDIR"/<bundle-root>/skills/mlops/hindsight-consolidation-operations "$HOME/.hermes/skills/mlops/"
rm -rf "$TMPDIR"
```

Then tell them to start a new Hermes session or run an equivalent skill reload/reset, and verify:

```bash
hermes skills list | grep hindsight
```

## Share Message

Share only:

- the `.tar.gz` archive
- the `.tar.gz.sha256` file

State plainly that the archive contains Hermes skills and helper scripts only. It does not contain Hindsight service binaries, Docker images, database state, `.env`, API keys, or LLM provider credentials. The recipient must configure their own Hindsight service and provider credentials.

On CLI, do not emit `MEDIA:/path` delivery tags; list absolute paths as plain text.