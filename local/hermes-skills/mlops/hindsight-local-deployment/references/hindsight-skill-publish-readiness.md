# Hindsight skill publish-readiness audit

Use this when deciding whether `hindsight-local-deployment` can be shared beyond the local operator environment.

## Verdict pattern

Split the answer into two gates:

1. Local/internal operational release: whether the scripts and workflow are safe to run in the current machine.
2. Public skill publication: whether the skill is self-contained, portable, and maintainable for another Hermes install.

A workflow can be safe for local release but still not publishable as a skill.

## Checks

1. Validate SKILL.md mechanics:
   - frontmatter starts at byte 0 and closes correctly;
   - `name` and `description` exist;
   - `description` <= 1024 chars;
   - content size is below `MAX_SKILL_CONTENT_CHARS=100000`.
2. Require release-quality frontmatter, even if not validator-enforced:
   - `version`, `author`, `license`;
   - `metadata.hermes.tags` and `metadata.hermes.related_skills`.
3. Keep size headroom:
   - do not publish with only a few bytes/chars below the 100k limit;
   - target 30k-50k chars in SKILL.md and move detailed runbooks to `references/`.
4. Verify all referenced files exist.
5. Verify self-contained scripts:
   - commands in SKILL.md/reference docs that call `~/.hermes/scripts/<script>.py` must either be packaged under `scripts/`, installed by a documented installer, or explicitly marked local-only.
6. Scan for portability blockers:
   - `$HOME`, `/tmp` shadowing risks, local PostgreSQL paths, local container names, provider-specific env vars, and private benchmark/report paths.
7. Scan for secret-like material before publishing:
   - `sk-...` including dots/dashes/underscores;
   - `api_key/token/secret/password/passwd` assignments;
   - bearer tokens;
   - AWS-style `AKIA...` ids.
8. For proposal-review release paths, verify the governance boundary:
   - proposal generation/review is local-file only;
   - advisory LLM calls require explicit confirm;
   - deterministic/secret-blocked proposals are not sent to external LLM;
   - final human go/no-go and rollback/quarantine remain separate.

## 2026-05-13 audit lesson

`hindsight-local-deployment` was safe for local/internal use after strict preflight and tests, but not ready for public publication because:

- SKILL.md was ~99,979 chars, leaving almost no headroom under the 100k limit.
- frontmatter lacked `version`, `author`, `license`, and `metadata.hermes` fields.
- the skill referenced many runtime scripts in `~/.hermes/scripts/` that were not packaged under the skill `scripts/` directory.
- many `$HOME` and local Hindsight/Postgres/container assumptions remained.
- the umbrella had grown into an operational archive; public release should split or slim it into class-level skills with concise SKILL.md files and detailed references.

## 2026-05-13 remediation pattern

The later publish pass converted the skill to `PASS` by applying these concrete fixes:

1. Move the old oversized umbrella `SKILL.md` out of the published skill tree or into a non-loaded backup location, then rewrite `SKILL.md` to a concise class-level guide (~17k chars) with full release frontmatter.
2. Package every runtime script used by the advertised workflows under `scripts/`, including transitive local imports. In this case, `hindsight_conflict_core.py` and `import_sqlite_to_hindsight.py` were required in addition to the obvious pipeline/proposal scripts.
3. Add an installer script (`scripts/install_hindsight_pipeline_scripts.sh`) that copies packaged scripts to `$HERMES_HOME/scripts`, backs up overwritten files, initializes config/tuning, and never edits `.env`.
4. Patch packaged tests to locate the packaged script directory via `Path(__file__).resolve().parents[1]`, not a local absolute path such as `/home/<user>/.hermes/scripts`.
5. Replace local/private paths in published code and references (`/home/<user>`, profile names like `profiles/<name>`) with `$HOME`, `$HERMES_HOME`, or `<profile>`.
6. Remove test/build artifacts from the skill tree after validation: `__pycache__/`, `.pytest_cache/`, tar backups, `.bak`, `.tmp`, `.orig`, `.swp`, and similar.
7. Re-run validation after cleanup because `pytest` can recreate `__pycache__`; use `PYTHONDONTWRITEBYTECODE=1` and `pytest -p no:cacheprovider` for package-clean tests.
8. Do an independent final review. A useful blocker check is: frontmatter/size, referenced files, packaged transitive scripts, hardcoded local paths, build artifacts, secret-like strings, and tests/temp-install.

Recommended answer shape:

- "Local/internal: yes/no" with evidence.
- "Public skill publication: yes/no" with blockers/caveats.
- A short remediation checklist, not a long rewrite plan.
