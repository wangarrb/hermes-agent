#!/bin/bash
# Wiki daily git snapshot — stage all tracked changes and commit.
# Runs as a no_agent cron job in the main Hermes profile.
# Exits 0 on success (including no-change), 1 on error.
# Silent when nothing changed (clean exit, no output).

set -euo pipefail

WIKI_DIR="/home/wyr/wiki"
cd "$WIKI_DIR"

# Ensure git identity
git config user.name "wyr" 2>/dev/null || true
git config user.email "wyr@local" 2>/dev/null || true

# Stage all changes (respecting .gitignore)
git add -A

# Check if there are staged changes
if git diff --cached --quiet; then
    # No changes — exit silently
    exit 0
fi

# Count changes for commit message
CHANGED=$(git diff --cached --name-only | wc -l)
ADDED=$(git diff --cached --diff-filter=A --name-only | wc -l)
DELETED=$(git diff --cached --diff-filter=D --name-only | wc -l)
MODIFIED=$(git diff --cached --diff-filter=M --name-only | wc -l)

TODAY=$(date +%Y-%m-%d)
git commit -m "wiki snapshot ${TODAY}: ${CHANGED} files (${ADDED} added, ${MODIFIED} modified, ${DELETED} deleted)" --quiet

echo "wiki snapshot ${TODAY}: ${CHANGED} files committed"
exit 0
