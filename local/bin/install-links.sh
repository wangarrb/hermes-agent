#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backup_root="${HERMES_LOCAL_LINK_BACKUP_ROOT:-/home/wyr/.local/state/hermes-agent/local-link-backups}"
stamp="$(date +%Y%m%d-%H%M%S)"
dry_run=0

usage() {
    cat <<'EOF'
Usage: install-links.sh [--dry-run]

Install symlinks from live local paths back to this Hermes checkout's local/
source tree. Existing non-matching files/directories are moved to a timestamped
backup directory before linking.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            dry_run=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "install-links.sh: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

run() {
    if [[ "$dry_run" -eq 1 ]]; then
        printf 'dry-run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

backup_path_for() {
    local path="$1"
    local safe
    safe="${path#/}"
    safe="${safe//\//__}"
    printf '%s/%s/%s' "$backup_root" "$stamp" "$safe"
}

link_path() {
    local source="$1"
    local target="$2"

    if [[ ! -e "$source" && ! -L "$source" ]]; then
        echo "missing source: $source" >&2
        exit 1
    fi

    local current=""
    if [[ -L "$target" ]]; then
        current="$(readlink "$target")"
        if [[ "$current" == "$source" ]]; then
            echo "ok: $target -> $source"
            return
        fi
    elif [[ -e "$target" ]]; then
        current=""
    fi

    if [[ -e "$target" || -L "$target" ]]; then
        local backup
        backup="$(backup_path_for "$target")"
        echo "backup: $target -> $backup"
        run mkdir -p "$(dirname "$backup")"
        run mv -T "$target" "$backup"
    fi

    echo "link: $target -> $source"
    run mkdir -p "$(dirname "$target")"
    run ln -s "$source" "$target"
}

link_path "$repo_root/local/bin/start-kanban.sh" /home/wyr/bin/start-kanban.sh
link_path "$repo_root/local/bin/stop-kanban.sh" /home/wyr/bin/stop-kanban.sh
link_path "$repo_root/local/bin/listen-kanban" /home/wyr/bin/listen-kanban
link_path "$repo_root/local/bin/reset-kanban" /home/wyr/bin/reset-kanban
link_path "$repo_root/local/bin/codex" /home/wyr/.local/bin/codex
link_path "$repo_root/local/bin/zellij" /home/wyr/.local/bin/zellij

link_path "$repo_root/local/zellij/config.kdl" /home/wyr/.config/zellij/config.kdl
link_path "$repo_root/local/zellij/layouts/hermes.kdl" /home/wyr/.config/zellij/layouts/hermes.kdl
link_path "$repo_root/local/zellij/layouts/hermes.yaml" /home/wyr/.config/zellij/layouts/hermes.yaml
link_path "$repo_root/local/zellij/layouts/kanban-launcher.kdl" /home/wyr/.config/zellij/layouts/kanban-launcher.kdl
link_path "$repo_root/local/zellij/layouts/multi-agent-codex-interactive-backup.kdl" /home/wyr/.config/zellij/layouts/multi-agent-codex-interactive-backup.kdl
link_path "$repo_root/local/zellij/layouts/multi-agent-planner-hermes-backup.kdl" /home/wyr/.config/zellij/layouts/multi-agent-planner-hermes-backup.kdl
link_path "$repo_root/local/zellij/layouts/multi-agent.kdl" /home/wyr/.config/zellij/layouts/multi-agent.kdl

link_path "$repo_root/local/hermes-scripts" /home/wyr/.hermes/scripts

link_path "$repo_root/local/hermes-skills/devops/kanban-orchestrator" /home/wyr/.hermes/skills/devops/kanban-orchestrator
link_path "$repo_root/local/hermes-skills/devops/kanban-worker" /home/wyr/.hermes/skills/devops/kanban-worker
link_path "$repo_root/local/hermes-skills/mlops/hindsight-consolidation-operations" /home/wyr/.hermes/skills/mlops/hindsight-consolidation-operations
link_path "$repo_root/local/hermes-skills/mlops/hindsight-local-deployment" /home/wyr/.hermes/skills/mlops/hindsight-local-deployment
link_path "$repo_root/local/hermes-skills/mlops/hindsight-external-import-design" /home/wyr/.hermes/skills/mlops/hindsight-external-import-design

link_path "$repo_root/local/agent-plugins/marketplace.json" /home/wyr/.agents/plugins/marketplace.json
link_path "$repo_root/plugins/kanban/codex_listener" /home/wyr/.agents/plugins/codex-kanban
link_path "$repo_root/plugins/kanban/codex_listener" /home/wyr/plugins/codex-kanban

link_path "$repo_root/plugins/kanban/codex_listener/bin/codex-kanban-interactive" /home/wyr/.local/bin/codex-kanban-interactive
link_path "$repo_root/plugins/kanban/codex_listener/bin/codex-kanban-listen" /home/wyr/.local/bin/codex-kanban-listen
link_path "$repo_root/plugins/kanban/deepseek_listener/bin/deepseek-kanban-interactive" /home/wyr/.local/bin/deepseek-kanban-interactive
link_path "$repo_root/plugins/kanban/deepseek_listener/bin/deepseek-kanban-interactive" /home/wyr/.local/bin/codewhale-kanban-interactive
link_path "$repo_root/plugins/kanban/deepseek_listener/bin/deepseek-kanban-listen" /home/wyr/.local/bin/deepseek-kanban-listen
link_path "$repo_root/plugins/kanban/reasonix_listener/bin/reasonix-kanban-interactive" /home/wyr/.local/bin/reasonix-kanban-interactive

echo "local Hermes orchestration links installed"
