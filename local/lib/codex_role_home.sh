#!/usr/bin/env bash
# codex_role_home.sh — ensure a per-role CODEX_HOME with shared symlinks.
#
# Called by start-kanban.sh to give each Codex pane its own session directory
# while sharing config, auth, skills and agents from the real ~/.codex.
#
# Usage (sourced or executed):
#   ensure_codex_role_home <real_home> <role_home>
#
# Creates <role_home>/sessions and symlinks shared files/directories from
# <real_home>/.codex, never overwriting existing entries.

ensure_codex_role_home() {
    local real_home="$1"
    local role_home="$2"

    [ -z "$real_home" ] && return 1
    [ -z "$role_home" ] && return 1

    local codex_src="${real_home}/.codex"
    mkdir -p "${role_home}/sessions"

    local f
    for f in config.toml auth.json hooks.json installation_id .personality_migration version.json AGENTS.md RTK.md models_cache.json; do
        [ -e "${codex_src}/$f" ] && [ ! -e "${role_home}/$f" ] && ln -sf "${codex_src}/$f" "${role_home}/$f"
    done

    local d
    for d in claude-skills superpowers skills plugins agents; do
        [ -d "${codex_src}/$d" ] && [ ! -e "${role_home}/$d" ] && ln -sf "${codex_src}/$d" "${role_home}/$d"
    done

    return 0
}
