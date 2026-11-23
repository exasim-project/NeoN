#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 NeoN authors
#
# List branches whose last commit is older than a given age, and the subset
# already merged into the base branch.
#
# Usage: scripts/staleBranches.sh [-d DAYS] [-b BASE] [-r REMOTE] [-l] [-m]
#   -d DAYS    age threshold in days (default: 60)
#   -b BASE    base branch to check merges against (default: develop)
#   -r REMOTE  remote to scan (default: origin)
#   -l         scan local branches instead of remote ones
#   -m         only print the stale + merged list
#
# Note: only regular/fast-forward merges are detected; squash- or
# rebase-merged branches appear as stale but not merged.

set -euo pipefail

days=60
base=develop
remote=origin
local=0
mergedOnly=0

while getopts "d:b:r:lmh" opt; do
    case $opt in
        d) days=$OPTARG ;;
        b) base=$OPTARG ;;
        r) remote=$OPTARG ;;
        l) local=1 ;;
        m) mergedOnly=1 ;;
        *) sed -n '5,17p' "$0"; exit 0 ;;
    esac
done

if [ "$local" -eq 1 ]; then
    scope=refs/heads
    baseRef=$base
else
    git fetch --quiet --prune "$remote" || echo "warning: fetch from $remote failed, using cached refs" >&2
    scope=refs/remotes/$remote
    baseRef=$remote/$base
fi

now=$(date +%s)
cutoff=$((now - days * 86400))

# list <extra for-each-ref args...>: print "<age>d <branch>" for stale branches
list() {
    git for-each-ref "$@" --format='%(refname:short) %(committerdate:unix)' "$scope" |
        while read -r branch ts; do
            case $branch in
                "$baseRef" | "$remote" | */HEAD | main | "$remote/main") continue ;;
            esac
            if [ "$ts" -lt "$cutoff" ]; then
                printf '%5dd  %s\n' $(((now - ts) / 86400)) "$branch"
            fi
        done | sort -rn
}

if [ "$mergedOnly" -eq 0 ]; then
    echo "== Stale (> ${days}d) =="
    list
    echo
fi
echo "== Stale (> ${days}d) and merged into $baseRef =="
list --merged "$baseRef"
