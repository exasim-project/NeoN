# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 NeoN authors

#!/usr/bin/env bash

## Prepare and publish a NeoN release.
##
## The release is a two-step process because `main` is a protected branch and needs an
## approving review, so the merge itself stays a human action:
##
##   scripts/release.sh prepare v0.3.0rc2   bump versions, open the release PR against main
##   <review and merge the PR on GitHub>
##   scripts/release.sh tag v0.3.0rc2       tag the merge commit and create the GitHub release
##
## Pushing the tag is what triggers everything downstream: python_wheels.yaml publishes CPU
## and CUDA wheels to PyPI, conda_packages.yaml publishes to prefix.dev, and build_doc.yaml
## publishes the documentation. Both package workflows take the version from the tag name,
## so the tag - not pyproject.toml - is the source of truth at build time.
##
##   scripts/release.sh check v0.3.0rc2     run the preflight checks only
##
## Add --dry-run to any subcommand to print the mutating commands instead of running them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REMOTE="origin"
SOURCE_BRANCH="develop"
TARGET_BRANCH="main"
PYPI_PACKAGE="neon-pde"

DRY_RUN=0

err() {
    echo "error: $*" >&2
    exit 1
}

info() { echo "==> $*"; }

# Run a mutating command, or print it under --dry-run.
run() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf 'dry-run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

usage() {
    sed -n 's/^## \{0,1\}//p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

# v0.3.0rc2 -> 0.3.0rc2
strip_v() { echo "${1#v}"; }

# 0.3.0rc2 -> 0.3.0
base_version() { echo "${1%%[a-z]*}"; }

# A prerelease carries an a/b/rc suffix, e.g. 0.3.0rc2.
is_prerelease() { [[ "$1" != "$(base_version "$1")" ]]; }

check_version_format() {
    local version="$1"
    # PEP 440 subset: the release workflows and PyPI both accept this form, and the
    # v*.*.* tag glob in the workflows matches the corresponding tag.
    if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
        err "'${version}' is not a supported version. Use 0.3.0 or a PEP 440 prerelease like 0.3.0rc2 (no dash)."
    fi
}

require_tools() {
    command -v git >/dev/null || err "git not found"
    command -v gh >/dev/null || err "gh not found; install the GitHub CLI"
    command -v python3 >/dev/null || err "python3 not found"
    command -v curl >/dev/null || err "curl not found"
    gh auth status >/dev/null 2>&1 || err "gh is not authenticated; run 'gh auth login'"
}

fetch() {
    git fetch --quiet "${REMOTE}" "$@" \
        || err "could not fetch ${*} from ${REMOTE}; check your network and git credentials"
}

check_clean_tree() {
    git diff --quiet && git diff --cached --quiet \
        || err "working tree has uncommitted changes; commit or stash them first"
}

check_tag_unused() {
    local tag="$1"
    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
        err "tag ${tag} already exists locally"
    fi
    if gh api "repos/{owner}/{repo}/git/ref/tags/${tag}" >/dev/null 2>&1; then
        err "tag ${tag} already exists on ${REMOTE}"
    fi
}

# PyPI burns a version permanently: once uploaded it can never be re-uploaded, not even
# after deleting the files. v0.3.0rc1 was spent this way by a rehearsal run, which is the
# failure mode this check exists to catch before a tag is pushed.
check_pypi_unused() {
    local version="$1" status
    status="$(curl -s -o /dev/null -w '%{http_code}' \
        "https://pypi.org/pypi/${PYPI_PACKAGE}/${version}/json")"
    case "${status}" in
    404) ;;
    200) err "${PYPI_PACKAGE} ${version} is already published on PyPI and cannot be re-uploaded; pick a new version" ;;
    *) echo "warning: could not reach PyPI to verify ${version} is unused (HTTP ${status})" >&2 ;;
    esac
}

check_source_branch() {
    local current
    current="$(git rev-parse --abbrev-ref HEAD)"
    [[ "${current}" == "${SOURCE_BRANCH}" ]] \
        || err "expected to be on ${SOURCE_BRANCH}, but on ${current}"

    fetch "${SOURCE_BRANCH}" "${TARGET_BRANCH}"
    [[ -z "$(git rev-list "${REMOTE}/${SOURCE_BRANCH}..HEAD")" ]] \
        || err "${SOURCE_BRANCH} has commits not pushed to ${REMOTE}"
    [[ -z "$(git rev-list "HEAD..${REMOTE}/${SOURCE_BRANCH}")" ]] \
        || err "${SOURCE_BRANCH} is behind ${REMOTE}/${SOURCE_BRANCH}; pull first"
}

# The CHANGELOG section for this release, used as the GitHub release notes.
changelog_section() {
    local base="$1"
    python3 - "${base}" <<'PY'
import re
import sys
from pathlib import Path

base = sys.argv[1]
text = Path("CHANGELOG.md").read_text(encoding="utf-8")
match = re.search(
    rf"^# Version {re.escape(base)}\b.*?$(.*?)(?=^# Version |\Z)",
    text,
    re.MULTILINE | re.DOTALL,
)
if not match:
    raise SystemExit(f"No '# Version {base}' section found in CHANGELOG.md")
print(match.group(1).strip())
PY
}

cmd_check() {
    local version="$1" tag="v$1"
    require_tools
    check_version_format "${version}"
    check_clean_tree
    check_tag_unused "${tag}"
    check_pypi_unused "${version}"
    changelog_section "$(base_version "${version}")" >/dev/null
    info "preflight checks passed for ${tag}"
}

cmd_prepare() {
    local version="$1" tag="v$1"
    local base branch today
    base="$(base_version "${version}")"
    branch="release/${tag}"

    check_source_branch
    cmd_check "${version}"

    today="$(date +%Y-%m-%d)"

    info "creating ${branch}"
    run git switch --create "${branch}"

    info "stamping pyproject.toml to ${version}"
    run python3 scripts/set_package_version.py "${version}"

    info "updating CITATION.cff"
    run python3 - "${tag}" "${today}" <<'PY'
import re
import sys
from pathlib import Path

tag, today = sys.argv[1], sys.argv[2]
path = Path("CITATION.cff")
text = path.read_text(encoding="utf-8")
text, n = re.subn(r"(?m)^version: .*$", f"version: {tag}", text)
if n != 1:
    raise SystemExit("Could not update 'version:' in CITATION.cff")
text, n = re.subn(r"(?m)^date-released: .*$", f"date-released: '{today}'", text)
if n != 1:
    raise SystemExit("Could not update 'date-released:' in CITATION.cff")
path.write_text(text, encoding="utf-8")
PY

    info "dating the CHANGELOG ${base} section"
    run python3 - "${base}" <<'PY'
import re
import sys
from datetime import date
from pathlib import Path

base = sys.argv[1]
path = Path("CHANGELOG.md")
text = path.read_text(encoding="utf-8")
# Accept either the open '(unreleased)' placeholder or an already dated header, so that a
# final release can re-date a section that a preceding release candidate already stamped.
text, n = re.subn(
    rf"(?m)^# Version {re.escape(base)} \((?:unreleased|[0-9]{{4}}/[0-9]{{2}}/[0-9]{{2}})\)$",
    f"# Version {base} ({date.today():%Y/%m/%d})",
    text,
)
if n != 1:
    raise SystemExit(f"Could not find a '# Version {base} (unreleased)' header to date")
path.write_text(text, encoding="utf-8")
PY

    echo
    info "review the staged version bump:"
    run git --no-pager diff -- pyproject.toml CITATION.cff CHANGELOG.md

    run git commit --all --message "release: ${tag}"
    run git push --set-upstream "${REMOTE}" "${branch}"

    info "opening the release PR against ${TARGET_BRANCH}"
    run gh pr create \
        --base "${TARGET_BRANCH}" \
        --head "${branch}" \
        --title "Release ${tag}" \
        --body "Release ${tag}.

Merging this PR brings \`${TARGET_BRANCH}\` up to \`${SOURCE_BRANCH}\` and stamps the release version.
After it is merged, run \`scripts/release.sh tag ${tag}\` to tag the merge commit and publish."

    echo
    info "next: get the PR approved and merged, then run: scripts/release.sh tag ${tag}"
    echo "note: the CITATION.cff abstract still describes the v0.1 release; update it by hand if it should reflect ${base}."
}

cmd_tag() {
    local version="$1" tag="v$1"
    local base sha notes prerelease_flag

    require_tools
    check_version_format "${version}"
    check_clean_tree
    check_tag_unused "${tag}"
    check_pypi_unused "${version}"

    base="$(base_version "${version}")"
    fetch "${TARGET_BRANCH}"
    sha="$(git rev-parse "${REMOTE}/${TARGET_BRANCH}")"

    # Guard against tagging a main that never received the release commit.
    local main_version
    main_version="$(git show "${sha}:pyproject.toml" | sed -n 's/^version = "\(.*\)"$/\1/p' | head -1)"
    [[ "${main_version}" == "${version}" ]] || err \
        "${REMOTE}/${TARGET_BRANCH} has pyproject version '${main_version}', expected '${version}'; is the release PR merged?"

    notes="$(changelog_section "${base}")"

    prerelease_flag=()
    if is_prerelease "${version}"; then
        prerelease_flag=(--prerelease)
        info "${version} is a prerelease; the GitHub release will be marked as such"
    fi

    info "tagging ${sha} as ${tag}"
    run git tag --annotate "${tag}" "${sha}" --message "NeoN ${tag}"
    run git push "${REMOTE}" "refs/tags/${tag}"

    info "creating the GitHub release"
    run gh release create "${tag}" \
        --title "NeoN ${tag}" \
        --notes "${notes}" \
        --verify-tag \
        "${prerelease_flag[@]}"

    echo
    info "the tag push triggers the wheel, conda and documentation workflows; watch them with:"
    echo "  gh run list --workflow python_wheels.yaml"
    echo "  gh run list --workflow conda_packages.yaml"
}

main() {
    local args=()
    for arg in "$@"; do
        case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h | --help) usage 0 ;;
        -*) err "unknown option ${arg}" ;;
        *) args+=("${arg}") ;;
        esac
    done

    [[ ${#args[@]} -eq 2 ]] || usage 1

    local subcommand version
    subcommand="${args[0]}"
    version="$(strip_v "${args[1]}")"

    case "${subcommand}" in
    check) cmd_check "${version}" ;;
    prepare) cmd_prepare "${version}" ;;
    tag) cmd_tag "${version}" ;;
    *) err "unknown subcommand '${subcommand}'" ;;
    esac
}

main "$@"
