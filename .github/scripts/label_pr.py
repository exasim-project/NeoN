# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Label a pull request by the parts of NeoN it touches and the kind of change.

Area labels (type:*) come from the changed paths, a PR can get several. The kind label
(bug, enhancement, ...) comes from the conventional-commit prefix of the title, then from
the branch prefix (feat/..., fix/...), then from the paths if they are all docs, CI or
build files. Labels are only ever added, so a maintainer can correct them by hand.
Labels that steer CI (skip-*, full-ci, benchmark, ready-*, auto-fix) are never set here.

Only reads the PR through the API and never checks out its code, so it is safe to run
from pull_request_target.

Usage:
    label_pr.py PR_NUMBER [PR_NUMBER ...] [--dry-run]

Environment:
    GH_REPO   owner/repo, e.g. exasim-project/NeoN
    GH_TOKEN  token with pull-requests: write (read access is enough for --dry-run)
"""

import json
import re
import subprocess
import sys

AREAS = {
    # Matched against the full path of every changed file.
    "type:distributed": [
        r"(^|/)distributed",
        r"[a-z]Distributed",
        r"(^|/)mpi",
        r"[Pp]rocessor\.(hpp|cpp)$",
        r"(^|/)communicat",
        r"(^|/)partition",
        r"(^|/)boundaryData\.(hpp|cpp)$",
        r"^cmake/DetectMpiThreadSupport\.cmake$",
    ],
    "type:core": [
        r"^(include/NeoN|src|test|benchmarks)/(core|executor|fields)/",
        r"^include/NeoN/helpers/",
    ],
    "type:dsl": [r"^(include/NeoN|src|test|benchmarks)/dsl/"],
    "type:fvm": [r"^(include/NeoN|src|test|benchmarks)/finiteVolume/"],
    "type:solver": [r"^(include/NeoN|src|test|benchmarks)/linearAlgebra/"],
    "type:mesh": [r"^(include/NeoN|src|test)/mesh/"],
    "type:time-integration": [r"^(include/NeoN|src|test)/timeIntegration/"],
    "type:python": [
        r"^(src|test)/bindings/",
        r"^src/neon/",
        r"^examples/python/",
        r"^pyproject\.toml$",
        r"^recipe/",
        r"^ci/check_(installed_wheel|wheel_rpath)\.py$",
        r"^doc/python",
    ],
}

# Conventional-commit types, also accepted as branch prefixes (feat/..., fix/...).
KINDS = {
    "feat": "enhancement",
    "enh": "enhancement",
    "fix": "bug",
    "perf": "performance",
    "refactor": "refactor",
    "docs": "documentation",
    "doc": "documentation",
    "test": "test",
    "tests": "test",
    "ci": "ci",
    "build": "build",
    "bump": "build",
    "chore": "chore",
    "release": "chore",
}

# Titles without a prefix that start with one of these verbs, e.g. "Add waveTransmissive BC".
VERBS = {"add": "enhancement", "implement": "enhancement", "support": "enhancement", "fix": "bug"}

# Release PRs and main -> develop back-merges touch everything, so they get no area labels.
RELEASE = re.compile(r"^(release v|(chore: )?merge main)", re.IGNORECASE)

# Last fallback: a kind that covers every changed file. Nearly every PR adds a CHANGELOG entry.
IGNORED_FOR_KIND = {"CHANGELOG.md"}
KIND_PATHS = {
    "documentation": [r"^doc/", r"\.(md|rst)$", r"^CITATION\.cff$", r"^AUTHORS", r"^assets/"],
    "ci": [r"^\.github/", r"^ci/"],
    "build": [r"CMakeLists\.txt$", r"^cmake/", r"^CMakePresets\.json$", r"^recipe/", r"^pyproject\.toml$"],
}

# A Ginkgo version bump changes what downstream users need to install.
GINKGO_VERSION = re.compile(r"^[+-]\s*set\(NeoN_GINKGO_(VERSION|TAG)\b", re.MULTILINE)


def gh(*args):
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def kind_from_text(text, separator):
    match = re.match(rf"\s*([A-Za-z]+)(\([^)]*\))?!?{separator}", text)
    return KINDS.get(match.group(1).lower()) if match else None


def labels_for(title, branch, files):
    paths = [f["filename"] for f in files]
    labels = set()
    release = RELEASE.match(title)
    for label, patterns in AREAS.items():
        if not release and any(re.search(p, path) for p in patterns for path in paths):
            labels.add(label)

    kind = kind_from_text(title, ":") or kind_from_text(branch, "/")
    if kind is None and release:
        kind = "chore"
    if kind is None:
        kind = VERBS.get(title.split(maxsplit=1)[0].lower()) if title.strip() else None
    kind_paths = [path for path in paths if path not in IGNORED_FOR_KIND]
    if kind is None and kind_paths:
        for label, patterns in KIND_PATHS.items():
            if all(any(re.search(p, path) for p in patterns) for path in kind_paths):
                kind = label
                break
    if kind:
        labels.add(kind)

    if any(
        f["filename"] == "cmake/Versions.cmake" and GINKGO_VERSION.search(f.get("patch", ""))
        for f in files
    ):
        labels.add("ginkgo-version")
    return labels


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    numbers = [a for a in args if a != "--dry-run"]
    if not numbers:
        sys.exit(__doc__)

    for number in numbers:
        pr = json.loads(gh("pr", "view", number, "--json", "title,headRefName,labels"))
        lines = gh(
            "api",
            "--paginate",
            f"repos/{{owner}}/{{repo}}/pulls/{number}/files",
            "--jq",
            ".[] | {filename, patch}",
        )
        files = [json.loads(line) for line in lines.splitlines()]

        present = {label["name"] for label in pr["labels"]}
        missing = sorted(labels_for(pr["title"], pr["headRefName"], files) - present)
        print(f"#{number} {pr['title']}: {', '.join(missing) or '-'}")
        if missing and not dry_run:
            gh("pr", "edit", number, "--add-label", ",".join(missing))


if __name__ == "__main__":
    main()
