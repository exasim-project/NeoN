#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

# Prints the CMake arguments that select the Ginkgo for the checked-out NeoN: none if the
# Ginkgo of the image contains the commit requested in cmake/Versions.cmake, otherwise
# -DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON, so that CPM builds the requested commit. The
# images record the commit they were built from in /opt/ginkgo/share/ginkgo-revision.
#   ginkgo-cmake-args.sh [<NeoN source dir>]
# Diagnostics go to stderr, use as: cmake ... $(./ci/ginkgo-cmake-args.sh)

set -euo pipefail

src_dir="${1:-.}"
revision_file="${Ginkgo_ROOT:-/opt/ginkgo}/share/ginkgo-revision"
use_cpm="-DCMAKE_DISABLE_FIND_PACKAGE_Ginkgo=ON"

requested=$(sed -nE 's/^set\(NeoN_GINKGO_TAG "([0-9a-f]+)"\)/\1/p' \
    "${src_dir}/cmake/Versions.cmake")
if [[ -z "${requested}" ]]; then
    echo "Ginkgo: no NeoN_GINKGO_TAG in cmake/Versions.cmake, using the image's Ginkgo" >&2
    exit 0
fi

if [[ ! -r "${revision_file}" ]]; then
    echo "Ginkgo: ${revision_file} not found, building ${requested} with CPM" >&2
    echo "${use_cpm}"
    exit 0
fi
installed=$(tr -d '[:space:]' < "${revision_file}")

if [[ "${installed}" == "${requested}" ]]; then
    echo "Ginkgo: image provides the requested commit ${requested}" >&2
    exit 0
fi

# Fetch only the commit graph of both revisions to compare them.
repo=$(mktemp -d)
trap 'rm -rf "${repo}"' EXIT
git -C "${repo}" init -q
if ! git -C "${repo}" fetch -q --filter=tree:0 \
    https://github.com/ginkgo-project/ginkgo.git "${requested}" "${installed}"; then
    echo "Ginkgo: cannot compare image ${installed} with requested ${requested}," \
        "building the requested commit with CPM" >&2
    echo "${use_cpm}"
    exit 0
fi

if git -C "${repo}" merge-base --is-ancestor "${requested}" "${installed}"; then
    echo "Ginkgo: image ${installed} is newer than the requested ${requested}," \
        "using the image's Ginkgo" >&2
else
    echo "Ginkgo: requested ${requested} is not part of the image's ${installed}," \
        "building it with CPM" >&2
    echo "${use_cpm}"
fi
