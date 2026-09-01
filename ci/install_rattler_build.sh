#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Install a pinned rattler-build release into a directory on PATH.
# Usage: install_rattler_build.sh [version] [install_dir]

set -euo pipefail

version="${1:-${RATTLER_BUILD_VERSION:-0.75.0}}"
install_dir="${2:-${RATTLER_BUILD_INSTALL_DIR:-${HOME}/.local/bin}}"

case "$(uname -s)/$(uname -m)" in
    # The musl builds are used on Linux so the binary does not depend on the runner's glibc.
    Linux/x86_64) target="x86_64-unknown-linux-musl" ;;
    Linux/aarch64 | Linux/arm64) target="aarch64-unknown-linux-musl" ;;
    Darwin/x86_64) target="x86_64-apple-darwin" ;;
    Darwin/arm64) target="aarch64-apple-darwin" ;;
    *)
        echo "Unsupported host for rattler-build: $(uname -s)/$(uname -m)" >&2
        exit 1
        ;;
esac

mkdir -p "${install_dir}"
url="https://github.com/prefix-dev/rattler-build/releases/download/v${version}/rattler-build-${target}"
curl -fsSL "${url}" -o "${install_dir}/rattler-build"
curl -fsSL "${url}.sha256" -o "${install_dir}/rattler-build.sha256"

# The release publishes "<sha>  rattler-build-<target>", so verify from the download directory
# with the name the checksum file refers to.
(
    cd "${install_dir}"
    mv rattler-build "rattler-build-${target}"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum --check "rattler-build.sha256"
    else
        shasum --algorithm 256 --check "rattler-build.sha256"
    fi
    mv "rattler-build-${target}" rattler-build
    rm -f rattler-build.sha256
)
chmod +x "${install_dir}/rattler-build"

"${install_dir}/rattler-build" --version
