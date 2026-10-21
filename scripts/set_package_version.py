#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Unlicense

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERSION_LINE = re.compile(r'(?m)^version = "([^"]+)"$')
RELEASE_VERSION = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)")
SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._!+-]*$")


def read_text() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def current_version() -> str:
    match = VERSION_LINE.search(read_text())
    if not match:
        raise SystemExit("Could not find project.version in pyproject.toml")
    return match.group(1)


def write_version(version: str) -> str:
    if not SAFE_VERSION.fullmatch(version):
        raise SystemExit(f"Refusing unsafe package version: {version}")

    text = read_text()
    updated, count = VERSION_LINE.subn(f'version = "{version}"', text, count=1)
    if count != 1:
        raise SystemExit("Could not update project.version in pyproject.toml")
    PYPROJECT.write_text(updated, encoding="utf-8")
    return version


def nightly_version(base_version: str, serial: str) -> str:
    match = RELEASE_VERSION.match(base_version)
    if not match:
        raise SystemExit(f"Cannot derive nightly version from {base_version}")

    major, minor, patch = (int(part) for part in match.groups())
    return f"{major}.{minor}.{patch + 1}.dev{serial}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read or update the NeoN Python package version.")
    parser.add_argument("version", nargs="?", help="Exact package version to write")
    parser.add_argument("--print", action="store_true", dest="show", help="Print the current version")
    parser.add_argument(
        "--nightly",
        metavar="SERIAL",
        help="Write the next patch dev version using SERIAL, e.g. 202605140001",
    )
    args = parser.parse_args()

    if args.show and (args.version or args.nightly):
        raise SystemExit("Use --print by itself")
    if args.version and args.nightly:
        raise SystemExit("Use either an exact version or --nightly, not both")

    if args.show:
        print(current_version())
    elif args.nightly:
        print(write_version(nightly_version(current_version(), args.nightly)))
    elif args.version:
        print(write_version(args.version))
    else:
        print(current_version())


if __name__ == "__main__":
    main()
