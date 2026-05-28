#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Unlicense

"""Smoke test for an already installed NeoN wheel."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re

import neon


PROJECT_NAME = re.compile(r'(?m)^name = "([^"]+)"$')


def project_name() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = PROJECT_NAME.search(pyproject.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("Could not find project.name in pyproject.toml")
    return match.group(1)


def main() -> None:
    dist_name = project_name()
    dist_version = metadata.version(dist_name)

    if neon.__version__ != dist_version:
        raise SystemExit(
            f"neon.__version__ ({neon.__version__}) does not match "
            f"installed {dist_name} distribution version ({dist_version})"
        )

    for attr in ("__has_serial__", "__has_cpu__", "__has_gpu__"):
        value = getattr(neon, attr, None)
        if not isinstance(value, bool):
            raise SystemExit(f"neon.{attr} should be a bool, got {value!r}")

    if not neon.__has_serial__:
        raise SystemExit("Installed wheel does not report serial executor support")


if __name__ == "__main__":
    main()
