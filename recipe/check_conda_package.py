#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Smoke test for an installed NeoN conda package.

The conda package is installed by CMake rather than pip, so it carries no
``.dist-info`` and cannot be checked with ``importlib.metadata`` the way
``ci/check_installed_wheel.py`` checks a wheel. The recipe version is passed
through ``PKG_VERSION`` instead.
"""

from __future__ import annotations

import os

import neon


def main() -> None:
    expected = os.environ.get("PKG_VERSION")
    if expected and neon.__version__ != expected:
        raise SystemExit(
            f"neon.__version__ ({neon.__version__}) does not match "
            f"the conda package version ({expected})"
        )

    for attr in ("__has_serial__", "__has_cpu__", "__has_gpu__"):
        value = getattr(neon, attr, None)
        if not isinstance(value, bool):
            raise SystemExit(f"neon.{attr} should be a bool, got {value!r}")

    if not neon.__has_serial__:
        raise SystemExit("Installed package does not report serial executor support")

    neon.initialize()
    try:
        executor = neon.SerialExecutor()
        vector = neon.ScalarVector(executor, 8, 1.0)
        size = vector.size()
        # Kokkos aborts if an allocation outlives Kokkos::finalize, so drop the containers
        # before finalizing rather than leaving them to the interpreter's teardown.
        del vector, executor
    finally:
        neon.finalize()

    if size != 8:
        raise SystemExit(f"Expected a vector of size 8, got {size}")


if __name__ == "__main__":
    main()
