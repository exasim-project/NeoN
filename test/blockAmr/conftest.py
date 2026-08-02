# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import os

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")
# Three consumers share the one device and must TOGETHER stay under 100 %:
# JAX/XLA, AMReX's arena (grown on demand from init size 0), and the Kokkos /
# Ginkgo GPU executors these tests construct in-process via ``_executors``.
# Left unset, JAX preallocates its own 75 % default and starves the other two,
# so pin it to the same fraction ``benchmarks/blockAmr/bench_backends.py`` uses.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

import pytest

import blockamr


@pytest.fixture(scope="session", autouse=True)
def blockamr_session():
    """Initialize and finalize blockAMR once for all tests in this directory."""
    with blockamr.runtime():
        yield
