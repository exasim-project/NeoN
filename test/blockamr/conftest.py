# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import os

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import pytest

import neon.blockamr as blockamr


@pytest.fixture(scope="session", autouse=True)
def blockamr_session():
    """Initialize and finalize blockAMR once for all tests in this directory."""
    with blockamr.runtime():
        yield
