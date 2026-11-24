# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest

import blockamr


@pytest.fixture(scope="session", autouse=True)
def blockamr_session():
    """Initialize and finalize blockAMR once for all tests in this directory."""
    with blockamr.runtime():
        yield
