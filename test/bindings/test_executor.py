# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest


def test_import():
    try:
        import neon  # noqa: F401

        assert True  # If import succeeds, the test passes

    except ImportError:
        assert False  # If import fails, the test fails

import neon as nn

def test_array_hip():
    pass
