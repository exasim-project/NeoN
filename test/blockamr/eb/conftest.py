# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""EB-specific pytest fixtures.

The session-scoped ``blockamr_session`` fixture in
``test/blockamr/conftest.py`` opens ``blockamr.runtime()`` once for the
whole test directory and is automatically inherited here. We add a
function-scope ``eb2_clean`` fixture so each EB test starts with an
empty ``EB2::IndexSpace`` stack — otherwise an implicit-function
geometry built by one test would still be the "top" geometry seen by
the next ``make_eb_factory`` call.
"""

import pytest

import neon.blockamr as blockamr


@pytest.fixture(autouse=True)
def eb2_clean():
    """Reset the EB2 IndexSpace stack between tests."""
    blockamr.eb2_clear()
    yield
    blockamr.eb2_clear()
