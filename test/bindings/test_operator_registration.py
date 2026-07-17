# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Regression test for runtime-selection operator registration in ``_neon``.

The Gauss ``div``/``laplacian`` operators self-register only when their (``extern
template``) instantiation fires. ``src/bindings/dsl.cpp`` forces that instantiation
inside ``_neon`` — without it the ``-fvisibility=hidden`` module gets an empty factory
table and scheme resolution aborts with ``Could not find constructor for Gauss``.
"""

import neon

# The schemes each operator factory is expected to have registered in ``_neon``.
# Equality (below) makes this the source of truth: force-instantiate another scheme in
# ``dsl.cpp`` and this must be updated in lockstep, so registration drift is caught.
EXPECTED_SCHEMES = {
    "div<scalar>": {"Gauss"},
    "div<Vector>": {"Gauss"},
    "div<Vector,scalar>": {"Gauss"},
    "laplacian<scalar>": {"Gauss"},
    "laplacian<Vector>": {"Gauss"},
    "laplacian<Vector,scalar>": {"Gauss"},
}


def test_registered_operator_schemes_match_expected():
    """Every operator factory has exactly its expected schemes registered in _neon."""
    registered = {k: set(v) for k, v in neon.registered_operator_schemes().items()}
    assert registered == EXPECTED_SCHEMES
