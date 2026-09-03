# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Regression tests for runtime-selection operator registration in the ``_neon`` module.

The Gauss ``div``/``laplacian`` operators self-register into a runtime-selection
factory table only when their template is instantiated. Their headers are declared
``extern template``, so that instantiation normally happens once, in ``libNeoN``.
Because the ``_neon`` extension is compiled with ``-fvisibility=hidden``, it would
otherwise get a private, empty copy of that table and scheme resolution would abort
with ``Could not find constructor for Gauss``. ``src/bindings/dsl.cpp`` forces the
explicit instantiations inside ``_neon`` so its table is populated.

These tests drive ``Expression.read`` — the call that performs the factory lookup
(``create("Gauss")``) — so they fail loudly if that registration ever regresses.
A scalar equation exercises the ``<scalar>`` instantiation; a vector equation
additionally exercises the ``<Vec3>`` and ``<Vec3, scalar>`` instantiations (the
vector ``div``/``laplacian`` build both same-type and scalar-matrix strategies).

``test_registered_operator_schemes_match_expected`` additionally asserts the exact
contents of each factory table, so a newly force-instantiated scheme (or one that
silently disappears) has to be reflected here.
"""

import neon
from neon import imp

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
    # Surface-interpolation schemes register from their headers, so they are not part of the
    # explicit-instantiation workaround above -- but they share the table with it, and
    # linearUpwindV is vector-only by design, so pin both sets here too.
    "surfaceInterpolation<scalar>": {"linear", "upwind", "linearUpwind"},
    "surfaceInterpolation<Vector>": {"linear", "upwind", "linearUpwind", "linearUpwindV"},
}


def test_registered_operator_schemes_match_expected():
    """Every operator factory has exactly its expected schemes registered in _neon."""
    registered = {k: set(v) for k, v in neon.registered_operator_schemes().items()}
    assert registered == EXPECTED_SCHEMES


def _make_schemes() -> neon.Dictionary:
    """divSchemes/laplacianSchemes for a ``div(phi,*) + laplacian(gamma,*)`` equation."""
    div_schemes = neon.Dictionary()
    div_schemes.insert_token_list("div(phi,U)", neon.TokenList(["Gauss", "linear"]))
    div_schemes.insert_token_list("div(phi,T)", neon.TokenList(["Gauss", "linear"]))

    laplacian_schemes = neon.Dictionary()
    laplacian_schemes.insert_token_list(
        "laplacian(gamma,U)", neon.TokenList(["Gauss", "linear", "uncorrected"])
    )
    laplacian_schemes.insert_token_list(
        "laplacian(gamma,T)", neon.TokenList(["Gauss", "linear", "uncorrected"])
    )

    schemes = neon.Dictionary()
    schemes.insert_dict("divSchemes", div_schemes)
    schemes.insert_dict("laplacianSchemes", laplacian_schemes)
    return schemes


def test_gauss_scalar_operators_resolve(executor):
    """Scalar Gauss div/laplacian resolve through the _neon factory table."""
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 10, 1.0)

    field = neon.ScalarVolumeField(exec, "T", mesh)
    phi = neon.ScalarSurfaceField(exec, "phi", mesh)
    gamma = neon.ScalarSurfaceField(exec, "gamma", mesh)
    neon.fill(phi.internal_vector(), 1.0)
    neon.fill(gamma.internal_vector(), 1.0)

    eqn = imp.div(phi, field) + imp.laplacian(gamma, field)
    assert eqn.size() == 2

    # Without the explicit instantiation in _neon this raises
    # "Could not find constructor for Gauss".
    eqn.read(_make_schemes())


def test_gauss_vector_operators_resolve(executor):
    """Vector Gauss div/laplacian resolve — covers the <Vec3> and <Vec3, scalar> tables."""
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 10, 1.0)

    field = neon.VectorVolumeField(exec, "U", mesh)
    phi = neon.ScalarSurfaceField(exec, "phi", mesh)
    gamma = neon.ScalarSurfaceField(exec, "gamma", mesh)
    neon.fill(phi.internal_vector(), 1.0)
    neon.fill(gamma.internal_vector(), 1.0)

    eqn = imp.div(phi, field) + imp.laplacian(gamma, field)
    assert eqn.size() == 2

    # Without the explicit instantiation in _neon this raises
    # "Could not find constructor for Gauss".
    eqn.read(_make_schemes())
