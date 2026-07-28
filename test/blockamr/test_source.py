# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The two source terms the DSL spells with one name.

``exp.source`` arity-dispatches (decision Q23/P2, ``plans/IBM/review.md`` §4),
exactly the way NeoN's C++ DSL does:

* ``exp.source(coeff_func, phi)`` — the **implicit (Sp)** form ``coeff * phi``,
  a callable coefficient times the solved field. jax only; the cpp backend
  raises for it, pinned by
  ``test_backend_dispatch.py::test_source_term_on_cpp_raises_naming_term``.
* ``exp.source(S)`` — the **explicit (Su)** form, where the one ``CellField``
  operand *is* the coefficient (``dsl::exp::source(coeff)``, ``sourceTerm.cpp``).
  Schemed (``PointwiseSource``) and therefore runs on both backends: ``cpp``
  through the compiled ``source_acc``, ``jax`` through ``Source3D``.
"""

import math

import numpy as np
import pytest

import blockamr
import jax.numpy as jnp
from blockamr.dsl import Equation, evaluate, exp, solve
from blockamr.field import CellField
from blockamr.mesh import Mesh
from blockamr.operators.source import Source


def _make_mesh(n_cell=64, max_size=32):
    """Create a periodic Mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _init_sin3d(phi, geom):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = (lo[2] + k + 0.5) * dx[2]
                    arr[i, j, k, 0] = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)


def test_source_exact():
    """Source(coeff_func, phi) = coeff_func * phi at cell centers (no stencil)."""
    mesh, geom = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _init_sin3d(phi, geom)

    def coeff_func(x, y, z, t):
        return x**2 + y

    source_op = Source(coeff_func, phi)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = jnp.asarray(phi.mf[0].grown_array(mfi)[:, :, :, 0])
        kernel = source_op.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        lo = mfi.valid_box().small_end()
        dx = geom.cell_size()
        prob_lo = geom.prob_lo()
        valid_arr = phi.mf[0].copy_to_host(mfi)
        nx, ny, nz = valid_arr.shape[:3]
        for i in range(nx):
            x = prob_lo[0] + (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = prob_lo[1] + (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = prob_lo[2] + (lo[2] + k + 0.5) * dx[2]
                    phi_val = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
                    exact = (x**2 + y) * phi_val
                    assert abs(float(result[i, j, k]) - exact) < 1e-14, (
                        f"At ({x:.3f},{y:.3f},{z:.3f}): "
                        f"got {float(result[i, j, k])}, expected {exact}"
                    )


# ---------------------------------------------------------------------------
# The explicit (Su) form — ``exp.source(S)`` (B41)
# ---------------------------------------------------------------------------

# 16 cells on the unit box: every cell centre ``(i + 0.5)/16`` is a dyadic
# rational, so the analytic fills below are exact in binary64 and the
# assertions can be ``==`` rather than a tolerance.
NS = 16


def _fill(field, geom, func):
    """Fill every valid cell (and component) of *field* from ``func(X, Y, Z)``."""
    mf = field.mf[0]
    dx = geom.cell_size()
    plo = geom.prob_lo()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        axes = [
            np.array([plo[d] + (lo[d] + i + 0.5) * dx[d] for i in range(arr.shape[d])])
            for d in range(3)
        ]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        for n in range(arr.shape[3]):
            arr[:, :, :, n] = func(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)


def _valid(field):
    """Per-box valid cells of a one-component field, as ``(nx, ny, nz)`` arrays."""
    mf = field.mf[0]
    out = []
    for mfi in blockamr.MFIterator(mf):
        arr = np.asarray(mf.copy_to_host(mfi))
        out.append(arr.reshape(arr.shape[:3]).copy())
    return out


def _boxes(result_level):
    """Per-box ``evaluate`` arrays of a one-component field, shape ``(nx, ny, nz)``."""
    return [np.asarray(a).reshape(np.asarray(a).shape[:3]) for a in result_level]


def _linear(X, Y, Z):
    return X + 2.0 * Y - Z


def _source_field(mesh, geom, ncomp=1):
    S = CellField(mesh, ncomp=ncomp, ngrow=1, name="S")
    _fill(S, geom, _linear)
    return S


def test_explicit_source_evaluates_to_the_field_itself():
    """``exp.source(S)`` *is* ``S`` — the single operand is the coefficient.

    ``2.0 * exp.source(S)`` therefore evaluates to exactly ``2*S`` in every
    valid cell: the term reads no neighbour and does no arithmetic beyond the
    scale, so there is nothing to round and the assertion is bitwise.
    """
    mesh, geom = _make_mesh(n_cell=NS, max_size=8)
    S = _source_field(mesh, geom)

    out = evaluate(Equation(2.0 * exp.source(S)), t=0.0, solution={"backend": "cpp"})

    for got, want in zip(_boxes(out[0]), _valid(S)):
        np.testing.assert_array_equal(got, 2.0 * want)


def test_explicit_source_matches_between_backends():
    """The Su term is schemed, so ``cpp`` and ``jax`` are two launches of the
    same arithmetic — at ``test_backend_parity.py``'s tolerances.

    Beside a laplacian on a *different* field, so the test also pins that the
    source reads its own operand rather than the equation's solved field.
    """
    mesh, geom = _make_mesh(n_cell=NS, max_size=8)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T")
    _fill(T, geom, lambda X, Y, Z: np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y))
    S = _source_field(mesh, geom)
    eqn = Equation(exp.laplacian(0.5, T) + 2.0 * exp.source(S))

    on_cpp = evaluate(eqn, t=0.0, solution={"backend": "cpp"})
    on_jax = evaluate(eqn, t=0.0, solution={"backend": "jax"})

    for got, want in zip(_boxes(on_jax[0]), _boxes(on_cpp[0])):
        np.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-9)


def test_a_source_term_drives_the_field_through_solve():
    """One forward-Euler step of ``ddt(T) + source(S)`` moves ``T`` by ``-dt*S``.

    This is the sign convention the whole explicit path is built on
    (``phi -= dt * sum(coeff_i * op_i)``), and the reason the convergence study
    next door writes ``+ALPHA * exp.source(S)`` for a residual-form
    ``ddt(T) - alpha lap(T) + alpha (lap T_exact) = 0``. ``dt`` is a negative
    power of two, so ``dt*S`` is exact and so is the assertion.
    """
    mesh, geom = _make_mesh(n_cell=NS, max_size=8)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T")
    _fill(T, geom, lambda X, Y, Z: X * X)
    S = _source_field(mesh, geom)
    before = _valid(T)
    dt = 0.25

    solve(
        Equation(exp.ddt(T) + exp.source(S), schemes={"ddt": "Euler"}),
        dt=dt,
        t=0.0,
        solution={"backend": "cpp"},
    )

    for got, was, s in zip(_valid(T), before, _valid(S)):
        np.testing.assert_array_equal(got, was - dt * s)


def test_a_multi_component_explicit_source_runs_on_cpp_and_is_refused_on_jax():
    """The one asymmetry between the two backends, stated rather than hidden.

    ``source_acc`` loops components natively, so ``cpp`` takes a vector source.
    The jax path cannot yet: ``parallel_for`` shifts the *phi* tiles per
    component and the source buffer is not shifted with them, so an ``ncomp>1``
    source would silently read component 0 into every component. It raises
    instead (decision Q23/P4 — every §4 study case is scalar).
    """
    mesh, geom = _make_mesh(n_cell=NS, max_size=8)
    S = _source_field(mesh, geom, ncomp=3)

    out = evaluate(Equation(exp.source(S)), t=0.0, solution={"backend": "cpp"})
    for got, mfi in zip(out[0], blockamr.MFIterator(S.mf[0])):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(S.mf[0].copy_to_host(mfi)))

    with pytest.raises(NotImplementedError, match="ncomp"):
        evaluate(Equation(exp.source(S)), t=0.0, solution={"backend": "jax"})
