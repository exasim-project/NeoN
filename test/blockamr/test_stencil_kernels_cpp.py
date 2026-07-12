# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Binding-level smoke test for the composable accumulate C++ stencil kernels.

Exercises the kernels added in plan 03 §4 (``div_{upwind,linear,vanleer,
quick}_acc``, ``laplacian_acc``, ``grad_acc``, ``source_acc``,
``euler_update``) directly through the nanobind bindings — no jax / Pallas
path. Verifies accumulate semantics (``out +=``), linear coeff scaling,
ncomp>1 coverage, and euler_update axpy against numpy. Numerical parity vs
the jax backend is covered separately in slice 3.
"""

import numpy as np
import pytest

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh
from neon.blockamr.field import CellField, FaceField


def _make_mesh(N):
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])  # periodic
    ba = blockamr.BoxArray(box)
    ba.max_size(N)  # single box → simple gather
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _seed_random(mf, geom, seed):
    """Fill the valid region with reproducible random data, then fill ghosts."""
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)  # (nx, ny, nz, ncomp) numpy, valid region
        arr[...] = rng.standard_normal(arr.shape)
        mf.copy_from(mfi, arr)
    mf.fill_boundary(geom)


def _gather(mf):
    """Return the valid-region data of every box as a list of numpy arrays."""
    return [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]


def _maxabs(arrs):
    return max(float(np.max(np.abs(a))) for a in arrs)


DIV_ACC = {
    "Upwind": "div_upwind_acc",
    "Linear": "div_linear_acc",
    "VanLeer": "div_vanleer_acc",
    "QUICK": "div_quick_acc",
}


def _make_div_inputs(ncomp):
    N = 16
    mesh, geom = _make_mesh(N)
    # ngrow=2 covers the wide VanLeer/QUICK stencils (± 2 cells).
    phi = CellField(mesh, ncomp=ncomp, ngrow=2, name="phi")
    out = CellField(mesh, ncomp=ncomp, ngrow=0, name="out")
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")
    _seed_random(phi.mf[0], geom, seed=1234)
    # Non-uniform, mixed-sign face flux exercises both upwind branches.
    ff[0][0].mf.set_val(1.0)
    ff[0][1].mf.set_val(-1.0)
    ff[0][2].mf.set_val(0.5)
    faces = (ff[0][0].mf, ff[0][1].mf, ff[0][2].mf)
    return geom, phi, out, faces


@pytest.mark.parametrize("scheme", ["Upwind", "Linear", "VanLeer", "QUICK"])
@pytest.mark.parametrize("ncomp", [1, 3])
def test_div_acc_accumulate_coeff_ncomp(blockamr_session, scheme, ncomp):
    geom, phi, out, faces = _make_div_inputs(ncomp)
    fn = getattr(blockamr, DIV_ACC[scheme])

    # (nonzero) single application with coeff=1
    out.mf[0].set_val(0.0)
    fn(out.mf[0], phi.mf[0], faces[0], faces[1], faces[2], geom, 1.0, ncomp)
    once = _gather(out.mf[0])
    assert _maxabs(once) > 1e-6, f"{scheme} div_acc produced a near-zero source"

    # (d) ncomp>1: every component is written non-trivially
    if ncomp > 1:
        for n in range(ncomp):
            comp_max = max(float(np.max(np.abs(a[:, :, :, n]))) for a in once)
            assert comp_max > 1e-6, f"{scheme} div_acc comp {n} not written"

    # (a) accumulate: calling twice doubles the increment
    out.mf[0].set_val(0.0)
    fn(out.mf[0], phi.mf[0], faces[0], faces[1], faces[2], geom, 1.0, ncomp)
    fn(out.mf[0], phi.mf[0], faces[0], faces[1], faces[2], geom, 1.0, ncomp)
    twice = _gather(out.mf[0])
    for a1, a2 in zip(once, twice):
        np.testing.assert_allclose(a2, 2.0 * a1, atol=1e-12, rtol=1e-12)

    # (b) coeff scales the increment linearly
    out.mf[0].set_val(0.0)
    fn(out.mf[0], phi.mf[0], faces[0], faces[1], faces[2], geom, 2.5, ncomp)
    scaled = _gather(out.mf[0])
    for a1, a in zip(once, scaled):
        np.testing.assert_allclose(a, 2.5 * a1, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("ncomp", [1, 3])
def test_laplacian_acc_accumulate_coeff(blockamr_session, ncomp):
    N = 16
    mesh, geom = _make_mesh(N)
    phi = CellField(mesh, ncomp=ncomp, ngrow=1, name="phi")
    out = CellField(mesh, ncomp=ncomp, ngrow=0, name="out")
    _seed_random(phi.mf[0], geom, seed=42)

    out.mf[0].set_val(0.0)
    blockamr.laplacian_acc(out.mf[0], phi.mf[0], geom, 1.0, ncomp)
    once = _gather(out.mf[0])
    assert _maxabs(once) > 1e-6

    out.mf[0].set_val(0.0)
    blockamr.laplacian_acc(out.mf[0], phi.mf[0], geom, 1.0, ncomp)
    blockamr.laplacian_acc(out.mf[0], phi.mf[0], geom, 1.0, ncomp)
    twice = _gather(out.mf[0])
    for a1, a2 in zip(once, twice):
        np.testing.assert_allclose(a2, 2.0 * a1, atol=1e-12, rtol=1e-12)

    out.mf[0].set_val(0.0)
    blockamr.laplacian_acc(out.mf[0], phi.mf[0], geom, 3.0, ncomp)
    scaled = _gather(out.mf[0])
    for a1, a in zip(once, scaled):
        np.testing.assert_allclose(a, 3.0 * a1, atol=1e-12, rtol=1e-12)


def test_grad_acc_accumulate_coeff_three_components(blockamr_session):
    N = 16
    mesh, geom = _make_mesh(N)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    out = CellField(mesh, ncomp=3, ngrow=0, name="grad")  # scalar -> 3-vector
    _seed_random(phi.mf[0], geom, seed=7)

    out.mf[0].set_val(0.0)
    blockamr.grad_acc(out.mf[0], phi.mf[0], geom, 1.0)
    once = _gather(out.mf[0])
    for n in range(3):
        comp_max = max(float(np.max(np.abs(a[:, :, :, n]))) for a in once)
        assert comp_max > 1e-6, f"grad_acc comp {n} not written"

    out.mf[0].set_val(0.0)
    blockamr.grad_acc(out.mf[0], phi.mf[0], geom, 1.0)
    blockamr.grad_acc(out.mf[0], phi.mf[0], geom, 1.0)
    twice = _gather(out.mf[0])
    for a1, a2 in zip(once, twice):
        np.testing.assert_allclose(a2, 2.0 * a1, atol=1e-12, rtol=1e-12)

    out.mf[0].set_val(0.0)
    blockamr.grad_acc(out.mf[0], phi.mf[0], geom, -2.0)
    scaled = _gather(out.mf[0])
    for a1, a in zip(once, scaled):
        np.testing.assert_allclose(a, -2.0 * a1, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("ncomp", [1, 3])
def test_source_acc_exact_and_accumulate(blockamr_session, ncomp):
    N = 16
    mesh, geom = _make_mesh(N)
    phi = CellField(mesh, ncomp=ncomp, ngrow=0, name="phi")
    out = CellField(mesh, ncomp=ncomp, ngrow=0, name="out")
    _seed_random(phi.mf[0], geom, seed=99)
    phi_arr = _gather(phi.mf[0])

    # out += coeff*phi, exactly
    out.mf[0].set_val(0.0)
    blockamr.source_acc(out.mf[0], phi.mf[0], 1.5, ncomp)
    once = _gather(out.mf[0])
    for a, p in zip(once, phi_arr):
        np.testing.assert_allclose(a, 1.5 * p, atol=1e-12, rtol=0.0)

    # accumulate: second call adds again
    blockamr.source_acc(out.mf[0], phi.mf[0], 1.5, ncomp)
    twice = _gather(out.mf[0])
    for a, p in zip(twice, phi_arr):
        np.testing.assert_allclose(a, 3.0 * p, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("ncomp", [1, 3])
def test_euler_update_axpy(blockamr_session, ncomp):
    N = 16
    mesh, geom = _make_mesh(N)
    phi = CellField(mesh, ncomp=ncomp, ngrow=0, name="phi")
    src = CellField(mesh, ncomp=ncomp, ngrow=0, name="src")
    _seed_random(phi.mf[0], geom, seed=1)
    _seed_random(src.mf[0], geom, seed=2)

    phi_before = _gather(phi.mf[0])
    src_arr = _gather(src.mf[0])

    dt_over_coeff = 0.375
    blockamr.euler_update(phi.mf[0], src.mf[0], dt_over_coeff, ncomp)
    phi_after = _gather(phi.mf[0])

    for pa, pb, s in zip(phi_after, phi_before, src_arr):
        np.testing.assert_allclose(pa, pb - dt_over_coeff * s, atol=1e-12, rtol=0.0)
