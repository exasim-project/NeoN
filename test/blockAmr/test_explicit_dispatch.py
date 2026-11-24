# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Explicit-operator dispatch through the AMReX C++ kernels.

Covers: (a) a per-field split where U advances through the C++ explicit kernels
while p is solved by MLMG; (b) an unsupported term/scheme (variable Laplacian
gamma) raises ``NotImplementedError`` naming both; (c) a pointwise source term
advances through the C++ kernels.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import exp, imp, solve
from blockamr.field import CellField, FaceField
from blockamr.mesh import Mesh
from blockamr.operators.div import Div
from blockamr.schemes.div_schemes import Upwind


def _make_mesh(n=16, max_size=16):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _set_face_const(ff):
    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            arr = ff[0][d].mf.copy_to_host(mfi)
            arr[:] = 1.0
            ff[0][d].mf.copy_from(mfi, arr)


# (a) --------------------------------------------------------------------------
def test_per_field_split_cpp_explicit_plus_mlmg_pressure(blockamr_session):
    """U advances through the C++ explicit kernels; p is solved by MLMG."""
    mesh, geom = _make_mesh()
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U_face")

    pi = np.pi
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(U.mf[0]):
        arr = U.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        for i in range(arr.shape[0]):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = np.sin(2 * pi * x)
        U.mf[0].copy_from(mfi, arr)
    U.fill_patch(0, 0.0)
    _set_face_const(ff)

    before = [U.mf[0].copy_to_host(mfi).copy() for mfi in blockamr.MFIterator(U.mf[0])]

    # Explicit momentum step through the C++ kernels.
    solve(
        exp.ddt(U) + Div(ff, U, scheme=Upwind()),
        t=0.0,
        dt=0.01,
    )
    after = [U.mf[0].copy_to_host(mfi) for mfi in blockamr.MFIterator(U.mf[0])]
    assert any(not np.allclose(a, b) for a, b in zip(after, before)), (
        "C++ explicit step did not modify U"
    )

    # Pressure solve — routed to MLMG.
    dt = 0.01
    solve(
        imp.laplacian(dt, p) == exp.div(U),
        dt=dt,
        solution={"rtol": 1e-10, "atol": 1e-12, "maxIter": 200},
    )
    assert p._imp_cache is not None, "MLMG pressure solve did not build its cache"


# (b) --------------------------------------------------------------------------
def test_variable_gamma_laplacian_raises_naming_term_and_scheme(blockamr_session):
    mesh, geom = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    phi.fill_patch(0, 0.0)

    def gamma(x, y, z, t):
        return 0.01 * (1.0 + x)  # genuinely variable

    with pytest.raises(NotImplementedError) as excinfo:
        solve(
            exp.ddt(phi) + exp.laplacian(gamma, phi),
            t=0.0,
            dt=0.01,
        )
    msg = str(excinfo.value)
    assert "Laplacian" in msg and "CentralDiffLaplacian" in msg


# (c) --------------------------------------------------------------------------
def test_source_term_advances_through_cpp_kernels(blockamr_session):
    """ddt(phi) + S*phi with S = 1 + x: one Euler step gives phi = 1 - dt*(1 + x)."""
    mesh, geom = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    phi.fill_patch(0, 0.0)
    phi.mf[0].set_val(1.0)

    def s(x, y, z, t):
        return 1.0 + x

    dt = 0.01
    solve(exp.ddt(phi) + exp.source(s, phi), t=0.0, dt=dt)

    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(phi.mf[0]):
        lo = mfi.valid_box().small_end()
        valid = phi.mf[0].copy_to_host(mfi)[..., 0]
        x = (lo[0] + np.arange(valid.shape[0]) + 0.5) * dx[0]
        expected = 1.0 - dt * (1.0 + x)
        np.testing.assert_allclose(
            valid, np.broadcast_to(expected[:, None, None], valid.shape), rtol=1e-12
        )
