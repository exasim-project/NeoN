# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Backend dispatch behaviour for ``solution["backend"]`` (plan 03 §Tests).

Covers: (a) an unknown backend errors listing the valid set; (b) a per-field
split where U advances on the cpp explicit backend while p is solved by MLMG
(which ignores ``backend`` by design); (c) an unsupported term/scheme on cpp
raises ``NotImplementedError`` naming both; (d) the cpp backend imports no jax
on its hot path.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import blockamr
from blockamr.backends import cpp_backend
from blockamr.backends.cpp_backend import CppBackend
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
def test_unknown_backend_lists_valid_set(blockamr_session):
    mesh, geom = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")
    _set_face_const(ff)
    phi.fill_patch(0, 0.0)

    with pytest.raises(KeyError) as excinfo:
        solve(
            exp.ddt(phi) + Div(ff, phi, scheme=Upwind()),
            t=0.0,
            dt=0.01,
            solution={"backend": "quantum"},
        )
    msg = str(excinfo.value)
    assert "quantum" in msg and "cpp" in msg and "jax" in msg


# (b) --------------------------------------------------------------------------
def test_per_field_split_cpp_explicit_plus_mlmg_pressure(blockamr_session):
    """U advances on the cpp explicit backend; p is solved by MLMG (backend ignored)."""
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

    # Explicit momentum step on cpp.
    solve(
        exp.ddt(U) + Div(ff, U, scheme=Upwind()),
        t=0.0,
        dt=0.01,
        solution={"backend": "cpp"},
    )
    after = [U.mf[0].copy_to_host(mfi) for mfi in blockamr.MFIterator(U.mf[0])]
    assert any(not np.allclose(a, b) for a, b in zip(after, before)), (
        "cpp explicit step did not modify U"
    )

    # Pressure solve — routed to MLMG; backend key is ignored, no NotImplementedError.
    dt = 0.01
    solve(
        imp.laplacian(dt, p) == exp.div(U),
        dt=dt,
        solution={"backend": "cpp", "rtol": 1e-10, "atol": 1e-12, "maxIter": 200},
    )
    assert p._imp_cache is not None, "MLMG pressure solve did not build its cache"


# (c) --------------------------------------------------------------------------
def test_variable_gamma_laplacian_on_cpp_raises_naming_term_and_scheme(blockamr_session):
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
            solution={"backend": "cpp"},
        )
    msg = str(excinfo.value)
    assert "Laplacian" in msg and "CentralDiffLaplacian" in msg


def test_source_term_on_cpp_raises_naming_term(blockamr_session):
    """The **Sp** overload specifically: ``exp.source(coeff_func, phi)``.

    Unchanged by B41, which built the *Su* overload — ``exp.source(S)``, a
    separate term class (``ExplicitSource``) with its own ``PointwiseSource``
    scheme and therefore its own cpp kernel. The Sp ``Source`` still carries
    ``scheme = None``, so this raise is still the cpp backend refusing a term
    with no kernel, still naming ``'Source'`` (decision Q23, ``plans/IBM/review.md``
    §4). A cpp kernel for the callable-coefficient form is not in B41's scope.
    """
    mesh, geom = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    phi.fill_patch(0, 0.0)

    def s(x, y, z, t):
        return jnp.ones_like(x)

    with pytest.raises(NotImplementedError, match="Source"):
        solve(
            exp.ddt(phi) + exp.source(s, phi),
            t=0.0,
            dt=0.01,
            solution={"backend": "cpp"},
        )


# (d) --------------------------------------------------------------------------
def test_cpp_backend_module_imports_no_jax():
    """The cpp backend's hot path pulls in no jax (plan §Verification)."""
    assert not hasattr(cpp_backend, "jax")
    assert not hasattr(cpp_backend, "jnp")
    assert isinstance(CppBackend(), CppBackend)
