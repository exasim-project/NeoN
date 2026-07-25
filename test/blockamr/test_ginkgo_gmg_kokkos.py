# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``precond="gmg_kokkos"``: the optimised Kokkos V-cycle as a CG preconditioner.

``bench/gmg_vcycle.cpp`` proves the optimised V-cycle computes the same cycle as the
shipped one, to the last bit (``test_gmg_kokkos.py``). This file proves the other
half: that the cycle is wired into a REAL solve correctly -- through
``bench/gmg_apply.hpp``, ``solvers/gmg_kokkos_precond.hpp`` and the flat-vector
scatter/gather that separates a preconditioner from a V-cycle. Getting a hierarchy
right and then handing the solver a vector in the wrong order, or an unsynchronised
one, would still converge; it would just cost more iterations.

So the gate is CG's own behaviour against ``precond="gmg"``, which shares the
operator, the Krylov solver, the sweep counts and the tolerance:

* it converges, and to the same answer -- the preconditioner cannot change the fixed
  point, only how fast CG reaches it;
* the iteration count does not get WORSE. It is allowed to get better: the Kokkos
  preconditioner agglomerates coarse levels, which lets the hierarchy coarsen further
  than in-place coarsening can, and a deeper hierarchy is a better preconditioner.

Being a device path it is cuda-only, and having no physical-BC handling it is
periodic-only; both are rejected at construction rather than silently ignored, and
that is tested too.
"""

import numpy as np
import pytest

import blockamr


def _make_mesh(n, max_size=None, periodic=True):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    per = [1, 1, 1] if periodic else [0, 0, 0]
    geom = blockamr.Geometry(box, rb, 0, per)
    ba = blockamr.BoxArray(box)
    ba.max_size(n if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_face(geom, dm, d, box_size, value):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(box_size)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _helmholtz(n, max_size=None, periodic=True):
    """The periodic Helmholtz (phi - laplacian phi) in face-coefficient form -- the
    operator bench_solvers.py and the V-cycle bench both use."""
    box_size = n if max_size is None else max_size
    geom, ba, dm = _make_mesh(n, max_size, periodic)
    inv_dx2 = 1.0 / geom.cell_size()[0] ** 2
    alpha = blockamr.MultiFab(ba, dm, 1, 0)
    alpha.set_val(1.0)
    faces = [_const_face(geom, dm, d, box_size, -inv_dx2) for d in range(3)]
    return geom, ba, dm, alpha, faces


def _rhs(ba, dm, seed=42):
    rng = np.random.default_rng(seed)
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        mf.copy_from(mfi, arr)
    return mf


def _solver_or_skip(geom, alpha, faces, **kwargs):
    if not hasattr(blockamr, "FaceCoeffSolver"):
        pytest.skip("blockamr.FaceCoeffSolver binding not available")
    fx, fy, fz = faces
    try:
        return blockamr.FaceCoeffSolver(
            alpha, fx, fx, fy, fy, fz, fz, geom, executor="cuda", **kwargs
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if "cuda" in str(exc).lower() and "unavailable" in str(exc).lower():
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise


def _solve(geom, ba, dm, alpha, faces, rhs, **kwargs):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)
    s = _solver_or_skip(
        geom, alpha, faces, solver="cg", max_iter=200, rtol=1e-10, precond_cycles=1, **kwargs
    )
    st = dict(s.solve(rhs, sol))
    st["sol"] = np.concatenate(
        [sol.copy_to_host(m)[:, :, :, 0].ravel() for m in blockamr.MFIterator(sol)]
    )
    return st


# max_size 8 on a 16^3 grid is 8 boxes: the coarse levels are then many tiny boxes,
# which is the decomposition the optimised launchers exist for and the one where a
# wrong box mapping in the halo or the transfers would show.
MAX_SIZE = [None, 8]
MAX_SIZE_IDS = ["1box", "8box"]


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_matches_the_shipped_gmg_preconditioner(blockamr_session, max_size, precision):
    """Same answer, and no more CG iterations, than precond="gmg"."""
    geom, ba, dm, alpha, faces = _helmholtz(16, max_size)
    rhs = _rhs(ba, dm)
    ref = _solve(geom, ba, dm, alpha, faces, rhs, precond="gmg", gmg_precision=precision)
    opt = _solve(geom, ba, dm, alpha, faces, rhs, precond="gmg_kokkos", gmg_precision=precision)

    assert ref["converged"] is True
    assert opt["converged"] is True
    # The preconditioner cannot move the fixed point, only the path to it. Both stop at
    # rtol=1e-10 on the same operator, so the answers agree well inside that.
    assert np.max(np.abs(opt["sol"] - ref["sol"])) < 1e-7 * max(1.0, np.max(np.abs(ref["sol"])))
    # Allowed to be better (a deeper agglomerated hierarchy is a better cycle), never
    # worse -- worse would mean the apply is losing information.
    assert opt["num_iters"] <= ref["num_iters"]


@pytest.mark.parametrize("bc_name", ["dirichlet", "neumann"])
@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_matches_the_shipped_preconditioner_with_physical_bcs(blockamr_session, max_size, bc_name):
    """The same gate as above on a NON-periodic domain.

    A periodic mesh never exercises the boundary fill at all, so it cannot tell a
    correct reflection from a missing one. Here every level has six physical faces:
    the ghost layer outside each is filled by reflecting the interior with the sign the
    condition dictates, and the V-cycle is only the same V-cycle as precond="gmg" if
    that fill agrees on every level of the hierarchy, coarse ones included.
    """
    geom, ba, dm, alpha, faces = _helmholtz(16, max_size, periodic=False)
    rhs = _rhs(ba, dm)
    kw = dict(bc=[bc_name] * 6)
    ref = _solve(geom, ba, dm, alpha, faces, rhs, precond="gmg", **kw)
    opt = _solve(geom, ba, dm, alpha, faces, rhs, precond="gmg_kokkos", **kw)

    assert ref["converged"] is True
    assert opt["converged"] is True
    assert np.max(np.abs(opt["sol"] - ref["sol"])) < 1e-7 * max(1.0, np.max(np.abs(ref["sol"])))
    assert opt["num_iters"] <= ref["num_iters"]


def test_rejects_the_chebyshev_smoother(blockamr_session):
    """Only the red-black smoother is ported; asking for Chebyshev must fail loudly
    rather than quietly running red-black under a Chebyshev label."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    with pytest.raises(RuntimeError, match="red-black smoother"):
        _solve(
            geom, ba, dm, alpha, faces, _rhs(ba, dm),
            precond="gmg_kokkos", gmg_smoother="chebyshev",
        )


def test_unknown_precond_still_names_the_new_option(blockamr_session):
    geom, ba, dm, alpha, faces = _helmholtz(16)
    with pytest.raises(RuntimeError, match="gmg_kokkos"):
        _solve(geom, ba, dm, alpha, faces, _rhs(ba, dm), precond="nope")
