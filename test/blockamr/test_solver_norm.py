# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``norm="l2" | "linf"``: which norm a solve is allowed to stop on.

Ginkgo's stopping criteria measure the residual in the 2-norm; AMReX MLMG measures
it in the INFINITY norm (``AMReX_MLMG.H``: ``MLResNormInf``/``MLRhsNormInf``, and
``res_target = max(atol, max(rtol,1e-16) * max_norm)``). Two solvers stopping on
different tests are answering different questions, so an iteration count from one is
not directly comparable with the other's -- which matters precisely because the
interesting comparisons are close (``mlmg`` at 9 iterations against ``mf-gmgk`` at
10). ``norm="linf"`` exists to hold the norm fixed across that comparison.

What is actually checked here is the STOPPING TEST, not a norm helper: for each norm,
the residual of the returned solution, recomputed in numpy from the operator, must
satisfy the criterion the caller asked for -- and must NOT satisfy it for the norm it
did not ask for when the two disagree. A criterion that was quietly ignored, or that
measured Ginkgo's recursive residual and reported something else, fails that.

The two solver paths are separate implementations of the same test and both are
covered: the Krylov path (a custom ``gko::stop::Criterion``, since Ginkgo's Dense has
no inf-norm) and the native stationary ``solver="gmg"`` loop (both norms out of the
one fused residual reduction).
"""

import numpy as np
import pytest

import blockamr

N = 16


def _mesh(n=N, max_size=None):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])  # triply periodic
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


def _helmholtz(n=N, max_size=None):
    """phi - laplacian phi, periodic, in face-coefficient form."""
    box_size = n if max_size is None else max_size
    geom, ba, dm = _mesh(n, max_size)
    inv_dx2 = 1.0 / geom.cell_size()[0] ** 2
    alpha = blockamr.MultiFab(ba, dm, 1, 0)
    alpha.set_val(1.0)
    faces = [_const_face(geom, dm, d, box_size, -inv_dx2) for d in range(3)]
    return geom, ba, dm, alpha, faces, inv_dx2


def _spiky_rhs(ba, dm, n=N):
    """A right-hand side whose max/rms ratio is far from the residual's, so the two
    criteria are genuinely different thresholds and not the same number twice: one
    tall spike on an otherwise smooth field pushes ||b||_inf well above its rms."""
    rng = np.random.default_rng(7)
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    field = 0.01 * rng.standard_normal((n, n, n))
    field[n // 2, n // 2, n // 2] = 100.0
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        sh = arr.shape[:3]
        arr[:, :, :, 0] = field[
            lo[0] : lo[0] + sh[0], lo[1] : lo[1] + sh[1], lo[2] : lo[2] + sh[2]
        ]
        mf.copy_from(mfi, arr)
    return mf, field


def _to_array(mf, n=N):
    out = np.zeros((n, n, n))
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)[:, :, :, 0]
        lo = mfi.valid_box().small_end()
        out[
            lo[0] : lo[0] + arr.shape[0],
            lo[1] : lo[1] + arr.shape[1],
            lo[2] : lo[2] + arr.shape[2],
        ] = arr
    return out


def _apply(x, inv_dx2):
    """The same operator, in numpy: (1 + 6/dx^2) x - sum(neighbours)/dx^2, periodic."""
    lap = -6.0 * x
    for axis in range(3):
        lap += np.roll(x, 1, axis=axis) + np.roll(x, -1, axis=axis)
    return x - inv_dx2 * lap


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
    s = _solver_or_skip(geom, alpha, faces, **kwargs)
    st = dict(s.solve(rhs, sol))
    st["x"] = _to_array(sol)
    return st


# rtol loose enough that the two criteria stop at DIFFERENT iterations on this rhs:
# at 1e-10 both paths run to the same near-exact answer and the test would pass
# without the criterion being in effect at all.
RTOL = 1e-4

# Both stopping-test implementations: the Krylov criterion and the native loop.
PATHS = [
    dict(solver="cg", precond="gmg", precond_cycles=1),
    dict(solver="gmg", gmg_coarsest_sweeps=100),
]
PATH_IDS = ["cg", "native-gmg"]


@pytest.mark.parametrize("path", PATHS, ids=PATH_IDS)
@pytest.mark.parametrize("norm", ["l2", "linf"])
def test_stops_on_the_norm_it_was_given(blockamr_session, path, norm):
    """The returned solution satisfies the criterion asked for, in the asker's norm."""
    geom, ba, dm, alpha, faces, inv_dx2 = _helmholtz()
    rhs, b = _spiky_rhs(ba, dm)
    st = _solve(
        geom, ba, dm, alpha, faces, rhs, max_iter=500, rtol=RTOL, norm=norm, **path
    )
    assert st["converged"] is True

    r = b - _apply(st["x"], inv_dx2)
    measured = np.max(np.abs(r)) if norm == "linf" else np.linalg.norm(r)
    baseline = np.max(np.abs(b)) if norm == "linf" else np.linalg.norm(b)
    assert measured <= RTOL * baseline
    # And res_norm is reported in that same norm, not always the 2-norm: the number a
    # caller compares against its own rtol has to be the one the solve stopped on.
    assert st["res_norm"] == pytest.approx(measured, rel=2e-2)


@pytest.mark.parametrize("path", PATHS, ids=PATH_IDS)
def test_the_two_norms_are_not_the_same_test(blockamr_session, path):
    """The criteria are genuinely different thresholds on this rhs, so the tests above
    are not both passing for the same reason. Whichever norm is the stricter one here
    is a property of the fields, not something to assert -- what must hold is that the
    looser one stops sooner and its residual would NOT pass the stricter test."""
    geom, ba, dm, alpha, faces, inv_dx2 = _helmholtz()
    rhs, b = _spiky_rhs(ba, dm)
    res = {}
    for norm in ("l2", "linf"):
        st = _solve(
            geom, ba, dm, alpha, faces, rhs, max_iter=500, rtol=RTOL, norm=norm, **path
        )
        res[norm] = st

    assert res["l2"]["num_iters"] != res["linf"]["num_iters"]
    loose, strict = sorted(("l2", "linf"), key=lambda k: res[k]["num_iters"])
    r = b - _apply(res[loose]["x"], inv_dx2)
    if strict == "linf":
        assert np.max(np.abs(r)) > RTOL * np.max(np.abs(b))
    else:
        assert np.linalg.norm(r) > RTOL * np.linalg.norm(b)


def test_l2_is_the_default_and_unchanged(blockamr_session):
    """norm defaults to l2, bit-for-bit the historical behaviour."""
    geom, ba, dm, alpha, faces, _ = _helmholtz()
    rhs, _ = _spiky_rhs(ba, dm)
    kw = dict(solver="cg", precond="gmg", precond_cycles=1, max_iter=500, rtol=RTOL)
    default = _solve(geom, ba, dm, alpha, faces, rhs, **kw)
    explicit = _solve(geom, ba, dm, alpha, faces, rhs, norm="l2", **kw)
    assert default["num_iters"] == explicit["num_iters"]
    assert np.array_equal(default["x"], explicit["x"])


@pytest.mark.parametrize("solver", ["cg", "gmg"])
def test_unknown_norm_is_rejected(blockamr_session, solver):
    """Naming a norm that does not exist must fail loudly, not fall back to l2 --
    both on the Krylov path and on the native one, which validate separately."""
    geom, ba, dm, alpha, faces, _ = _helmholtz()
    rhs, _ = _spiky_rhs(ba, dm)
    with pytest.raises(RuntimeError, match="unknown norm"):
        _solve(geom, ba, dm, alpha, faces, rhs, solver=solver, norm="l1")
