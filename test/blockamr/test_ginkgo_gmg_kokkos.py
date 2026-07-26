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


def _variable_helmholtz(n, max_size=None):
    """The same operator with a SMOOTHLY VARYING b: face coefficient -b/dx^2 with
    b = 1 + 0.5 sin(2 pi x), periodic on [0, 1] so the wrap face stays consistent.

    Needed by the coefficient-precision tests specifically. The constant-coefficient
    problem above cannot exercise a narrowed coefficient at all -- see
    test_bf16_coeffs_are_exact_on_a_power_of_two_operator.
    """
    box_size = n if max_size is None else max_size
    geom, ba, dm = _make_mesh(n, max_size)
    dx = geom.cell_size()
    alpha = blockamr.MultiFab(ba, dm, 1, 0)
    alpha.set_val(1.0)
    faces = []
    for d in range(3):
        dom = geom.domain()
        face_box = blockamr.Box(dom.small_end(), dom.big_end())
        face_box.surrounding_nodes(d)
        face_ba = blockamr.BoxArray(face_box)
        face_ba.max_size(box_size)
        mf = blockamr.MultiFab(face_ba, dm, 1, 0)
        for mfi in blockamr.MFIterator(mf):
            arr = mf.copy_to_host(mfi)
            lo = mfi.valid_box().small_end()
            # Faces sit at integer multiples of dx along d, cell centres across it.
            xi = lo[0] + np.arange(arr.shape[0]) + (0.0 if d == 0 else 0.5)
            b = 1.0 + 0.5 * np.sin(2.0 * np.pi * xi * dx[0])
            arr[:, :, :, 0] = (-b / dx[d] ** 2)[:, None, None]
            mf.copy_from(mfi, arr)
        faces.append(mf)
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
    # solver defaults to cg but is overridable: the mixed-precision tests below need
    # solver="mpir" over the same fields, tolerance and iteration cap.
    kwargs.setdefault("solver", "cg")
    s = _solver_or_skip(geom, alpha, faces, max_iter=200, rtol=1e-10, precond_cycles=1, **kwargs)
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


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_level0_agglomeration_is_the_same_preconditioner(blockamr_session, precision):
    """gmg_agg_l0_size gives level 0 its own boxes, at the cost of one copy per apply
    in each direction -- the flat vectors CG hands the preconditioner are in the
    caller's cell order, not level 0's. A copy that lost the ordering would still
    precondition something, just worse, so the gate is that CG cannot tell: same
    answer, same iteration count."""
    geom, ba, dm, alpha, faces = _helmholtz(16, 8)
    rhs = _rhs(ba, dm)
    kw = dict(precond="gmg_kokkos", gmg_precision=precision)
    ref = _solve(geom, ba, dm, alpha, faces, rhs, **kw)
    agg = _solve(geom, ba, dm, alpha, faces, rhs, gmg_agg_l0_size=16, **kw)
    assert agg["num_iters"] == ref["num_iters"]
    assert np.array_equal(agg["sol"], ref["sol"])


@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_bf16_preconditions_the_same_system(blockamr_session, max_size):
    """A bf16 hierarchy is a deliberately approximate preconditioner, and the point of
    a preconditioner is that being approximate is allowed.

    It cannot move the fixed point -- the operator CG applies and the residual CG
    stops on are fp64 whatever the V-cycle is stored in -- so the answer has to match
    the fp32 hierarchy's to the same tolerance the two of them stop at. What it IS
    allowed to cost is iterations, since ~3 decimal digits per stored value make the
    cycle a few percent weaker. A couple of extra iterations is the deal on offer; an
    order more would mean the apply is losing information rather than precision, so
    the margin is bounded rather than waived.
    """
    geom, ba, dm, alpha, faces = _helmholtz(16, max_size)
    rhs = _rhs(ba, dm)
    kw = dict(precond="gmg_kokkos")
    ref = _solve(geom, ba, dm, alpha, faces, rhs, gmg_precision="fp32", **kw)
    bf = _solve(geom, ba, dm, alpha, faces, rhs, gmg_precision="bf16", **kw)

    assert bf["converged"] is True
    assert np.max(np.abs(bf["sol"] - ref["sol"])) < 1e-7 * max(1.0, np.max(np.abs(ref["sol"])))
    assert bf["num_iters"] <= ref["num_iters"] + 3
    # And it really is the bf16 path: a weaker preconditioner may not be free, and
    # silently running fp32 would be.
    assert not np.array_equal(bf["sol"], ref["sol"])


def test_gmg_config_carries_bf16(blockamr_session):
    """The pydantic layer says which precisions EXIST; the solver says which precond
    supports each. So GmgConfig has to accept bf16 and hand it through unchanged --
    otherwise the only way to reach the bf16 hierarchy is the raw kwarg."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    cfg = blockamr.GmgConfig(precision="bf16")
    assert cfg.kwargs()["gmg_precision"] == "bf16"
    # precond_cycles is _solve's own argument; everything else splats.
    kw = {k: v for k, v in cfg.kwargs().items() if k != "precond_cycles"}
    st = _solve(geom, ba, dm, alpha, faces, _rhs(ba, dm), precond="gmg_kokkos", **kw)
    assert st["converged"] is True


def test_bf16_needs_the_kokkos_precond(blockamr_session):
    """The shipped GmgPrecondT hierarchy is fp64/fp32; asking it for bf16 has to name
    the precond that has one rather than fall back to fp64 under a bf16 label."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    with pytest.raises(RuntimeError, match="bf16.*gmg_kokkos"):
        _solve(geom, ba, dm, alpha, faces, _rhs(ba, dm), precond="gmg", gmg_precision="bf16")


# ---------------------------------------------------------------------------
# gmg_coeff_precision: the coefficients narrowed independently of the fields
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_bf16_coeffs_are_exact_on_a_power_of_two_operator(blockamr_session, max_size):
    """On the constant-coefficient Helmholtz, bf16 coefficients change NOTHING --
    and that is a statement about the test problem, not about bf16.

    Every coefficient this operator has is a power of two: alpha = 1, and the face
    coefficient is -1/dx^2 = -256 at 16^3 on the unit cube. The restriction weights
    are 1/4 (faces) and 1/8 (alpha), so every coarse level holds a power-of-two
    multiple as well. bf16 has 8 exponent bits and stores all of them exactly, so
    the hierarchy is bit-for-bit the fp32 one.

    Pinned deliberately, because it is the trap in measuring this option: the
    benchmark problem is a constant-coefficient Laplacian on a power-of-two grid, so
    it reports the full bandwidth saving at zero iteration cost and would recommend
    bf16 coefficients on evidence that cannot distinguish them from fp32. The
    variable-coefficient test below is the one that can.
    """
    geom, ba, dm, alpha, faces = _helmholtz(16, max_size)
    rhs = _rhs(ba, dm)
    kw = dict(precond="gmg_kokkos", gmg_precision="fp32")
    ref = _solve(geom, ba, dm, alpha, faces, rhs, **kw)
    bf = _solve(geom, ba, dm, alpha, faces, rhs, gmg_coeff_precision="bf16", **kw)
    assert bf["num_iters"] == ref["num_iters"]
    assert np.array_equal(bf["sol"], ref["sol"])


@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_bf16_coeffs_on_a_variable_operator(blockamr_session, max_size):
    """With a coefficient bf16 cannot represent, the preconditioner really is a
    different one -- and the answer must not move regardless.

    That is the whole argument for narrowing the coefficients rather than the
    fields: CG applies the fp64 operator and stops on the fp64 residual, so a
    rounded coefficient perturbs only the preconditioner. It may cost iterations; it
    may not cost accuracy. Both halves are asserted, plus the negative -- that the
    solution DIFFERS from the fp32-coefficient one, so this cannot be passing
    because the option was silently ignored.
    """
    geom, ba, dm, alpha, faces = _variable_helmholtz(16, max_size)
    rhs = _rhs(ba, dm)
    kw = dict(precond="gmg_kokkos", gmg_precision="fp32")
    ref = _solve(geom, ba, dm, alpha, faces, rhs, **kw)
    bf = _solve(geom, ba, dm, alpha, faces, rhs, gmg_coeff_precision="bf16", **kw)

    assert bf["converged"] is True
    assert not np.array_equal(bf["sol"], ref["sol"])
    assert np.max(np.abs(bf["sol"] - ref["sol"])) < 1e-7 * max(1.0, np.max(np.abs(ref["sol"])))
    assert bf["num_iters"] <= ref["num_iters"] + 3


def test_coeff_precision_may_not_be_wider_than_the_fields(blockamr_session):
    """fp32 coefficients under an fp32 hierarchy is the default; fp64 ones would buy
    accuracy in the array that needs it least and pay traffic for it. Rejected rather
    than instantiated."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    with pytest.raises(RuntimeError, match="wider than precision"):
        _solve(
            geom, ba, dm, alpha, faces, _rhs(ba, dm),
            precond="gmg_kokkos", gmg_precision="fp32", gmg_coeff_precision="fp64",
        )


def test_coeff_precision_needs_the_kokkos_precond(blockamr_session):
    """The shipped GmgPrecondT stores one type per level, so accepting the option
    there would report a narrowed-coefficient timing for a hierarchy that never
    narrowed anything."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    with pytest.raises(RuntimeError, match="gmg_coeff_precision.*gmg_kokkos"):
        _solve(
            geom, ba, dm, alpha, faces, _rhs(ba, dm),
            precond="gmg", gmg_coeff_precision="fp32",
        )


def test_gmg_config_carries_coeff_precision(blockamr_session):
    """Same reason as bf16 above: without the pydantic field the only way to reach a
    narrowed coefficient hierarchy is the raw kwarg."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    cfg = blockamr.GmgConfig(precision="fp32", coeff_precision="bf16")
    assert cfg.kwargs()["gmg_coeff_precision"] == "bf16"
    kw = {k: v for k, v in cfg.kwargs().items() if k != "precond_cycles"}
    st = _solve(geom, ba, dm, alpha, faces, _rhs(ba, dm), precond="gmg_kokkos", **kw)
    assert st["converged"] is True


# ---------------------------------------------------------------------------
# solver="mpir": an fp32 Krylov inside an fp64 refinement loop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_mpir_reaches_the_fp64_answer(blockamr_session, max_size):
    """The inner solve is fp32; the answer must still be the fp64 one.

    That is the entire claim of iterative refinement and it is not obvious: fp32 CG
    on its own stalls around 1e-7 relative, well short of the 1e-10 asked for here.
    What rescues it is that the OUTER loop never leaves fp64 -- r = b - A x uses the
    fp64 operator and the stopping test measures that r -- so the fp32 solve is only
    ever computing a correction, and a correction is allowed to be wrong.

    Checked against the fp64 CG's own answer rather than a tolerance in the
    abstract, so a refinement loop that quietly stopped at fp32's floor would fail.
    """
    geom, ba, dm, alpha, faces = _variable_helmholtz(16, max_size)
    rhs = _rhs(ba, dm)
    ref = _solve(geom, ba, dm, alpha, faces, rhs, precond="gmg_kokkos", gmg_precision="fp32")
    mp = _solve(
        geom, ba, dm, alpha, faces, rhs,
        solver="mpir", precond="gmg_kokkos", gmg_precision="fp32",
        mp_inner_rtol=1e-2, mp_inner_max_iter=20,
    )
    assert mp["converged"] is True
    # 1e-10 is the rtol both stop at; the two answers must agree to about that,
    # with a decade of slack for a different iteration order.
    assert np.max(np.abs(mp["sol"] - ref["sol"])) < 1e-9 * max(1.0, np.max(np.abs(ref["sol"])))


def test_mpir_needs_the_kokkos_precond(blockamr_session):
    """gmg_kokkos is the only preconditioner with an fp32 apply, so mpir names it
    rather than silently building an fp64 inner solve under a mixed-precision label."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    with pytest.raises(RuntimeError, match="mpir.*gmg_kokkos"):
        _solve(geom, ba, dm, alpha, faces, _rhs(ba, dm), solver="mpir", precond="gmg")


def test_mpir_inner_tolerance_changes_the_outer_count(blockamr_session):
    """The outer contraction factor IS the inner tolerance, so a looser inner solve
    must cost more outer steps. Pins that mp_inner_rtol is actually plumbed through
    -- a knob that reached nothing would give the same count for both."""
    geom, ba, dm, alpha, faces = _variable_helmholtz(16)
    rhs = _rhs(ba, dm)
    kw = dict(solver="mpir", precond="gmg_kokkos", gmg_precision="fp32", mp_inner_max_iter=40)
    loose = _solve(geom, ba, dm, alpha, faces, rhs, mp_inner_rtol=1e-1, **kw)
    tight = _solve(geom, ba, dm, alpha, faces, rhs, mp_inner_rtol=1e-4, **kw)
    assert loose["converged"] and tight["converged"]
    assert loose["num_iters"] > tight["num_iters"]


def test_rejects_an_unknown_precision(blockamr_session):
    """gmg_kokkos parses the spelling itself (the string goes straight through), so a
    typo must raise there too rather than quietly select fp64."""
    geom, ba, dm, alpha, faces = _helmholtz(16)
    with pytest.raises(RuntimeError, match="unknown precision 'fp8'"):
        _solve(
            geom, ba, dm, alpha, faces, _rhs(ba, dm),
            precond="gmg_kokkos", gmg_precision="fp8",
        )


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
