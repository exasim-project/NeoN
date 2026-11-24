# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Boundary conditions on the ``blockamr::la`` path: left on the face coefficients.

A domain boundary reaches the mat-vec as a GHOST value — ``FaceCoeffOp``
reflecting the ghost layer — and ``ops::Laplacian`` writes coefficients that carry
no boundary condition of their own:

    aF      -> -0.5*(gC + gC)/dx**2   the boundary CELL's gamma, twice: a domain
                                      face has no second cell to average over
    rhs(C)  -= aF * scale * g         with bc_data; g read from ITS ghost cell

``(sign, scale)`` is ``(-1, 2)`` for Dirichlet and ``(+1, dx[d])`` for Neumann,
straight out of ``core/bc.hpp``'s ghost fill. ``sign`` never appears in what the
operator writes: the consumer derives ``(sign-1)*aF`` from the live ``aF`` — the
stencil as ``aF*(sign*pC)`` against a diagonal of ``alpha - sum(faces)``.

S6b briefly had ``ops::Laplacian`` fold instead (``aF -> 0`` plus ``(sign-1)*aF``
on ``alpha``), which gives the same FINE matrix and was reverted for the coarse
ones: ``(sign-1)*aF`` is ``2*gamma/dx**2``, dx-DEPENDENT, and the GMG hierarchy
coarsens ``alpha`` as a dx-INDEPENDENT density. Measured 12/13/14 CG+GMG
iterations at 64/128/256**3 folded against 8/8/8 unfolded. See ``laplacian.cpp``.

What is checked here, on periodic / Dirichlet / Neumann / a mixed array:

* the coefficients ``ops::Laplacian`` writes, BITWISE, per BC kind — the domain
  faces and the ``alpha`` it must leave alone, on a symmetric AND on an
  asymmetric matrix (the latter is the only row in the suite that reaches the
  operator's low-side write at all);
* the format reproduces the LEGACY ``FaceCoeffSolver``. That is the load-bearing
  test: it is the only one that can tell a self-consistent wrong convention from
  the right one;
* the inhomogeneous rhs term, bitwise against the datum, and again against the
  legacy path's ``bc_data`` answer — with an anti-vacuity check that dropping the
  datum moves the answer.

The BC contract against a reference that shares no C++ code at all —
``test_la_dense_oracle.py`` — is where the fold is checked against an independent
numpy operator rather than against a second path through ``stencil.hpp``.

The legacy path is deliberately NOT changed and never folded into its
coefficients: it shares them with the GMG hierarchy, which reflects ghosts of its
own per level, so folding into them would apply every BC twice per level
(plans/blockamr-la-implementation.md, "S6b — RESCOPED"). ``test_ginkgo_bc.py`` is
that path's test and is untouched; this path now agrees with it on the stored
coefficients as well as on the answer.

The entry points are ``blockamr._blockamr._la_system_solve`` / ``_la_system_probe``
— test-facing bindings, since blockAmr has no C++ test target that builds.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor

_ext = getattr(blockamr, "_blockamr", None)

N = 16
_SOLVE_KWARGS = dict(solver="cg", max_iter=5000, rtol=1e-14, atol=0.0)

# The same bar S4 and S5 set for their agreement rows. Two CG solves of two
# roundings of one operator -- the legacy path folds an inhomogeneous datum by
# applying the operator to a zero vector and subtracting the result inside the
# solver, this one folds the same constant at assembly -- so this cannot be
# bitwise. Measured worst case across every row below is 1.6e-15 on solutions
# whose peak is 6.1e-1, i.e. ~600x of headroom; the bitwise assertions elsewhere
# in this file are what actually pin the arithmetic.
_AGREE_TOL = 1e-12


def _require_bindings():
    if _ext is None or not hasattr(_ext, "_la_system_solve"):
        pytest.skip("blockamr._la_system_solve binding not available")


# Every BC kind the fold has a branch for, plus one array that mixes all three.
# `periodic` is the control: it folds nothing, and it is the case that would let a
# broken fold through unnoticed if it were the only one here.
_BC_CASES = [
    ("periodic", [1, 1, 1], ["periodic"] * 6),
    ("dirichlet", [0, 0, 0], ["dirichlet"] * 6),
    ("neumann", [0, 0, 0], ["neumann"] * 6),
    (
        "mixed",
        [0, 0, 1],
        ["dirichlet", "dirichlet", "neumann", "neumann", "periodic", "periodic"],
    ),
]

_INHOM_CASES = [c for c in _BC_CASES if c[0] != "periodic"]


def _make_mesh(periodic):
    """Single-box mesh on [0,1]^3, N cells per side, with the given periodicity."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, periodic)
    ba = blockamr.BoxArray(box)
    ba.max_size(N)  # single box -> face fabs align 1:1 with the cell fab
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _face_field(geom, dm, d, value):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(N)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _inv_dx2(geom, d):
    """1/dx**2 spelled as ops::Laplacian spells it: 1.0 / (dx*dx), not dx**-2.

    The coefficient comparisons below are bitwise, so the expression and not just
    the value has to match.
    """
    dx = geom.cell_size()
    return 1.0 / (dx[d] * dx[d])


def _raw_faces(geom, dm):
    """The unfolded faces the LEGACY solver takes: -1/dx**2 on every face."""
    return [_face_field(geom, dm, d, -_inv_dx2(geom, d)) for d in range(3)]


def _out_fields(geom, ba, dm):
    """Zeroed receivers for the assembled (alpha, ux, uy, uz, lx, ly, lz).

    The three LOW-side receivers are written by the binding only when the matrix
    reports a ``lower`` view — a symmetric one does not — so on a symmetric row
    they stay zero, and that is itself the assertion that the operator wrote
    nothing to the low side.
    """
    return (
        _const_cell(ba, dm, 0.0),
        *(_face_field(geom, dm, d, 0.0) for d in range(3)),
        *(_face_field(geom, dm, d, 0.0) for d in range(3)),
    )


def _one_box(mf):
    """The single box of a single-box MultiFab, as (i, j, k) without the component."""
    boxes = [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]
    assert len(boxes) == 1
    return boxes[0][:, :, :, 0]


def _grown_box(mf):
    boxes = [mf.copy_grown_to_host(mfi) for mfi in blockamr.MFIterator(mf)]
    assert len(boxes) == 1
    return boxes[0][:, :, :, 0]


def _max_abs_diff(a, b):
    return float(np.max(np.abs(_one_box(a) - _one_box(b))))


def _random_rhs(ba, dm, seed=42):
    """Seeded random rhs -- the full spectrum, so CG has to iterate across it.

    A smooth rhs on this problem is nearly an eigenvector and converges in three
    iterations, which barely exercises the mat-vec (S4 handoff §9).
    """
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        mf.copy_from(mfi, arr)
    return mf


def _zero_sol(ba, dm):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return sol


def _bc_data(ba, dm, seed=1607):
    """MLMG-style carrier: cell MultiFab, 1 ghost, the datum in the GHOST layer.

    Seeded random over the whole grown region rather than a constant: both paths
    read the datum at one specific ghost cell per boundary face, and a constant
    would agree with every wrong choice of cell. Only the domain-boundary ghost
    layer is ever read; the rest is noise neither path consults.
    """
    mf = blockamr.MultiFab(ba, dm, 1, 1)
    mf.set_val(0.0)
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        arr[:, :, :, 0] = rng.uniform(-1.0, 1.0, arr.shape[:3])
        mf.copy_grown_from(mfi, arr)
    return mf


def _skip_on_missing_ginkgo(exc, executor="reference"):
    if "without Ginkgo" in str(exc):
        pytest.skip("blockamr built without Ginkgo")
    if executor == "cuda":
        pytest.skip(f"cuda executor unavailable: {exc}")
    raise exc


def _probe(gamma, alpha, geom, rhs, bc, out, **kwargs):
    try:
        return _ext._la_system_probe(
            gamma,
            alpha,
            geom,
            rhs,
            *out,
            executor=gko_executor("reference"),
            bc=bc,
            **kwargs,
        )
    except RuntimeError as exc:
        _skip_on_missing_ginkgo(exc)


def _system_solve(gamma, alpha, geom, rhs, sol, bc, out, **kwargs):
    try:
        return _ext._la_system_solve(
            gamma,
            alpha,
            geom,
            rhs,
            sol,
            *out,
            executor=gko_executor("reference"),
            bc=bc,
            **_SOLVE_KWARGS,
            **kwargs,
        )
    except RuntimeError as exc:
        _skip_on_missing_ginkgo(exc)


def _reference_solve(geom, ba, dm, rhs, bc, bc_data=None):
    """The legacy FaceCoeffSolver: RAW coefficients plus `bc`, folded at apply time."""
    alpha = _const_cell(ba, dm, 1.0)
    fx, fy, fz = _raw_faces(geom, dm)
    sol = _zero_sol(ba, dm)
    kwargs = {} if bc_data is None else dict(bc_data=bc_data)
    try:
        s = blockamr.FaceCoeffSolver(
            alpha,
            fx,
            fx,
            fy,
            fy,
            fz,
            fz,
            geom,
            executor=gko_executor("reference"),
            bc=bc,
            **_SOLVE_KWARGS,
            **kwargs,
        )
    except RuntimeError as exc:
        _skip_on_missing_ginkgo(exc)
    stats = s.solve(rhs, sol)
    assert stats["converged"] is True, f"reference did not converge: {dict(stats)}"
    return sol, stats


def _sides(bc):
    """(direction, side index, kind) for every side that is not periodic."""
    return [(s // 2, s, bc[s]) for s in range(6) if bc[s] != "periodic"]


def _expected_faces(geom, dm, bc):
    """The face fields ops::Laplacian writes for gamma = 1 under `bc`.

    -1/dx**2 on every face, a non-periodic domain one included: the boundary
    condition is not in the coefficient, only the gamma AVERAGE is -- and there
    the operator averages the boundary cell's value with itself for want of a
    second cell. Written as the operator spells it, since the comparison is
    bitwise; at gamma = 1 it lands on the same number as an interior face, which
    is why the varying-gamma row in ``test_la_linear_system.py`` is what pins
    WHICH cell it read.
    """
    faces = _raw_faces(geom, dm)
    for _, s, _kind in _sides(bc):
        d = s // 2
        edge = 0 if s % 2 == 0 else -1
        face = faces[d]
        for mfi in blockamr.MFIterator(face):
            arr = face.copy_to_host(mfi)
            sl = [slice(None)] * 3
            sl[d] = edge
            arr[tuple(sl) + (0,)] = -0.5 * (1.0 + 1.0) * _inv_dx2(geom, d)
            face.copy_from(mfi, arr)
    return faces


def _expected_alpha(alpha_val=1.0):
    """alpha, untouched: the operator writes NO diagonal source, under ANY bc.

    This is the assertion that would object to a boundary term being folded back
    onto the diagonal source. `(sign-1)*aF` there is dx-DEPENDENT and ends up
    coarsened as a density by the GMG hierarchy; on the face it is coarsened by
    the law it obeys. Dirichlet is the kind that would move (`sign-1 == -2`);
    Neumann (`sign-1 == 0`) never could, and is in this parametrization so that
    the two kinds are held to the same statement rather than only the one where a
    mistake is expressible.
    """
    return np.full((N, N, N), alpha_val)


def _expected_rhs_fold(geom, bc, datum):
    """-aF * scale * g per non-periodic side, accumulated in the operator's order.

    `datum` is the GROWN bc_data array, so index 0 along a direction is the ghost
    below the domain and index N+1 the one above -- exactly the cells the ghost
    fill of core/bc.hpp writes and this fold reads.
    """
    dx = geom.cell_size()
    out = np.zeros((N, N, N))
    for d in range(3):
        coef = -0.5 * (1.0 + 1.0) * _inv_dx2(geom, d)
        for s in (2 * d, 2 * d + 1):
            kind = bc[s]
            if kind == "periodic":
                continue
            low = s % 2 == 0
            scale = 2.0 if kind == "dirichlet" else dx[d]
            sl_out = [slice(None)] * 3
            sl_out[d] = 0 if low else N - 1
            sl_g = [slice(1, N + 1)] * 3
            sl_g[d] = 0 if low else N + 1
            out[tuple(sl_out)] -= coef * scale * datum[tuple(sl_g)]
    return out


@pytest.mark.parametrize("symmetry", ["symmetric", "asymmetric"])
@pytest.mark.parametrize("case, periodic, bc", _BC_CASES)
def test_laplacian_writes_the_boundary_face_coefficient(
    blockamr_session, symmetry, case, periodic, bc
):
    """What the operator wrote, per BC kind, asserted BITWISE.

    The claim is that the domain-face coefficient is LIVE and the diagonal source
    is untouched -- the boundary condition belongs to whoever applies the matrix,
    which is the only place it can be re-derived per GMG level. Bitwise because
    the claim is exactness: a tolerance would pass a fold that put `(sign-1)*aF`
    back on alpha on a mesh where 1/dx**2 happened to be small.

    NEUMANN reaches nothing else in this suite at coefficient level, and it is the
    kind where the two conventions coincide (`sign-1 == 0`), so it is the row that
    says the statement holds for reasons rather than by luck; DIRICHLET is the row
    where the old fold was expressible and so the one that can regress.

    The `asymmetric` row is what exercises ops::Laplacian's LOW-side write at all:
    every other test in this suite runs the default symmetric matrix, where the
    matrix reports no ``lower`` and the operator writes only ``upper``. One face
    value carries both roles, so the low side must come out bitwise equal to the
    high side -- and on the symmetric rows it must be untouched, because there
    ``lower[d]`` aliases ``upper[d]`` and a second write would double every
    coefficient.
    """
    _require_bindings()
    geom, ba, dm = _make_mesh(periodic)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    out = _out_fields(geom, ba, dm)
    tag = f"{symmetry}/{case}"

    _probe(gamma, alpha, geom, _random_rhs(ba, dm), bc, out, symmetry=symmetry)

    np.testing.assert_array_equal(
        _one_box(out[0]), _expected_alpha(), err_msg=f"{tag}: alpha must be untouched"
    )
    want = _expected_faces(geom, dm, bc)
    for d, name in enumerate("xyz"):
        np.testing.assert_array_equal(
            _one_box(out[1 + d]), _one_box(want[d]), err_msg=f"{tag}: u{name}"
        )
    for d, name in enumerate("xyz"):
        high = _one_box(want[d])
        want_low = high if symmetry == "asymmetric" else np.zeros_like(high)
        np.testing.assert_array_equal(_one_box(out[4 + d]), want_low, err_msg=f"{tag}: l{name}")


@pytest.mark.parametrize("case, periodic, bc", _BC_CASES)
def test_folded_system_matches_the_legacy_face_coeff_solver(blockamr_session, case, periodic, bc):
    """The assembled matrix IS the matrix the legacy ghost reflection applies.

    The reference gets HAND-WRITTEN raw coefficients plus `bc`; this path gets the
    same `bc` and coefficients ops::Laplacian assembled. Two independently written
    coefficient sets under one boundary convention, so this is the test that can
    tell a self-consistent wrong convention (a gamma read off the wrong side of a
    domain face, a scale of 1 where 2 belongs) from the right one.
    """
    _require_bindings()
    geom, ba, dm = _make_mesh(periodic)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)

    ref, ref_stats = _reference_solve(geom, ba, dm, _random_rhs(ba, dm), bc)

    sol = _zero_sol(ba, dm)
    out = _out_fields(geom, ba, dm)
    stats = _system_solve(gamma, alpha, geom, _random_rhs(ba, dm), sol, bc, out)

    assert stats["converged"] is True, f"{case} did not converge: {dict(stats)}"
    assert stats["num_iters"] > 10, f"{case}: too few CG iterations to mean anything"
    diff = _max_abs_diff(ref, sol)
    assert diff < _AGREE_TOL, (
        f"{case}: |FaceCoeffSolver - folded LinearSystem| = {diff} "
        f"exceeds {_AGREE_TOL} (ref {dict(ref_stats)}, got {dict(stats)})"
    )


@pytest.mark.parametrize("case, periodic, bc", _INHOM_CASES)
def test_laplacian_writes_the_inhomogeneous_datum_into_the_rhs(
    blockamr_session, case, periodic, bc
):
    """bc_data lands on the rhs as -aF*scale*g, BITWISE, from the right ghost cell.

    The rhs goes in at zero so what comes out is the fold alone. scale is 2 for
    Dirichlet (the datum is u ON the face) and dx for Neumann (it is du/dn), which
    is the one place the two kinds differ by more than a sign -- so a test with
    only Dirichlet would not see a scale mistake on the Neumann branch.
    """
    _require_bindings()
    geom, ba, dm = _make_mesh(periodic)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    data = _bc_data(ba, dm)
    rhs = _const_cell(ba, dm, 0.0)
    out = _out_fields(geom, ba, dm)

    _probe(gamma, alpha, geom, rhs, bc, out, bc_data=data)

    np.testing.assert_array_equal(
        _one_box(rhs),
        _expected_rhs_fold(geom, bc, _grown_box(data)),
        err_msg=f"{case}: inhomogeneous rhs fold",
    )


@pytest.mark.parametrize("case, periodic, bc", _INHOM_CASES)
def test_inhomogeneous_system_matches_the_legacy_face_coeff_solver(
    blockamr_session, case, periodic, bc
):
    """The affine term, folded at assembly, gives the legacy path's answer.

    The legacy path keeps `apply` linear and folds c0 = L(0) into the rhs once per
    solve; this one folds the same constant while assembling. The datum fab is the
    SAME object for both, which is what makes this a comparison of the fold rather
    than of two data conventions.
    """
    _require_bindings()
    geom, ba, dm = _make_mesh(periodic)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    data = _bc_data(ba, dm)

    ref, ref_stats = _reference_solve(geom, ba, dm, _random_rhs(ba, dm), bc, bc_data=data)

    sol = _zero_sol(ba, dm)
    out = _out_fields(geom, ba, dm)
    stats = _system_solve(gamma, alpha, geom, _random_rhs(ba, dm), sol, bc, out, bc_data=data)

    assert stats["converged"] is True, f"{case} did not converge: {dict(stats)}"
    diff = _max_abs_diff(ref, sol)
    assert diff < _AGREE_TOL, (
        f"{case}: |FaceCoeffSolver(bc_data) - LinearSystem(bc_data)| = {diff} "
        f"exceeds {_AGREE_TOL} (ref {dict(ref_stats)}, got {dict(stats)})"
    )


def test_dropping_bc_data_moves_the_answer(blockamr_session):
    """The anti-vacuity check: a silently ignored datum would pass everything above.

    Every inhomogeneous test compares two paths, and two paths that both ignored
    bc_data would agree perfectly. So: the same problem with and without the datum
    must give different solutions, by much more than the agreement bar.
    """
    _require_bindings()
    geom, ba, dm = _make_mesh([0, 0, 0])
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    bc = ["dirichlet"] * 6

    homogeneous = _zero_sol(ba, dm)
    _system_solve(
        gamma, alpha, geom, _random_rhs(ba, dm), homogeneous, bc, _out_fields(geom, ba, dm)
    )
    inhomogeneous = _zero_sol(ba, dm)
    _system_solve(
        gamma,
        alpha,
        geom,
        _random_rhs(ba, dm),
        inhomogeneous,
        bc,
        _out_fields(geom, ba, dm),
        bc_data=_bc_data(ba, dm),
    )

    diff = _max_abs_diff(homogeneous, inhomogeneous)
    assert diff > 1e-4, f"bc_data may be a no-op: the two solutions differ by only {diff}"
