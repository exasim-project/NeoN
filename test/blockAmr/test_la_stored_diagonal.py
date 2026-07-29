# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``MFFaceCoeffs``' stored fine-level diagonal.

Before S7 the matrix diagonal ``alpha - (aE+aW+aN+aS+aT+aB)`` was never stored:
both matrix-free stencils re-derived it per cell on every apply. It is now a
field of the matrix, computed once, and the stencils read it. Everything that
consumes the operator — ``test_ginkgo_face_coeffs.py``, ``test_ginkgo_bc.py``,
``test_la_matrix_formats.py``, … — is the regression check that the mat-vec did
not change. What is checked *here* is the thing those cannot see, because a
stored diagonal is invisible from the outside until it is stale or wrong:

* the stored value is exactly ``alpha - sum(faces)``, in that association order,
  so this is asserted BITWISE — a tolerance would pass a diagonal built from the
  wrong six faces on constant coefficients;
* it is BC-independent. Domain BCs enter the mat-vec through the ghost
  reflection, i.e. through the off-diagonal term, so the same coefficients must
  give the same diagonal under periodic and Dirichlet. That is the reason S7
  could land before the BC move (S6b) rather than after it;
* it distinguishes the low-side coefficients from the high-side ones, which
  constant coefficients cannot show;
* the freshness rule — it is recomputed after a write through ``coefficients()``
  or ``zero()``, including a write through a COPY of the matrix, which shares the
  freshness state exactly as ``CsrMatrix``'s assembly state is shared.

The coefficients are seeded random rather than constant throughout: with
``ux == lx == uy == ...`` every wrong choice of the six faces yields the same
number, and a bitwise assertion over that proves nothing.

The entry point is ``blockamr._blockamr._la_stored_diagonal`` — a test-facing
binding, since blockAmr has no C++ test target that builds and the diagonal is
deliberately not part of ``MatrixCoefficients``.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor

_ext = getattr(blockamr, "_blockamr", None)

N = 8
_SEED = 20260728


def _require_bindings():
    if _ext is None or not hasattr(_ext, "_la_stored_diagonal"):
        pytest.skip("blockamr._la_stored_diagonal binding not available")


def _make_mesh(periodic):
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, periodic)
    ba = blockamr.BoxArray(box)
    ba.max_size(N)  # single box -> face fabs align 1:1 with the cell fab
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _face_ba(geom, d):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    fba = blockamr.BoxArray(face_box)
    fba.max_size(N)
    return fba


def _filled(mf, values):
    """Write a host array into the (single-box) MultiFab and return the array."""
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = values
        mf.copy_from(mfi, arr)
    return values


def _random_cell(ba, dm, rng):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    return mf, _filled(mf, rng.standard_normal((N, N, N)))


def _random_face(geom, dm, d, rng):
    shape = [N, N, N]
    shape[d] = N + 1
    mf = blockamr.MultiFab(_face_ba(geom, d), dm, 1, 0)
    return mf, _filled(mf, rng.standard_normal(tuple(shape)))


def _empty_cell(ba, dm):
    return blockamr.MultiFab(ba, dm, 1, 0)


def _host(mf):
    (arr,) = [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]
    return arr[:, :, :, 0]


def _expected_diag(alpha, ux, lx, uy, ly, uz, lz):
    """alpha - (aE+aW+aN+aS+aT+aB), in the operator's own association order.

    aE is the HIGH x face of the cell, aW the LOW one, and so on — the same
    ux(i+1)/lx(i) split the stencils spell. The parenthesisation is copied from
    the kernel deliberately: float addition is not associative, and a bitwise
    comparison against a differently-ordered sum would be luck, not agreement.
    """
    a_e = ux[1:, :, :]
    a_w = lx[:-1, :, :]
    a_n = uy[:, 1:, :]
    a_s = ly[:, :-1, :]
    a_t = uz[:, :, 1:]
    a_b = lz[:, :, :-1]
    return alpha - (a_e + a_w + a_n + a_s + a_t + a_b)


def _problem(periodic, rng, symmetric=True):
    """Random alpha and six random face fields on a single-box mesh."""
    geom, ba, dm = _make_mesh(periodic)
    alpha, alpha_h = _random_cell(ba, dm, rng)
    faces = {}
    for name, d in (("x", 0), ("y", 1), ("z", 2)):
        upper, upper_h = _random_face(geom, dm, d, rng)
        if symmetric:
            lower, lower_h = upper, upper_h
        else:
            lower, lower_h = _random_face(geom, dm, d, rng)
        faces[f"u{name}"] = (upper, upper_h)
        faces[f"l{name}"] = (lower, lower_h)
    return geom, ba, dm, (alpha, alpha_h), faces


def _call(geom, alpha, faces, **kwargs):
    order = ("ux", "lx", "uy", "ly", "uz", "lz")
    try:
        _ext._la_stored_diagonal(
            alpha,
            *[faces[k][0] for k in order],
            geom,
            executor=gko_executor("reference"),
            **kwargs,
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise


_BC_CASES = [
    ("periodic", [1, 1, 1], ["periodic"] * 6),
    ("dirichlet", [0, 0, 0], ["dirichlet"] * 6),
]


@pytest.mark.parametrize("case, periodic, bc", _BC_CASES)
def test_stored_diagonal_is_alpha_minus_face_sum(blockamr_session, case, periodic, bc):
    """The stored diagonal is exactly alpha - sum(faces), under either BC.

    Bitwise, because the expectation is computed with the kernel's own operator
    order. Running it under Dirichlet as well as periodic is the BC-independence
    claim S7 rests on: the reflection that folds a domain BC in sets a GHOST
    value, which the mat-vec consumes through the off-diagonal term, so the same
    coefficients must produce the same diagonal either way.
    """
    _require_bindings()
    rng = np.random.default_rng(_SEED)
    geom, ba, dm, (alpha, alpha_h), faces = _problem(periodic, rng)
    out = _empty_cell(ba, dm)

    _call(geom, alpha, faces, diag_out=out, bc=bc)

    expected = _expected_diag(alpha_h, *[faces[k][1] for k in ("ux", "lx", "uy", "ly", "uz", "lz")])
    np.testing.assert_array_equal(_host(out), expected, err_msg=f"{case}: stored diagonal")


def test_stored_diagonal_uses_the_low_face_coefficients(blockamr_session):
    """aW/aS/aB come from l{x,y,z}, not from a second read of u{x,y,z}.

    An asymmetric matrix is the only place this is visible: the symmetric factory
    ALIASES lower[d] onto upper[d], so a diagonal that read the high-side field
    six times would agree with the correct one everywhere. Here the two are
    independent random fields.
    """
    _require_bindings()
    rng = np.random.default_rng(_SEED + 1)
    geom, ba, dm, (alpha, alpha_h), faces = _problem([1, 1, 1], rng, symmetric=False)
    out = _empty_cell(ba, dm)

    _call(geom, alpha, faces, diag_out=out, symmetry="asymmetric")

    order = ("ux", "lx", "uy", "ly", "uz", "lz")
    got = _host(out)
    np.testing.assert_array_equal(got, _expected_diag(alpha_h, *[faces[k][1] for k in order]))
    # The control: the same formula with the low fields replaced by the high ones
    # is a genuinely different diagonal, so the assertion above had something to
    # discriminate.
    high_only = _expected_diag(
        alpha_h, *[faces[k][1] for k in ("ux", "ux", "uy", "uy", "uz", "uz")]
    )
    assert np.max(np.abs(got - high_only)) > 1.0


@pytest.mark.parametrize("through_copy", [False, True], ids=["direct", "via_copy"])
def test_stored_diagonal_refreshes_after_a_coefficient_write(blockamr_session, through_copy):
    """A write through coefficients() invalidates the stored diagonal.

    This is the property S7 had to create on the matrix-free format, which needed
    no freshness rule before: the operator used to derive the diagonal from the
    fields on every apply, so an in-place coefficient update was picked up for
    free. `via_copy` writes through a COPY of the matrix and reads the diagonal
    back through the ORIGINAL — copies share the coefficient fields, so they must
    share the freshness state too, or one copy hands out a diagonal for
    coefficients another has already replaced.
    """
    _require_bindings()
    rng = np.random.default_rng(_SEED + 2)
    geom, ba, dm, (alpha, alpha_h), faces = _problem([1, 1, 1], rng)
    alpha2, alpha2_h = _random_cell(ba, dm, rng)
    first = _empty_cell(ba, dm)
    second = _empty_cell(ba, dm)

    _call(
        geom,
        alpha,
        faces,
        diag_out=first,
        alpha2=alpha2,
        diag2_out=second,
        rewrite_through_copy=through_copy,
    )

    order = ("ux", "lx", "uy", "ly", "uz", "lz")
    face_h = [faces[k][1] for k in order]
    np.testing.assert_array_equal(_host(first), _expected_diag(alpha_h, *face_h))
    np.testing.assert_array_equal(_host(second), _expected_diag(alpha2_h, *face_h))


def test_stored_diagonal_is_zero_after_zero(blockamr_session):
    """zero() invalidates it too — a zeroed matrix has a zero diagonal.

    zero() writes the fields directly rather than through coefficients(), so it
    is a second, independent place the flag has to be set; without it the matrix
    would report the diagonal of the coefficients it no longer holds.
    """
    _require_bindings()
    rng = np.random.default_rng(_SEED + 3)
    geom, ba, dm, (alpha, _), faces = _problem([1, 1, 1], rng)
    first = _empty_cell(ba, dm)
    zeroed = _empty_cell(ba, dm)

    _call(geom, alpha, faces, diag_out=first, diag_zero_out=zeroed)

    assert np.max(np.abs(_host(first))) > 0.0
    np.testing.assert_array_equal(_host(zeroed), np.zeros((N, N, N)))
