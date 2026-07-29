# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The IBM **API surface**: error messages, the staleness guard, the methods
that do not exist yet, and the uniformity claim.

Companion to ``test_ibm_rungs.py`` (the ordered ladder) and
``test_ibm_convergence.py`` (the MMS suite). This file holds what those two do
not: the parts of ``plans/IBM/ibm-verification-plan.md`` §8 they leave
uncovered, the §3 staleness rule of ``plans/IBM/ibm-row-format.md``, and the
API for ``penalization``, ``cutCell``, moving bodies (T17) and per-patch force
diagnostics (T18) — written against the API we *want*, so most of it is red.

**Error paths already covered elsewhere, deliberately not duplicated here:**

===========================================  ===========================================
verification plan §8 row                     where it lives
===========================================  ===========================================
unknown ibm name + the valid list            ``test_ibm_rungs.py``
``ibm_bc`` keys vs ``mesh.bodies``           ``test_ibm_rungs.py``
ibm requested with no bodies                 ``test_ibm_rungs.py``
step method asked for operator evaluation    ``test_ibm_rungs.py``
deferred method refuses to run (evaluate)    ``test_ibm_rungs.py``
**thin gap between two bodies**              **here** — nothing covered it
===========================================  ===========================================

``cutCell``'s *schema* half (the name is legal, and it is legal at
``IBM.lookup`` rather than only at ``evaluate``) is also here: the rungs file
asserts the refusal through ``evaluate``, not that the name validates.

Region masks and cell counts are computed test-side from the analytic body —
with no access to the implementation's classification that is an independent
oracle, not duplication (verification plan §10).
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp
from blockamr.field import CellField
from blockamr.ibm import IBM, Cylinder, FixedValue, Plane
from blockamr.mesh import AmrMesh, Mesh
from blockamr.schemes.registry import SCHEME_REGISTRY

from .ibm_gaps import (
    CUT_CELL,
    PENALIZATION,
    T17_MOVING_BODIES,
    T18_FORCES,
)

BACKEND = "cpp"

N = 32  # nothing here needs resolution
NZ = 4  # thin in the cylinder axis; every probe is z-invariant
DX = 1.0 / N

CONST = 3.0

# "Exactly zero" for a laplacian is a rounding bound on the terms the stencil
# cancels (``O(CONST/dx**2) ~ 3e3`` here), not on 1. Fixed, resolution- and
# physics-free — it is not a tolerance the method gets to hide an error in.
_MACHINE_ZERO = 1.0e-11

# MMS quadratic about a body centre: T = A + B*(r^2 - R^2), laplacian = 4B,
# trace on r = R is exactly A.
A_MMS = 0.3
B_MMS = 0.5
R = 0.2

# Rung-5 geometry, reused by the fresh-cell probe: a plane wall at X_WALL and a
# field that is linear along its normal, so the surface trace is a constant a
# scalar ``FixedValue`` can express exactly.
X_WALL = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mesh(n=N, nz=NZ, bodies=None, periodic=(1, 1, 1)):
    """Single-box ``Mesh`` on the unit cube, ``n x n x nz`` cells."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, list(periodic))
    ba = blockamr.BoxArray(box)
    ba.max_size(max(n, nz))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {} if bodies is None else bodies
    return mesh


def _coords(mesh, lo, shape, lev=0):
    """Cell-centre coordinate meshgrid for a box whose first cell is ``lo``."""
    geom = mesh.geom(lev)
    dx = geom.cell_size()
    plo = geom.prob_lo()
    axes = [
        np.array([plo[d] + (lo[d] + i + 0.5) * dx[d] for i in range(shape[d])]) for d in range(3)
    ]
    return np.meshgrid(*axes, indexing="ij")


def _fill(field, mesh, func, lev=0):
    """Fill the valid cells from ``func(X, Y, Z)`` — solid cells included.

    The IBM must reconstruct its near-surface stencil from its own BC, never
    lean on the values it happens to find inside the body.
    """
    mf = field.mf[lev]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3], lev)
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(lev, 0.0)


def _fill_halo(field, mesh, func):
    """Seed the domain-exterior ghosts analytically, after ``fill_patch``.

    Only needed on a non-periodic mesh, where ``fill_boundary`` leaves them
    untouched — an unfilled halo would contaminate every edge cell for reasons
    that have nothing to do with the IBM.
    """
    mf = field.mf[0]
    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_grown_from(mfi, arr)


def _poison_column(field, i, value=1.0e6):
    """Overwrite the x-column ``i`` of the (single-box) field.

    A cell that was solid and has just become fluid carries no history — this
    is what "no history" looks like in a test: a number that is not a sample of
    the solution anywhere.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[i - mfi.valid_box().small_end()[0], :, :, 0] = value
        mf.copy_from(mfi, arr)


def _flat(results):
    """Every valid cell of every level as one flat array."""
    return np.concatenate([np.asarray(a).ravel() for lev in results for a in lev])


def _assemble(field, results, n=N, nz=NZ):
    """Stitch the per-box level-0 result into one global ``(n, n, nz)`` array."""
    out = np.full((n, n, nz), np.nan)
    for bi, mfi in enumerate(blockamr.MFIterator(field.mf[0])):
        lo = mfi.valid_box().small_end()
        arr = np.asarray(results[0][bi])
        arr = arr.reshape(arr.shape[:3])
        out[
            lo[0] : lo[0] + arr.shape[0],
            lo[1] : lo[1] + arr.shape[1],
            lo[2] : lo[2] + arr.shape[2],
        ] = arr
    assert not np.isnan(out).any(), "box decomposition did not cover the domain"
    return out


def _sol(method=None, backend=BACKEND):
    """The ``fvSolution.solvers[field]`` block; no ``"ibm"`` key means no IBM."""
    return {"backend": backend} if method is None else {"ibm": method, "backend": backend}


def _cylinder(centre, radius=R):
    return Cylinder(centre=centre, radius=radius, axis=2)


def _mms(centre):
    """``T(r) = A + B*(r^2 - R^2)`` about ``centre``; ``laplacian(T) = 4B``."""

    def func(X, Y, Z):
        return A_MMS + B_MMS * ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 - R**2)

    return func


def _mms_case(centre, n=N):
    mesh = _make_mesh(n=n, bodies={"cyl": _cylinder(centre)})
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(A_MMS)})
    _fill(T, mesh, _mms(centre))
    return mesh, T, Equation(exp.laplacian(1.0, T))


def _non_fluid_count(centre, radius=R, n=N, nz=NZ):
    """Cells whose centre is inside the cylinder — analytic, from the body.

    This is the row count ``directForcing`` owes: its recipe writes one row per
    non-fluid cell (``b = 1``, no donors, ``gamma = u_body``).
    """
    xs = (np.arange(n) + 0.5) / n
    zs = (np.arange(nz) + 0.5) / nz
    X, Y, _Z = np.meshgrid(xs, xs, zs, indexing="ij")
    return int((np.hypot(X - centre[0], Y - centre[1]) <= radius).sum())


# ---------------------------------------------------------------------------
# 1. Error paths (verification plan §8) — the row nothing else covers
# ---------------------------------------------------------------------------


def _two_bodies_a_gap_apart(gap):
    """Two cylinders whose surfaces are ``gap`` apart, straddling the centre."""
    return {
        "left": _cylinder((0.5 - R - gap / 2.0, 0.5)),
        "right": _cylinder((0.5 + R + gap / 2.0, 0.5)),
    }


def test_thin_gap_between_two_bodies_raises_naming_the_patch_and_the_region(blockamr_session):
    """§8, the row the other files leave open.

    Two bodies half a cell apart leave a sub-cell fluid channel: the ghost
    cells facing it have no fluid cell to mirror into — the image point and the
    one-cell-out fallback both land in the *other* body. The design's rule is
    that this fails loudly rather than silently interpolating from solid, and
    the message must localise it: **which patch**, and **where**.

    "Where" is the offending cell's index, which is the only region name the
    row builder has — it works per cell, not per named region. The regex
    demands both halves, so a regression to a generic "IBM geometry error"
    fails this test rather than passing it.
    """
    mesh = _make_mesh(bodies=_two_bodies_a_gap_apart(0.5 * DX))
    T = CellField(
        mesh,
        ncomp=1,
        ngrow=1,
        name="T",
        ibm_bc={"left": FixedValue(CONST), "right": FixedValue(CONST)},
    )
    _fill(T, mesh, lambda X, Y, Z: np.full(X.shape, CONST))
    eqn = Equation(exp.laplacian(1.0, T))

    with pytest.raises(ValueError, match=r"\[\d+, \d+, \d+\] on patch '(left|right)'") as excinfo:
        evaluate(eqn, t=0.0, solution=_sol("ghostCell"))

    msg = str(excinfo.value)
    # the diagnosis, not just the location: the caller has to be told the fluid
    # is under one cell deep there, or the message is unactionable.
    assert "fluid" in msg
    assert "cell" in msg


def test_a_gap_wider_than_a_cell_between_two_bodies_is_accepted(blockamr_session):
    """The complement of the test above, and the reason its bound is a real
    bound rather than "two bodies are unsupported".

    One cell of fluid between the two surfaces is enough for the mirror image
    point to land in fluid, so the same configuration must go through — and a
    constant field consistent with both patches' data must still be annihilated
    exactly, in both bands.

    ``_MACHINE_ZERO`` rather than a literal ``1e-12``: the stencil cancels
    terms of size ``CONST/dx**2``, and two bands facing each other across a
    one-cell channel amplify their rows more than a lone body does. The bound
    is still a rounding bound (``~3e-15`` relative), not a physical tolerance —
    it does not move with resolution and nothing about the method is being
    excused by it.
    """
    mesh = _make_mesh(bodies=_two_bodies_a_gap_apart(2.0 * DX))
    T = CellField(
        mesh,
        ncomp=1,
        ngrow=1,
        name="T",
        ibm_bc={"left": FixedValue(CONST), "right": FixedValue(CONST)},
    )
    _fill(T, mesh, lambda X, Y, Z: np.full(X.shape, CONST))

    out = evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("ghostCell"))
    np.testing.assert_allclose(_flat(out), 0.0, atol=_MACHINE_ZERO)


# ---------------------------------------------------------------------------
# 2. The staleness guard (row format §3)
# ---------------------------------------------------------------------------


def _tag_all(lev, tags, time, ngrow):
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        shape = tuple(hi[d] - lo[d] + 1 for d in range(3))
        tbi.set_tags(np.ones(shape, dtype=np.int32))


def _amr_case(centre=(0.5, 0.5)):
    """A two-level-capable ``AmrMesh`` carrying the MMS field and its equation."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)

    mesh = AmrMesh(geom, info)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(A_MMS)})
    mesh.register_field(T)
    mesh.init_from_scratch(0.0)
    mesh.bodies = {"cyl": _cylinder(centre)}
    return mesh, T, Equation(exp.laplacian(1.0, T))


def _fill_levels(field, mesh, func):
    for lev in range(mesh.n_levels()):
        _fill(field, mesh, func, lev)


def test_regridding_rebuilds_the_wall_rows_instead_of_reusing_a_stale_table(blockamr_session):
    """The equation-level half of the row format's staleness rule (§3).

    "A table built for a stale grid produces plausible wrong numbers, so this
    check is not optional and has no 'skip' value." There are two acceptable
    outcomes for an ``evaluate`` after a regrid — rebuild, or raise naming both
    versions — and **the design intends the rebuild**: the table cache is keyed
    on ``mesh.grid_version`` precisely so the rebuild happens, and the
    ``RuntimeError`` the kernels raise on a mismatch is the backstop for the
    day someone forgets to key it. The third outcome, plausible wrong numbers,
    is what this test exists to exclude.

    Excluding it needs a probe that a stale table would get *wrong*, which a
    constant field is not — every table annihilates a constant, stale or not.
    So the field is the MMS quadratic and the oracle is a second mesh that was
    regridded to the same grid **before** any table was built on it: with no old
    generation to reuse, its rows can only be the right ones. The two runs must
    then agree **bitwise**; a tolerance here would permit exactly the "plausible"
    part of "plausible wrong numbers".

    The first ``evaluate`` on the regridded mesh is load-bearing — it is what
    populates the cache for the generation that is about to become stale.
    """
    body = (0.5, 0.5)
    regridded, T_a, eqn_a = _amr_case(body)
    _fill_levels(T_a, regridded, _mms(body))
    evaluate(eqn_a, t=0.0, solution=_sol("ghostCell"))  # caches generation 0

    before = regridded.grid_version
    regridded.regrid(0.0, tag=_tag_all)
    assert regridded.grid_version != before, "regrid must bump the grid generation"
    assert regridded.n_levels() == 2, "the regrid did not actually change the grid"

    _fill_levels(T_a, regridded, _mms(body))
    after = evaluate(eqn_a, t=0.0, solution=_sol("ghostCell"))
    assert len(after) == regridded.n_levels(), "the new level was not evaluated"

    fresh, T_b, eqn_b = _amr_case(body)
    fresh.regrid(0.0, tag=_tag_all)  # same grid, but no table was ever built here
    assert fresh.n_levels() == regridded.n_levels()
    _fill_levels(T_b, fresh, _mms(body))
    reference = evaluate(eqn_b, t=0.0, solution=_sol("ghostCell"))

    for lev, (got, want) in enumerate(zip(after, reference)):
        assert len(got) == len(want), f"level {lev}: different box counts"
        for bi, (g, w) in enumerate(zip(got, want)):
            np.testing.assert_array_equal(
                np.asarray(g), np.asarray(w), err_msg=f"level {lev}, box {bi}: stale wall rows"
            )


# ---------------------------------------------------------------------------
# 3. penalization — row recipe b = theta, gamma = u_body, w_self = 1 - theta,
#    RestrictMode.AddSource (row format §5)
# ---------------------------------------------------------------------------


@PENALIZATION
def test_penalization_is_registered_as_an_add_source_operator_method(blockamr_session):
    """The registry half. ``penalization`` is a *source* method: unlike
    ``ghostCell`` (which zeroes the operator result in the band) and
    ``directForcing`` (which overwrites it), it **adds** ``b*gamma`` — that is
    the whole difference between the three, and it is one declared attribute.

    It is an ``"operator"`` method, not a ``"step"`` one: the forcing enters the
    equation as a source term, so asking for operator-level evaluation is the
    normal way to use it (contrast ``directForcing``, §8).
    """
    method = IBM.lookup("penalization")

    assert method.kind == "operator"
    assert method.restrict_mode == "AddSource"
    assert method.requires_bodies is True


@PENALIZATION
def test_penalization_body_datum_enters_the_result_linearly(blockamr_session):
    """The behavioural half, and it needs no geometry.

    ``theta`` is a per-cell wall fraction this test cannot know, but ``gamma``
    is not: the whole IBM apply is affine in it (row format §5 — ``gamma`` is
    never folded into ``w``, which is rule R1) and the operator is linear, so
    the result must be **affine in the body datum**, exactly:

        E(2u) - E(u) == E(u) - E(0)

    That is a tolerance-free assertion which fails the moment ``gamma`` is
    baked into the weights, scaled twice, or applied per term instead of per
    evaluate. The second assertion — that the datum changes the answer at all —
    is what stops the first from passing vacuously if ``gamma`` is dropped.

    The body datum is spelled the way every other method spells its surface
    datum, as the field's ``ibm_bc`` entry; ``FixedValue(u)`` is the immersed
    surface velocity ``u_body`` of the row recipe.
    """

    def result(datum):
        mesh = _make_mesh(bodies={"cyl": _cylinder((0.5, 0.5))})
        T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(datum)})
        _fill(T, mesh, lambda X, Y, Z: np.full(X.shape, CONST))
        return _assemble(
            T, evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("penalization"))
        )

    zero = result(0.0)
    one = result(1.0)
    two = result(2.0)

    np.testing.assert_allclose(two - one, one - zero, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(one - zero)) > 0.0, "the body datum never reached the result"


# ---------------------------------------------------------------------------
# 4. cutCell — deferred (T19), gated on the T16 accuracy gate
# ---------------------------------------------------------------------------


def test_cutcell_is_a_schema_valid_name_that_refuses_to_run(blockamr_session):
    """§8's deferred-method row, at the registry rather than at ``evaluate``.

    Two distinct promises live here and only one of them is about failing.
    ``cutCell`` **validates**: it is a legal ``fvSolution`` value, so it must
    appear in the valid list an unknown name is offered, and it must raise
    ``NotImplementedError`` rather than the ``ValueError`` an unknown name
    gets — a user who typed a real-but-deferred name is in a different
    situation from one who made a typo, and the exception type is what tells
    the two apart. ``test_ibm_rungs.py`` covers the refusal through
    ``evaluate``; nothing covered the schema half.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        IBM.lookup("cutCell")

    msg = str(excinfo.value)
    assert "cutCell" in msg
    assert "not implemented" in msg.lower() or "defer" in msg.lower()

    with pytest.raises(ValueError) as unknown:
        IBM.lookup("noSuchMethod")
    assert "cutCell" in str(unknown.value), "a deferred name must still validate as a schema value"


@CUT_CELL
def test_cutcell_annihilates_a_constant_like_every_other_method(blockamr_session):
    """What ``cutCell`` owes on the day the gate (T16) sends us to T19.

    Flux rows are a different discretisation, not a different contract: the row
    consistency identity ``sum(w) + b*alpha == 1`` is the one property every
    method in the registry shares, so a constant field consistent with its wall
    BC must come out of the laplacian exactly zero — band included — for
    ``cutCell`` exactly as for ``ghostCell``.

    Red until T19 lands; it is deliberately the *cheapest* possible claim, so
    that when the method arrives this is the first thing it has to satisfy.
    """
    mesh = _make_mesh(bodies={"cyl": _cylinder((0.5, 0.5))})
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(CONST)})
    _fill(T, mesh, lambda X, Y, Z: np.full(X.shape, CONST))

    out = evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("cutCell"))
    np.testing.assert_allclose(_flat(out), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. directForcing as rows (T6) — DELETED with `wall_table.cpp`
#
# `test_direct_forcing_is_expressed_as_wall_rows` was a strict xfail asserting
# `isinstance(tables[0], blockamr.WallTable)` — the ONLY reference to
# `WallTable` anywhere in the tree, and to a `DirectForcing.build_tables` that
# was never written. `wall_table.cpp` is deleted (plans/IBM/design.md §1.3): the
# row format it bound is gone, so the shape T6 asked `directForcing` to take no
# longer exists and the xfail could never flip by being implemented — only by
# being rewritten into a different claim.
#
# Deleted for that stated reason and NOT relaxed. `directForcing` keeps its own
# coverage: it is a STEP method (design §7's table), it is refused by an
# operator-level evaluate — asserted in `test_ibm_rungs.py` — and it runs end to
# end in `test_cylinder_ibm.py`. `T6_DIRECT_FORCING_ROWS` stays in
# `ibm_gaps.py`: `test_ibm_combinations.py` still uses it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Moving bodies and fresh cells (T17)
# ---------------------------------------------------------------------------


def test_moving_a_body_gives_the_same_result_as_building_it_there(blockamr_session):
    """A moved body is indistinguishable from one that was always there.

    A prescribed body motion is expressed by re-assigning ``mesh.bodies`` — the
    geometry is data, and a moved body is different data. What must then hold
    is the only statement that does not depend on how the rebuild is
    implemented: **a moved body is indistinguishable from a body that was
    always there.** Bitwise, not to a tolerance, because the two runs execute
    the same code on the same numbers and any difference is a stale table.

    The first ``evaluate`` is load-bearing: it populates the wall-table cache
    for the *old* position. Without it the tables would be built fresh from the
    new bodies and the test would pass without proving anything.

    What makes it hold is that every IBM cache — classification, band, method
    data, rows — is keyed on the **IBM** generation, which re-assigning
    ``mesh.bodies`` bumps, and not on ``mesh.grid_version``, which only a
    regrid does. Fresh cells (the other half of moving bodies) are still open.
    """
    start, end = (0.4, 0.5), (0.6, 0.5)

    _ref_mesh, T_ref, eqn_ref = _mms_case(end)
    reference = _assemble(T_ref, evaluate(eqn_ref, t=0.0, solution=_sol("ghostCell")))

    mesh, T, eqn = _mms_case(start)
    evaluate(eqn, t=0.0, solution=_sol("ghostCell"))  # caches the table at `start`

    mesh.bodies = {"cyl": _cylinder(end)}
    _fill(T, mesh, _mms(end))
    moved = _assemble(T, evaluate(eqn, t=0.0, solution=_sol("ghostCell")))

    np.testing.assert_array_equal(moved, reference)


@T17_MOVING_BODIES
def test_a_fresh_cell_is_reconstructed_and_not_read_from_its_solid_history(blockamr_session):
    """T17's fresh cells — "just another row type (``b = 0``, extrapolation
    weights); if they need a new kernel, the format is leaking".

    A plane wall steps one cell in ``-x``, so exactly one column of cells goes
    from solid to fluid. Those cells have no history, and this test says so
    literally: their contents are overwritten with a number that is not a
    sample of the solution anywhere. A fresh-cell row must extrapolate over
    that, and its fluid neighbours — whose stencils read it — must never see it.

    The probe is a field that is **linear along the wall normal**, so the claim
    is exact: an extrapolation row that is linear-exact reproduces it to
    machine precision and the laplacian is machine-zero over the whole domain
    (the solid side is zeroed by ``R``, the fresh cell is a row target and is
    zeroed too, the fluid is exact). No tolerance to argue about.

    The two lines differ before and after the step — different slope, different
    wall datum — so a table left over from the old position reconstructs the
    fresh cell against the *old* wall and the *old* datum, which is a plausible
    number and the wrong one. That is precisely the failure this exists to
    catch, and it is today's failure.
    """
    wall_before, wall_after = X_WALL, X_WALL - DX
    fresh_i = int(wall_after * N)  # the one column between the two wall positions

    def line(a, b, x_wall):
        return lambda X, Y, Z: a + b * (X - x_wall)

    before = line(0.0, 2.0, wall_before)
    after = line(1.0, -3.0, wall_after)

    mesh = _make_mesh(
        bodies={"wall": Plane(point=(wall_before, 0.0, 0.0), normal=(1.0, 0.0, 0.0))},
        periodic=(0, 1, 1),
    )
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"wall": FixedValue(0.0)})
    _fill(T, mesh, before)
    _fill_halo(T, mesh, before)
    evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("ghostCell"))

    mesh.bodies = {"wall": Plane(point=(wall_after, 0.0, 0.0), normal=(1.0, 0.0, 0.0))}
    T.ibm_bc = {"wall": FixedValue(1.0)}  # the new line's trace on the new wall
    _fill(T, mesh, after)
    _fill_halo(T, mesh, after)
    _poison_column(T, fresh_i)

    out = evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("ghostCell"))
    np.testing.assert_allclose(_assemble(T, out), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Per-patch force and torque diagnostics (T18)
# ---------------------------------------------------------------------------

# The API this section is written against, and the reason it is a free function
# next to ``evaluate`` rather than a flag on it: a per-patch surface integral is
# a different *quantity*, not a different rendering of the field ``evaluate``
# returns. Making ``evaluate`` return a tuple would tax every call site that
# does not want diagnostics; hanging the numbers off the field after the fact
# would put hidden per-evaluate state exactly where rule R5 forbids it.
#
#     from blockamr.ibm import wall_diagnostics
#     diag = wall_diagnostics(eqn, t=0.0, solution=..., about=(x, y, z))
#     diag["cyl"].flux      # ndarray (ncomp,) — A1's wall flux, A3's force
#     diag["cyl"].torque    # ndarray (3,)     — the moment of that flux about `about`
#
# ``patch[r]`` is already carried on every row and already documented as
# "host-side diagnostics (per-patch forces); no kernel reads it" — this is the
# consumer it was put there for.


def _two_body_case(near_datum=1.0, far_datum=0.0):
    """One body inside the domain, one far outside — with distinct data."""
    mesh = _make_mesh(
        bodies={"near": _cylinder((0.5, 0.5)), "far": _cylinder((99.0, 99.0))},
    )
    T = CellField(
        mesh,
        ncomp=1,
        ngrow=1,
        name="T",
        ibm_bc={"near": FixedValue(near_datum), "far": FixedValue(far_datum)},
    )
    _fill(T, mesh, lambda X, Y, Z: np.full(X.shape, CONST))
    return mesh, T, Equation(exp.laplacian(1.0, T))


@T18_FORCES
def test_wall_diagnostics_are_keyed_per_patch_and_attributed_per_body(blockamr_session):
    """T18 — the diagnostic layer ``patch[r]`` exists for.

    Two bodies, one of them entirely outside the domain. The contract is that
    the result is a dict over exactly the patch names of ``mesh.bodies`` (the
    same key set as ``ibm_bc`` — one vocabulary for geometry, BCs and
    diagnostics), and that attribution is real: a patch with no rows reports
    **exactly** zero, not a share of the other body's flux.

    This is the shape A1's wall flux and A3's wall torque are read through, and
    the two-body case is the one that distinguishes per-patch attribution from
    a single global integral that happens to be right when there is one body.
    """
    from blockamr.ibm import wall_diagnostics

    mesh, _T, eqn = _two_body_case()
    diag = wall_diagnostics(eqn, t=0.0, solution=_sol("ghostCell"))

    assert set(diag) == set(mesh.bodies)
    np.testing.assert_array_equal(diag["far"].flux, np.zeros_like(diag["far"].flux))
    np.testing.assert_array_equal(diag["far"].torque, np.zeros(3))
    assert np.any(diag["near"].flux != 0.0), "the immersed body reported no wall flux at all"


@T18_FORCES
def test_wall_flux_is_affine_in_that_patch_s_own_surface_datum(blockamr_session):
    """T18. The flux is a linear functional of a result that is affine in
    ``gamma`` (rule R1: ``gamma`` is never folded into ``w``), so it is affine
    in the datum — exactly, with no geometry entering:

        F(2g) - F(g) == F(g) - F(0)

    and, because attribution is per patch, moving the *other* body's datum must
    not move this one's at all. Both halves are tolerance-free, and together
    they are the sharpest statement about the diagnostic that does not require
    an analytic solution (A1 supplies that separately).
    """
    from blockamr.ibm import wall_diagnostics

    def near_flux(near_datum, far_datum=0.0):
        _mesh, _T, eqn = _two_body_case(near_datum=near_datum, far_datum=far_datum)
        return np.asarray(wall_diagnostics(eqn, t=0.0, solution=_sol("ghostCell"))["near"].flux)

    f0, f1, f2 = near_flux(0.0), near_flux(1.0), near_flux(2.0)
    np.testing.assert_allclose(f2 - f1, f1 - f0, rtol=1e-13, atol=1e-13)
    assert np.any(f1 != f0), "the surface datum never reached the wall flux"

    np.testing.assert_array_equal(near_flux(1.0, far_datum=7.0), f1)


@T18_FORCES
def test_wall_torque_moves_with_the_reference_point_by_exactly_r_cross_f(blockamr_session):
    """T18, the torque half — and the one identity that pins it without knowing
    a single surface traction.

    For any distribution of wall loads, the moment about two different points
    differs by the cross product of the offset with the total load:

        torque(p2) - torque(p1) == (p1 - p2) x flux

    So this test needs no analytic solution and no geometry, and it fails for
    every wrong lever arm — a torque taken about the domain origin when the
    caller asked for the body centre, a sign error in the cross product, a
    moment arm measured to the cell centre instead of the wall. A3's absolute
    number is checked in the analytic suite; the *definition* is checked here.

    ``ncomp=3``: a torque is the moment of a vector load, so the diagnostic is
    only meaningful on a vector field, and that is the case A3 runs.
    """
    from blockamr.ibm import wall_diagnostics

    mesh = _make_mesh(bodies={"cyl": _cylinder((0.5, 0.5))})
    U = CellField(mesh, ncomp=3, ngrow=1, name="U", ibm_bc={"cyl": FixedValue(0.0)})
    for comp, value in enumerate((1.0, -2.0, 0.5)):
        mf = U.mf[0]
        for mfi in blockamr.MFIterator(mf):
            arr = mf.copy_to_host(mfi)
            arr[:, :, :, comp] = value
            mf.copy_from(mfi, arr)
    U.fill_patch(0, 0.0)
    eqn = Equation(exp.laplacian(1.0, U))

    p1 = (0.5, 0.5, 0.0)
    p2 = (0.0, 0.0, 0.0)
    d1 = wall_diagnostics(eqn, t=0.0, solution=_sol("ghostCell"), about=p1)["cyl"]
    d2 = wall_diagnostics(eqn, t=0.0, solution=_sol("ghostCell"), about=p2)["cyl"]

    expected = np.cross(np.asarray(p1) - np.asarray(p2), np.asarray(d1.flux))
    np.testing.assert_allclose(np.asarray(d2.torque) - np.asarray(d1.torque), expected, atol=1e-12)
    np.testing.assert_array_equal(d2.flux, d1.flux)  # the load itself is not a moment


# ---------------------------------------------------------------------------
# 8. The uniformity claim (verification plan §5, §10)
# ---------------------------------------------------------------------------

# "Adding a test body for a new method or scheme" is the anti-pattern: both axes
# are supposed to enter as parametrize *data*, generated from the registries, so
# that neither a new scheme nor a new method can be added without entering the
# grid. These two tests assert that of the grid in ``test_ibm_rungs.py`` — the
# only place the grid exists — rather than restating the grid here.

# ``ddt`` is a time scheme: it has no operator-level ``evaluate``, so it is not
# part of a *spatial* scheme axis.
_TIME_OPERATOR = "ddt"


def test_the_scheme_axis_of_the_grid_is_generated_from_the_scheme_registry(blockamr_session):
    """§5: "The ``SCHEMES`` list should be generated from ``SCHEME_REGISTRY`` so
    a new scheme cannot be added without entering the grid."

    Asserted as an equality against the registry rather than against a literal:
    a hand-maintained copy that happens to be correct today passes a literal
    check and drifts tomorrow, which is the failure mode §10 names.
    """
    from .test_ibm_rungs import SCHEMES

    expected = [
        (op, name)
        for op, table in sorted(SCHEME_REGISTRY.items())
        if op != _TIME_OPERATOR
        for name in table
    ]
    assert sorted(SCHEMES) == sorted(expected)


def test_the_method_axis_of_the_grid_is_generated_from_the_ibm_registry(blockamr_session):
    """The other axis of the same claim.

    ``SCHEME_REGISTRY`` is a public dict, so the scheme axis generates itself.
    The IBM registry used to be a module-private ``_METHODS`` behind
    ``IBM.lookup`` with no way to ask it what it holds, so the grid's method
    list was hand-written and "adding a method touches no C++" still cost a
    test edit — exactly the uniformity the design claims it does not.
    ``IBM.names()`` (B13) is the fix, and it lists **every** name the schema
    accepts, sorted, deferred ones included: the unknown-name message offers
    that list, so a name missing from it is a name a user cannot discover.

    A deferred name is therefore a cell of the grid too, and what it owes there
    is its refusal sentence rather than a number — carried as data in the
    grid's ``METHOD_EXPECTS`` table, so a method still enters the grid without
    a new test body.
    """
    from .test_ibm_rungs import METHODS

    assert sorted(METHODS) == IBM.names()
    assert "cutCell" in IBM.names(), "a deferred name must still be offered"
