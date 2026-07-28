# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The compiled ``ghostCell`` preprocessing — ``ibm/ghost_cell.{H,cpp}`` (B31).

**Conformance, not acceptance**, exactly like ``test_ibm_cell_type.py``: no row
of the equation suite may read a donor table, now or later. What this file
asserts is the one thing tasks.md §3's verify column asks for —

    the compiled ``preprocess`` produces the same donors, weights, image
    points and distances as v1's numpy ``ghost_cell_data``, **bitwise**.

*Bitwise* is taken literally: the float arrays are compared through their raw
``int64`` views, so a last-ulp difference is a failure and not a rounding
detail. review.md §4 Q29(d) refuses the ULP fallback — a residual mismatch
stays red and is escalated, never absorbed into a tolerance.

**The oracle is v1, and v1 is pinned elsewhere.** Nothing here is asserted
against the implementation's own output: ``test_ibm_ghost_cell.py``'s eight
numpy rows are what fix the oracle (the image-point rule, the half-cell reach,
the weight normalisation, Invariant F), and this file only asks whether the
transcription moved a bit. That is why the inputs are made bit-identical first,
in two pinned links (Q29(d)): the compiled side's geometry is
``IbmMesh.geometry_fab``, uploaded from ``packed_box_geometry``, while the
numpy side's ``GhostCell.preprocess`` reads v1's ``box_geometry`` — two
different builders, made bitwise-equal by ``test_ibm_mesh.py``'s
``test_the_packed_geometry_is_the_v1_geometry_of_the_same_bodies``. A red row
here therefore means ``ghost_cell.cpp``'s own arithmetic *or* a drift between
the two geometry builders — check that mesh row first to tell them apart.

**Why a separate file** (the plan's OP-3, decided here): ``test_ibm_ghost_cell``
is one of the four files of the O3 fence, whose whole point is that its count
does not move while the v2 port lands. Appending compiled rows to it would move
that count and would also force an edit to its docstring's "none of it needs the
compiled extension". A new file keeps both invariants and costs one import.
"""

import numpy as np
import pytest

import blockamr
from blockamr.ibm.body import Cylinder, Plane
from blockamr.ibm.classify import _patches, box_grids
from blockamr.ibm.geometry import GEOM_NORMAL, packed_geometry_on_grids
from blockamr.ibm.ghost_cell import GhostCell, K
from blockamr.mesh import Mesh

# Underscore-private test binding (api §4). `from ._blockamr import *` skips
# underscore names, so it is reached on the extension module itself.
_ghost_cell_numpy = blockamr._blockamr._ghost_cell_numpy

N = 16
DX = 1.0 / N

#: The same three geometries ``test_ibm_ghost_cell.py`` drives v1 with, on the
#: same non-periodic unit cube: a cylinder, a tilted plane whose every normal
#: component is nonzero, and two bodies four cells apart.
RADIUS = 0.2
TILTED = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])


def _cylinder(centre=(0.5, 0.5), radius=RADIUS):
    return Cylinder(centre=centre, radius=radius, axis=2)


BODIES = {
    "cylinder": {"cyl": _cylinder()},
    "tilted-plane": {"wall": Plane(point=(0.5, 0.5, 0.5), normal=tuple(TILTED))},
    "two-bodies": {
        "left": _cylinder((0.5 - RADIUS - 2.0 * DX, 0.5)),
        "right": _cylinder((0.5 + RADIUS + 2.0 * DX, 0.5)),
    },
}

#: An axis-aligned wall: its image point lands exactly on a cell face, so half
#: the trilinear weights are exactly ``0.0`` and the dead-slot rule (a weight of
#: exactly zero points its donor at the row's own cell) is exercised rather than
#: assumed. Without it the parity rows never see a dead slot.
FACE_WALL = {"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))}

#: The mesh no body ever meets — the empty-band path.
FAR_AWAY = {"far": _cylinder(centre=(99.0, 99.0))}

#: The anisotropic cell: ``dx = (DX, DX, 8 DX)``, from a domain eight times as
#: tall. This is the row that exercises the image step's ``max_d |n_d| / dx_d``
#: hardest — a reassociation into ``0.5 * min_d dx_d / |n_d|`` is algebraically
#: equal and rounds differently, and on an isotropic grid it very nearly hides.
TALL = (1.0, 1.0, 8.0)

#: **The load-bearing grid.** Every other grid in this file has ``prob_lo = 0``
#: and a power-of-two ``dx`` (``1/16``, ``1/2``), and on such a grid most of the
#: arithmetic under test is *exact*: multiplying and dividing by a power of two
#: only moves the exponent, and ``0 + x`` is ``x``. Measured on this file's own
#: cylinder, all three reassociation hazards — H-c, H-d, H-f — agree with the
#: reference to the last bit on **every one of the 320 rows**, so those rows
#: cannot tell a correct transcription from a reassociated one.
#:
#: This grid removes the exactness: a non-zero ``prob_lo`` and extents 0.9 / 0.7
#: / 1.3 over 16 cells, i.e. ``dx = 0.05625 / 0.04375 / 0.08125``, none of them
#: dyadic. Measured on 4000 sampled cells and unit normals, the same three
#: reassociations then differ on 26 % (H-d), 34 % (H-c) and 16 % (H-f) of
#: entries. This is the row that makes "bitwise" mean something.
SKEW_LO = (-0.37, 0.11, 0.23)
SKEW_HI = (0.53, 0.81, 1.53)
SKEW_BODY = {"cyl": _cylinder(centre=(0.08, 0.46), radius=0.2)}


def _mesh(bodies, hi=(1.0, 1.0, 1.0), max_size=None, periodic=(0, 0, 0), lo=(0.0, 0.0, 0.0)):
    """A single-level mesh on ``[lo, hi]`` at ``N^3`` cells, with ``bodies``."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox(list(lo), list(hi))
    geom = blockamr.Geometry(box, rb, 0, list(periodic))
    ba = blockamr.BoxArray(box)
    ba.max_size(N if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
    return mesh, geom, ba, dm


def _compiled(mesh, geom, ba, dm, ngrow=1):
    """``preprocess`` on the compiled side: the v2 marker, the v2 geometry."""
    names, _bodies = _patches(mesh.bodies)
    g = mesh.ibm.geometry_fab(0, ngrow=ngrow)
    ct = blockamr.CellTypeFab(ba, dm, ngrow)
    blockamr.classify_default(ct, g, geom)
    return _ghost_cell_numpy(ct, g, geom, names)


def _assert_bitwise(got, want, name):
    """Equality of the raw bits, with a message that names the first offender.

    ``np.testing.assert_array_equal`` on f64 would already be bit equality up
    to ``-0.0 == 0.0``; comparing the ``int64`` views removes even that escape
    and makes the intent unmistakable. No weight or distance here is a negative
    zero (every trilinear factor is in ``[0, 1]``), so the two never disagree —
    which is the point: the stricter one costs nothing.
    """
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    assert got.dtype == want.dtype, f"{name}: dtype {got.dtype} != {want.dtype}"
    lhs = got.view(np.int64) if got.dtype == np.float64 else got
    rhs = want.view(np.int64) if want.dtype == np.float64 else want
    if np.array_equal(lhs, rhs):
        return
    bad = np.argwhere(lhs != rhs)
    at = tuple(int(v) for v in bad[0])
    raise AssertionError(
        f"{name}: {len(bad)} of {got.size} entries differ bitwise; first at {at}: "
        f"compiled {got[at]!r} vs numpy {want[at]!r} "
        f"(raw {int(lhs[at])} vs {int(rhs[at])})"
    )


def _assert_parity(mesh, geom, ba, dm):
    """The whole verify column, on one configuration."""
    v1 = GhostCell.preprocess(mesh, 0)
    ip, donor, weight, distance = _compiled(mesh, geom, ba, dm)

    assert v1.nrows > 0, "vacuous: this configuration has no wall layer at all"
    assert ip.shape == (v1.nrows, 3)
    assert donor.shape == (v1.nrows, K, 3)
    assert weight.shape == (v1.nrows, K)
    assert distance.shape == (v1.nrows,)
    assert donor.dtype == np.int32

    _assert_bitwise(donor, v1.donor, "donor")
    _assert_bitwise(weight, v1.weight, "weight")
    _assert_bitwise(ip, v1.image_point, "image_point")
    _assert_bitwise(distance, v1.distance, "distance")
    return v1, ip, weight


# ---------------------------------------------------------------------------
# 1. The verify column: bitwise parity with the numpy preprocessing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BODIES))
def test_the_compiled_preprocess_matches_the_numpy_rows_bitwise(blockamr_session, name):
    """tasks.md §3's verify column, on the three geometries v1 is pinned on."""
    mesh, geom, ba, dm = _mesh(BODIES[name])
    _assert_parity(mesh, geom, ba, dm)


def test_the_compiled_preprocess_matches_the_numpy_rows_on_an_anisotropic_cell(
    blockamr_session,
):
    """The same, with ``dx_z`` eight times ``dx_x``.

    The image step is ``0.5 / max_d(|n_d| / dx_d)``: on an isotropic grid the
    three quotients are near each other and a reassociated ``0.5 * min_d(dx_d /
    |n_d|)`` agrees to the last bit far more often than it deserves to. Here
    they are an order of magnitude apart.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"], hi=TALL)
    np.testing.assert_array_equal(mesh.geom(0).cell_size(), [DX, DX, 8.0 * DX])
    _assert_parity(mesh, geom, ba, dm)


def test_the_compiled_preprocess_matches_the_numpy_rows_on_a_non_dyadic_grid(blockamr_session):
    """Parity where the arithmetic actually **rounds** — see ``SKEW_LO``.

    On the unit cube at ``n = 16`` every factor is a power of two and
    ``prob_lo`` is zero, so the reassociations this transcription is guarding
    against are bit-for-bit invisible: they were measured to agree on all 320
    rows. Here ``dx`` is non-dyadic and ``prob_lo`` is not zero, so the cell
    centre (H-a), the index round trip (H-c/H-g), the image step (H-d) and the
    weight product (H-f) each round, and each would show up as a differing bit
    if the C++ had reassociated any of them.
    """
    mesh, geom, ba, dm = _mesh(SKEW_BODY, lo=SKEW_LO, hi=SKEW_HI)
    dx = np.asarray(mesh.geom(0).cell_size())
    plo = np.asarray(mesh.geom(0).prob_lo())
    # non-vacuous: the premise of this row is that nothing here is exact
    assert (np.log2(dx) % 1.0 != 0.0).all(), f"a dx is a power of two: {dx}"
    assert (plo != 0.0).all(), f"prob_lo is zero: {plo}"
    _assert_parity(mesh, geom, ba, dm)


def test_the_compiled_preprocess_matches_the_numpy_rows_across_eight_boxes(blockamr_session):
    """Parity **and** the cross-box row order in one statement.

    v1 concatenates its per-box blocks in ``MFIterator`` order; the compiled
    side offsets each box's ranks by the running total of the boxes before it,
    walking the same iterator. A decomposition changes which cells land in
    which block and therefore the whole row order — so on eight boxes a wrong
    concatenation is a total mismatch, not a last-ulp one.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"], max_size=8)
    assert sum(1 for _ in blockamr.MFIterator(mesh.ibm.geometry_fab(0, 1))) == 8
    _assert_parity(mesh, geom, ba, dm)


def test_the_compiled_preprocess_matches_the_numpy_rows_when_slots_are_dead(blockamr_session):
    """The dead-slot rule, which the three ``BODIES`` never reach.

    A wall normal to x puts the image point exactly on a cell face, so four of
    the eight trilinear weights are ``0.0`` *exactly* — and v1 then points those
    donors at the row's own cell rather than leaving them where the stencil
    fell. The rule is ``w == 0.0``, not ``w < eps``, and a port that used a
    tolerance would diverge in the *donors* here rather than in the weights.
    """
    mesh, geom, ba, dm = _mesh(FACE_WALL)
    v1, _ip, weight = _assert_parity(mesh, geom, ba, dm)
    # non-vacuous: dead slots really do occur, and they are exactly half
    assert (weight == 0.0).any()
    assert np.array_equal(weight == 0.0, v1.weight == 0.0)


# ---------------------------------------------------------------------------
# 2. The row order, stated on its own
# ---------------------------------------------------------------------------


def test_the_compiled_rows_are_the_wall_layer_in_band_order(blockamr_session):
    """The contract B32 indexes on, asserted against ``np.argwhere`` directly.

    Per local box in ``MFIterator`` order, and within a box sorted by ``i``,
    then ``j``, then ``k`` — C order, **k fastest**. AMReX's own linear index
    runs ``i`` fastest and an atomic append would be a third order and a
    non-deterministic one, so this is a real hazard and not a restatement.

    Stated without v1's ``ghost_cell_data``: the expectation is the cell list
    ``np.argwhere(depth == 1)`` yields, and the check is that row ``r``'s image
    point lies within half a cell of that cell's centre in every direction,
    which is exactly the image step's cap.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"], max_size=8)
    ip, _donor, _weight, _distance = _compiled(mesh, geom, ba, dm)

    grids = box_grids(mesh, 0)
    dx = np.asarray(grids[0].dx)
    plo = np.asarray(grids[0].prob_lo)

    cells = []
    for grid, geometry in zip(grids, mesh.ibm.geometry(0)):
        cells.append(np.argwhere(geometry.depth == 1) + np.asarray(grid.lo))
    expected = np.concatenate(cells)

    assert len(ip) == len(expected) > 0
    centre = plo + (expected + 0.5) * dx
    assert (np.abs(ip - centre) <= 0.5 * dx + 1e-15).all()


def test_the_compiled_preprocess_is_deterministic(blockamr_session):
    """verification §10's exactness rule: two runs, bitwise equal.

    The rank of a row comes from an exclusive scan, never from an atomic
    append, so repeating the call has to reproduce every bit — including the
    row order, which an append would shuffle differently each time.
    """
    mesh, geom, ba, dm = _mesh(BODIES["two-bodies"])
    first = _compiled(mesh, geom, ba, dm)
    second = _compiled(mesh, geom, ba, dm)
    for a, b, name in zip(first, second, ("image_point", "donor", "weight", "distance")):
        _assert_bitwise(a, b, name)


# ---------------------------------------------------------------------------
# 3. The empty band
# ---------------------------------------------------------------------------


def test_the_compiled_preprocess_leaves_the_data_empty_when_no_body_is_met(blockamr_session):
    """No wall layer, no rows — and correctly *shaped* empties, not ``None``.

    The empty band is the zero-overhead path (verification §2), and B32 will
    index these arrays without a special case, so the shapes have to be
    ``(0, 3)``, ``(0, K, 3)``, ``(0, K)``, ``(0,)`` exactly as v1 returns them.
    """
    mesh, geom, ba, dm = _mesh(FAR_AWAY)
    v1 = GhostCell.preprocess(mesh, 0)
    ip, donor, weight, distance = _compiled(mesh, geom, ba, dm)

    assert v1.nrows == 0
    assert ip.shape == (0, 3) and ip.dtype == np.float64
    assert donor.shape == (0, K, 3) and donor.dtype == np.int32
    assert weight.shape == (0, K) and weight.dtype == np.float64
    assert distance.shape == (0,) and distance.dtype == np.float64


# ---------------------------------------------------------------------------
# 4. The error surface (design §10: a sentence naming the offending object)
# ---------------------------------------------------------------------------


def test_a_non_fluid_donor_names_the_cell_and_the_patch(blockamr_session):
    """Invariant F is a *loud* failure on the compiled side too.

    The violation has to be injected — over ~1800 generated geometries v1 never
    produced one naturally (``test_ibm_ghost_cell.py``'s own note) — and it is
    injected the same way: hand ``preprocess`` a geometry whose normals point
    into the body. The marker is unaffected, because the classification reads
    only ``sdf``, so the rows are selected exactly as before and every image
    point lands inside the solid.

    The sentence is v1's word for word, because that is what B36's rewire will
    keep showing the user: which cell, which patch, and what rule.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"])
    names, _bodies = _patches(mesh.bodies)

    blocks = packed_geometry_on_grids(box_grids(mesh, 0), mesh.bodies, 1)
    inward = blockamr.MultiFab(ba, dm, blockamr.IBM_GEOM_NCOMP, 1)
    for mfi, block in zip(blockamr.MFIterator(inward), blocks):
        flipped = np.array(block, copy=True)
        flipped[..., GEOM_NORMAL : GEOM_NORMAL + 3] *= -1.0
        inward.copy_grown_from(mfi, np.asfortranarray(flipped))

    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, inward, geom)

    with pytest.raises(RuntimeError, match=r"\[\d+, \d+, \d+\] on patch 'cyl'") as excinfo:
        _ghost_cell_numpy(ct, inward, geom, names)

    message = str(excinfo.value)
    assert "Invariant F" in message
    assert "fluid" in message


def test_a_marker_with_no_ghost_cell_is_refused_naming_both_widths(blockamr_session):
    """A donor reaches one cell out, so Invariant F reads the marker's ghost.

    Without one the check would read out of bounds — silent garbage in a
    release build, i.e. a spurious Invariant-F sentence or a segfault.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"])
    g = mesh.ibm.geometry_fab(0, ngrow=1)
    ct = blockamr.CellTypeFab(ba, dm, 0)
    ct.set_val(int(blockamr.CellType.FLUID))
    with pytest.raises(RuntimeError, match=r"at least 1.*CellTypeFab has 0"):
        _ghost_cell_numpy(ct, g, geom, ["cyl"])


def test_a_geometry_with_the_wrong_component_count_is_refused(blockamr_session):
    """The packed layout is a contract here too, not only at classification."""
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"])
    g = mesh.ibm.geometry_fab(0, ngrow=1)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, g, geom)

    narrow = blockamr.MultiFab(ba, dm, 1, 1)
    narrow.set_val(1.0)
    with pytest.raises(RuntimeError, match=r"8 components.*has 1"):
        _ghost_cell_numpy(ct, narrow, geom, ["cyl"])


# ---------------------------------------------------------------------------
# 5. The stencil size crosses the boundary
# ---------------------------------------------------------------------------


def test_the_trilinear_stencil_size_is_the_one_the_compiled_side_expects(blockamr_session):
    """``K`` is declared on both sides; this is where they are held together."""
    assert K == blockamr.GHOST_CELL_K == 8


# ---------------------------------------------------------------------------
# 6. The cell -> row map (B32)
#
# `GhostCellData` gained one member, `iMultiFab row`: the rank per cell, `-1`
# where the marker is not `WALL`. A wall functor is called at a *cell* and this
# data is indexed by *rank*, so without the map the two cannot meet — the
# exclusive scan that produces the ranks used to be a transient per-box vector
# that `preprocess` freed on return (review.md §4 Q49(d)).
#
# The rows below pin the map against the same contract section 2 above pins the
# arrays against, and against nothing else: this file's subject is still the
# preprocessing, and what a pair *does* with a rank is B32's own file.
# ---------------------------------------------------------------------------

_cell_type_numpy = blockamr._blockamr._cell_type_numpy


def _preprocessed(mesh, geom, ba, dm, ngrow=1):
    """``(data, ct, g)`` — the opaque `GhostCellData` and what it was built from."""
    names, _bodies = _patches(mesh.bodies)
    g = mesh.ibm.geometry_fab(0, ngrow=ngrow)
    ct = blockamr.CellTypeFab(ba, dm, ngrow)
    blockamr.classify_default(ct, g, geom)
    return blockamr.ghost_cell_preprocess(ct, g, geom, names), ct, g


def test_the_row_map_is_minus_one_at_every_cell_that_is_not_wall(blockamr_session):
    """``-1``, not ``0``, and that is the whole point of the sentinel.

    A non-``WALL`` cell has no row. Were the map zero-filled there, a functor
    that reached such a cell would read **row 0's** donors — another cell's, and
    on a decomposed level possibly another box's — which is a plausible wrong
    answer rather than a crash. The marker is read back independently, so this
    is a comparison against the classification and not against the map itself.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"], max_size=8)
    data, ct, _g = _preprocessed(mesh, geom, ba, dm)

    wall = int(blockamr.CellType.WALL)

    # The marker is materialised before any `row_at` call. That used to be
    # mandatory — `row_at` opened an `MFIter` of its own and AMReX refuses a
    # nested one with an abort rather than an exception — and since B33 it is
    # merely tidy: `rowAt` resolves the box through `IndexArray()` +
    # `atLocalIdx()` and opens nothing. The two rows at the end of this file
    # pin that, so the workaround is not restored by habit.
    marker = {}
    probe = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(probe):
        lo = tuple(mfi.valid_box().small_end())
        block = _cell_type_numpy(ct, mfi)
        for local in np.ndindex(block.shape):
            marker[tuple(lo[d] + local[d] for d in range(3))] = int(block[local])

    counts = {"wall": 0, "other": 0}
    for cell, value in marker.items():
        rank = data.row_at(*cell)
        if value == wall:
            counts["wall"] += 1
            assert 0 <= rank < data.nrows, f"{cell} is WALL but its rank is {rank}"
        else:
            counts["other"] += 1
            assert rank == -1, f"{cell} is not WALL but carries rank {rank}"
    assert counts["wall"] == data.nrows > 0
    assert counts["other"] > 0


def test_the_row_map_is_the_exclusive_scan_rank_across_eight_boxes(blockamr_session):
    """The map is the row ORDER of section 2, published per cell.

    Per local box in ``MFIterator`` order and, within a box, by ``i`` then ``j``
    then ``k``. On eight boxes a wrong cross-box concatenation — the offset that
    used to be a host-side running total — is a total mismatch rather than an
    off-by-one, and it is asserted against ``np.argwhere(depth == 1)`` directly
    rather than against the compiled arrays it is supposed to index.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"], max_size=8)
    data, _ct, _g = _preprocessed(mesh, geom, ba, dm)

    grids = box_grids(mesh, 0)
    assert len(grids) == 8
    expected = np.concatenate(
        [
            np.argwhere(geometry.depth == 1) + np.asarray(grid.lo)
            for grid, geometry in zip(grids, mesh.ibm.geometry(0))
        ]
    )
    assert len(expected) == data.nrows > 0

    for rank, cell in enumerate(expected):
        got = data.row_at(*(int(v) for v in cell))
        assert got == rank, f"cell {tuple(cell)} has rank {got}, expected {rank}"


# ===========================================================================
# 8. `row_at` from inside an `MFIter` (B33, rider D-2)
# ===========================================================================


def test_row_at_returns_the_same_ranks_after_the_mfiter_free_rewrite(blockamr_session):
    """`rowAt`'s **answers** are unchanged by B33's rewrite — only its mechanism
    is (review.md §4 Q52(f), rider D-2).

    `row` carries `ngrow = 0`, so its valid box *is* its fab box and there is no
    valid/fab distinction for this lookup to get wrong; the rewrite swaps an
    `MFIter` loop for `boxArray()[IndexArray()[li]]` + `atLocalIdx(li)`, which
    walks the same local boxes in the same order. On **eight boxes** a wrong
    local-index resolution would read another box's fab and mismatch wholesale,
    so this is the row that would catch it.

    Deliberately independent of the map's *construction*: the expected ranks
    come from `np.argwhere(depth == 1)`, exactly as the section-2 row above.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"], max_size=8)
    data, _ct, _g = _preprocessed(mesh, geom, ba, dm)

    grids = box_grids(mesh, 0)
    assert len(grids) == 8
    expected = np.concatenate(
        [
            np.argwhere(geometry.depth == 1) + np.asarray(grid.lo)
            for grid, geometry in zip(grids, mesh.ibm.geometry(0))
        ]
    )
    assert len(expected) == data.nrows > 0
    got = [data.row_at(*(int(v) for v in cell)) for cell in expected]
    assert got == list(range(len(expected)))


def test_row_at_is_callable_from_inside_an_mfiterator_loop(blockamr_session):
    """The call that **aborted** at B32, made a passing row (rider D-2).

    AMReX refuses a nested `MFIter` with an `Abort`, not an exception, so before
    B33 this loop killed the interpreter rather than failing a test — which is
    why B32 could only record the limitation in a comment. `rowAt` now opens no
    `MFIter`, so a caller may ask per cell from inside its own iteration, and the
    ranks it hands back are the same ones the materialise-first loop gets.

    The comparison against the materialised answers is what stops this from
    passing vacuously if the lookup ever silently returned `-1` everywhere.
    """
    mesh, geom, ba, dm = _mesh(BODIES["cylinder"], max_size=8)
    data, ct, _g = _preprocessed(mesh, geom, ba, dm)

    probe = blockamr.MultiFab(ba, dm, 1, 0)

    outside = {}
    for mfi in blockamr.MFIterator(probe):
        lo = tuple(mfi.valid_box().small_end())
        block = _cell_type_numpy(ct, mfi)
        for local in np.ndindex(block.shape):
            outside[tuple(lo[d] + local[d] for d in range(3))] = int(block[local])
    materialised = {cell: data.row_at(*cell) for cell in outside}

    inside = {}
    for mfi in blockamr.MFIterator(probe):
        lo = tuple(mfi.valid_box().small_end())
        block = _cell_type_numpy(ct, mfi)
        for local in np.ndindex(block.shape):
            cell = tuple(lo[d] + local[d] for d in range(3))
            inside[cell] = data.row_at(*cell)  # <- the nested call

    assert inside == materialised
    assert sum(1 for r in inside.values() if r >= 0) == data.nrows > 0
