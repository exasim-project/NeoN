# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The band and ``mesh.ibm`` — the boundary-cell set and the lazy IBM cache.

This is the layer between the classification (``test_ibm_classify.py``) and
the boundary schemes: which cells a width-``w`` scheme has to treat specially,
and when that answer is recomputed. It knows no method, no operator and no
kernel.

**Why this file is unit-level while the rest of the IBM suite is not.** Same
reason as ``test_ibm_classify.py``: the equation-only rule of
``plans/IBM/verification.md`` §1 governs the transferred *equation* suite,
which asserts physics through ``evaluate``. A band is not physics — it is a set
of cells and a CSR offset array, and a cache is a question about *when* work
happens, which an equation-level assertion cannot see at all. Routing either
through a laplacian would test the laplacian.

Every expectation is a literal or a formula in the cell index, written from the
body's geometry, never read back from the implementation. The mesh is the unit
cube at ``n = 16``, so a cell is ``dx = 1/16`` wide and the centre of cell
``i`` sits at ``(i + 0.5)/16``. All bodies here are planes normal to x, so
every field is constant in y and z and one column of 16 numbers describes it.

Laziness is asserted the only way that does not depend on internals: the bodies
count the times they are evaluated.
"""

import numpy as np
import pytest

import blockamr
from blockamr.ibm.band import BOX, CROSS, band_on_grids
from blockamr.ibm.body import Cylinder, Plane
from blockamr.ibm.classify import MAX_DEPTH, BoxGrid
from blockamr.ibm.geometry import (
    GEOM_NCOMP,
    GEOM_NORMAL,
    GEOM_PATCH,
    GEOM_SDF,
    GEOM_WALL_POINT,
    box_geometry,
    geometry_on_grids,
    packed_geometry_on_grids,
)
from blockamr.mesh import Mesh

N = 16
DX = 1.0 / N

#: Non-periodic in x — the walls are normal to x, and a periodic seam would
#: put solid on the far side of the domain edge.
PERIODIC = (False, True, True)

#: The slab: fluid between the two walls, solid outside both. Patch ids are
#: indices into ``sorted(bodies)``, so "lower" is 0 and "upper" is 1.
SLAB = {
    "lower": Plane(point=(0.25, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
    "upper": Plane(point=(0.75, 0.0, 0.0), normal=(-1.0, 0.0, 0.0)),
}

#: The slab's depth, by hand. Fluid is cells 4..11 (centres 0.28..0.72); a
#: fluid cell's depth is its cell count to the nearer wall, a non-fluid cell's
#: is minus its count to the nearer fluid cell, both clamped at 4.
SLAB_DEPTH = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, -1, -2, -3], dtype=np.int8)

#: The owning patch of every column: the nearest surface owns the cell, and
#: the domain's mid-plane x = 0.5 is exactly halfway between the two walls.
SLAB_PATCH = np.where(np.arange(N) <= 7, 0, 1).astype(np.int8)


def _grid(lo=(0, 0, 0), hi=(N - 1, N - 1, N - 1)):
    """One local box of the unit cube, in global index space."""
    return BoxGrid(
        lo=lo,
        hi=hi,
        dx=(DX, DX, DX),
        prob_lo=(0.0, 0.0, 0.0),
        domain_lo=(0, 0, 0),
        domain_hi=(N - 1, N - 1, N - 1),
        periodic=PERIODIC,
    )


def _band(bodies, width, grids=None, shape=CROSS):
    """The band of the given boxes, without a mesh or a compiled extension."""
    grids = grids if grids is not None else [_grid()]
    return band_on_grids(grids, geometry_on_grids(grids, bodies), width, shape)


def _expected_cells(column_depth, width, lo=(0, 0, 0), hi=(N - 1, N - 1, N - 1)):
    """The cells of ``{depth <= width}``, from a hand-written depth column.

    Built the way the row order is defined — C order over the box's valid
    cells, plus the box's lower corner — and independent of the code under
    test, which only ever sees the bodies.
    """
    shape = tuple(hi[d] - lo[d] + 1 for d in range(3))
    depth = np.broadcast_to(column_depth[lo[0] : hi[0] + 1, np.newaxis, np.newaxis], shape)
    return np.argwhere(depth <= width) + np.asarray(lo)


def _wall_depth(wall_cell):
    """The depth column of a plane wall with fluid at and above ``wall_cell``."""
    i = np.arange(N)
    return np.where(
        i >= wall_cell,
        np.minimum(i - wall_cell + 1, MAX_DEPTH),
        np.maximum(1 - (wall_cell - i), -MAX_DEPTH),
    ).astype(np.int8)


def _wall(wall_cell):
    """A plane wall on the face below cell ``wall_cell``, fluid above it."""
    return {"wall": Plane(point=(wall_cell * DX, 0.0, 0.0), normal=(1.0, 0.0, 0.0))}


# ---------------------------------------------------------------------------
# 1. the band — {depth <= width}, per box, in MFIterator order
# ---------------------------------------------------------------------------


def test_the_band_is_exactly_the_cells_whose_stencil_leaves_the_fluid():
    """The definition, on the slab: ``band(w) = {depth <= w}``.

    A width-1 scheme reads its six face neighbours, so it is disturbed exactly
    in the cells one step from a non-fluid cell and in the non-fluid cells
    themselves — columns 0..4 and 11..15 of the slab, and nothing in between.
    """
    expected = _expected_cells(SLAB_DEPTH, width=1)

    band = _band(SLAB, width=1)

    assert band.width == 1
    assert band.shape == CROSS
    assert band.nrows == expected.shape[0]
    np.testing.assert_array_equal(band.cell, expected)
    np.testing.assert_array_equal(np.unique(band.cell[:, 0]), [0, 1, 2, 3, 4, 11, 12, 13, 14, 15])


def test_a_band_row_carries_the_depth_and_the_owning_patch_of_its_cell():
    """What a row knows about itself, and the only two things it knows.

    ``depth`` is what tells the scheme a row is non-fluid (``<= 0``) rather
    than a fluid cell to reconstruct; ``patch`` is what makes a per-body force
    a sum over rows. Both are the classification's, per band cell, in row
    order — so the assertion is the hand column indexed by the row's own cell.
    """
    band = _band(SLAB, width=1)

    column = band.cell[:, 0]
    np.testing.assert_array_equal(band.depth, SLAB_DEPTH[column])
    np.testing.assert_array_equal(band.patch, SLAB_PATCH[column])
    assert band.depth.dtype == np.int8
    assert band.patch.dtype == np.int8
    assert band.cell.dtype == np.int32


def test_a_wider_stencil_widens_the_band_by_one_column_per_wall():
    """Nesting: the width-2 band is the width-1 band plus ``{depth == 2}``.

    One classification serves every stencil width — that is what makes
    ``depth`` a signed cell count rather than a boolean mask, and it is why a
    scheme of any width costs no extra preprocessing.
    """
    narrow = _band(SLAB, width=1)
    wide = _band(SLAB, width=2)

    narrow_cells = {tuple(c) for c in narrow.cell}
    wide_cells = {tuple(c) for c in wide.cell}
    assert narrow_cells < wide_cells
    added = np.unique(np.array(sorted(wide_cells - narrow_cells))[:, 0])
    np.testing.assert_array_equal(added, [5, 10])


def test_the_band_of_a_mesh_without_bodies_is_empty():
    """No bodies, no rows — and an empty band is a valid one, not ``None``.

    This is what makes the ``noIbm`` path cost nothing while going through the
    same code: the offsets are still one entry per box, all zero.
    """
    band = _band({}, width=1)

    assert band.nrows == 0
    assert band.cell.shape == (0, 3)
    np.testing.assert_array_equal(band.box_offset, [0, 0])


def test_box_offset_addresses_the_rows_of_each_local_box_in_iterator_order():
    """The CSR half of the contract, which is how a kernel finds its rows.

    Two boxes split in y; the geometry is y-invariant, so both hold the same
    number of band cells and the split is exactly in half. The rows of box
    ``i`` are ``[box_offset[i], box_offset[i + 1])`` and they are the cells of
    that box — no row may be filed under a box that does not contain it.
    """
    grids = [_grid(hi=(N - 1, 7, N - 1)), _grid(lo=(0, 8, 0))]
    per_box = 10 * 8 * N  # 10 band columns x 8 y-rows x 16 z-rows

    band = _band(SLAB, width=1, grids=grids)

    np.testing.assert_array_equal(band.box_offset, [0, per_box, 2 * per_box])
    assert band.box_offset.dtype == np.int32
    assert band.box_offset[-1] == band.nrows
    lower = band.cell[band.box_offset[0] : band.box_offset[1]]
    upper = band.cell[band.box_offset[1] : band.box_offset[2]]
    assert lower[:, 1].max() == 7
    assert upper[:, 1].min() == 8


# ---------------------------------------------------------------------------
# 2. what the band refuses
# ---------------------------------------------------------------------------


def test_a_corner_reading_stencil_refuses_and_names_the_shapes_that_exist():
    """``depth`` is an axis-ray count, so it cannot answer for a box stencil.

    A corner neighbour is one ray step in two directions at once; measuring it
    needs the Chebyshev depth, which arrives with the first scheme that reads
    corners. Until then a "box" band would be silently too small.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        _band(SLAB, width=1, shape=BOX)

    message = str(excinfo.value)
    assert BOX in message
    assert CROSS in message


def test_an_unknown_stencil_shape_names_the_shapes_that_exist():
    with pytest.raises(ValueError) as excinfo:
        _band(SLAB, width=1, shape="star")

    message = str(excinfo.value)
    assert "star" in message
    assert BOX in message and CROSS in message


def test_a_band_wider_than_the_depth_clamp_refuses_instead_of_guessing():
    """Past the clamp, "far from a body" and "at the clamp" are the same number.

    Thresholding there would sweep the whole bulk into the band and call it a
    boundary cell, which is a plausible wrong answer rather than a loud one.
    """
    with pytest.raises(ValueError) as excinfo:
        _band(SLAB, width=MAX_DEPTH)

    assert str(MAX_DEPTH) in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. mesh.ibm — lazy, cached per generation
# ---------------------------------------------------------------------------


class _CountingBody:
    """A body that records how often it has been evaluated."""

    def __init__(self, body):
        self._body = body
        self.evaluations = 0

    def sdf(self, x, y, z):
        self.evaluations += 1
        return self._body.sdf(x, y, z)

    def normal(self, x, y, z):
        return self._body.normal(x, y, z)


class _CountingMethod:
    """A stand-in method: all the mesh may do with it is call ``preprocess``."""

    def __init__(self):
        self.calls = 0

    def preprocess(self, mesh, lev):
        self.calls += 1
        # an object of a shape no other layer knows — that is the point of it
        return {"call": self.calls}


def _make_mesh(bodies=None):
    """Single-box ``Mesh`` on the unit cube, ``16^3`` cells."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [int(p) for p in PERIODIC])
    ba = blockamr.BoxArray(box)
    ba.max_size(N)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {} if bodies is None else bodies
    return mesh


def test_setting_the_bodies_classifies_nothing_until_something_asks(blockamr_session):
    """Preprocessing is triggered by the first consumer, not by the assignment.

    A mesh that carries bodies but is never evaluated must not pay for them —
    and, less obviously, a field that names no IBM method must not be able to
    fail on a body it never uses.
    """
    body = _CountingBody(Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0)))
    mesh = _make_mesh({"wall": body})

    assert body.evaluations == 0

    mesh.ibm.geometry(0)

    assert body.evaluations > 0


def test_the_geometry_of_a_generation_is_built_once(blockamr_session):
    """The second consumer of a level's geometry gets the first one's."""
    body = _CountingBody(Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0)))
    mesh = _make_mesh({"wall": body})

    first = mesh.ibm.geometry(0)
    evaluations = body.evaluations
    second = mesh.ibm.geometry(0)

    assert second is first
    assert body.evaluations == evaluations


def test_the_band_of_a_mesh_is_cached_per_width(blockamr_session):
    """Two schemes of the same width share a band; a wider one gets its own."""
    mesh = _make_mesh(SLAB)

    narrow = mesh.ibm.band(0, width=1)

    assert mesh.ibm.band(0, width=1) is narrow
    assert mesh.ibm.band(0, width=2) is not narrow


def test_the_mesh_band_holds_the_hand_checked_cells_in_global_indices(blockamr_session):
    """The mesh path and the pure-numpy path are the same band.

    ``cell`` is a global index, not a box-local one, so a row is addressable
    without knowing which box produced it — the same convention the wall rows
    use for their targets.
    """
    expected = _expected_cells(SLAB_DEPTH, width=1)
    mesh = _make_mesh(SLAB)

    band = mesh.ibm.band(0, width=1)

    np.testing.assert_array_equal(band.cell, expected)
    np.testing.assert_array_equal(band.depth, SLAB_DEPTH[band.cell[:, 0]])


def test_moving_a_body_rebuilds_the_geometry_and_the_band(blockamr_session):
    """A moved body invalidates exactly what a moved grid does.

    Re-assigning ``mesh.bodies`` is how a prescribed motion is expressed, so it
    starts a new generation: everything keyed on the old one is dropped, and
    what comes back describes the wall where it is *now*. Serving the old band
    here would be the "plausible wrong numbers" failure — a band of the right
    shape, one cell off.
    """
    mesh = _make_mesh(_wall(8))
    before = mesh.ibm.band(0, width=1)
    version = mesh.ibm.grid_version

    mesh.bodies = _wall(7)
    after = mesh.ibm.band(0, width=1)

    assert mesh.ibm.grid_version != version
    assert after is not before
    expected = _expected_cells(_wall_depth(7), width=1)
    np.testing.assert_array_equal(after.cell, expected)
    np.testing.assert_array_equal(np.unique(after.cell[:, 0]), [0, 1, 2, 3, 4, 5, 6, 7])


def test_the_bodies_a_mesh_carries_are_reachable_from_its_ibm(blockamr_session):
    """One geometry, two spellings — ``mesh.ibm`` never holds its own copy."""
    mesh = _make_mesh(SLAB)

    assert mesh.ibm.bodies == SLAB

    moved = _wall(7)
    mesh.bodies = moved

    assert mesh.ibm.bodies == moved


def test_method_data_is_preprocessed_once_and_handed_back_untouched(blockamr_session):
    """The mesh stores what the method returned, and never looks inside it.

    The method declares its own data type; the mesh's job is a cache key and a
    lifetime. Identity is the whole assertion: anything the mesh inspected,
    copied or normalised would show up here as a different object.
    """
    method = _CountingMethod()
    mesh = _make_mesh(SLAB)

    data = mesh.ibm.data(method, 0)

    assert mesh.ibm.data(method, 0) is data
    assert method.calls == 1


def test_method_data_is_cached_per_method_and_level(blockamr_session):
    """Two methods on one mesh do not share a cache entry.

    A field on ``ghostCell`` and a field on ``directForcing`` are the reason
    the method is part of the key rather than a mesh-wide setting.
    """
    one, other = _CountingMethod(), _CountingMethod()
    mesh = _make_mesh(SLAB)

    assert mesh.ibm.data(one, 0) is not mesh.ibm.data(other, 0)
    assert one.calls == 1
    assert other.calls == 1


def test_moving_a_body_rebuilds_the_method_data_too(blockamr_session):
    """Method data is pure geometry, so it goes stale exactly when geometry does."""
    method = _CountingMethod()
    mesh = _make_mesh(_wall(8))
    first = mesh.ibm.data(method, 0)

    mesh.bodies = _wall(7)
    second = mesh.ibm.data(method, 0)

    assert second is not first
    assert method.calls == 2


def test_invalidate_starts_a_new_generation(blockamr_session):
    """The escape hatch for a geometry change the mesh cannot see.

    Mutating the ``bodies`` dict in place — or moving a body's own attributes —
    goes unnoticed by the setter, so it is spelled as an explicit invalidation
    rather than silently served from the cache.
    """
    mesh = _make_mesh(SLAB)
    before = mesh.ibm.geometry(0)
    version = mesh.ibm.grid_version

    mesh.ibm.invalidate()

    assert mesh.ibm.grid_version != version
    assert mesh.ibm.geometry(0) is not before


# ---------------------------------------------------------------------------
# the v2 packed geometry fab (B29) — ``mesh.ibm.geometry_fab``
# ---------------------------------------------------------------------------
#
# The v2 side of the same cache: one packed 8-component MultiFab per level
# (review.md §4, Q29(b)), filled over the GROWN box and uploaded from this very
# numpy evaluation (Q29(d)). It coexists with ``geometry(lev)`` above, which
# keeps returning v1's per-box dataclasses — ``depth`` and all — until B36/B37.
#
# These rows are here rather than in ``test_ibm_cell_type.py`` because they ask
# nothing of a kernel: they build a fab and read it back. The marker file owns
# the row that *classifies* against this fab.

#: A cylinder straddling the periodic ``y = 0`` seam. The wrapped and unwrapped
#: evaluations of its sdf differ sharply in the first y-ghost plane, which is
#: what makes the wrap assertion below discriminating rather than decorative.
SEAM = {"cyl": Cylinder(centre=(0.5, 0.08), radius=0.12, axis=2)}


def _analytic_packed(body, lo, hi, ngrow, periodic=PERIODIC):
    """The packed geometry of ``[lo - ngrow, hi + ngrow]``, written from the body.

    Independent of the implementation (verification §4): the coordinates, the
    wrap and the arithmetic are all spelled out here. ``periodic`` is a
    parameter so the row below can also build the *unwrapped* expectation and
    show that it is a different array.
    """
    axes = []
    for d in range(3):
        idx = np.arange(lo[d] - ngrow, hi[d] + ngrow + 1)
        if periodic[d]:
            idx = np.mod(idx, N)
        axes.append((idx + 0.5) * DX)
    x, y, z = np.meshgrid(*axes, indexing="ij")

    sdf = body.sdf(x, y, z)
    normal = np.asarray(body.normal(x, y, z))
    out = np.zeros(sdf.shape + (GEOM_NCOMP,), dtype=float)
    out[..., GEOM_SDF] = sdf
    out[..., GEOM_NORMAL : GEOM_NORMAL + 3] = normal
    out[..., GEOM_WALL_POINT : GEOM_WALL_POINT + 3] = (
        np.stack([x, y, z], axis=-1) - sdf[..., np.newaxis] * normal
    )
    out[..., GEOM_PATCH] = 0.0  # one body, so every cell is owned by patch 0
    return out


def test_the_packed_layout_is_the_one_the_compiled_side_expects(blockamr_session):
    """The layout is named twice — here and in ``ibm/geometry_view.H``.

    All five names, not just the count (B29-R, I-2). Only ``GEOM_NCOMP`` used
    to cross the language boundary, so the four *offsets* were declared
    independently on each side: swapping ``GEOM_NORMAL`` and
    ``GEOM_WALL_POINT`` in one file alone would have kept every row green while
    the compiled kernels read the wall point as a normal. B31 exports them, and
    this row is where the two declarations are held together.
    """
    assert GEOM_NCOMP == blockamr.IBM_GEOM_NCOMP
    assert GEOM_SDF == blockamr.GEOM_SDF
    assert GEOM_NORMAL == blockamr.GEOM_NORMAL
    assert GEOM_WALL_POINT == blockamr.GEOM_WALL_POINT
    assert GEOM_PATCH == blockamr.GEOM_PATCH


def test_the_v2_geometry_fab_is_filled_analytically_over_the_grown_box(blockamr_session):
    """F10 honoured: every ghost cell carries the analytic geometry, exactly.

    ``atol = 0``. The x direction is non-periodic here, so the x-ghost planes
    lie outside the domain and no fill step could ever have populated them; the
    y and z ghosts wrap. Both are the same statement — the geometry is evaluated
    on the grown box, never read back from a MultiFab.
    """
    mesh = _make_mesh(SEAM)
    mf = mesh.ibm.geometry_fab(0, ngrow=2)

    assert mf.num_comp() == GEOM_NCOMP
    assert mf.n_grow() == 2

    boxes = 0
    for mfi in blockamr.MFIterator(mf):
        vb = mfi.valid_box()
        lo = tuple(int(v) for v in vb.small_end())
        hi = tuple(int(v) for v in vb.big_end())
        got = mf.copy_grown_to_host(mfi)
        assert got.shape == (N + 4,) * 3 + (GEOM_NCOMP,)
        assert np.array_equal(got, _analytic_packed(SEAM["cyl"], lo, hi, 2))
        boxes += 1
    assert boxes == 1


def test_the_geometry_ghost_across_a_periodic_seam_is_the_wrapped_cell(blockamr_session):
    """The consistency requirement the always-on M5 check imposes (B28-R, I2).

    ``classify_default`` fills the marker's ghosts with ``FillBoundary``, i.e.
    from the **wrapped** valid cell, and then compares that marker against this
    fab's ghost sdf at the same index. So the geometry's ghost must be the
    wrapped cell's geometry too — evaluating the analytic body at the unwrapped
    coordinate is the natural implementation and it would throw M5 for every
    body near a periodic boundary. This row pins the convention *and* shows the
    two evaluations really disagree, so it cannot pass by accident.
    """
    mesh = _make_mesh(SEAM)
    mf = mesh.ibm.geometry_fab(0, ngrow=1)

    for mfi in blockamr.MFIterator(mf):
        vb = mfi.valid_box()
        lo = tuple(int(v) for v in vb.small_end())
        hi = tuple(int(v) for v in vb.big_end())
        got = mf.copy_grown_to_host(mfi)[..., GEOM_SDF]
        wrapped = _analytic_packed(SEAM["cyl"], lo, hi, 1)[..., GEOM_SDF]
        unwrapped = _analytic_packed(SEAM["cyl"], lo, hi, 1, periodic=(False,) * 3)[..., GEOM_SDF]

        seam = got[:, 0, :]  # the j = -1 ghost plane, across the periodic seam
        assert np.array_equal(seam, wrapped[:, 0, :])
        assert not np.allclose(seam, unwrapped[:, 0, :])


def test_the_v2_geometry_fab_grows_monotonically_and_is_never_shrunk(blockamr_session):
    """Why design §8's ``(lev, grid_version)`` cache key survives with no ngrow.

    A wider request rebuilds; a narrower one is served the wider fab, because
    shrinking it under the caller that asked for more is the one thing the
    ghost contract forbids. A new generation drops it like everything else.
    """
    mesh = _make_mesh(SLAB)

    narrow = mesh.ibm.geometry_fab(0, ngrow=1)
    assert narrow.n_grow() == 1
    assert mesh.ibm.geometry_fab(0, ngrow=1) is narrow

    wide = mesh.ibm.geometry_fab(0, ngrow=2)
    assert wide is not narrow
    assert wide.n_grow() == 2
    assert mesh.ibm.geometry_fab(0, ngrow=1) is wide

    mesh.ibm.invalidate()
    assert mesh.ibm.geometry_fab(0, ngrow=1) is not wide


def test_the_v2_geometry_fab_leaves_the_v1_geometry_untouched(blockamr_session):
    """The coexistence guarantee: v1 keeps ``depth`` until B36/B37.

    "``IbmGeometry`` loses ``depth``" is a statement about the *compiled* view,
    which has no such member. The v1 dataclass and its cache entry are not
    disturbed by building the v2 fab beside them.
    """
    mesh = _make_mesh(SLAB)

    v1 = mesh.ibm.geometry(0)
    mesh.ibm.geometry_fab(0, ngrow=1)

    assert mesh.ibm.geometry(0) is v1
    assert np.array_equal(v1[0].depth[:, 0, 0], SLAB_DEPTH)


def _centres(lo, hi, ngrow, periodic=PERIODIC):
    """Cell centres of ``[lo - ngrow, hi + ngrow]``, shape ``(nx, ny, nz, 3)``.

    Wrapped in a periodic direction and extended in a non-periodic one — the
    convention written out here rather than taken from the implementation.
    """
    axes = []
    for d in range(3):
        idx = np.arange(lo[d] - ngrow, hi[d] + ngrow + 1)
        if periodic[d]:
            idx = np.mod(idx, N)
        axes.append((idx + 0.5) * DX)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def test_the_v2_geometry_fab_of_a_mesh_without_bodies_is_the_documented_empty_fill(
    blockamr_session,
):
    """The ``noIbm`` fill: ``sdf = +inf``, a zero normal, the cell's own centre.

    Every other row here carries at least one body, so the branch a body-less
    mesh takes was never executed. It has to *fill*, not skip:
    ``classify_default`` reads component 0 of every grown cell it visits, and
    ``+inf > 0`` is what makes all of them fluid — an unwritten ghost there is a
    garbage marker, not an unused one. ``wall_point`` is the cell centre because
    there is no surface to project onto, which is also v1's answer for an empty
    ``body_list``.
    """
    mesh = _make_mesh()

    mf = mesh.ibm.geometry_fab(0, ngrow=1)

    boxes = 0
    for mfi in blockamr.MFIterator(mf):
        vb = mfi.valid_box()
        lo = tuple(int(v) for v in vb.small_end())
        hi = tuple(int(v) for v in vb.big_end())
        got = mf.copy_grown_to_host(mfi)
        assert got.shape == (N + 2,) * 3 + (GEOM_NCOMP,)
        assert (got[..., GEOM_SDF] == np.inf).all()
        assert (got[..., GEOM_NORMAL : GEOM_NORMAL + 3] == 0.0).all()
        assert (got[..., GEOM_PATCH] == 0.0).all()
        assert np.array_equal(got[..., GEOM_WALL_POINT : GEOM_WALL_POINT + 3], _centres(lo, hi, 1))
        boxes += 1
    assert boxes == 1


# ---------------------------------------------------------------------------
# more than one body: the owner rule, and v1/v2 parity (B29-R, I-3)
# ---------------------------------------------------------------------------
#
# Every row above carries a single body, so ``patch`` is 0 everywhere and the
# owner rule never has to choose between two bodies. These two rows are the
# multi-body half: one against a hand-written owner rule, one against v1's
# ``box_geometry`` — because ``packed_box_geometry`` re-derives v1's four values
# from the same primitives rather than calling it, and Q29(d)'s premise for
# B31's bitwise bar is that the two are the *same* geometry.

#: Two half-spaces whose solid regions overlap: solid where ``x < 0.5`` OR
#: ``y < 0.25``. Patch ids are indices into ``sorted(bodies)``, so "x-wall" is 0
#: and "y-wall" is 1, and the two owners have different normals — which is what
#: makes ``normal`` and ``wall_point`` discriminating here and not just
#: ``patch``.
CORNER = {
    "x-wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
    "y-wall": Plane(point=(0.0, 0.25, 0.0), normal=(0.0, 1.0, 0.0)),
}


def _flat_grid(lo=(0, 0, 0), hi=(N - 1, N - 1, N - 1)):
    """:func:`_grid` with **no** periodic direction.

    Both ``CORNER`` walls are half-spaces, so a periodic direction would put
    their solid across the seam from fluid and v1's ``_check_adjacent`` would
    refuse the configuration outright — and the parity row has to be able to run
    v1 on exactly the grid it runs v2 on.
    """
    return BoxGrid(
        lo=lo,
        hi=hi,
        dx=(DX, DX, DX),
        prob_lo=(0.0, 0.0, 0.0),
        domain_lo=(0, 0, 0),
        domain_hi=(N - 1, N - 1, N - 1),
        periodic=(False, False, False),
    )


def _analytic_packed_pair(first, second, lo, hi, ngrow, periodic=(False, False, False)):
    """The packed geometry of exactly **two** bodies, from the owner rule by hand.

    The same independence as :func:`_analytic_packed` — coordinates, owner rule,
    union sdf and intercept are all spelled out here and the only thing taken
    from the implementation's side is the *input* bodies. The rule
    (``ibm/classify.py``'s header: "the owner is the nearest **containing**
    surface, not the deepest … for a fluid cell — which no body contains — it is
    the nearest surface", ties to the lowest patch id) is written as booleans on
    the two signed distances rather than as an ``argmin`` over a stack, so
    neither of the two one-branch rules an implementation might have written
    instead can agree with it everywhere.
    """
    coords = _centres(lo, hi, ngrow, periodic)
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    s0, s1 = first.sdf(x, y, z), second.sdf(x, y, z)
    inside0, inside1 = s0 <= 0.0, s1 <= 0.0
    # inside a body ``|s| = -s``, so among the containing bodies the nearest
    # surface is the LARGER (least negative) signed distance.
    patch = np.where(
        inside0 & inside1,
        np.where(s0 >= s1, 0, 1),
        np.where(inside0, 0, np.where(inside1, 1, np.where(s0 <= s1, 0, 1))),
    )
    normal = np.where(
        patch[..., np.newaxis] == 0,
        np.asarray(first.normal(x, y, z)),
        np.asarray(second.normal(x, y, z)),
    )
    sdf = np.minimum(s0, s1)

    out = np.zeros(sdf.shape + (GEOM_NCOMP,), dtype=float)
    out[..., GEOM_SDF] = sdf
    out[..., GEOM_NORMAL : GEOM_NORMAL + 3] = normal
    out[..., GEOM_WALL_POINT : GEOM_WALL_POINT + 3] = coords - sdf[..., np.newaxis] * normal
    out[..., GEOM_PATCH] = patch
    return out


def test_the_packed_geometry_of_two_bodies_follows_the_two_branch_owner_rule():
    """Two bodies: a nonzero ``patch``, and each owner's own normal (B29-R, I-3).

    All three situations the owner rule distinguishes occur on ``CORNER``, and
    two of them are decided *differently* by each of the one-branch rules a
    packed builder might have been written with (``ngrow = 1``, so the array
    index of global cell ``[i, j, k]`` is ``[i + 1, j + 1, k + 1]``):

    * global ``[0, 3, 5]`` is inside **both** walls (``x = 0.03125``,
      ``y = 0.21875``). The y-wall's surface is ``0.03125`` away and the
      x-wall's ``0.46875``, so the nearest *containing* surface is the y-wall,
      patch **1** — whereas the deepest body, which a plain ``argmin(s)`` picks,
      is the x-wall.
    * global ``[0, 4, 5]`` is inside the x-wall only (``y = 0.28125``). The
      y-wall's surface is fifteen times nearer (``0.03125`` against
      ``0.46875``) but it does not contain the cell, so the owner is the x-wall,
      patch **0** — whereas ``argmin(|s|)`` picks the y-wall.
    * the fluid corner is contained by nothing and is owned by the nearest
      surface.

    Because the two walls have different normals, ``normal`` and ``wall_point``
    carry the same disagreement — a wrong owner is not a bookkeeping detail
    here, it mirrors the cell across the wrong plane.
    """
    grid = _flat_grid()

    (got,) = packed_geometry_on_grids([grid], CORNER, ngrow=1)

    want = _analytic_packed_pair(CORNER["x-wall"], CORNER["y-wall"], grid.lo, grid.hi, 1)
    assert got.shape == (N + 2,) * 3 + (GEOM_NCOMP,)
    assert np.array_equal(got, want)

    patch = got[..., GEOM_PATCH]
    assert patch[1, 4, 6] == 1.0  # global [0, 3, 5]: inside both, y-wall nearer
    assert patch[1, 5, 6] == 0.0  # global [0, 4, 5]: inside the x-wall only
    np.testing.assert_array_equal(got[1, 4, 6, GEOM_NORMAL : GEOM_NORMAL + 3], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(got[1, 5, 6, GEOM_NORMAL : GEOM_NORMAL + 3], [1.0, 0.0, 0.0])
    # non-vacuous: both patch ids are produced, and the fluid branch is reached
    np.testing.assert_array_equal(np.unique(patch), [0.0, 1.0])
    assert (got[..., GEOM_SDF] > 0.0).any()


def test_the_packed_geometry_is_the_v1_geometry_of_the_same_bodies():
    """v1 and v2 are pinned to each other, exactly, on two bodies (B29-R, I-3).

    ``packed_box_geometry`` does not call :func:`box_geometry`; it rebuilds the
    same four values from the same primitives. Q29(d)'s premise for B31's
    bitwise parity bar is that the v2 fab *is* v1's numpy evaluation, uploaded —
    so without this row the two can drift apart silently and B31 would be
    comparing its kernel against a geometry no other layer builds.

    ``atol = 0``: this is the same arithmetic on the same inputs, not two
    approximations of it. The valid box only — ``box_geometry`` has no grown
    form, which is exactly what B29 added.
    """
    grid = _flat_grid()
    # the patch-id convention, re-derived: ids are indices into ``sorted``
    names = sorted(CORNER)
    body_list = [CORNER[name] for name in names]

    (packed,) = packed_geometry_on_grids([grid], CORNER, ngrow=0)

    v1 = box_geometry(grid, names, body_list)
    assert np.array_equal(packed[..., GEOM_SDF], v1.sdf)
    assert np.array_equal(packed[..., GEOM_PATCH], v1.patch)
    assert np.array_equal(packed[..., GEOM_NORMAL : GEOM_NORMAL + 3], v1.normal)
    assert np.array_equal(packed[..., GEOM_WALL_POINT : GEOM_WALL_POINT + 3], v1.wall_point)
    # non-vacuous: v1 really does hand both patches out on this configuration
    np.testing.assert_array_equal(np.unique(v1.patch), [0, 1])
