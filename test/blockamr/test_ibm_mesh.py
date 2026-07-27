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
from blockamr.ibm.body import Plane
from blockamr.ibm.classify import MAX_DEPTH, BoxGrid
from blockamr.ibm.geometry import geometry_on_grids
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
