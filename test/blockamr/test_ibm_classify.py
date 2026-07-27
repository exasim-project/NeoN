# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cell classification and wall geometry — ``depth``, ``patch``, ``IbmGeometry``.

This is the method-agnostic IBM layer (``plans/IBM/design.md`` §2.1): what the
mesh will store, what the band is derived from, and what every boundary scheme
reads. It knows no method, no operator and no scheme, and it runs on explicit
per-box index ranges — plain numpy, no MultiFab, no backend, no kernel.

**Why this file is unit-level while the rest of the IBM suite is not.** The
equation-only rule of ``plans/IBM/verification.md`` §1 governs the transferred
*equation* suite: those tests assert physics through ``evaluate`` and survive
an implementation change precisely because they never touch an internal.
``depth`` is not physics — it is a cell count, and the task that introduces it
asks for exactly this evidence: hand-checked depths on a cylinder, a tilted
plane and two bodies, with no compiled extension needed to get them. A count
of cells is worth asserting as a count of cells; routing it through a laplacian
would test the laplacian.

Every expectation below is a literal or an analytic formula in the cell index,
computed by hand from the body's geometry. Nothing is read back from the
implementation.

The mesh is the unit cube at ``n = 16`` in every direction, so a cell is
``dx = 1/16`` wide and the centre of cell ``i`` sits at ``(i + 0.5)/16``.
"""

import numpy as np
import pytest

from blockamr.ibm.body import Cylinder, Plane
from blockamr.ibm.classify import MAX_DEPTH, BoxGrid
from blockamr.ibm.geometry import geometry_on_grids

N = 16
DX = 1.0 / N

#: Cell centres along one axis: the coordinate of cell ``i`` is ``CENTRE[i]``.
CENTRE = (np.arange(N) + 0.5) * DX

NON_PERIODIC = (False, False, False)
PERIODIC = (True, True, True)

#: The cylinder of the rung suite: radius 0.2 = 3.2 cells, centred in the domain.
RADIUS = 0.2
CENTRED = (0.5, 0.5)

#: Any row of cells away from the domain edge; every case here is z-invariant.
J_MID = 8
K_MID = 8


def _grid(lo=(0, 0, 0), hi=(N - 1, N - 1, N - 1), periodic=NON_PERIODIC):
    """One local box of the unit cube, in global index space."""
    return BoxGrid(
        lo=lo,
        hi=hi,
        dx=(DX, DX, DX),
        prob_lo=(0.0, 0.0, 0.0),
        domain_lo=(0, 0, 0),
        domain_hi=(N - 1, N - 1, N - 1),
        periodic=periodic,
    )


def _geometry(bodies, periodic=NON_PERIODIC):
    """The wall geometry of the single-box unit cube."""
    return geometry_on_grids([_grid(periodic=periodic)], bodies)[0]


def _cylinder(centre=CENTRED, radius=RADIUS):
    return Cylinder(centre=centre, radius=radius, axis=2)


# ---------------------------------------------------------------------------
# 1. depth — the signed, clamped, axis-ray cell count
# ---------------------------------------------------------------------------


def test_depth_along_a_wall_normal_is_the_signed_cell_count_to_the_wall():
    """The definition, on the simplest body there is.

    A plane at ``x = 0.5`` makes cell 8 the first fluid cell (its centre is at
    ``0.53125``) and cell 7 the last non-fluid one. Reading outward from the
    surface, the fluid side counts ``1, 2, 3, 4`` — ``depth = 1`` meaning "a
    face neighbour is not fluid" — and the solid side counts ``0, -1, -2, -3``,
    ``0`` meaning "a face neighbour is fluid". Both saturate at ``MAX_DEPTH``,
    which is what makes one array serve every stencil width.

    The plane is invariant in ``y`` and ``z``, so the four rays perpendicular
    to the normal never meet a state change and the whole field is this one
    column repeated — asserted, because a ray that leaked a shorter count from
    a perpendicular direction would be an off-by-one nobody notices in a
    single row.
    """
    geom = _geometry({"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))})

    expected = [-4, -4, -4, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4, 4, 4, 4]
    np.testing.assert_array_equal(geom.depth[:, J_MID, K_MID], expected)
    np.testing.assert_array_equal(
        geom.depth, np.broadcast_to(np.array(expected)[:, None, None], (N, N, N))
    )
    assert geom.depth.dtype == np.int8
    assert geom.patch.dtype == np.int8


def test_depth_without_a_body_is_the_clamped_fluid_value_everywhere():
    """No body means no wall to be near: every cell is bulk, at the clamp.

    The empty-``bodies`` case is not hypothetical — it is a mesh before its
    geometry is assigned, and it must classify rather than raise.
    """
    geom = _geometry({})

    np.testing.assert_array_equal(geom.depth, np.full((N, N, N), MAX_DEPTH))
    np.testing.assert_array_equal(geom.patch, np.zeros((N, N, N)))


# The cylinder's solid cells, from the body alone: the centre is at cell corner
# (8, 8) so cell (i, j) sits (i - 7.5, j - 7.5) cells from the axis, and the
# radius is 3.2 cells. Rounding those two comparisons by hand gives the solid
# set  j = 5: i in 6..9,  j = 6..9: i in 5..10,  j = 10: i in 6..9  — six rows,
# 32 cells per z-slice. Every depth below is counted off that picture.
_CYLINDER_DEPTHS = [
    (7, 8, -2, "deepest cell of the body: three cells to fluid in every direction"),
    (8, 8, -2, "its mirror across the axis"),
    (6, 8, -1, "one cell in from the first solid layer"),
    (5, 8, 0, "first solid layer: cell 4 is fluid"),
    (10, 8, 0, "the same layer on the far side"),
    (4, 8, 1, "first fluid layer: a face neighbour is solid"),
    (3, 8, 2, "second fluid layer"),
    (2, 8, 3, "third"),
    (1, 8, 4, "fourth — the last one the clamp still resolves exactly"),
    (0, 8, 4, "fifth: clamped, and indistinguishable from the far field"),
    (5, 5, 1, "the cross stencil reaches the solid cell (6, 5) sideways"),
    (4, 4, 4, "diagonally two cells from the body, but no axis ray meets it"),
    (0, 0, 4, "the far corner"),
]


@pytest.mark.parametrize("i, j, expected, reason", _CYLINDER_DEPTHS)
def test_cylinder_depth_matches_the_hand_counted_cell_layers(i, j, expected, reason):
    """Depths counted by hand off the analytic solid set, cell by cell.

    Two of the rows carry the whole point of the *cross* (axis-ray) definition
    (``plans/IBM/design.md`` §4): cell (5, 5) is at depth 1 because the ray in
    ``+x`` hits solid, while cell (4, 4) — a shorter Euclidean distance from
    the body along the diagonal — is at the clamp, because no axis ray from it
    meets a solid cell within reach. A depth built from a distance rather than
    from rays would swap those two, and a scheme with a corner-reading stencil
    would need the other definition (which is why declaring the shape is
    mandatory there).

    The deepest cell of this body is only ``-2``: a radius of 3.2 cells cannot
    reach the ``-MAX_DEPTH`` clamp, which the plane cases cover instead.
    """
    geom = _geometry({"cyl": _cylinder()})

    assert geom.depth[i, j, K_MID] == expected, reason


def test_tilted_plane_depth_counts_axis_ray_cells_rather_than_wall_distance():
    """The tilted case, where "cells to the wall" and "distance to the wall"
    genuinely differ — a 45° plane puts the surface ``1/sqrt(2)`` cell widths
    away along the diagonal but a whole cell away along either axis.

    The plane ``x + y = 1`` makes cell ``(i, j)`` fluid exactly when
    ``i + j >= 16`` (cell centres are half-integers, so ``i + j = 15`` is the
    last non-fluid diagonal). Walking ``-x`` from a fluid cell leaves the fluid
    after ``i + j - 15`` steps and walking ``+x`` from a non-fluid one enters
    it after ``16 - i - j`` steps, and the ``y`` rays give the same counts by
    symmetry — so the whole field is one closed-form expression in ``i + j``,
    clamped. Asserted over every cell, not a sample: this is the case where an
    error in the ray walk shows up as a band that is one diagonal too thin.
    """
    geom = _geometry({"tilt": Plane(point=(0.5, 0.5, 0.0), normal=(1.0, 1.0, 0.0))})

    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    fluid_depth = np.minimum(i + j - 15, MAX_DEPTH)
    solid_depth = -np.minimum(15 - i - j, MAX_DEPTH)
    expected = np.where(i + j >= 16, fluid_depth, solid_depth)

    np.testing.assert_array_equal(geom.depth[:, :, K_MID], expected)
    # the two ends of the clamp, so the assertion above cannot pass vacuously
    assert geom.depth[15, 15, K_MID] == MAX_DEPTH
    assert geom.depth[0, 0, K_MID] == -MAX_DEPTH


def test_the_classification_does_not_depend_on_the_box_decomposition():
    """The reason the sdf is evaluated analytically on a grown box instead of
    being read back from a MultiFab: a cell on a box edge must see the same
    rays as it does when the box is the whole domain, including the ones that
    leave the box entirely.

    Splitting the domain in two across the cylinder — so the split plane cuts
    the body's band — and stitching the halves back together must reproduce the
    single-box classification bit for bit.
    """
    bodies = {"cyl": _cylinder()}
    whole = _geometry(bodies)
    left, right = geometry_on_grids(
        [_grid(hi=(7, N - 1, N - 1)), _grid(lo=(8, 0, 0))],
        bodies,
    )

    np.testing.assert_array_equal(np.concatenate([left.depth, right.depth]), whole.depth)
    np.testing.assert_array_equal(np.concatenate([left.patch, right.patch]), whole.patch)
    assert left.depth.shape == (8, N, N)


# ---------------------------------------------------------------------------
# 2. Wall geometry — sdf, normal, wall point
# ---------------------------------------------------------------------------


def test_wall_geometry_of_a_plane_is_the_analytic_distance_normal_and_foot():
    """``sdf``, ``normal`` and ``wall_point``, each against its closed form.

    For a plane at ``x = 0.5`` the signed distance of a cell is ``x - 0.5``,
    the into-fluid normal is ``+x`` everywhere (including *inside* the body:
    the normal is the body's, not the cell's), and the foot of the normal is
    the cell's own ``(y, z)`` at ``x = 0.5``. All three are exact in binary,
    so this is an equality and not a tolerance.
    """
    geom = _geometry({"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))})

    np.testing.assert_array_equal(geom.sdf[:, J_MID, K_MID], CENTRE - 0.5)
    np.testing.assert_array_equal(
        geom.normal, np.broadcast_to(np.array([1.0, 0.0, 0.0]), (N, N, N, 3))
    )
    np.testing.assert_array_equal(geom.wall_point[..., 0], np.full((N, N, N), 0.5))
    np.testing.assert_array_equal(
        geom.wall_point[5, J_MID, K_MID], [0.5, CENTRE[J_MID], CENTRE[K_MID]]
    )


def test_wall_point_of_a_cylinder_lands_on_the_surface():
    """``wall_point = x - sdf * n̂`` is the body intercept, so for a cylinder it
    is exactly one radius from the axis — on both sides of the surface, which
    is what makes it usable from a solid cell and from a fluid one.

    The tolerance is a rounding bound on ``hypot`` and the division that
    normalises the radial normal, not a geometric one.
    """
    geom = _geometry({"cyl": _cylinder()})
    band = np.abs(geom.depth) <= 1

    offset = geom.wall_point[band][:, :2] - np.asarray(CENTRED)
    np.testing.assert_allclose(np.hypot(offset[:, 0], offset[:, 1]), RADIUS, atol=1e-15)
    # the axial coordinate is untouched: an axis-2 cylinder's normal has no z
    np.testing.assert_array_equal(geom.wall_point[..., 2], np.broadcast_to(CENTRE, (N, N, N)))
    assert band.sum() > 0


# ---------------------------------------------------------------------------
# 3. Two bodies — patch attribution
# ---------------------------------------------------------------------------

# Two cylinders of equal radius, placed so that the centre of cell 7 is exactly
# equidistant from both axes (0.46875 - 0.25 == 0.6875 - 0.46875 == 0.21875,
# all exact in binary) and inside both bodies. Patch ids are indices into
# sorted(bodies), so "left" is 0 and "right" is 1.
_LEFT = (0.25, 0.5)
_RIGHT = (0.6875, 0.5)
_OVERLAP_RADIUS = 0.25
_TIE_CELL = 7


def _two_cylinders():
    return {
        "left": _cylinder(centre=_LEFT, radius=_OVERLAP_RADIUS),
        "right": _cylinder(centre=_RIGHT, radius=_OVERLAP_RADIUS),
    }


def test_a_cell_inside_two_bodies_at_equal_depth_is_owned_by_the_lower_id():
    """The tie rule, on a cell built to tie exactly.

    Both bodies contain cell 7 of the mid row and its distance to the two
    surfaces is the same floating-point number, so the owner is decided by the
    tie rule alone: the lowest patch id, which is the first name in
    ``sorted(bodies)``. Without a rule this cell's diagnostics, forces and
    surface datum would depend on dict ordering.
    """
    geom = _geometry(_two_cylinders())

    assert geom.depth[_TIE_CELL, J_MID, K_MID] < 0, "the tie cell must be inside both bodies"
    assert geom.patch[_TIE_CELL, J_MID, K_MID] == 0
    # its mirror leans to the right body, so the tie above is a real tie and
    # not simply "patch 0 everywhere"
    assert geom.patch[_TIE_CELL + 1, J_MID, K_MID] == 1


def test_the_nearest_containing_surface_owns_a_cell_not_the_deepest_one():
    """Attribution is by ``|s|``, not by ``min s`` — the distinction only shows
    up where two bodies overlap, and it is the one the row builders depend on:
    a cell reconstructs against the surface it is *near*, which is the one its
    fluid neighbours are on the other side of.

    A small body sits inside a big one. Cell (7, 11) is 0.056 inside the small
    body's surface and 0.079 inside the big one's, so the small body owns it
    while the union signed distance — which is a property of the fluid region,
    not of a body — stays the deeper of the two.
    """
    bodies = {"big": _cylinder(radius=0.3), "small": _cylinder(centre=(0.5, 0.75), radius=0.1)}
    geom = _geometry(bodies)

    assert geom.patch[7, 11, K_MID] == 1  # "small" is the second name in sorted order
    np.testing.assert_allclose(geom.sdf[7, 11, K_MID], -0.079029, atol=1e-6)
    np.testing.assert_allclose(geom.normal[7, 11, K_MID], [-np.sqrt(0.5), -np.sqrt(0.5), 0.0])


def test_two_bodies_share_one_depth_field_measured_to_the_nearer_of_them():
    """Depth is a property of the fluid region, so it is counted against the
    union of the bodies and not per body.

    In the mid row the two overlapping cylinders leave only cell 15 fluid, and
    cell 0's ``-x`` ray steps *outside the domain* — where, the mesh being
    non-periodic here, the analytic body is simply evaluated at the extended
    coordinate and reports fluid. Both edge cells are therefore at depth 0.
    Cell 7 is the overlap cell: eight cells from fluid along ``x``, but only
    two along ``y``, so the ``y`` ray decides it.
    """
    geom = _geometry(_two_cylinders())
    row = geom.depth[:, J_MID, K_MID]

    assert row[0] == 0
    assert row[7] == -1
    assert row[14] == 0
    assert row[15] == 1
    np.testing.assert_array_equal(geom.patch[:, J_MID, K_MID], [0] * 8 + [1] * 8)


# ---------------------------------------------------------------------------
# 4. The body the mesh cannot carry
# ---------------------------------------------------------------------------


def test_a_body_meeting_fluid_across_the_periodic_seam_names_the_cell_and_patch():
    """The "body incompatible with this mesh" error, which lives at the
    classification layer because that is where it is detectable.

    For a true signed distance the invariant ``|s| < dx`` holds at every
    ``depth = 0`` cell — the function is 1-Lipschitz and such a cell has a
    fluid face neighbour one cell away. The way to break it is to make the
    neighbour something other than a neighbour: on a periodic mesh the state of
    a cell outside the domain is judged at its *wrapped* position, so a body
    that is not itself periodic has its solid region meet fluid across the
    seam. Here a wall at ``x = 0.9`` leaves cells 14 and 15 fluid, and cell 0
    — 0.87 away from that surface, fourteen cell widths — sees them as its
    ``-x`` neighbour.

    The message must localise it to a cell and a patch; a generic "IBM geometry
    error" is unactionable on a mesh with several bodies.
    """
    bodies = {"wall": Plane(point=(0.9, 0.0, 0.0), normal=(1.0, 0.0, 0.0))}

    with pytest.raises(ValueError, match=r"\[0, \d+, \d+\] on patch 'wall'") as excinfo:
        _geometry(bodies, periodic=PERIODIC)

    message = str(excinfo.value)
    assert "cell size" in message
    assert "incompatible with this mesh" in message
    # the same body on the same mesh, minus the seam, classifies normally
    assert _geometry(bodies).depth[0, J_MID, K_MID] == -MAX_DEPTH
