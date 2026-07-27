# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``ghostCell``'s preprocessing — image points, their stencils, Invariant F.

The method's own data (``plans/IBM/design.md`` §2.4): one image point per fluid
wall-layer cell, its trilinear stencil, and the assertion that every live donor
of that stencil is a fluid cell.

**Why this file is unit-level while the equation suite is not.** The same
reason ``test_ibm_classify.py`` is: the task that introduces this asks for
Invariant F to hold on a cylinder, a tilted plane and two bodies, and that is a
statement about a stencil, not about physics. Routing it through a laplacian
would test the laplacian, and would not distinguish "no violation" from "no
stencil". Every expectation below is a literal or an analytic formula in the
cell index; nothing is read back from the implementation.

Plain numpy on explicit per-box index ranges, so none of it needs the compiled
extension. The mesh is the unit cube at ``n = 16``.
"""

from dataclasses import replace

import numpy as np
import pytest

from blockamr.ibm.body import Cylinder, Plane
from blockamr.ibm.classify import BoxGrid, _patches
from blockamr.ibm.geometry import geometry_on_grids
from blockamr.ibm.ghost_cell import K, ghost_cell_data, image_step

N = 16
DX = 1.0 / N

NON_PERIODIC = (False, False, False)

RADIUS = 0.2
CENTRED = (0.5, 0.5)

#: The tilted normal of the rung-5 probe, normalised at import.
TILTED = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])


def _grid(dx=(DX, DX, DX), periodic=NON_PERIODIC):
    """One local box covering the unit cube, in global index space."""
    return BoxGrid(
        lo=(0, 0, 0),
        hi=(N - 1, N - 1, N - 1),
        dx=dx,
        prob_lo=(0.0, 0.0, 0.0),
        domain_lo=(0, 0, 0),
        domain_hi=(N - 1, N - 1, N - 1),
        periodic=periodic,
    )


def _data(bodies, grid=None):
    """``GhostCellData`` of the single-box unit cube, and its geometry."""
    grid = grid or _grid()
    names, body_list = _patches(bodies)
    geometry = geometry_on_grids([grid], bodies)
    return ghost_cell_data([grid], geometry, names, body_list), geometry[0], grid


def _cylinder(centre=CENTRED, radius=RADIUS):
    return Cylinder(centre=centre, radius=radius, axis=2)


def _two_cylinders(gap):
    """Two cylinders whose surfaces are ``gap`` apart, straddling the centre."""
    return {
        "left": _cylinder((0.5 - RADIUS - gap / 2.0, 0.5)),
        "right": _cylinder((0.5 + RADIUS + gap / 2.0, 0.5)),
    }


BODIES = {
    "cylinder": {"cyl": _cylinder()},
    "tilted-plane": {"wall": Plane(point=(0.5, 0.5, 0.5), normal=tuple(TILTED))},
    "two-bodies": _two_cylinders(4.0 * DX),
}


# ---------------------------------------------------------------------------
# 1. Invariant F — every live donor is a fluid cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bodies", list(BODIES.values()), ids=list(BODIES))
def test_every_live_donor_of_an_image_stencil_is_a_fluid_cell(bodies):
    """Invariant F (``plans/IBM/row-contract.md`` §8) on the three geometries
    the ladder used to be needed for.

    ``preprocess`` asserts it and raises otherwise, so reaching this line at
    all is half the statement; the fluid test is repeated here from the
    analytic body — an oracle independent of the classification — so that a
    build which stopped asserting would still be caught.
    """
    data, geometry, grid = _data(bodies)
    assert data.nrows == int((geometry.depth == 1).sum()) > 0

    live = data.weight != 0.0
    centres = np.asarray(grid.prob_lo) + (data.donor + 0.5) * np.asarray(grid.dx)
    for body in bodies.values():
        outside = body.sdf(centres[..., 0], centres[..., 1], centres[..., 2]) > 0.0
        assert (outside | ~live).all(), f"a live donor lies inside {body!r}"


def test_a_non_fluid_donor_names_the_cell_and_the_patch():
    """The other half of Invariant F: it is a *loud* failure.

    The violation has to be injected, and that is itself a result worth
    recording: over ~1800 generated geometries — cylinders of radius 0.3 to 5
    cells about all three axes, cell aspect ratios 1 to 8, planes at 63
    orientations and 8 offsets, two-body gaps and fluid slabs from 0.2 to 4
    cells — **not one** produced a non-fluid live donor. The step is capped at
    half a cell and points along ``n̂``, whose every component points away from
    the body, so the donor block leans away from the solid in all three
    directions at once.

    Un-asserted is not the same as unnecessary: a second body on the outward
    side, or a different image-point rule in a later method, breaks that
    argument, and the failure would otherwise be a wrong number rather than a
    crash. So the guard is driven directly, by handing preprocessing a geometry
    whose normals point the wrong way, and what is asserted is the sentence it
    owes: which cell, which patch, and what is wrong.
    """
    bodies = {"cyl": _cylinder()}
    grid = _grid()
    names, body_list = _patches(bodies)
    geometry = geometry_on_grids([grid], bodies)[0]
    inward = replace(geometry, normal=-geometry.normal)

    with pytest.raises(ValueError, match=r"\[\d+, \d+, \d+\] on patch 'cyl'") as excinfo:
        ghost_cell_data([grid], [inward], names, body_list)

    message = str(excinfo.value)
    assert "Invariant F" in message
    assert "fluid" in message


# ---------------------------------------------------------------------------
# 2. The image point — where it is, and what reads it
# ---------------------------------------------------------------------------


def test_the_image_point_of_an_axis_aligned_wall_is_the_outward_face_centre():
    """``h = 0.5 / max_d(|n_d| / dx_d)`` is half a cell for an axis normal.

    A plane at ``x = 0.5`` with ``n = +x``: cell 8 is the first fluid cell, its
    centre sits at ``0.53125``, i.e. ``0.5 dx`` from the wall, and its image
    point is half a cell further out — the centre of the face between cells 8
    and 9, at ``x = 9/16``. The closure distance is then ``1 dx`` exactly, and
    the trilinear stencil is the two cells either side of that face with weight
    ``1/2`` each.
    """
    data, geometry, _box = _data({"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))})

    wall_layer = np.argwhere(geometry.depth == 1)
    assert (wall_layer[:, 0] == 8).all(), "the plane's wall layer is the x = 8 column"

    np.testing.assert_allclose(data.image_point[:, 0], 9.0 / N)
    np.testing.assert_allclose(data.distance, DX)
    live = data.weight > 0.0
    np.testing.assert_allclose(data.weight[live], 0.5)
    assert sorted(np.unique(data.donor[live][:, 0])) == [8, 9]


def test_the_image_point_lies_on_the_wall_normal_through_its_cell():
    """The property linear exactness rests on: the closure reads the field at a
    known distance **along the normal**, so a field that varies linearly along
    ``n̂`` is reproduced with no error at all.

    Asserted on the tilted plane, where every component of the normal is
    nonzero and a construction that quietly snapped to an axis would show up.
    """
    bodies = BODIES["tilted-plane"]
    data, geometry, grid = _data(bodies)

    wall_layer = np.argwhere(geometry.depth == 1) + np.asarray(grid.lo)
    centre = np.asarray(grid.prob_lo) + (wall_layer + 0.5) * np.asarray(grid.dx)
    offset = data.image_point - centre

    step = np.linalg.norm(offset, axis=1)
    np.testing.assert_allclose(offset, step[:, np.newaxis] * TILTED, atol=1e-14)
    # ...and the distance to the surface is the cell's own plus that step
    np.testing.assert_allclose(data.distance, geometry.sdf[geometry.depth == 1] + step)


def test_every_donor_is_within_one_cell_so_a_single_ghost_layer_carries_it():
    """The bound that fixes the field's ghost width, on the worst geometry
    available here: a cylinder on cells eight times longer in ``z`` than in
    ``x``. The step is capped at half a cell **in index units**, so no donor is
    more than one cell from the row's own cell in any direction — which is why
    ``ngrow = 1`` is enough and the old mirror geometry's 7-cell reach is gone.
    """
    grid = _grid(dx=(DX, DX, 8.0 * DX))
    data, geometry, _g = _data({"cyl": _cylinder()}, grid=grid)

    cell = np.argwhere(geometry.depth == 1) + np.asarray(grid.lo)
    reach = np.abs(data.donor - cell[:, np.newaxis, :])
    assert reach.max() <= 1


def test_the_image_step_is_the_longest_that_stays_within_half_a_cell():
    """The rule itself, on unit-length normals and an anisotropic cell.

    An axis-aligned normal gives half that axis' cell; a diagonal normal gives
    a longer step, because the same half-cell budget is spent across three
    directions at once. Longer is better — the closure divides by it.
    """
    dx = np.array([1.0, 2.0, 4.0])
    axis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(image_step(axis, dx), [0.5, 2.0])

    diagonal = np.full((1, 3), 1.0 / np.sqrt(3.0))
    np.testing.assert_allclose(image_step(diagonal, dx), [0.5 * np.sqrt(3.0)])


# ---------------------------------------------------------------------------
# 3. Shape and ordering — the contract a boundary scheme indexes on
# ---------------------------------------------------------------------------


def test_the_data_lists_the_wall_layer_in_band_order_with_weights_summing_to_one():
    """A boundary scheme selects these rows out of its band with
    ``band.depth == 1`` and no lookup table, so the order must be the band's:
    per local box, in the order ``np.argwhere`` yields the cells. Trilinear
    weights sum to one, which is what makes the interpolation reproduce a
    constant exactly and the wall closure annihilate it.
    """
    data, geometry, _box = _data(BODIES["cylinder"])

    assert data.donor.shape == (data.nrows, K, 3)
    assert data.weight.shape == (data.nrows, K)
    assert data.donor.dtype == np.int32
    np.testing.assert_allclose(data.weight.sum(axis=1), 1.0)

    # band order: the same cells the geometry's own mask yields, in that order
    expected = np.argwhere(geometry.depth == 1)
    centre = (expected + 0.5) * DX
    np.testing.assert_array_less(np.linalg.norm(data.image_point - centre, axis=1), DX)


def test_a_body_the_mesh_never_meets_leaves_the_data_empty():
    """No wall layer, no image points — and no failure. The empty band is the
    zero-overhead path, and preprocessing has to agree with it.
    """
    data, geometry, _box = _data({"far": _cylinder(centre=(99.0, 99.0))})

    assert (geometry.depth == 1).sum() == 0
    assert data.nrows == 0
    assert data.donor.shape == (0, K, 3)
