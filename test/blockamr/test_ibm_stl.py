# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""An STL triangulation as an immersed body (``blockamr.ibm.stl``).

**Why a cube.** A triangulated cube is an *exact* oracle, which almost no other
STL is: the union of its twelve triangles is the box's boundary exactly, so the
distance to the triangulation and the distance to the analytic box agree to the
last bit — there is no faceting error to hide behind a tolerance. Every
signed-distance assertion below is therefore an equality at machine epsilon
against :func:`cube_sdf`, a closed form written here and never read back from
:class:`~blockamr.ibm.stl.Stl`.

**The fixture is generated, not checked in.** ``_write_cube`` emits ASCII STL
from the corner coordinates, so the body's geometry is visible in the test
rather than in an opaque binary blob, and the file lives in ``tmp_path``. It is
the format's own writer, so ``scale``/``center``/``reverse_normal`` are varied
by re-reading a file the test wrote, never by patching text.

**What is exact and what is not.** The signed distance is exact — it comes
straight from AMReX's BVH point-to-triangle distance at the queried coordinate,
with no interpolation anywhere. The *normal* is a central difference of that
distance (``STLtools`` exposes no surface normal), so it is exact next to a
face, where the difference is taken across a plane, and carries the difference's
``O(h^2)`` truncation next to an edge or a corner, where the distance is
curved. The two are asserted at their two different tolerances below, and the
end-to-end rows inherit the second one.

**The strongest row is byte-identical.** The marker is classified from the sign
of the sampled distance alone, so an STL cube and an analytic cube of the same
size must produce the *same bytes* — not close markers, the same ones. That is
``test_the_marker_of_an_stl_cube_is_the_analytic_cubes``, and it is the row that
proves an STL body reaches the compiled classification on the ordinary path.
"""

import itertools

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp
from blockamr.field import CellField
from blockamr.ibm import FixedValue, GhostCell, Stl
from blockamr.mesh import Mesh

N = 16  # cells per side of the unit cube; dx = 1/16
DX = 1.0 / N

#: The body: an axis-aligned cube from 0.3 to 0.7. No face lies on a cell face
#: and no cell centre lies on a face, so the solid cells are unambiguous — the
#: centres inside are ``(i + 0.5)/16 in (0.3, 0.7)``, i.e. ``i`` in ``5..10``.
CENTRE = (0.5, 0.5, 0.5)
HALF = 0.2
SOLID_RANGE = range(5, 11)

#: Periodic, like the cylinder cases: the cube is strictly interior, so no
#: solid region meets fluid across a seam.
PERIODIC = (True, True, True)

BACKEND = "cpp"


# ---------------------------------------------------------------------------
# the fixture: an ASCII STL of an axis-aligned cube, and its closed form
# ---------------------------------------------------------------------------


def _write_cube(path, centre, half, name="cube"):
    """Write a watertight ASCII STL of an axis-aligned cube.

    Twelve triangles, two per face, wound counter-clockwise seen from outside
    so the facet normals point out of the solid. AMReX's ASCII reader wants
    exactly seven lines per facet and no blank lines.
    """
    lo = np.asarray(centre, dtype=float) - half
    hi = np.asarray(centre, dtype=float) + half
    corner = {
        s: np.array([lo[d] if s[d] == 0 else hi[d] for d in range(3)])
        for s in itertools.product((0, 1), repeat=3)
    }
    #: (outward normal, the face's four corners counter-clockwise from outside)
    faces = [
        ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
        ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
    ]
    lines = [f"solid {name}"]
    for normal, quad in faces:
        for triangle in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
            lines.append("facet normal {:.17g} {:.17g} {:.17g}".format(*normal))
            lines.append("outer loop")
            for key in triangle:
                lines.append("vertex {:.17g} {:.17g} {:.17g}".format(*corner[key]))
            lines.append("endloop")
            lines.append("endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")
    return path


def cube_sdf(x, y, z, centre=CENTRE, half=HALF):
    """Exact signed distance to an axis-aligned cube; positive outside it."""
    offset = np.stack(np.broadcast_arrays(x, y, z), axis=-1) - np.asarray(centre, dtype=float)
    outside = np.abs(offset) - half
    return np.linalg.norm(np.maximum(outside, 0.0), axis=-1) + np.minimum(outside.max(axis=-1), 0.0)


def cube_normal(x, y, z, centre=CENTRE, half=HALF):
    """Exact outward unit normal of the cube's signed distance, ``(..., 3)``.

    Outside, it is the direction from the nearest surface point; inside, the
    axis of the nearest face. Both are the gradient of :func:`cube_sdf`, and
    both are well defined at every cell centre of this mesh (none sits on a
    diagonal of the cube's interior, where the nearest face is ambiguous).
    """
    offset = np.stack(np.broadcast_arrays(x, y, z), axis=-1) - np.asarray(centre, dtype=float)
    outside = np.abs(offset) - half
    nearest_face = np.eye(3)[outside.argmax(axis=-1)]
    direction = np.where(
        (outside > 0.0).any(axis=-1)[..., np.newaxis], np.maximum(outside, 0.0), nearest_face
    ) * np.sign(offset)
    return direction / np.linalg.norm(direction, axis=-1, keepdims=True)


class AnalyticCube:
    """The same cube as a closed-form body — the oracle every STL row is against."""

    def sdf(self, x, y, z):
        return cube_sdf(x, y, z)

    def normal(self, x, y, z):
        return cube_normal(x, y, z)


@pytest.fixture
def cube_stl(tmp_path):
    """The STL of :data:`CENTRE` / :data:`HALF`, written into ``tmp_path``."""
    return _write_cube(tmp_path / "cube.stl", CENTRE, HALF)


def _cell_centres(ngrow=0):
    """Cell centres of the level, grown by ``ngrow``, shape ``(n, n, n, 3)``."""
    axis = np.arange(-ngrow, N + ngrow)
    index = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return (index + 0.5) * DX


def _mesh(bodies):
    """One box on the unit cube, ``16^3`` cells, periodic."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [int(p) for p in PERIODIC])
    ba = blockamr.BoxArray(box)
    ba.max_size(N)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
    return mesh


# ---------------------------------------------------------------------------
# 1. the signed distance
# ---------------------------------------------------------------------------


def test_the_signed_distance_of_a_triangulated_cube_is_the_analytic_one(blockamr_session, cube_stl):
    """The exactness that makes a cube the oracle: no faceting error at all.

    ``atol`` is machine epsilon on values of order ``0.5``, i.e. the row asserts
    equality and nothing weaker.
    """
    points = _cell_centres()
    body = Stl(str(cube_stl))

    got = body.sdf(points[..., 0], points[..., 1], points[..., 2])

    np.testing.assert_allclose(
        got,
        cube_sdf(points[..., 0], points[..., 1], points[..., 2]),
        rtol=0.0,
        atol=1e-15,
        err_msg="STL signed distance differs from the analytic cube's",
    )


def test_the_signed_distance_is_negative_inside_the_solid(blockamr_session, cube_stl):
    """The sign convention ``ibm/body.py`` documents: ``s > 0`` is fluid.

    Asserted on the *classification* that convention drives — the cells whose
    centres are inside 0.3..0.7 are exactly ``i`` in ``5..10`` per axis — rather
    than on one sampled value, so a global sign flip cannot pass.
    """
    points = _cell_centres()
    body = Stl(str(cube_stl))

    got = body.sdf(points[..., 0], points[..., 1], points[..., 2])

    inside = np.zeros((N, N, N), dtype=bool)
    inside[np.ix_(SOLID_RANGE, SOLID_RANGE, SOLID_RANGE)] = True
    assert np.array_equal(got < 0.0, inside)


def test_the_body_is_evaluated_at_a_half_cell_offset_the_same_way(blockamr_session, cube_stl):
    """The face centres ``_check_resolvable_gap`` asks about are a lattice too.

    That check shifts the cell centres half a cell onto a face, which is the
    one query in the pipeline that is *not* a cell centre. It must be answered
    exactly, not by interpolating between the two cells it lies between.
    """
    points = _cell_centres()
    body = Stl(str(cube_stl))
    face_x = points[..., 0] + 0.5 * DX

    got = body.sdf(face_x, points[..., 1], points[..., 2])

    np.testing.assert_allclose(
        got,
        cube_sdf(face_x, points[..., 1], points[..., 2]),
        rtol=0.0,
        atol=1e-15,
        err_msg="STL signed distance on the x-face lattice",
    )


def test_scale_stretches_the_body_and_center_translates_it(blockamr_session, tmp_path):
    """AMReX's spelling: every vertex becomes ``v * scale + center``.

    Read the *unit* cube at the origin and place it with the two arguments; the
    result must be the cube the other rows read straight from the file, which
    pins both arguments at once and pins ``center`` as a translation rather
    than a centring.
    """
    unit = _write_cube(tmp_path / "unit.stl", centre=(0.0, 0.0, 0.0), half=0.5)
    points = _cell_centres()

    body = Stl(str(unit), scale=2.0 * HALF, center=CENTRE)

    np.testing.assert_allclose(
        body.sdf(points[..., 0], points[..., 1], points[..., 2]),
        cube_sdf(points[..., 0], points[..., 1], points[..., 2]),
        rtol=0.0,
        atol=1e-15,
        err_msg="scale/center did not place the unit cube on 0.3..0.7",
    )


def test_reverse_normal_swaps_the_solid_and_the_fluid(blockamr_session, cube_stl):
    """The fix for a file whose facet normals point inward.

    Flipping the winding negates the signed distance exactly — same distance,
    other side — so the assertion is against the *negated* analytic cube and is
    just as tight as the un-flipped one.
    """
    points = _cell_centres()
    body = Stl(str(cube_stl), reverse_normal=True)

    got = body.sdf(points[..., 0], points[..., 1], points[..., 2])

    np.testing.assert_allclose(
        got,
        -cube_sdf(points[..., 0], points[..., 1], points[..., 2]),
        rtol=0.0,
        atol=1e-15,
        err_msg="reverse_normal did not swap solid and fluid",
    )


# ---------------------------------------------------------------------------
# 2. the normal — the one derived quantity
# ---------------------------------------------------------------------------


def test_the_normal_next_to_a_face_is_that_faces_normal(blockamr_session, cube_stl):
    """Exact where it matters most: the wall cells the ghost-cell method uses.

    Next to a face the signed distance is linear along the face normal and
    constant across it, so the central difference is the face normal to
    round-off — the two shifted samples differ by exactly ``2h`` on one axis and
    by exactly nothing on the other two.
    """
    points = _cell_centres()
    body = Stl(str(cube_stl))
    # the column of cells just outside the +x face, away from every edge
    face = (slice(11, 12), slice(6, 10), slice(6, 10))
    x, y, z = (points[..., d][face] for d in range(3))

    got = body.normal(x, y, z)

    np.testing.assert_allclose(
        got,
        np.broadcast_to([1.0, 0.0, 0.0], got.shape),
        rtol=0.0,
        atol=1e-12,
        err_msg="the normal of the cells outside the +x face",
    )


def test_the_normal_is_the_gradient_of_the_signed_distance_everywhere_outside(
    blockamr_session, cube_stl
):
    """Away from the faces the difference is second order, and this is its size.

    ``atol`` bounds the central difference's ``O(h^2/r)`` truncation with
    ``h = 1e-3 * dx`` on a body whose sharpest feature is a corner ``~dx``
    away; the measured worst case is ``5.4e-8``, an order and a half inside it,
    and it is what the rows below inherit. A *wrong* gradient — a mixed-up
    axis, an unnormalised vector, a sign — is wrong by ``O(1)`` here, so the
    bound is loose only against its own truncation.
    """
    points = _cell_centres()
    body = Stl(str(cube_stl))
    x, y, z = (points[..., d] for d in range(3))
    outside = cube_sdf(x, y, z) > 0.0

    got = body.normal(x, y, z)

    np.testing.assert_allclose(
        got[outside],
        cube_normal(x, y, z)[outside],
        rtol=0.0,
        atol=1e-6,
        err_msg="the STL normal differs from the analytic cube's outside the body",
    )
    np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1.0, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. what the pipeline gets — the marker, the geometry, an evaluate
# ---------------------------------------------------------------------------


def test_the_marker_of_an_stl_cube_is_the_analytic_cubes(blockamr_session, cube_stl):
    """Byte-identical: the classification is the sign of the sampled distance.

    The strongest honest assertion available, and the one that shows an STL
    body reaches ``blockamr.classify_default`` on the ordinary path — the
    marker is read back with the one test binding api §4 allows, over the grown
    box so the ghost shell is compared too.
    """
    stl_mesh = _mesh({"cube": Stl(str(cube_stl))})
    analytic_mesh = _mesh({"cube": AnalyticCube()})

    def marker(mesh):
        cell_type = mesh.ibm.cell_type(GhostCell, 0)
        grown = mesh.ibm.geometry_fab(0, 1)
        return np.stack(
            [
                blockamr._blockamr._cell_type_numpy(cell_type, mfi, True)
                for mfi in blockamr.MFIterator(grown)
            ]
        )

    assert np.array_equal(marker(stl_mesh), marker(analytic_mesh))


def test_the_wall_geometry_of_an_stl_cube_is_the_analytic_cubes(blockamr_session, cube_stl):
    """Depth, patch and signed distance identically; the wall point to the
    normal's own accuracy.

    ``wall_point`` is ``x - s * n``, so it carries the normal's ``O(h^2)``
    truncation scaled by the distance to the surface — under one cell — and
    nothing else.

    **On fluid cells only, and that is not a weakening.** A cube's *interior*
    diagonals are its medial set: the centre of a corner cell like ``[5, 5, 5]``
    is the same distance from three faces, so "the nearest face" is not a
    question with an answer there. The closed form breaks the tie by axis order
    and returns the ``+x`` face; the central difference returns the mean of the
    three, ``(1, 1, 1)/sqrt(3)``. Both are correct subgradients of a function
    that has no gradient at that point. Those cells are ``SOLID`` under
    ``classify_default`` — the ghost-cell method reads wall geometry at ``WALL``
    cells, which are fluid — so nothing downstream ever asks. Restricting the
    row to the fluid side asserts every value that is read, and the solid side
    is covered by the ``sdf`` equality above, which holds everywhere.
    """
    stl_geometry = _mesh({"cube": Stl(str(cube_stl))}).ibm.geometry(0)[0]
    analytic_geometry = _mesh({"cube": AnalyticCube()}).ibm.geometry(0)[0]

    assert np.array_equal(stl_geometry.depth, analytic_geometry.depth)
    assert np.array_equal(stl_geometry.patch, analytic_geometry.patch)
    np.testing.assert_allclose(stl_geometry.sdf, analytic_geometry.sdf, rtol=0.0, atol=1e-15)

    fluid = analytic_geometry.sdf > 0.0
    np.testing.assert_allclose(
        stl_geometry.wall_point[fluid],
        analytic_geometry.wall_point[fluid],
        rtol=0.0,
        # the normal's truncation times |s| < dx; measured worst case 4.5e-9
        atol=1e-7,
        err_msg="the body intercept of the STL cube",
    )


def test_a_ghost_cell_evaluate_on_an_stl_body_matches_the_analytic_body(blockamr_session, cube_stl):
    """End to end: the same laplacian, the same wall condition, two bodies.

    The equation surface, which is the suite's vocabulary
    (``plans/IBM/verification.md`` §1) — an STL body is not a special path, so
    an ``evaluate`` through ``ghostCell`` has to land where the analytic body
    lands.

    The agreement is round-off, not the normal's ``O(h^2)``, and the reason is
    worth writing down: a ``WALL`` cell is a fluid cell with a *face* neighbour
    in the solid, so for an axis-aligned cube every ``WALL`` cell sits directly
    outside a face — the one place the central difference is exact. The
    corner-diagonal fluid cells, where the normal does carry truncation, touch
    the solid only edge-on and are never ``WALL``. Measured worst case is
    ``2e-13`` on values up to ``270``, i.e. ``1.3e-15`` relative; the tolerance
    is set two orders of magnitude above that.
    """

    def result(bodies):
        mesh = _mesh(bodies)
        field = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cube": FixedValue(0.0)})
        mf = field.mf[0]
        for mfi in blockamr.MFIterator(mf):
            arr = mf.copy_to_host(mfi)
            lo = mfi.valid_box().small_end()
            i = np.arange(arr.shape[0])[:, None, None] + lo[0]
            x = (i + 0.5) * DX
            arr[:, :, :, :] = (x * x)[..., None]
            mf.copy_from(mfi, arr)
        field.fill_patch(0, 0.0)
        out = evaluate(
            Equation(exp.laplacian(1.0, field)),
            t=0.0,
            solution={"ibm": "ghostCell", "backend": BACKEND},
        )
        array = np.asarray(out[0][0])
        return array.reshape(array.shape[:3])

    np.testing.assert_allclose(
        result({"cube": Stl(str(cube_stl))}),
        result({"cube": AnalyticCube()}),
        rtol=1e-13,
        atol=1e-11,
        err_msg="the ghost-cell laplacian on an STL cube",
    )


# ---------------------------------------------------------------------------
# 4. reading the file, and refusing what cannot be answered
# ---------------------------------------------------------------------------


def test_the_stl_file_is_read_once_and_never_again(blockamr_session, cube_stl):
    """The body rides ``mesh.ibm``'s caches, and it keeps its reader.

    Asserted by deleting the file: a second generation's geometry — everything
    ``mesh.ibm`` holds is dropped when ``bodies`` is re-assigned — still has to
    be built, which it can only do from a triangulation already in memory.
    """
    body = Stl(str(cube_stl))
    mesh = _mesh({"cube": body})
    first = mesh.ibm.geometry(0)[0].sdf

    cube_stl.unlink()
    mesh.bodies = {"cube": body}

    np.testing.assert_array_equal(mesh.ibm.geometry(0)[0].sdf, first)


def test_a_second_evaluate_does_not_fill_the_signed_distance_again(blockamr_session, cube_stl):
    """The fills belong to the classification, not to the evaluate.

    The compiled surface is wrapped in a counter — the real ``StlSurface``
    still does every fill, so nothing under test is replaced — and the second
    ``mesh.ibm`` consumer of the same generation must not add to it.
    """

    class CountingSurface:
        def __init__(self, surface):
            self._surface = surface
            self.fills = 0

        def signed_distance_block(self, **kwargs):
            self.fills += 1
            return self._surface.signed_distance_block(**kwargs)

    body = Stl(str(cube_stl))
    counting = CountingSurface(body.surface)
    body._surface = counting
    mesh = _mesh({"cube": body})

    mesh.ibm.cell_type(GhostCell, 0)
    after_first = counting.fills
    mesh.ibm.cell_type(GhostCell, 0)

    assert after_first > 0
    assert counting.fills == after_first


def test_a_missing_stl_file_is_refused_when_the_body_is_built(blockamr_session, tmp_path):
    """Before anything else, and by name.

    AMReX's reader calls ``amrex::Abort`` on a file it cannot open, which takes
    the process down instead of raising; the body checks first so a typo in a
    path is an exception at the line that wrote it.
    """
    missing = tmp_path / "not-here.stl"

    with pytest.raises(FileNotFoundError, match="no STL file at"):
        Stl(str(missing))


def test_a_query_off_a_regular_lattice_is_refused_rather_than_interpolated(
    blockamr_session, cube_stl
):
    """The honest half of the grid-fill design.

    ``STLtools`` answers "signed distance on this lattice", and every point the
    IBM geometry asks about is on one. A point set that is not gets an
    exception, never a value interpolated from neighbouring samples — that
    value would be wrong by ``O(dx^2)`` exactly where the ghost-cell method
    reconstructs from it.
    """
    body = Stl(str(cube_stl))
    scattered = np.array([0.1, 0.15, 0.2001])
    on_lattice = np.array([0.1, 0.15, 0.2])

    with pytest.raises(ValueError, match="regular axis-aligned lattice"):
        body.sdf(scattered, on_lattice, on_lattice)


def test_a_normal_at_a_single_point_is_refused(blockamr_session, cube_stl):
    """One point names no spacing, and the normal is a difference over one.

    The signed distance at one point is still exact and still served — only the
    derived quantity refuses, which is the whole rule of this module: never
    approximate silently.
    """
    body = Stl(str(cube_stl))

    assert body.sdf(0.5, 0.5, 0.9) == pytest.approx(cube_sdf(0.5, 0.5, 0.9))
    with pytest.raises(ValueError, match="needs more than one point"):
        body.normal(0.5, 0.5, 0.9)
