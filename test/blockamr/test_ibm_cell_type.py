# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Conformance test for the v2 marker — ``ibm/cell_type.{H,cpp}`` (B28).

**This is a conformance test, not an acceptance test.** The acceptance suite is
verification §1's equation-level tests and it stays equation-only: no row in
``test_ibm_rungs.py``, ``test_ibm_convergence.py``, ``test_ibm_validation_*.py``
or any other acceptance file may read the marker, now or later. This file exists
because **M4** and **M5** (design §2.4) are conformance checks *no equation can
reach* — api §4's stated criterion — and it is the direct peer of
``test_stencil_kernels_cpp.py``: a binding-level test that exercises the compiled
surface directly, beside an acceptance suite that never mentions a kernel name.

**Every expectation here is computed in numpy from the analytic body**, never
from ``classify_default``'s output and never from ``ibm/classify.py`` — that is
verification §4's rule, and it is why ``_cell_type_numpy`` is the only readback
binding and ``set_val`` the only write.

The marker under test is the v2 (compiled) classification. It shares nothing
with the v1 numpy classification that drives the rest of the suite: no Python
production path calls ``blockamr.classify_default`` yet (B36 rewires it).
"""

import numpy as np
import pytest

import blockamr
from blockamr.ibm.body import Cylinder, Plane

# Underscore-private test binding (api §4). `from ._blockamr import *` skips
# underscore names, so it is reached on the extension module itself.
_cell_type_numpy = blockamr._blockamr._cell_type_numpy

SOLID = int(blockamr.CellType.SOLID)
WALL = int(blockamr.CellType.WALL)
FLUID = int(blockamr.CellType.FLUID)

N = 16
LO = (0.0, 0.0, 0.0)
HI = (1.0, 1.0, 1.0)
PERIODIC = (1, 1, 1)

CYLINDER = [Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)]
PLANE = [Plane(point=(0.5, 0.5, 0.5), normal=(1.0, 0.0, 0.0))]
TWO_CYLINDERS = [
    Cylinder(centre=(0.3, 0.3), radius=0.12, axis=2),
    Cylinder(centre=(0.78, 0.78), radius=0.12, axis=2),
]
FAR_AWAY = [Cylinder(centre=(5.0, 5.0), radius=0.2, axis=2)]
#: A body that cuts the x-lo domain face, for the non-periodic F10 row: its
#: solid region reaches into ghost cells that lie OUTSIDE the domain, where
#: ``FillBoundary`` has nothing to copy from and only an analytic geometry fill
#: puts a meaningful sdf.
X_FACE_CYLINDER = [Cylinder(centre=(0.0, 0.5), radius=0.25, axis=2)]

BODIES = {"cylinder": CYLINDER, "plane": PLANE, "two-cylinders": TWO_CYLINDERS}


# --------------------------------------------------------------------------
# the analytic oracle — numpy only, no compiled call anywhere below
# --------------------------------------------------------------------------


def _grid(lo, hi, n, periodic):
    """Cell centres of the inclusive index block ``[lo, hi]``.

    Indices outside the domain are **wrapped** in a periodic direction — that
    halo cell *is* the wrapped cell, body and all (``ibm/classify.py``'s
    documented convention, and exactly what ``FillBoundary`` propagates). In a
    non-periodic direction the analytic body is evaluated where the index says.
    """
    axes = []
    for d in range(3):
        idx = np.arange(lo[d], hi[d] + 1)
        if periodic[d]:
            idx = np.mod(idx, n)
        dx = (HI[d] - LO[d]) / n
        axes.append(LO[d] + (idx + 0.5) * dx)
    return np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")


def _sdf_block(bodies, lo, hi, n, periodic):
    """``min_b sdf_b`` on the inclusive index block ``[lo, hi]``."""
    x, y, z = _grid(lo, hi, n, periodic)
    return np.min(np.stack([b.sdf(x, y, z) for b in bodies]), axis=0)


def _geometry_block(bodies, lo, hi, n, periodic):
    """The packed 8-component geometry (Q29b) on ``[lo, hi]``, Fortran-ordered.

    Only component 0 (``sdf``) is read by the default rule; the rest are filled
    so the packed layout is exercised end to end rather than assumed.
    """
    x, y, z = _grid(lo, hi, n, periodic)
    per_body = np.stack([b.sdf(x, y, z) for b in bodies])
    owner = np.argmin(per_body, axis=0)
    sdf = _sdf_block(bodies, lo, hi, n, periodic)
    normals = np.stack([np.asarray(b.normal(x, y, z)) for b in bodies])
    normal = np.take_along_axis(normals, owner[None, ..., None], axis=0)[0]

    out = np.zeros(sdf.shape + (8,), dtype=float)
    out[..., 0] = sdf
    out[..., 1:4] = normal
    out[..., 4:7] = np.stack([x, y, z], axis=-1) - sdf[..., None] * normal
    out[..., 7] = owner
    return np.asfortranarray(out)


def _marker_block(bodies, lo, hi, n, periodic):
    """SOLID / WALL / FLUID on ``[lo, hi]`` — design §2.2's default rule.

    Needs the sdf one cell wider so the six-neighbour test is available on the
    outermost expected layer.
    """
    lo1 = tuple(v - 1 for v in lo)
    hi1 = tuple(v + 1 for v in hi)
    solid = _sdf_block(bodies, lo1, hi1, n, periodic) <= 0.0
    inner = solid[1:-1, 1:-1, 1:-1]
    touches = (
        solid[:-2, 1:-1, 1:-1]
        | solid[2:, 1:-1, 1:-1]
        | solid[1:-1, :-2, 1:-1]
        | solid[1:-1, 2:, 1:-1]
        | solid[1:-1, 1:-1, :-2]
        | solid[1:-1, 1:-1, 2:]
    )
    return np.where(inner, SOLID, np.where(touches, WALL, FLUID)).astype(np.uint8)


def _expected_grown(bodies, lo, hi, n, periodic, ngrow):
    """The expectation on the box ``[lo, hi]`` grown by ``ngrow``.

    Pass 1 (fluid / non-fluid from the sdf) holds on the whole grown box. Pass 2
    (the WALL upgrade) holds on the valid box, and reaches a ghost only through
    ``FillBoundary`` — which has a neighbour across a box edge and across a
    periodic seam, and none at a non-periodic domain edge.
    """
    glo = tuple(v - ngrow for v in lo)
    ghi = tuple(v + ngrow for v in hi)
    grown = np.where(_sdf_block(bodies, glo, ghi, n, periodic) <= 0.0, SOLID, FLUID)
    grown = grown.astype(np.uint8)
    if all(periodic):
        # every ghost is either box-internal or a periodic wrap: both carry the
        # neighbouring valid cell's marker, so the global valid marker answers.
        glob = _marker_block(bodies, (0, 0, 0), (n - 1,) * 3, n, periodic)
        ii, jj, kk = np.meshgrid(
            np.mod(np.arange(glo[0], ghi[0] + 1), n),
            np.mod(np.arange(glo[1], ghi[1] + 1), n),
            np.mod(np.arange(glo[2], ghi[2] + 1), n),
            indexing="ij",
        )
        return glob[ii, jj, kk]
    core = (slice(ngrow, -ngrow),) * 3
    grown[core] = _marker_block(bodies, lo, hi, n, periodic)
    return grown


# --------------------------------------------------------------------------
# the fixture — direct on the compiled surface
# --------------------------------------------------------------------------


def _level(n=N, max_size=None, periodic=PERIODIC):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox(list(LO), list(HI))
    geom = blockamr.Geometry(box, rb, 0, list(periodic))
    ba = blockamr.BoxArray(box)
    ba.max_size(n if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _geometry_fab(bodies, ba, dm, n, periodic, ngrow):
    g = blockamr.MultiFab(ba, dm, blockamr.IBM_GEOM_NCOMP, ngrow)
    for mfi in blockamr.MFIterator(g):
        vb = mfi.valid_box()
        lo = tuple(v - ngrow for v in vb.small_end())
        hi = tuple(v + ngrow for v in vb.big_end())
        g.copy_grown_from(mfi, _geometry_block(bodies, lo, hi, n, periodic))
    return g


def _classified(bodies, n=N, max_size=None, periodic=PERIODIC, ngrow=1, geom_ngrow=None):
    """``(ct, g, geom)`` — the v2 marker of ``bodies`` on an ``n^3`` level."""
    geom, ba, dm = _level(n, max_size, periodic)
    g = _geometry_fab(bodies, ba, dm, n, periodic, ngrow if geom_ngrow is None else geom_ngrow)
    ct = blockamr.CellTypeFab(ba, dm, ngrow)
    blockamr.classify_default(ct, g, geom)
    return ct, g, geom


def _boxes(g):
    """Yield ``(mfi, lo, hi)`` per local box, in MFIter order.

    A generator on purpose: ``MFIterator.__next__`` returns *itself* and drops
    its ``MFIter`` when the loop ends, so a collected list of them would all be
    the same, dead, object.
    """
    for mfi in blockamr.MFIterator(g):
        vb = mfi.valid_box()
        yield mfi, tuple(vb.small_end()), tuple(vb.big_end())


def _nboxes(g):
    return sum(1 for _ in _boxes(g))


def _shell(shape):
    """A mask of the outermost (ghost) layer of a grown readback."""
    mask = np.ones(shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = False
    return mask


def _global(ct, g, n=N):
    """The valid-region marker of every box assembled into one ``(n, n, n)``."""
    out = np.full((n, n, n), 255, dtype=np.uint8)
    for mfi, lo, hi in _boxes(g):
        out[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = _cell_type_numpy(ct, mfi)
    return out


# --------------------------------------------------------------------------
# 1-8 — the green rows
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BODIES))
def test_the_marker_holds_only_the_three_values(blockamr_session, name):
    """M4 — three values, valid cells and ghosts alike."""
    bodies = BODIES[name]
    ct, g, _ = _classified(bodies)
    for mfi, _lo, _hi in _boxes(g):
        grown = _cell_type_numpy(ct, mfi, grown=True)
        assert np.isin(grown, [SOLID, WALL, FLUID]).all()


@pytest.mark.parametrize("name", list(BODIES))
def test_wall_is_exactly_a_fluid_cell_with_a_solid_face_neighbour(blockamr_session, name):
    """The verify column's headline predicate, against the analytic oracle."""
    bodies = BODIES[name]
    ct, g, _ = _classified(bodies)
    for mfi, lo, hi in _boxes(g):
        got = _cell_type_numpy(ct, mfi)
        want = _marker_block(bodies, lo, hi, N, PERIODIC)
        assert np.array_equal(got, want)
    # and it is not a vacuous agreement: all three values occur
    glob = _global(ct, g)
    assert (glob == WALL).any() and (glob == SOLID).any() and (glob == FLUID).any()


@pytest.mark.parametrize("name", list(BODIES))
def test_solid_is_exactly_where_the_sdf_is_not_positive(blockamr_session, name):
    """M5's green side, on the valid region."""
    bodies = BODIES[name]
    ct, g, _ = _classified(bodies)
    for mfi, lo, hi in _boxes(g):
        got = _cell_type_numpy(ct, mfi)
        sdf = _sdf_block(bodies, lo, hi, N, PERIODIC)
        assert np.array_equal(got == SOLID, sdf <= 0.0)
        assert ((got == WALL) <= (sdf > 0.0)).all()


def test_ghost_cells_carry_the_marker_of_the_neighbouring_box(blockamr_session):
    """The verify column's *ghosts filled*, on an eight-box periodic level."""
    ct, g, _ = _classified(CYLINDER, max_size=8)
    assert _nboxes(g) == 8
    saw_wall_in_a_ghost = False
    for mfi, lo, hi in _boxes(g):
        got = _cell_type_numpy(ct, mfi, grown=True)
        want = _expected_grown(CYLINDER, lo, hi, N, PERIODIC, ngrow=1)
        assert np.array_equal(got, want)
        # a WALL in the ghost region can only have arrived through FillBoundary:
        # pass 2 runs on the valid box.
        saw_wall_in_a_ghost |= bool((got[_shell(got.shape)] == WALL).any())
    assert saw_wall_in_a_ghost


def test_the_marker_is_independent_of_the_box_decomposition(blockamr_session):
    """FillBoundary is doing real work — one box and eight agree bitwise."""
    ct1, g1, _ = _classified(CYLINDER, max_size=N)
    ct8, g8, _ = _classified(CYLINDER, max_size=8)
    assert _nboxes(g1) == 1 and _nboxes(g8) == 8
    assert np.array_equal(_global(ct1, g1), _global(ct8, g8))


def test_a_body_outside_the_domain_marks_every_cell_fluid(blockamr_session):
    """verification §2's *an all-FLUID marker means zero overhead* precondition."""
    ct, g, _ = _classified(FAR_AWAY)
    for mfi, _lo, _hi in _boxes(g):
        assert (_cell_type_numpy(ct, mfi, grown=True) == FLUID).all()


def test_the_outer_ghost_of_a_non_periodic_domain_is_solid_or_fluid_never_wall(blockamr_session):
    """A deliberate asymmetry, pinned so nobody "fixes" it.

    Pass 2 runs on the valid box and ``FillBoundary`` has no neighbour outside a
    non-periodic domain, so the outer ghost keeps pass 1's SOLID/FLUID — the
    correct fluid/solid state, and never a garbage value.
    """
    flat = (0, 0, 0)
    ct, g, _ = _classified(CYLINDER, periodic=flat)
    assert _nboxes(g) == 1
    for mfi, lo, hi in _boxes(g):
        grown = _cell_type_numpy(ct, mfi, grown=True)
        assert np.array_equal(grown, _expected_grown(CYLINDER, lo, hi, N, flat, ngrow=1))

        shell = _shell(grown.shape)
        assert not (grown[shell] == WALL).any()
        sdf = _sdf_block(CYLINDER, tuple(v - 1 for v in lo), tuple(v + 1 for v in hi), N, flat)
        assert np.array_equal(grown[shell] == SOLID, (sdf <= 0.0)[shell])


def test_classification_is_deterministic(blockamr_session):
    """verification §10's exactness rule — two classifications, bitwise equal."""
    geom, ba, dm = _level()
    g = _geometry_fab(CYLINDER, ba, dm, N, PERIODIC, 1)
    first = blockamr.CellTypeFab(ba, dm, 1)
    second = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(first, g, geom)
    blockamr.classify_default(second, g, geom)
    for mfi, _lo, _hi in _boxes(g):
        assert np.array_equal(
            _cell_type_numpy(first, mfi, grown=True),
            _cell_type_numpy(second, mfi, grown=True),
        )


# --------------------------------------------------------------------------
# B29 — the frozen geometry view, the honoured ngrow contract (review F10)
# --------------------------------------------------------------------------


def test_the_ghost_outside_a_non_periodic_face_follows_the_analytic_sdf(blockamr_session):
    """F10's *value* half, and it is invisible in a periodic test.

    A body cutting the x-lo domain face with ``is_periodic = (0, 1, 1)``. The
    ghost plane at ``i = -1`` lies outside the domain, so ``FillBoundary`` has
    no neighbour and no periodic image to copy from: the marker there is pass
    1's answer and nothing else, and it is right only because the *geometry*
    fab's ghost carries the analytically extended sdf. A ``FillBoundary``-based
    geometry fill cannot satisfy this row at all.
    """
    flat = (0, 1, 1)
    ct, g, _ = _classified(X_FACE_CYLINDER, periodic=flat)
    assert _nboxes(g) == 1
    for mfi, lo, hi in _boxes(g):
        grown = _cell_type_numpy(ct, mfi, grown=True)
        sdf = _sdf_block(
            X_FACE_CYLINDER, tuple(v - 1 for v in lo), tuple(v + 1 for v in hi), N, flat
        )
        outside = grown[0]  # the i = -1 plane: outside a non-periodic face
        want_solid = (sdf <= 0.0)[0]
        assert np.array_equal(outside == SOLID, want_solid)
        assert not (outside == WALL).any()
        # non-vacuous: that ghost plane really straddles the surface
        assert want_solid.any() and (~want_solid).any()


def test_the_classification_runs_at_a_wider_marker(blockamr_session):
    """A green ``ngrow = 2`` row — ``MARKER_NGROW`` is a floor, not a size.

    ``MARKER_NGROW = 1`` is the default rule's requirement; W1's degrade (B35)
    reads ``m(i +- 2s)`` and will allocate a two-cell marker. Nothing proved a
    wider marker worked, and the guard only enforces ``>= 1``, so a marker
    widened at B35 would have been the first one ever tried (B28-R, I3). Here it
    is exercised end to end: two ghost layers, eight boxes, classified against a
    two-cell geometry and validated by M4/M5 over the whole grown box.
    """
    ct, g, _ = _classified(CYLINDER, max_size=8, ngrow=2)
    assert _nboxes(g) == 8
    assert g.n_grow() == 2
    blockamr.validate_cell_type(ct, g)  # always-on inside classify; explicit here too
    for mfi, lo, hi in _boxes(g):
        got = _cell_type_numpy(ct, mfi, grown=True)
        assert got.shape == (8 + 4,) * 3
        assert np.array_equal(got, _expected_grown(CYLINDER, lo, hi, N, PERIODIC, ngrow=2))


def test_the_mesh_built_geometry_classifies_a_grown_box_in_one_pass(blockamr_session):
    """B29's verify column, on the geometry the **production builder** makes.

    ``IbmMesh.geometry_fab`` is the v2 path Q29(d) rules on: the packed fab is
    uploaded from the v1 numpy analytic evaluation through ``copy_grown_from``.
    Because that fill is analytic and grown, ``classify_default``'s first pass
    writes correct SOLID/FLUID markers into the ghost region directly, which is
    what "classification runs on grown boxes" asserts. This level is fully
    periodic, so every ghost here *also* has a ``FillBoundary`` source — the
    shell no exchange can reach is the non-periodic one, and it is the row
    below that pins it (B29-R, I-1). The expectation is still the numpy oracle
    above, never the builder's own output.
    """
    from blockamr.mesh import Mesh

    geom, ba, dm = _level(max_size=8)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {"cyl": CYLINDER[0]}

    g = mesh.ibm.geometry_fab(0, ngrow=1)
    assert g.num_comp() == blockamr.IBM_GEOM_NCOMP
    assert g.n_grow() == 1

    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, g, geom)

    assert _nboxes(g) == 8
    for mfi, lo, hi in _boxes(g):
        got = _cell_type_numpy(ct, mfi, grown=True)
        assert np.array_equal(got, _expected_grown(CYLINDER, lo, hi, N, PERIODIC, ngrow=1))


def test_the_mesh_built_geometry_classifies_a_non_periodic_grown_box(blockamr_session):
    """The production builder and the classification, on a NON-periodic domain.

    The two halves of F10 are proved separately above and this row joins them.
    ``…_classifies_a_grown_box_in_one_pass`` runs the production builder, but on
    a fully periodic level where every ghost also has a ``FillBoundary`` source,
    so nothing there depends on the builder having filled the ghosts;
    ``…_ghost_outside_a_non_periodic_face_follows_the_analytic_sdf`` does depend
    on it, but its geometry is this file's ``_geometry_fab`` fixture rather than
    ``IbmMesh.geometry_fab``.

    Here the geometry is the **builder's** and the domain is non-periodic in
    every direction, with a body cutting the x-lo face: the whole ghost shell
    lies outside the domain, ``FillBoundary`` has nothing to copy from anywhere
    on it, and the marker there is pass 1's answer read off the geometry fab. A
    builder that filled only the valid box would leave that shell at whatever
    the fab was allocated with, and the grown comparison would fail.
    """
    from blockamr.mesh import Mesh

    flat = (0, 0, 0)
    geom, ba, dm = _level(periodic=flat)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {"cyl": X_FACE_CYLINDER[0]}

    g = mesh.ibm.geometry_fab(0, ngrow=1)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, g, geom)

    assert _nboxes(g) == 1
    for mfi, lo, hi in _boxes(g):
        grown = _cell_type_numpy(ct, mfi, grown=True)
        assert np.array_equal(grown, _expected_grown(X_FACE_CYLINDER, lo, hi, N, flat, ngrow=1))
        # non-vacuous: the ghost plane outside the x-lo face really does
        # straddle the surface, so both SOLID and FLUID are asserted there
        outside = grown[0]
        assert (outside == SOLID).any() and (outside == FLUID).any()


# --------------------------------------------------------------------------
# 9-13 — the error surface (design §10: a sentence naming the offending object)
# --------------------------------------------------------------------------


def test_a_value_outside_the_three_is_named_with_its_cell(blockamr_session):
    """M4 red — a fourth value, named with the cell that carries it."""
    ct, g, _ = _classified(CYLINDER)
    ct.set_val(7)
    with pytest.raises(RuntimeError, match=r"value 7 at cell \[-?\d+, -?\d+, -?\d+\]"):
        blockamr.validate_cell_type(ct, g)


def test_a_wall_cell_inside_the_body_is_named_with_its_cell(blockamr_session):
    """M5 red — WALL where the sdf is not positive."""
    ct, g, _ = _classified(CYLINDER)
    ct.set_val(WALL)
    with pytest.raises(
        RuntimeError, match=r"cell \[-?\d+, -?\d+, -?\d+\] is marked WALL but its sdf is"
    ):
        blockamr.validate_cell_type(ct, g)


def test_a_solid_cell_in_the_fluid_is_named_with_its_cell(blockamr_session):
    """M5 red, the mirror arm — SOLID where the sdf is positive."""
    ct, g, _ = _classified(FAR_AWAY)
    ct.set_val(SOLID)
    with pytest.raises(
        RuntimeError, match=r"cell \[-?\d+, -?\d+, -?\d+\] is marked SOLID but its sdf is"
    ):
        blockamr.validate_cell_type(ct, g)


def test_a_marker_with_no_ghost_cell_is_refused_naming_both_widths(blockamr_session):
    """The F10 host guard, marker side."""
    geom, ba, dm = _level()
    g = _geometry_fab(CYLINDER, ba, dm, N, PERIODIC, 1)
    ct = blockamr.CellTypeFab(ba, dm, 0)
    with pytest.raises(RuntimeError, match=r"at least 1.*CellTypeFab has 0"):
        blockamr.classify_default(ct, g, geom)


def test_a_geometry_narrower_than_the_marker_is_refused_naming_both_widths(blockamr_session):
    """The F10 host guard, geometry side."""
    geom, ba, dm = _level()
    g = _geometry_fab(CYLINDER, ba, dm, N, PERIODIC, 1)
    ct = blockamr.CellTypeFab(ba, dm, 2)
    with pytest.raises(
        RuntimeError,
        match=r"classified against this geometry.*at least the marker's 2.*MultiFab has 1",
    ):
        blockamr.classify_default(ct, g, geom)


def test_validate_cell_type_refuses_a_geometry_narrower_than_the_marker(blockamr_session):
    """The F10 host guard on the **standalone** validation entry point.

    ``validate_cell_type`` is bound on its own so the M4/M5 red paths are
    reachable, and it iterates the *marker's* fab box while reading the
    *geometry's* ``Array4`` at the same indices. A marker wider than the
    geometry is therefore an out-of-bounds read — silent garbage in a release
    build, i.e. a spurious M5 sentence or a segfault, which is precisely what
    design §10's error surface exists to prevent (B28-R, I1). It must raise, and
    name both widths, exactly as ``classify_default`` already does.
    """
    _geom, ba, dm = _level()
    g = _geometry_fab(CYLINDER, ba, dm, N, PERIODIC, 1)
    wide = blockamr.CellTypeFab(ba, dm, 2)
    wide.set_val(FLUID)
    with pytest.raises(
        RuntimeError,
        match=r"classified against this geometry.*at least the marker's 2.*MultiFab has 1",
    ):
        blockamr.validate_cell_type(wide, g)


def test_a_geometry_with_the_wrong_component_count_is_refused(blockamr_session):
    """Q29b's packed layout is a contract, not a convention."""
    geom, ba, dm = _level()
    g = blockamr.MultiFab(ba, dm, 1, 1)
    g.set_val(1.0)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    with pytest.raises(RuntimeError, match=r"8 components.*has 1"):
        blockamr.classify_default(ct, g, geom)


# --------------------------------------------------------------------------
# 14-15 — pin_solid (design §7, Q3; joins this TU by Q29c)
# --------------------------------------------------------------------------


def test_pin_solid_writes_the_pin_value_at_solid_cells_and_nowhere_else(blockamr_session):
    """The v2 SOLID pin: the only write this architecture makes to a field."""
    geom, ba, dm = _level()
    g = _geometry_fab(CYLINDER, ba, dm, N, PERIODIC, 1)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, g, geom)

    phi = blockamr.MultiFab(ba, dm, 2, 1)
    phi.set_val(3.5)
    blockamr.pin_solid(phi, ct, -1.25, 2)

    for mfi, lo, hi in _boxes(g):
        got = phi.copy_to_host(mfi)
        marker = _marker_block(CYLINDER, lo, hi, N, PERIODIC)
        assert (marker == SOLID).any()
        for n in range(2):
            assert np.array_equal(got[..., n] == -1.25, marker == SOLID)
            assert np.array_equal(got[..., n] == 3.5, marker != SOLID)


def test_pin_solid_leaves_a_marker_without_a_solid_cell_bitwise_unchanged(blockamr_session):
    """An all-FLUID marker means the pin is a no-op (verification §2)."""
    geom, ba, dm = _level()
    g = _geometry_fab(FAR_AWAY, ba, dm, N, PERIODIC, 1)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, g, geom)

    phi = blockamr.MultiFab(ba, dm, 1, 1)
    rng = np.random.default_rng(28)
    before = []
    for mfi, _lo, _hi in _boxes(g):
        arr = np.asfortranarray(rng.standard_normal(phi.copy_to_host(mfi).shape))
        phi.copy_from(mfi, arr)
        before.append(arr)

    blockamr.pin_solid(phi, ct, 0.0, 1)

    for (mfi, _lo, _hi), want in zip(_boxes(g), before):
        assert np.array_equal(phi.copy_to_host(mfi), want)
