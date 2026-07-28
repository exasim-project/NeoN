# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""W1's degrade — the compiled siblings ``div_{vanleer,quick}_acc_ibm`` (B35).

**Conformance, not acceptance**, like ``test_ibm_cell_type.py``,
``test_ibm_ghost_cell_cpp.py`` and ``test_ibm_wall_functors.py``: nothing on an
evaluate path reaches these kernels. The scheme dispatch, the production marker
allocation and the ``noIbm`` routing are B36's; here the siblings are called
directly through the bindings.

W1 (design §5, ruled at review.md §4 **Q42(a)**): *a width-``w > 1`` interior
scheme falls back to its width-1 formula at any cell whose stencil would read a
``SOLID`` cell*, tested **per cell** on the stencil's own twelve axis offsets —
not per face. The fallback is ``div_upwind``, and the undegraded arm is a call
to the parent's own device function inside a statement copied token for token
from the parent kernel.

**The bitwise rows here are branch-selection claims, not floating-point parity
claims** — Q35's rule applied honestly. Both sides of every equality are the
*same* compiled device function, in one translation unit, under one set of
flags; there is no second implementation to reassociate and no expression whose
contraction could differ, so the dyadic-grid vacuity that bit B31 has no
purchase. What these rows can fail on is the branch: a sibling that degrades a
cell it should not, or fails to degrade one it should, moves the bits.

Which makes the *other* vacuity the one to guard against: **a sibling that never
degrades — a verbatim copy of the parent — passes the two "bitwise the parent"
rows.** The falsifiers are the all-``SOLID`` row (the fallback is pinned against
``div_upwind_acc``), the non-vacuity row (the degrade demonstrably fires), and
the plane-wall probe. That probe carries its own control:
``test_the_undegraded_kernel_fails_the_same_probe`` runs the **parent** through
it and asserts it comes out wrong, so the probe can never quietly become a row
that anything passes.
"""

import itertools

import numpy as np
import pytest

import blockamr
from blockamr.field import FaceField
from blockamr.ibm.body import Cylinder, Plane
from blockamr.mesh import Mesh

_cell_type_numpy = blockamr._blockamr._cell_type_numpy

SOLID = int(blockamr.CellType.SOLID)
WALL = int(blockamr.CellType.WALL)
FLUID = int(blockamr.CellType.FLUID)

N = 16
#: The reach of a width-2 div stencil, and so the ghost width the siblings
#: demand of both the field and the marker. ``MARKER_NGROW`` stays 1: it is the
#: default classification's floor, not an allocation size (``cell_type.H``).
NGROW = 2

#: ``(the parent, the W1 sibling)`` per wide scheme. The parents are the
#: reference for every "bitwise" row below; ``div_upwind_acc`` is the reference
#: for the fallback.
SIBLINGS = {
    "vanleer": ("div_vanleer_acc", "div_vanleer_acc_ibm"),
    "quick": ("div_quick_acc", "div_quick_acc_ibm"),
}
SCHEMES = sorted(SIBLINGS)

CYLINDER = Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)
#: Axis-asymmetric on purpose: a z-uniform cylinder on the diagonal survives
#: ``x <-> y``, so it cannot see an axis mix-up in the reach test.
TILTED = Plane(point=(0.5, 0.5, 0.5), normal=(1.0, 2.0, 3.0))

#: The plane-wall probe (D1's v2 form). ``X_WALL`` falls on a cell face, so no
#: cell centre sits on the surface and the marker is unambiguous.
X_WALL = 0.25
A_LIN = 2.0
B_LIN = 1.5
PLANE_WALL = Plane(point=(X_WALL, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
#: The first ``FLUID`` cell of the plane fixture: ``i = 4`` is ``WALL`` (its
#: ``i = 3`` neighbour is ``SOLID``) and ``i = 5`` is ``FLUID`` but still reads
#: ``i = 3`` at its ``-2`` offset. That cell is where the whole per-cell /
#: per-face question lives, and it is asserted on by index.
DEPTH_TWO = 5
#: The value pinned into every ``SOLID`` cell before the probe runs. Without it
#: the analytic field continues linearly into the body and a wide stencil that
#: reads the solid gets the right answer for the wrong reason (§7.1).
POISON = 1e30


# ---------------------------------------------------------------------------
# the level, the marker, the field
# ---------------------------------------------------------------------------


def _level(max_size=None):
    """``(mesh, geom, ba, dm)`` — one non-periodic unit cube at ``N^3``."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(N if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom, ba, dm


def _classified(body, max_size=None):
    """``(mesh, geom, ba, dm, ct)`` with the marker built the **production** way.

    ``IbmMesh.geometry_fab(0, ngrow=2)`` then ``classify_default`` — B29's I3
    path, the ``ngrow = 2`` classification that was made green precisely so the
    wide marker W1 needs would not first be tried here. The expectations in this
    file are never the marker: they are numpy, from the analytic body.
    """
    mesh, geom, ba, dm = _level(max_size)
    mesh.bodies = {"body": body}
    g = mesh.ibm.geometry_fab(0, ngrow=NGROW)
    ct = blockamr.CellTypeFab(ba, dm, NGROW)
    blockamr.classify_default(ct, g, geom)
    return mesh, geom, ba, dm, ct


def _valid(mf):
    """The valid-region numpy block of the (single-box) MultiFab."""
    got = [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]
    assert len(got) == 1
    return got[0]


def _marker(ct, mf):
    """The **grown** marker block, read back through the test binding."""
    got = [_cell_type_numpy(ct, mfi, True) for mfi in blockamr.MFIterator(mf)]
    assert len(got) == 1
    return got[0]


def _field(ba, dm, block, ncomp=1):
    """A field carrying ``block`` over its whole grown box, ghosts included.

    Filled grown rather than filled-and-exchanged: this level is non-periodic,
    so a ``FillBoundary`` would leave the domain-edge ghosts undefined, and the
    wide stencils read two of them.
    """
    mf = blockamr.MultiFab(ba, dm, ncomp, NGROW)
    for mfi in blockamr.MFIterator(mf):
        mf.copy_grown_from(mfi, np.asfortranarray(block))
    return mf


def _random_block(ncomp, seed):
    """Seeded random data on the grown box.

    Not a constant and not a linear field: a constant is annihilated by every
    scheme at every width and a linear field is reproduced exactly by every
    scheme at every width, so neither can tell a degraded stencil from an intact
    one. Only the plane-wall probe below uses a linear field, and there the
    poison is what gives it teeth.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N + 2 * NGROW,) * 3 + (ncomp,))


def _flux(mesh, values=(1.0, -1.0, 0.5)):
    """The three face-flux MultiFabs. Mixed signs exercise both upwind arms."""
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")
    for d, v in enumerate(values):
        ff[0][d].mf.set_val(v)
    return tuple(ff[0][d].mf for d in range(3))


def _evaluate(name, ba, dm, phi, faces, geom, ct=None, coeff=1.0, ncomp=1):
    """Run one kernel into a freshly zeroed ``out`` and return the valid block.

    ``ct = None`` selects the plain (parent / width-1) signature. Both kernels
    are always run on the same ``phi`` object in the same process, so "the same
    data" is literally the same data.
    """
    out = blockamr.MultiFab(ba, dm, ncomp, 0)
    out.set_val(0.0)
    fn = getattr(blockamr, name)
    if ct is None:
        fn(out, phi, faces[0], faces[1], faces[2], geom, coeff, ncomp)
    else:
        fn(out, phi, ct, faces[0], faces[1], faces[2], geom, coeff, ncomp)
    return _valid(out)


# ---------------------------------------------------------------------------
# the analytic oracles — numpy, from the body, never from the implementation
# ---------------------------------------------------------------------------


def _centres(ngrow):
    """Cell-centre coordinates of the box grown by ``ngrow``."""
    c = (np.arange(-ngrow, N + ngrow) + 0.5) / N
    return np.meshgrid(c, c, c, indexing="ij")


def _solid_grown(body):
    """The analytic ``SOLID`` mask over the grown box (``sdf <= 0``)."""
    return body.sdf(*_centres(NGROW)) <= 0.0


def _window(grown, offset):
    """The valid-box-sized window of a grown array, shifted by ``offset``."""
    return grown[tuple(slice(NGROW + offset[d], NGROW + offset[d] + N) for d in range(3))]


def _bulk_mask(body):
    """Valid cells whose full width-2 cross stencil lies in the fluid.

    Taken from the grown analytic body — no ``np.roll`` and so no wrap
    assumption — and computed with no access to the implementation's marker,
    which is what makes it an independent oracle (verification §10).
    """
    fluid = ~_solid_grown(body)
    mask = np.ones((N, N, N), dtype=bool)
    for d in range(3):
        for s in (-2, -1, 0, 1, 2):
            offset = [0, 0, 0]
            offset[d] = s
            mask &= _window(fluid, offset)
    return mask


# ---------------------------------------------------------------------------
# 1 — bitwise the parent where nothing reaches SOLID
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_an_all_fluid_marker_is_bitwise_the_undegraded_kernel(blockamr_session, scheme):
    """The verify column in its literal form.

    An all-``FLUID`` marker *is* the ``noIbm`` marker (design §6), so the
    sibling must be its parent over the whole valid box, every component,
    bit for bit. It holds by construction — the ``else`` arm is a call to
    ``divVanLeerCell`` / ``divQuickCell`` inside the parent's own accumulate
    statement — and this row is what turns "by construction" into a fact about
    the shipped object.
    """
    parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm = _level()
    ct = blockamr.CellTypeFab(ba, dm, NGROW)
    ct.set_val(FLUID)
    phi = _field(ba, dm, _random_block(3, seed=3501), ncomp=3)
    faces = _flux(mesh)

    plain = _evaluate(parent, ba, dm, phi, faces, geom, ncomp=3)
    ibm = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=3)

    assert np.abs(plain).max() > 1e-6, "the fixture produced a near-zero source"
    np.testing.assert_array_equal(ibm, plain)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_an_all_solid_marker_is_bitwise_the_width_one_kernel(blockamr_session, scheme):
    """The fallback **is** ``div_upwind``, exactly — and the first falsifier.

    Every cell degrades, so the sibling must equal ``div_upwind_acc`` bit for
    bit. Three things die here at once: a sibling that silently never degrades
    (it would return its parent, which differs from upwind on random data); a
    fallback wired to ``div_linear`` (width 1 in reach but centred, and exact on
    a linear field, so the plane-wall probe below would not see it); and an
    early-out at ``SOLID`` centre cells, which would leave ``out`` at zero where
    this row demands the upwind value.
    """
    _parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm = _level()
    ct = blockamr.CellTypeFab(ba, dm, NGROW)
    ct.set_val(SOLID)
    phi = _field(ba, dm, _random_block(3, seed=3502), ncomp=3)
    faces = _flux(mesh)

    upwind = _evaluate("div_upwind_acc", ba, dm, phi, faces, geom, ncomp=3)
    ibm = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=3)

    assert np.abs(upwind).max() > 1e-6, "the fixture produced a near-zero source"
    np.testing.assert_array_equal(ibm, upwind)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_bulk_is_bitwise_the_undegraded_kernel_around_a_body(blockamr_session, scheme):
    """The verify column in its geometric form, on a real classified marker.

    A cell whose full width-2 cross stencil lies in the fluid never reads a
    ``SOLID`` marker, so W1 cannot fire there and the value must be the parent's
    — bitwise, not to a tolerance, because a tolerance would permit exactly the
    coupling this forbids.

    The mask is analytic. Asking the implementation which cells it thinks are
    near the wall would make the row a tautology.
    """
    parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm, ct = _classified(CYLINDER)
    phi = _field(ba, dm, _random_block(3, seed=3503), ncomp=3)
    faces = _flux(mesh)

    plain = _evaluate(parent, ba, dm, phi, faces, geom, ncomp=3)
    ibm = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=3)

    bulk = _bulk_mask(CYLINDER)
    assert bulk.sum() == 2880  # the fixture is what it is meant to be
    np.testing.assert_array_equal(ibm[bulk], plain[bulk])


# ---------------------------------------------------------------------------
# 2 — and it does fire, and only where it may
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_degrade_fires_and_never_outside_the_reach(blockamr_session, scheme):
    """Non-vacuity, in the form Q35 asks for.

    The row above is satisfied by a sibling that is a verbatim copy of its
    parent. This one is not: it asserts that the two kernels **do** differ
    somewhere, and that every cell where they differ is a cell whose width-2
    stencil analytically reaches the body. Together the two rows bracket the
    degrade from both sides.

    The count is reported (O2): it is the measurement of how much of the domain
    W1 actually touches.
    """
    parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm, ct = _classified(CYLINDER)
    phi = _field(ba, dm, _random_block(3, seed=3503), ncomp=3)
    faces = _flux(mesh)

    plain = _evaluate(parent, ba, dm, phi, faces, geom, ncomp=3)
    ibm = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=3)

    diff = np.any(ibm != plain, axis=-1)
    bulk = _bulk_mask(CYLINDER)
    print(f"\n[B35 T4 {scheme}] degraded-and-different cells: {int(diff.sum())} of {diff.size}")

    assert diff.any(), "the sibling never differs from its parent — the degrade never fired"
    assert not (diff & bulk).any(), (
        f"{int((diff & bulk).sum())} bulk cells changed: the degrade reached cells whose "
        "width-2 stencil is entirely in the fluid"
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_degraded_set_is_exactly_the_cells_whose_stencil_reaches_the_body(
    blockamr_session, scheme
):
    """The verify column at full strength — the degraded set *is* the reach set.

    The two rows above bracket the degrade from outside: it fires somewhere, and
    not in the bulk. Between them sits a sibling that degrades only *some* of the
    cells it should. This row closes that gap, and it can, because at a degraded
    cell the sibling evaluates ``coeff * divUpwindCell(...)`` into a zeroed
    ``out`` — which is exactly what ``div_upwind_acc`` does. Comparing against
    **both** references makes the branch observable cell by cell:

    * every cell whose analytic width-2 stencil touches the body is bitwise
      ``div_upwind_acc``, and
    * every cell whose stencil does not is bitwise its parent.

    That is the whole partition, per cell rather than in aggregate, and it is
    what tells a reach-1 or a reach-3 test apart from the ruled reach-2 one.

    Note the two counts differ by design: *degraded* is not *different*. A cell
    can degrade and still land on its parent's value, which happens wherever van
    Leer's limiter zeroes on every face — that is the one cell separating this
    row's 1216 from the previous row's reported 1215.
    """
    parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm, ct = _classified(CYLINDER)
    phi = _field(ba, dm, _random_block(3, seed=3503), ncomp=3)
    faces = _flux(mesh)

    plain = _evaluate(parent, ba, dm, phi, faces, geom, ncomp=3)
    upwind = _evaluate("div_upwind_acc", ba, dm, phi, faces, geom, ncomp=3)
    ibm = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=3)

    reaches = ~_bulk_mask(CYLINDER)
    took_upwind = np.all(ibm == upwind, axis=-1)
    took_parent = np.all(ibm == plain, axis=-1)
    print(
        f"\n[B35 T4b {scheme}] degraded cells: {int(took_upwind[reaches].sum())} of "
        f"{int(reaches.sum())} that analytically reach the body"
    )

    assert took_upwind[reaches].all(), (
        f"{int((~took_upwind & reaches).sum())} cells whose width-2 stencil reaches the body "
        "did not take the width-1 value"
    )
    assert took_parent[~reaches].all(), (
        f"{int((~took_parent & ~reaches).sum())} cells whose width-2 stencil is entirely in the "
        "fluid did not take the parent's value"
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_degrade_does_not_reach_the_bulk_on_an_axis_asymmetric_body(blockamr_session, scheme):
    """Containment on a body with no axis symmetry: the bulk keeps the parent value.

    What this row can and cannot catch, measured at B35-R (I-1): it asserts
    *containment* only, so an axis mix-up in the reach test — a shrinking
    mutant — passes it; and a *pure* axis swap is a no-op anyway, because the
    twelve-offset set is permutation-invariant. The axis-mutant class this
    fixture was built for is caught instead by
    ``test_the_degraded_set_is_exactly_the_cells_whose_stencil_reaches_the_body``,
    whose oracle enumerates the reach set offset by offset. This row's real
    claim is the bulk half of design §5's consequence 3, on a body whose
    asserted asymmetry rules out coincidental agreement.
    """
    parent, sibling = SIBLINGS[scheme]
    solid = _solid_grown(TILTED)
    for perm in itertools.permutations(range(3)):
        if perm == (0, 1, 2):
            continue
        assert not np.array_equal(solid, np.transpose(solid, perm)), (
            f"the fixture body is invariant under the axis permutation {perm}, so an axis "
            "transposition in the reach test would be invisible here"
        )

    mesh, geom, ba, dm, ct = _classified(TILTED)
    phi = _field(ba, dm, _random_block(3, seed=3507), ncomp=3)
    faces = _flux(mesh)

    plain = _evaluate(parent, ba, dm, phi, faces, geom, ncomp=3)
    ibm = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=3)

    bulk = _bulk_mask(TILTED)
    assert bulk.sum() == 1518
    diff = np.any(ibm != plain, axis=-1)
    assert diff.any(), "the degrade never fired on the tilted body"
    np.testing.assert_array_equal(ibm[bulk], plain[bulk])


# ---------------------------------------------------------------------------
# 3 — the plane-wall probe, and its control
# ---------------------------------------------------------------------------


def _plane_probe(ba, dm, ct):
    """The poisoned linear field of the plane-wall probe.

    ``T = A + B (x - X_WALL)`` over the whole grown box, then every ``SOLID``
    cell overwritten with :data:`POISON` — v2's pin (``pin_solid``), which is
    what makes a wide stencil that reaches into the body give a wrong answer
    instead of accidentally the right one.
    """
    x = _centres(NGROW)[0]
    phi = _field(ba, dm, (A_LIN + B_LIN * (x - X_WALL))[..., None], ncomp=1)
    blockamr.pin_solid(phi, ct, POISON, 1)
    return phi


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_linear_field_at_a_plane_wall_is_exact_two_cells_out(blockamr_session, scheme):
    """**D1's v2 form**, and the row that decides per-cell against per-face.

    ``u = (1, 1, 1)`` is divergence-free and ``T`` is linear, so
    ``div(u T) = u . grad T = B_LIN`` exactly — for a width-1 scheme and a
    width-2 scheme alike, with no tolerance to argue about. v1 passed this in
    the *band*, which covered ``depth <= 2``; v2 has no band, its wall sweep
    writes ``WALL`` cells only, and the cell at ``depth 2`` keeps whatever the
    interior kernel wrote. So this row is exactly W1's output at
    ``i = DEPTH_TWO`` and nothing else.

    A **per-face** degrade fails here — it would take the upwind (cell-centre)
    value on the left face and the van Leer (face-centre) value on the right,
    leaving ``1.5 * B_LIN``. So does a reach-1 test, which would not degrade
    ``i = DEPTH_TWO`` at all. The control row below measures both numbers.

    ``WALL`` cells are excluded: ``i = 4`` degrades to upwind and upwind's left
    neighbour *is* the poisoned solid. Closing that cell is the wall sweep's
    job, not the interior scheme's.
    """
    _parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm, ct = _classified(PLANE_WALL)
    phi = _plane_probe(ba, dm, ct)
    faces = _flux(mesh, values=(1.0, 1.0, 1.0))

    marker = _window(_marker(ct, phi), [0, 0, 0])
    fluid = marker == FLUID
    assert fluid[DEPTH_TWO].all(), "the depth-2 slab is not FLUID — the fixture moved"
    assert (marker[DEPTH_TWO - 1] == WALL).all(), "the wall slab moved"

    out = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=1)[..., 0]
    np.testing.assert_allclose(out[fluid], B_LIN, atol=1e-12)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_undegraded_kernel_fails_the_same_probe(blockamr_session, scheme):
    """The probe's control — the in-suite falsification of "it never degrades".

    The cheapest mutant of the sibling is the parent itself. Rather than predict
    what the suite *would* do under that mutant, this row runs it: the parent,
    on the same poisoned fixture, at the same cells. It must come out wrong, and
    it must stay wrong for as long as the probe exists.

    The two schemes fail differently, and the difference is worth pinning:

    * **QUICK** has no limiter, so ``-0.125 * POISON`` comes straight through.
    * **van Leer**'s limiter sees ``d_up = T(x_4) - POISON < 0`` against
      ``d_down = B dx > 0``, returns 0, and leaves ``pl = T(x_4)`` — the upwind
      cell-centre value against a van Leer face-centre value on the other side.
      That is ``1.5 * B_LIN`` exactly: the *same* number the per-face degrade
      would produce, measured here rather than derived (Q42(a)).
    """
    parent, _sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm, ct = _classified(PLANE_WALL)
    phi = _plane_probe(ba, dm, ct)
    faces = _flux(mesh, values=(1.0, 1.0, 1.0))

    out = _evaluate(parent, ba, dm, phi, faces, geom, ncomp=1)[..., 0]
    depth_two = out[DEPTH_TWO]

    assert not np.allclose(depth_two, B_LIN, atol=1e-9), (
        "the undegraded kernel passed the probe — the probe has no teeth"
    )
    if scheme == "vanleer":
        np.testing.assert_allclose(depth_two, 1.5 * B_LIN, rtol=1e-12)
    else:
        assert np.abs(depth_two).min() >= 1e20


# ---------------------------------------------------------------------------
# 4 — the error surface (api §9): a sentence, never an illegal address
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_marker_narrower_than_two_is_refused_naming_both_widths(blockamr_session, scheme):
    """The marker arm of the ghost guard.

    ``Array4``'s own index assert is compiled out of a release build, so a
    marker at ``MARKER_NGROW = 1`` under a reach-2 test is an illegal address
    that surfaces at some unrelated later sync. The message carries the width it
    needs and the width it has, in its own sentence — ``wall_apply.H``'s guard
    narrates a functor's ``stencil_reach``, and an interior kernel has none.
    """
    _parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm = _level()
    narrow = blockamr.CellTypeFab(ba, dm, 1)
    narrow.set_val(FLUID)
    phi = _field(ba, dm, _random_block(1, seed=3508))
    faces = _flux(mesh)

    with pytest.raises(
        RuntimeError,
        match=rf"{sibling}: .*reads 2 cells outside the valid box.*cell_type marker has ngrow = 1",
    ):
        _evaluate(sibling, ba, dm, phi, faces, geom, ct=narrow)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_field_narrower_than_two_is_refused_naming_both_widths(blockamr_session, scheme):
    """The field arm of the same guard, checked first so a narrow field cannot
    hide behind a correctly sized marker."""
    _parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm = _level()
    ct = blockamr.CellTypeFab(ba, dm, NGROW)
    ct.set_val(FLUID)
    narrow = blockamr.MultiFab(ba, dm, 1, 1)
    narrow.set_val(0.0)
    faces = _flux(mesh)

    with pytest.raises(
        RuntimeError,
        match=rf"{sibling}: .*reads 2 cells outside the valid box.*the field has ngrow = 1",
    ):
        _evaluate(sibling, ba, dm, narrow, faces, geom, ct=ct)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_marker_on_a_foreign_boxarray_is_refused(blockamr_session, scheme):
    """``const_array(mfi)`` on a foreign layout indexes another box's memory.

    Measured as a segfault at B30a-R (I-2) when the box counts differ, and a
    plausible wrong answer when the counts agree and the extents do not —
    neither is an exception, which is why the check is host-side and explicit.
    """
    _parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm = _level()
    _other_mesh, _other_geom, other_ba, other_dm = _level(max_size=8)
    foreign = blockamr.CellTypeFab(other_ba, other_dm, NGROW)
    foreign.set_val(FLUID)
    phi = _field(ba, dm, _random_block(1, seed=3509))
    faces = _flux(mesh)

    with pytest.raises(
        RuntimeError,
        match=rf"{sibling}: .*share the field's BoxArray and DistributionMapping.*has 1 boxes.*has 8",
    ):
        _evaluate(sibling, ba, dm, phi, faces, geom, ct=foreign)


# ---------------------------------------------------------------------------
# 5 — the binding's own semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_sibling_accumulates_and_scales_like_its_parent(blockamr_session, scheme):
    """``out +=``, linear in ``coeff``, and every component written.

    The same three claims ``test_stencil_kernels_cpp.py`` makes of the parents,
    made of the siblings on a marker where the degrade is *active* — so the
    branch is inside the accumulate, not around it. Also pins the keyword
    spelling of the binding (``cell_type`` between ``phi`` and ``fx``, with
    ``coeff`` and ``ncomp`` defaulted), which B36's dispatch will call by name.
    """
    _parent, sibling = SIBLINGS[scheme]
    mesh, geom, ba, dm, ct = _classified(CYLINDER)
    phi = _field(ba, dm, _random_block(3, seed=3511), ncomp=3)
    faces = _flux(mesh)

    once = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, ncomp=3)
    assert np.abs(once).max() > 1e-6
    for n in range(3):
        assert np.abs(once[..., n]).max() > 1e-6, f"component {n} not written"

    out = blockamr.MultiFab(ba, dm, 3, 0)
    out.set_val(0.0)
    fn = getattr(blockamr, sibling)
    fn(out, phi, ct, faces[0], faces[1], faces[2], geom, 1.0, 3)
    fn(out, phi, ct, faces[0], faces[1], faces[2], geom, 1.0, 3)
    np.testing.assert_array_equal(_valid(out), 2.0 * once)

    scaled = _evaluate(sibling, ba, dm, phi, faces, geom, ct=ct, coeff=2.5, ncomp=3)
    np.testing.assert_allclose(scaled, 2.5 * once, rtol=1e-14, atol=0.0)

    kw = blockamr.MultiFab(ba, dm, 3, 0)
    kw.set_val(0.0)
    fn(kw, phi, cell_type=ct, fx=faces[0], fy=faces[1], fz=faces[2], geom=geom, ncomp=3)
    np.testing.assert_array_equal(_valid(kw), once)
