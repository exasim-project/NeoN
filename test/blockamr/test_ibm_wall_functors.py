# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The compiled wall FRAME — ``schemes/boundary/wall_{value,apply}.H``,
``robin_data.H`` and ``wall_frame.cpp`` (B30a).

**Conformance, not acceptance**, exactly like ``test_ibm_cell_type.py`` and
``test_ibm_ghost_cell_cpp.py``: no row of the equation suite may read a sink or
a functor, now or later. What this file asserts is tasks.md §3's verify column
for B30a —

    a functor frame is callable host-side against a ``RecordSink`` on one
    cell; the probe harness green

— together with the frame conformance checks no equation can reach: **S1** (only
``WALL`` cells are written), **S2** (the affine split — the BC datum reaches the
row through ``constant`` and nothing else), **S4** (the frame owns the mode),
**S6** (a mode that is declared and not implemented names itself), **S8** (a
stencil must fit inside the ghost region) and **R2** (``constant_scale = 0``
drops exactly the datum).

**There is no wall formula here, and that is deliberate.** The functor under
test is ``WallFrameProbe``, a conformance harness: two linear entries on the
x-neighbours and the patch's ``gamma(t)`` through ``constant``. It calls no
closure, reads no ``alpha`` or ``beta``, computes no distance and makes no
accuracy claim. The formula is ``robin.H``'s ``closure(alpha, beta, gamma, d)``
and it is **B30b**, blocked on the G1 re-judgement (review.md §4 Q30/Q31/Q32).
B32's real per-cell rows (S2 and S3 on ``laplacian x ghostCell``) land in this
same file and reuse this same ``RecordSink`` readback shape.

**Why several assertions are bitwise.** The probe's arithmetic on this grid is
*exact*: ``dx = 1/16``, so ``1/dx^2 = 256`` exactly; the field is
``(16 i + 4 j + k)/16 + n``, so every product ``256 * phi`` is a small integer;
and every datum used below is dyadic. Nothing here is a tolerance in disguise —
where the expected value is exactly representable the row asserts exactly it,
and where a transcendental is involved (the harmonic datum) it says so and uses
``pytest.approx``. Q35's rule applies: a bitwise claim must be able to fail.
Here it can — a contracted multiply-add would still be exact, but a wrong donor
index, a wrong sign, a datum routed through ``linear`` or a missed component
all move the bits.
"""

import numpy as np
import pytest

import blockamr
from blockamr.ibm.body import Cylinder, Plane
from blockamr.mesh import Mesh

# Underscore-private test bindings (api §4). `from ._blockamr import *` skips
# underscore names, so they are reached on the extension module itself.
_wall_frame_record = blockamr._blockamr._wall_frame_record
_wall_frame_apply = blockamr._blockamr._wall_frame_apply
_cell_type_numpy = blockamr._blockamr._cell_type_numpy
#: B32's pair, whose per-cell rows this file's docstring already promised.
_wall_row_lgc = blockamr._blockamr._wall_row_laplacian_ghost_cell
_ghost_cell_numpy = blockamr._blockamr._ghost_cell_numpy

SOLID = int(blockamr.CellType.SOLID)
WALL = int(blockamr.CellType.WALL)
FLUID = int(blockamr.CellType.FLUID)

CONSTANT = blockamr.GAMMA_CONSTANT
HARMONIC = blockamr.GAMMA_HARMONIC

N = 16
DX = 1.0 / N
#: ``1 / dx^2`` — exact, because ``dx`` is a power of two.
INV_H2 = 256.0
#: The value ``out`` is filled with before every apply, so that "not written"
#: is an assertion about bits and not about a plausible zero.
SENTINEL = -7.0

ONE_BODY = {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)}
TWO_BODIES = {
    "left": Cylinder(centre=(0.28, 0.5), radius=0.12, axis=2),
    "right": Cylinder(centre=(0.72, 0.5), radius=0.12, axis=2),
}

#: The two faces of ``_slab``'s fluid gap, in world coordinates. Cell 7's centre
#: sits ``dx/2`` above ``SLAB_LO`` and cell 8's ``dx/2`` below ``SLAB_HI``, so
#: the two bodies' bisector is exactly the face between them.
SLAB_LO = 7.0 / N
SLAB_HI = 9.0 / N


def _slab(axis):
    """Two half-spaces facing each other across a **two-cell** fluid gap.

    ``a`` is solid below ``SLAB_LO`` and ``b`` solid above ``SLAB_HI``, both
    perpendicular to ``axis``. Only two cells along ``axis`` are fluid, and both
    are ``WALL`` — each has one ``SOLID`` face neighbour. A cell's patch is the
    *nearest surface* (``ibm/classify.py``), so the low cell is patch 0, the high
    cell is patch 1, and the patch boundary runs through the face between them:
    each of the two ``WALL`` cells has a face neighbour on the **other** patch.
    """
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    lo_point, hi_point = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    lo_point[axis], hi_point[axis] = SLAB_LO, SLAB_HI
    return {
        "a": Plane(point=tuple(lo_point), normal=tuple(normal)),
        "b": Plane(point=tuple(hi_point), normal=tuple(-v for v in normal)),
    }


def _shifted(cell, offset):
    return tuple(cell[d] + offset[d] for d in range(3))


# ---------------------------------------------------------------------------
# the level, the marker, the field
# ---------------------------------------------------------------------------


def _level(bodies, max_size=None):
    """``(mesh, geom, ba, dm)`` — one non-periodic unit cube at ``N^3``."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(N if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
    return mesh, geom, ba, dm


def _classified(bodies, max_size=None):
    """``(g, ct, geom, ba, dm)`` — the v2 geometry and the v2 marker of it."""
    mesh, geom, ba, dm = _level(bodies, max_size)
    g = mesh.ibm.geometry_fab(0, ngrow=1)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, g, geom)
    return g, ct, geom, ba, dm


def _phi_block(lo, hi, ncomp):
    """``(16 i + 4 j + k) / 16 + n`` on the inclusive index block, F-ordered.

    Every value is a multiple of ``1/16`` below ``2^11``, so ``256 * phi`` is a
    small integer and the probe's row is exact in binary64.
    """
    idx = np.meshgrid(
        np.arange(lo[0], hi[0] + 1, dtype=float),
        np.arange(lo[1], hi[1] + 1, dtype=float),
        np.arange(lo[2], hi[2] + 1, dtype=float),
        indexing="ij",
    )
    base = (16.0 * idx[0] + 4.0 * idx[1] + idx[2]) / 16.0
    return np.asfortranarray(np.stack([base + n for n in range(ncomp)], axis=-1))


def _phi_at(i, j, k, n):
    """The same field, evaluated at one global index — the numpy oracle."""
    return (16.0 * i + 4.0 * j + k) / 16.0 + n


def _field(ba, dm, ncomp=1, ngrow=1):
    """``phi``, filled over its **grown** box so ghosts are known, not zero."""
    mf = blockamr.MultiFab(ba, dm, ncomp, ngrow)
    for mfi in blockamr.MFIterator(mf):
        vb = mfi.valid_box()
        lo = tuple(v - ngrow for v in vb.small_end())
        hi = tuple(v + ngrow for v in vb.big_end())
        mf.copy_grown_from(mfi, _phi_block(lo, hi, ncomp))
    return mf


def _out(ba, dm, ncomp=1):
    mf = blockamr.MultiFab(ba, dm, ncomp, 0)
    mf.set_val(SENTINEL)
    return mf


def _boxes(mf):
    """Yield ``(mfi, lo)`` per local box; a generator because ``MFIterator``
    hands back itself and drops its ``MFIter`` when the loop ends."""
    for mfi in blockamr.MFIterator(mf):
        yield mfi, tuple(mfi.valid_box().small_end())


def _marker(ct, mf):
    """``{global (i, j, k): marker}`` over every valid cell of every box."""
    out = {}
    for mfi, lo in _boxes(mf):
        block = _cell_type_numpy(ct, mfi)
        for local in np.ndindex(block.shape):
            out[tuple(lo[d] + local[d] for d in range(3))] = int(block[local])
    return out


def _patch_of(g, mf):
    """``{global (i, j, k): patch}`` over every valid cell, from the geometry."""
    out = {}
    for mfi, lo in _boxes(mf):
        block = g.copy_to_host(mfi)[..., blockamr.GEOM_PATCH]
        for local in np.ndindex(block.shape):
            out[tuple(lo[d] + local[d] for d in range(3))] = int(block[local])
    return out


def _readback(mf):
    """``{global (i, j, k, n): value}`` over every valid cell of every box."""
    out = {}
    for mfi, lo in _boxes(mf):
        block = mf.copy_to_host(mfi)
        for local in np.ndindex(block.shape):
            key = tuple(lo[d] + local[d] for d in range(3)) + (local[3],)
            out[key] = block[local]
    return out


# ---------------------------------------------------------------------------
# the Robin tables
# ---------------------------------------------------------------------------


def _robin(gammas, alpha=1.0, beta=0.0):
    """``RobinData`` from ``gammas[patch][comp] = (form, a0, ac, as, omega)``.

    ``alpha`` and ``beta`` are carried and never read by B30a — the frame does
    not close a row, and the closure that would is B30b.
    """
    npatch = len(gammas)
    ncomp = len(gammas[0])
    form = np.zeros((npatch, ncomp), dtype=np.int32)
    param = np.zeros((npatch, ncomp, 4), dtype=np.float64)
    for p, row in enumerate(gammas):
        for n, (f, a0, ac, asin, omega) in enumerate(row):
            form[p, n] = f
            param[p, n] = (a0, ac, asin, omega)
    return blockamr.RobinData(
        np.full(npatch, alpha, dtype=np.float64),
        np.full(npatch, beta, dtype=np.float64),
        form,
        param,
    )


def _constant(value, npatch=1, ncomp=1):
    return _robin([[(CONSTANT, value, 0.0, 0.0, 0.0)] * ncomp] * npatch)


def _gamma_numpy(form, a0, ac, asin, omega, t):
    """The datum, in numpy — the oracle for the compiled ``GammaExpr``."""
    if form == CONSTANT:
        return a0
    return a0 + ac * np.cos(omega * t) + asin * np.sin(omega * t)


def _a_wall_cell(ct, mf, patch_of=None, patch=None):
    """One global ``WALL`` index, optionally on a given patch."""
    for cell, value in sorted(_marker(ct, mf).items()):
        if value != WALL:
            continue
        if patch is not None and patch_of[cell] != patch:
            continue
        return cell
    raise AssertionError("vacuous: this configuration has no WALL cell")


# ===========================================================================
# 1. The verify column — the functor, on the host, at one cell
# ===========================================================================


def test_the_probe_row_is_callable_host_side_on_one_cell(blockamr_session):
    """tasks.md §3's verify column for B30a, literally.

    The same ``AMREX_GPU_HOST_DEVICE`` member the kernel calls is called from
    the host at one cell, and a ``RecordSink`` gives back what v1's deleted row
    objects used to carry: the linear entries and the constant, separately.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    i, j, k = _a_wall_cell(ct, phi)

    entries, c = _wall_frame_record(g, _constant(0.5), geom, 0.0, i, j, k, 0)

    assert entries == [(i - 1, j, k, INV_H2), (i + 1, j, k, -INV_H2)]
    assert c == 0.5


def test_the_datum_reaches_the_row_through_constant_and_nothing_else(blockamr_session):
    """**S2**, and the whole enforcement of R1.

    ``linear`` and ``constant`` are two methods with two signatures, so a
    functor *cannot* route the BC datum through the linear part. The row proves
    the split is real rather than conventional: the datum is a value no
    stencil coefficient on this grid can take, and it appears only in ``c``.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    i, j, k = _a_wall_cell(ct, phi)
    datum = 0.078125  # 2^-7: dyadic, and not +-256

    entries, c = _wall_frame_record(g, _constant(datum), geom, 3.5, i, j, k, 0)

    assert c == datum
    assert [a for *_, a in entries] == [INV_H2, -INV_H2]
    assert datum not in [a for *_, a in entries]


def test_the_row_reads_the_geometry_at_its_own_cell_only(blockamr_session):
    """The recorded row is invariant under the geometry's ghost width — and
    that is **all** it pins.

    B30a-R's finding I-1 measured what this row can and cannot fail for: both
    fabs are built from the same analytic body by the same builder, so they
    agree at *every shared index*, halo included, and six neighbour-reading
    mutants pass it. Growing ``ngrow`` extends the box; it changes no value a
    functor could read, at its own cell or at any other. So this row is a
    regression bar on the **builder and the staging path** — a geometry whose
    interior depended on how far it was grown, or a ``stageGeometryBox`` that
    copied the wrong extent, moves it — and it is *not* evidence for Q34.

    Q34's discriminating row is the next one.
    """
    mesh, geom, ba, dm = _level(ONE_BODY)
    narrow = mesh.ibm.geometry_fab(0, ngrow=1)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, narrow, geom)
    phi = _field(ba, dm)
    i, j, k = _a_wall_cell(ct, phi)
    robin = _constant(0.5)

    first = _wall_frame_record(narrow, robin, geom, 0.0, i, j, k, 0)
    wide = mesh.ibm.geometry_fab(0, ngrow=3)
    second = _wall_frame_record(wide, robin, geom, 0.0, i, j, k, 0)

    assert first == second


def test_the_row_reads_the_patch_at_its_own_cell_and_not_at_a_face_neighbour(blockamr_session):
    """**Q34**, made falsifiable (B30a-R I-1).

    The probe's only geometry read is ``patch(i, j, k)`` at the cell it is
    called on — never a ``normal`` or a ``wall_point`` at a neighbour or a ghost
    index, which is the question B29's freeze deliberately left open. This row
    is what a functor reading one cell over would fail: it records at ``WALL``
    cells that sit **on** a patch boundary, against a table whose two patches
    carry different data, and asserts the constant is the *own* cell's datum.

    ``_slab(axis)`` leaves exactly two fluid cells, adjacent along ``axis``,
    both ``WALL``, one owned by each body. Recording at each of them in turn,
    over the three axes, covers all six face offsets::

        offset       recorded cell   its patch   neighbour       its patch
        (-1, 0, 0)   (8, j, k)       1           (7, j, k)       0
        (+1, 0, 0)   (7, j, k)       0           (8, j, k)       1
        (0, -1, 0)   (i, 8, k)       1           (i, 7, k)       0
        (0, +1, 0)   (i, 7, k)       0           (i, 8, k)       1
        (0, 0, -1)   (i, j, 8)       1           (i, j, 7)       0
        (0, 0, +1)   (i, j, 7)       0           (i, j, 8)       1

    The coverage is asserted rather than assumed: for every offset the row must
    *find* a ``WALL`` cell whose neighbour there is on the other patch, must see
    the own patch's datum, and must not see the neighbour's — and all six
    offsets must be reached. A geometry change that flattened the patch field
    would make the row red, not vacuously green.
    """
    data = [0.5, -0.25]
    assert data[0] != data[1], "vacuous: the two patches would carry the same datum"
    robin = _robin([[(CONSTANT, value, 0.0, 0.0, 0.0)] for value in data])

    covered = set()
    for axis in (0, 1, 2):
        g, ct, geom, ba, dm = _classified(_slab(axis))
        phi = _field(ba, dm)
        marker = _marker(ct, phi)
        patch_of = _patch_of(g, phi)
        for sign in (-1, 1):
            offset = tuple(sign if d == axis else 0 for d in range(3))
            straddling = [
                cell
                for cell, value in sorted(marker.items())
                if value == WALL
                and _shifted(cell, offset) in patch_of
                and patch_of[_shifted(cell, offset)] != patch_of[cell]
            ]
            assert straddling, f"vacuous: no WALL cell straddles a patch boundary at {offset}"

            cell = straddling[0]
            neighbour = _shifted(cell, offset)
            _entries, c = _wall_frame_record(g, robin, geom, 0.0, *cell, 0)

            assert c == data[patch_of[cell]], f"offset {offset}, cell {cell}"
            assert c != data[patch_of[neighbour]], f"offset {offset}, cell {cell}"
            covered.add(offset)

    assert len(covered) == 6, f"only {sorted(covered)} of the six face offsets were exercised"


# ===========================================================================
# 2. gamma(t) — a compiled per-patch expression, never a callback (Q4, Q29e)
# ===========================================================================


def test_a_constant_datum_is_bitwise_independent_of_time(blockamr_session):
    """Why ``Constant`` is an explicit tag and not ``ac = as = 0``.

    verification §2's probes assert *exactly* zero on constant data. The tag
    keeps a transcendental off that path entirely instead of relying on
    ``0.0 * cos(x)`` being exactly ``+-0.0`` in every build, so the datum is the
    same bits at every ``t`` — including ``t`` values where ``cos`` would not
    be.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    i, j, k = _a_wall_cell(ct, phi)
    robin = _constant(0.5)

    values = [
        _wall_frame_record(g, robin, geom, t, i, j, k, 0)[1]
        for t in (0.0, 1.0e-9, 1.0, 1.0e6, -3.75)
    ]
    assert values == [0.5] * 5


def test_a_harmonic_datum_is_evaluated_inside_the_frame(blockamr_session):
    """The datum is read where design §8 says it is: inside the kernel.

    Exactly at ``t = 0`` (``cos 0 = 1``, ``sin 0 = 0``, both exact) and against
    numpy's own ``cos``/``sin`` elsewhere. The comparison away from zero is
    ``approx`` on purpose: two correctly-rounded libms need not agree in the
    last bit, and B30a claims a *respelling* of A4/A6's expression, not
    equality with numpy's transcendental.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    i, j, k = _a_wall_cell(ct, phi)
    a0, ac, asin, omega = 0.25, 2.0, -0.5, 3.0
    robin = _robin([[(HARMONIC, a0, ac, asin, omega)]])

    assert _wall_frame_record(g, robin, geom, 0.0, i, j, k, 0)[1] == a0 + ac

    for t in (0.3, 1.7, -2.2):
        got = _wall_frame_record(g, robin, geom, t, i, j, k, 0)[1]
        assert got == pytest.approx(_gamma_numpy(HARMONIC, a0, ac, asin, omega, t), rel=1e-14)


@pytest.mark.parametrize(
    "case, params",
    [
        # A4, Stokes' second problem: U0 cos(omega t)
        ("a4-oscillating-wall", (0.0, 1.5, 0.0, 4.0)),
        # A6, Womersley: -(G/omega) sin(omega t)
        ("a6-womersley", (0.0, 0.0, -2.5 / 4.0, 4.0)),
    ],
)
def test_the_unsteady_validation_data_respell_as_harmonics(blockamr_session, case, params):
    """OP-6 (Q22/Q25), on the compiled side.

    A4 and A6 are spelled on v1 as Python callables, which Q4 rules out of
    scope for v2. Both are pure ``cos``/``sin`` of ``t``, so the cos/sin basis
    respells them without an amplitude/phase conversion — the reason for
    choosing that basis over amplitude and phase, which would spell A6 as
    ``cos(omega t - pi/2)`` and move A6's fitted numbers in the last bits.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    i, j, k = _a_wall_cell(ct, phi)
    a0, ac, asin, omega = params
    robin = _robin([[(HARMONIC, a0, ac, asin, omega)]])

    for t in (0.0, 0.13, 0.9, 2.4):
        got = _wall_frame_record(g, robin, geom, t, i, j, k, 0)[1]
        want = a0 + ac * np.cos(omega * t) + asin * np.sin(omega * t)
        assert got == pytest.approx(want, rel=1e-14, abs=1e-300)


def test_the_datum_is_read_per_patch_and_per_component(blockamr_session):
    """``gamma`` is indexed ``(patch, component)`` and the patch comes from the
    geometry the row sits on.

    Two bodies, two components, four distinct data: a layout that swapped the
    two indices, or a functor that read patch 0 for every body, produces one of
    the other three numbers. The patch id is the position of the body in
    ``sorted(mesh.bodies)`` — ``left`` is 0 and ``right`` is 1.
    """
    g, ct, geom, ba, dm = _classified(TWO_BODIES)
    phi = _field(ba, dm, ncomp=2)
    patch_of = _patch_of(g, phi)
    data = [[0.5, 0.25], [-1.5, -0.125]]
    robin = _robin([[(CONSTANT, v, 0.0, 0.0, 0.0) for v in row] for row in data])

    for patch in (0, 1):
        i, j, k = _a_wall_cell(ct, phi, patch_of=patch_of, patch=patch)
        for n in (0, 1):
            _entries, c = _wall_frame_record(g, robin, geom, 1.0, i, j, k, n)
            assert c == data[patch][n], f"patch {patch}, component {n}"


# ===========================================================================
# 3. The frame, over real fabs
# ===========================================================================


def _expected(cell, n, coeff, constant_scale, datum):
    i, j, k = cell
    linear = INV_H2 * _phi_at(i - 1, j, k, n) - INV_H2 * _phi_at(i + 1, j, k, n)
    return coeff * (linear + constant_scale * datum)


def test_the_frame_writes_only_wall_cells(blockamr_session):
    """**S1**, over eight boxes.

    Every ``SOLID`` and ``FLUID`` cell keeps the sentinel *bitwise* — the frame
    returns before the sink exists, so "not written" is not "written zero".
    That is also the one behavioural difference from v1 the port records rather
    than reproduces: v1 emitted a row at every band cell, so a first term in
    Overwrite mode wrote ``0.0`` at ``SOLID``; v2 leaves whatever the interior
    sweep put there (design §4.1, plan §8 / OPEN-C). Any v1-vs-v2 whole-array
    comparison is therefore fluid-masked.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY, max_size=8)
    phi = _field(ba, dm)
    out = _out(ba, dm)
    _wall_frame_apply(
        out, phi, ct, g, _constant(0.5), geom, 0.0, 1.0, 1, blockamr.WallMode.Overwrite, 1.0
    )

    marker = _marker(ct, phi)
    got = _readback(out)
    assert sorted(set(marker.values())) == [SOLID, WALL, FLUID]
    nwall = 0
    for (i, j, k, n), value in got.items():
        if marker[(i, j, k)] == WALL:
            nwall += 1
            assert value == _expected((i, j, k), n, 1.0, 1.0, 0.5), f"WALL {(i, j, k)}"
        else:
            assert value == SENTINEL, f"non-WALL {(i, j, k)} was written: {value}"
    assert nwall > 0, "vacuous: no WALL cell was visited"


def test_overwrite_assigns_and_add_accumulates(blockamr_session):
    """**S4** — the frame owns the mode, and it is a per-call argument.

    design §6's composition rule: the first term of an equation applies with
    ``Overwrite`` and every later one with ``Add``, each row carrying the
    term's full value. A second ``Add`` over the same cells therefore doubles
    the first result exactly, and the non-wall cells are still untouched.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)
    robin = _constant(0.5)

    _wall_frame_apply(out, phi, ct, g, robin, geom, 0.0, 1.0, 1, blockamr.WallMode.Overwrite, 1.0)
    first = _readback(out)
    _wall_frame_apply(out, phi, ct, g, robin, geom, 0.0, 1.0, 1, blockamr.WallMode.Add, 1.0)
    second = _readback(out)

    marker = _marker(ct, phi)
    for key, value in second.items():
        if marker[key[:3]] == WALL:
            assert value == 2.0 * first[key]
        else:
            assert value == SENTINEL


def test_constant_scale_zero_leaves_exactly_the_linear_part(blockamr_session):
    """**R2** — the implicit track's matvec is one field on ``ApplySink``.

    ``constant_scale = 0`` must drop the datum *exactly*, not approximately:
    the Krylov matvec of an affine operator is the linear part alone, and a
    residual datum there is a wrong operator, not a small error.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    robin = _constant(0.5)
    marker = _marker(ct, phi)

    affine = _out(ba, dm)
    _wall_frame_apply(
        affine, phi, ct, g, robin, geom, 0.0, 2.0, 1, blockamr.WallMode.Overwrite, 1.0
    )
    linear = _out(ba, dm)
    _wall_frame_apply(
        linear, phi, ct, g, robin, geom, 0.0, 2.0, 1, blockamr.WallMode.Overwrite, 0.0
    )

    got_affine, got_linear = _readback(affine), _readback(linear)
    for key, value in got_linear.items():
        if marker[key[:3]] != WALL:
            continue
        assert value == _expected(key[:3], key[3], 2.0, 0.0, 0.5)
        assert got_affine[key] - value == 2.0 * 0.5


def test_every_component_gets_its_own_row(blockamr_session):
    """``ncomp`` is the launch's second extent, and ``n`` reaches the functor.

    A frame that launched one component, or that handed every component the
    same ``n``, produces component 1's answer for component 0 — both fields
    here differ by exactly 1 per component, so the miss is visible.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm, ncomp=2)
    out = _out(ba, dm, ncomp=2)
    robin = _robin([[(CONSTANT, 0.5, 0.0, 0.0, 0.0), (CONSTANT, -0.25, 0.0, 0.0, 0.0)]])
    _wall_frame_apply(out, phi, ct, g, robin, geom, 0.0, 1.0, 2, blockamr.WallMode.Overwrite, 1.0)

    marker = _marker(ct, phi)
    got = _readback(out)
    for key, value in got.items():
        datum = (0.5, -0.25)[key[3]]
        if marker[key[:3]] == WALL:
            assert value == _expected(key[:3], key[3], 1.0, 1.0, datum)
        else:
            assert value == SENTINEL


def test_two_identical_sweeps_are_bitwise_equal(blockamr_session):
    """``out`` and ``phi`` are different MultiFabs, so there are no atomics and
    no ordering constraint between cells — determinism is structural, not a
    property of the launch configuration."""
    g, ct, geom, ba, dm = _classified(ONE_BODY, max_size=8)
    phi = _field(ba, dm)
    robin = _constant(0.5)

    runs = []
    for _ in range(2):
        out = _out(ba, dm)
        _wall_frame_apply(
            out, phi, ct, g, robin, geom, 0.25, 1.5, 1, blockamr.WallMode.Overwrite, 1.0
        )
        runs.append(np.array([v for _k, v in sorted(_readback(out).items())]))
    np.testing.assert_array_equal(runs[0].view(np.int64), runs[1].view(np.int64))


def test_a_harmonic_datum_is_evaluated_on_the_device_path(blockamr_session):
    """B30a-R S-2: the compiled ``Harmonic`` on the *kernel's* side of the line.

    Every other ``_wall_frame_apply`` row uses a ``Constant`` datum, so
    ``std::cos``/``std::sin`` were only ever evaluated by the host through the
    record hook — while Q36's decision to take AMReX's ``--use_fast_math`` on
    this TU is argued precisely on the harmonic being f64. This row measures
    that path.

    The field is **constant**, deliberately: the probe's two entries are then
    ``+256*phi`` and ``-256*phi`` on equal values, so the linear part is exactly
    ``0``, the row asserts that, and the affine result is ``coeff * gamma(t)``
    with no cancellation against a large linear part to hide a wrong datum in.
    ``coeff`` is a power of two, so the scaling is exact and what remains is the
    datum alone. ``t`` is chosen where ``cos`` and ``sin`` are both far from
    ``0`` and from ``+-1``, so a swapped pair, a wrong ``omega`` or a dropped
    ``a0`` all move the answer by order one.

    The oracle is the same ``GammaExpr`` evaluated on the **host** by the record
    hook at the same cell. ``approx(rel=1e-14)`` and not ``==``: the two sides
    run different libms (the device's ``cos``/``sin`` against glibc's), which
    need not agree in the last bit — the same cross-libm bar the harmonic record
    rows above use, for the same reason.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = blockamr.MultiFab(ba, dm, 1, 1)
    phi.set_val(1.0)
    a0, ac, asin, omega = 0.25, 2.0, -0.5, 3.0
    t = 0.3  # omega * t = 0.9 rad: cos = 0.62..., sin = 0.78...
    coeff = 2.0
    robin = _robin([[(HARMONIC, a0, ac, asin, omega)]])

    affine = _out(ba, dm)
    _wall_frame_apply(
        affine, phi, ct, g, robin, geom, t, coeff, 1, blockamr.WallMode.Overwrite, 1.0
    )
    linear = _out(ba, dm)
    _wall_frame_apply(
        linear, phi, ct, g, robin, geom, t, coeff, 1, blockamr.WallMode.Overwrite, 0.0
    )

    marker = _marker(ct, phi)
    got_affine, got_linear = _readback(affine), _readback(linear)
    nwall = 0
    for key, value in got_linear.items():
        if marker[key[:3]] != WALL:
            continue
        nwall += 1
        host = _wall_frame_record(g, robin, geom, t, *key[:3], key[3])[1]
        assert host != a0, "vacuous: the harmonic collapsed onto its own mean"
        assert value == 0.0, f"the constant field's linear part is not exactly 0 at {key}"
        assert got_affine[key] == pytest.approx(coeff * host, rel=1e-14)
    assert nwall > 0, "vacuous: no WALL cell was visited"


def test_the_sweep_does_not_write_the_field_it_reads(blockamr_session):
    """verification §2's purity probe, at the frame's own scale: ``evaluate``
    reads ``phi`` and writes ``out``, and the ``SOLID`` pin is classification's
    (Q3), so the field is bitwise unchanged by a sweep."""
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    before = np.array([v for _k, v in sorted(_readback(phi).items())])
    out = _out(ba, dm)
    _wall_frame_apply(
        out, phi, ct, g, _constant(0.5), geom, 0.0, 1.0, 1, blockamr.WallMode.Add, 1.0
    )
    after = np.array([v for _k, v in sorted(_readback(phi).items())])
    np.testing.assert_array_equal(before.view(np.int64), after.view(np.int64))


# ===========================================================================
# 4. The error surface — where a wrong number becomes an exception (api §9)
# ===========================================================================


def test_a_field_narrower_than_the_stencil_reach_is_refused_naming_both_widths(blockamr_session):
    """**S8**. ``Array4``'s own index assert is compiled out of a release build,
    so this guard is the only thing between a too-narrow ``ngrow`` and an
    illegal address that surfaces at an unrelated later sync."""
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm, ngrow=0)
    out = _out(ba, dm)
    with pytest.raises(
        RuntimeError, match=r"stencil_reach = 1.*the field has ngrow = 0.*ghost region"
    ):
        _wall_frame_apply(
            out, phi, ct, g, _constant(0.5), geom, 0.0, 1.0, 1, blockamr.WallMode.Overwrite, 1.0
        )


def test_a_marker_narrower_than_the_stencil_reach_is_refused_naming_the_marker(blockamr_session):
    """The same guard, second subject. The marker is read at the launch cell
    only *today*; W1's degrade (B35) reads ``m(i +- 2s)``, so the guard is on
    the marker from the start rather than added when it first matters."""
    g, _ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)
    narrow = blockamr.CellTypeFab(ba, dm, 0)
    narrow.set_val(FLUID)
    with pytest.raises(
        RuntimeError, match=r"stencil_reach = 1.*the cell_type marker has ngrow = 0"
    ):
        _wall_frame_apply(
            out, phi, narrow, g, _constant(0.5), geom, 0.0, 1.0, 1, blockamr.WallMode.Overwrite, 1.0
        )


def test_assemble_mode_names_itself(blockamr_session):
    """**S6** — a declared-but-unimplemented capability raises naming itself
    rather than behaving like the nearest mode it can. ``AssembleSink`` is the
    implicit track's and the implicit track is not built (api §5.2)."""
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)
    with pytest.raises(RuntimeError, match=r"WallMode.Assemble.*not implemented.*AssembleSink"):
        _wall_frame_apply(
            out, phi, ct, g, _constant(0.5), geom, 0.0, 1.0, 1, blockamr.WallMode.Assemble, 1.0
        )
    assert all(v == SENTINEL for v in _readback(out).values())


def test_out_and_phi_must_be_different_multifabs(blockamr_session):
    """design §4.4: in place, a wall row would read cells another row had
    already overwritten — a race whose result depends on the launch."""
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    with pytest.raises(RuntimeError, match=r"different MultiFabs"):
        _wall_frame_apply(
            phi, phi, ct, g, _constant(0.5), geom, 0.0, 1.0, 1, blockamr.WallMode.Overwrite, 1.0
        )


def test_fabs_on_different_grids_are_refused_naming_the_entry_point(blockamr_session):
    """B30a-R's **I-2**, closed here (carried by B30b's build).

    ``applyWall`` resolves ``out``, ``phi`` and the marker by ``MFIter`` **local
    index**, so three fabs on different ``BoxArray``\\ s are paired by position
    and not by box. In a ``-DNDEBUG`` build that is a segfault (measured, exit
    11) or, at equal box counts, a silently wrong answer — the two cases the
    next row covers. The guard runs before the loop, so nothing is launched.

    A ``DistributionMapping`` mismatch is **not constructible on one rank** (one
    rank owns every box either way), so it is checked in the same condition and
    recorded here rather than given a row that could never fail.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)

    # a different max_grid_size: same domain, more boxes
    ba8 = blockamr.BoxArray(blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1]))
    ba8.max_size(8)
    out = _out(ba8, blockamr.DistributionMapping(ba8))

    with pytest.raises(RuntimeError, match=r"_wall_frame_apply: out, phi and the cell_type"):
        _wall_frame_apply(
            out, phi, ct, g, _constant(0.5), geom, 0.0, 1.0, 1, blockamr.WallMode.Overwrite, 1.0
        )


def test_a_marker_of_the_same_box_count_but_a_different_extent_is_refused(blockamr_session):
    """The quiet half of I-2: equal box counts pass any count-based check and
    still pair box 0 with a fab of a different extent, which reads past the end
    of it. ``ct`` here is one box like ``phi``, on a quarter of the domain."""
    g, _ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)

    small = blockamr.BoxArray(blockamr.Box([0, 0, 0], [N // 2 - 1, N // 2 - 1, N // 2 - 1]))
    small.max_size(N)
    dm_small = blockamr.DistributionMapping(small)
    ct_small = blockamr.CellTypeFab(small, dm_small, 1)

    probe = blockamr.MultiFab(small, dm_small, 1, 0)
    assert sum(1 for _ in _boxes(probe)) == sum(1 for _ in _boxes(phi)), (
        "vacuous: the two BoxArrays must have the same box count for this to be the quiet case"
    )

    with pytest.raises(RuntimeError, match=r"must share one BoxArray and one DistributionMapping"):
        _wall_frame_apply(
            out,
            phi,
            ct_small,
            g,
            _constant(0.5),
            geom,
            0.0,
            1.0,
            1,
            blockamr.WallMode.Overwrite,
            1.0,
        )


def test_a_robin_table_with_the_wrong_component_count_is_refused(blockamr_session):
    """A ``robin`` narrower than the field is a silent out-of-bounds read of
    ``gamma``, not a wrong number — exactly the class api §9 exists to turn
    into a sentence."""
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm, ncomp=2)
    out = _out(ba, dm, ncomp=2)
    with pytest.raises(RuntimeError, match=r"the field has 2 but the table has 1"):
        _wall_frame_apply(
            out, phi, ct, g, _constant(0.5), geom, 0.0, 1.0, 2, blockamr.WallMode.Overwrite, 1.0
        )


def test_robin_data_refuses_tables_that_disagree_on_their_shape(blockamr_session):
    """The four arrays are one table in four pieces; a disagreement between
    them is not recoverable and is not guessed at."""
    with pytest.raises(RuntimeError, match=r"must agree on \(npatch, ncomp\)"):
        blockamr.RobinData(
            np.zeros(2, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros((2, 1), dtype=np.int32),
            np.zeros((2, 1, 4), dtype=np.float64),
        )


def test_robin_data_refuses_an_unknown_gamma_form(blockamr_session):
    """Q4: a datum that is neither Constant nor Harmonic is not expressible
    device-side, and the refusal names the patch, the component and the form."""
    with pytest.raises(RuntimeError, match=r"patch 0 component 0 asks for form 7"):
        blockamr.RobinData(
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.full((1, 1), 7, dtype=np.int32),
            np.zeros((1, 1, 4), dtype=np.float64),
        )


def test_robin_data_refuses_a_table_with_no_patch_and_no_component(blockamr_session):
    """``robin_data.H``'s third named refusal (B30a-R S-4).

    Four empty arrays *agree* on their shape, so they pass the binding's ndim
    and shape checks and reach the constructor. What they agree on is nothing: a
    ``RobinView`` built from them has ``npatch = ncomp = 0`` and every
    ``gammaAt`` is an out-of-bounds read of a zero-length managed vector — the
    same silent class as a table one component too narrow, and refused by name
    for the same reason.
    """
    with pytest.raises(RuntimeError, match=r"at least one patch and one component"):
        blockamr.RobinData(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 0, 4), dtype=np.float64),
        )


def test_the_record_hook_refuses_a_cell_no_box_owns(blockamr_session):
    """The test hook is a binding like any other: it names the offending
    object rather than reading past the end of a fab."""
    g, _ct, geom, _ba, _dm = _classified(ONE_BODY)
    with pytest.raises(RuntimeError, match=r"lies in no local box"):
        _wall_frame_record(g, _constant(0.5), geom, 0.0, 999, 0, 0, 0)


def test_the_record_hook_refuses_a_component_the_table_lacks(blockamr_session):
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    i, j, k = _a_wall_cell(ct, phi)
    with pytest.raises(RuntimeError, match=r"component 3 is outside the Robin table's 1"):
        _wall_frame_record(g, _constant(0.5), geom, 0.0, i, j, k, 3)


# ===========================================================================
# 5. What the frame exports
# ===========================================================================


def test_the_wall_mode_enum_is_the_three_modes(blockamr_session):
    """``Assemble`` is *declared* though it is not implemented: S6 requires a
    missing capability to be able to name itself, and an enum that omitted it
    would fail with ``AttributeError`` instead of a sentence."""
    assert int(blockamr.WallMode.Overwrite) == 0
    assert int(blockamr.WallMode.Add) == 1
    assert int(blockamr.WallMode.Assemble) == 2


def test_the_record_sink_capacity_covers_the_widest_ported_row(blockamr_session):
    """``laplacian x ghostCell`` walks six arms and a wall arm emits one entry
    per trilinear donor, so a cell whose six face neighbours are all SOLID
    emits ``6 * 8 = 48`` linear entries. The sink is sized for that, and B32
    inherits the number rather than discovering it."""
    assert blockamr.WALL_RECORD_CAPACITY >= 6 * blockamr.GHOST_CELL_K


def test_the_gamma_form_tags_are_exported(blockamr_session):
    assert (blockamr.GAMMA_CONSTANT, blockamr.GAMMA_HARMONIC) == (0, 1)


# ===========================================================================
# 6. `laplacian x ghostCell` — the first real pair, per cell (B32)
#
# The docstring above promised this section: "B32's real per-cell rows (S2 and
# S3 on `laplacian x ghostCell`) land in this same file and reuse this same
# `RecordSink` readback shape."
#
# **Conformance, not acceptance.** v1<->v2 bitwise row parity, the falsification
# matrix and the sweep live in `test_ibm_laplacian_ghost_cell.py`, which has the
# heavy fixtures and a different vocabulary. What is here is what the shipped
# frame file is the natural home for: which cells a row may name (S3), how the
# BC datum reaches it (S2), where the geometry is read (Q34), what the row's
# shape is, and the error surface.
# ===========================================================================

#: A wall normal to x: its image point lands exactly on a cell face, so four of
#: the eight trilinear weights are exactly ``0.0`` and the dead-slot rule (a
#: dead donor points at the row's own cell) is exercised rather than assumed.
PLANE_X = {"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))}

#: The six face offsets, in the pair's own loop order: d ascending, +1 then -1.
ARMS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _ghost_data(ct, g, geom, bodies):
    """The method's own preprocessed rows, as the opaque ``GhostCellData``."""
    return blockamr.ghost_cell_preprocess(ct, g, geom, sorted(bodies))


def _row(ct, g, data, robin, geom, cell, n=0, t=0.0):
    """The pair's row at one cell as ``([(index, a)], c)``."""
    entries, c = _wall_row_lgc(ct, g, data, robin, geom, t, *cell, n)
    return [((i, j, k), a) for i, j, k, a in entries], c


def _wall_cells(ct, mf):
    """Every global ``WALL`` index, in sorted order."""
    return [cell for cell, value in sorted(_marker(ct, mf).items()) if value == WALL]


def _marker_grown(ct, mf, ngrow=1):
    """``{global (i, j, k): marker}`` over each box's **fab** box, ghosts included.

    This is what the functor reads. A ``WALL`` cell on the edge of the domain
    has a face neighbour in the ghost region, and outside a non-periodic face
    that ghost is classified from the *analytic* body — ``FLUID`` where the body
    is not — so its arm is a live one. v1 does the same thing by evaluating
    ``_fluid_at_index`` at the same index, which is why the parity rows agree
    there; a map over valid cells only would report a hole that neither side
    has.
    """
    out = {}
    for mfi, lo in _boxes(mf):
        block = _cell_type_numpy(ct, mfi, True)
        base = tuple(v - ngrow for v in lo)
        for local in np.ndindex(block.shape):
            out[tuple(base[d] + local[d] for d in range(3))] = int(block[local])
    return out


def _fluid_arm_count(marker, cell):
    """How many of the six face neighbours are **not** ``SOLID``."""
    return sum(1 for off in ARMS if marker[_shifted(cell, off)] != SOLID)


def _perturbed_geometry(mesh, ba, dm, cell, delta, ngrow=1):
    """The packed geometry rebuilt with ``normal_x`` moved at ONE index.

    Built through v1's own ``packed_geometry_on_grids`` — the grown blocks the
    uploader expects — because the perturbation has to survive into the ghost
    region as well: the point of the row that uses this is to move a value the
    functor *could* read and show that it does not.
    """
    from blockamr.ibm.classify import box_grids
    from blockamr.ibm.geometry import packed_geometry_on_grids

    grids = box_grids(mesh, 0)
    blocks = packed_geometry_on_grids(grids, mesh.bodies, ngrow)
    out = blockamr.MultiFab(ba, dm, blockamr.IBM_GEOM_NCOMP, ngrow)
    moved = 0
    for mfi, block, grid in zip(blockamr.MFIterator(out), blocks, grids):
        block = np.array(block, copy=True)
        local = tuple(cell[d] - (grid.lo[d] - ngrow) for d in range(3))
        if all(0 <= local[d] < block.shape[d] for d in range(3)):
            block[local + (blockamr.GEOM_NORMAL,)] += delta
            moved += 1
        out.copy_grown_from(mfi, np.asfortranarray(block))
    assert moved > 0, f"vacuous: {cell} is in no block, so nothing was perturbed"
    return out


def test_the_pair_row_is_callable_host_side_on_one_wall_cell(blockamr_session):
    """**F-1** — tasks.md §3's verify column, for a pair that computes a wall.

    The same ``AMREX_GPU_HOST_DEVICE`` functor the kernel launches, called from
    the host at one cell against a ``RecordSink``: ``([(i, j, k, a), ...], c)``
    is exactly the shape v1's deleted row objects carried, recovered from the
    shipped device code rather than from a numpy builder written beside it.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    cell = _a_wall_cell(ct, phi)

    entries, c = _row(ct, g, data, _constant(0.5), geom, cell)

    assert data.nrows == len(_wall_cells(ct, phi)) > 0
    assert isinstance(c, float)
    assert entries and all(len(index) == 3 for index, _a in entries)
    assert entries[0][0] == cell, "the first linear entry is the diagonal, at the row's own cell"


def test_the_bc_datum_reaches_the_pair_row_through_constant_and_nothing_else(blockamr_session):
    """**F-2 / S2**, and stronger than "the number is not in the list".

    Two Robin tables differing in ``gamma`` and in nothing else: the linear
    entries must be **bitwise identical** and only ``c`` may move. That is the
    affine split (R1) as a measurement — a functor that leaked the datum into
    any coefficient, however small, moves a bit here.

    ``Mixed``-shaped ``(alpha, beta)`` on purpose, so ``grad_constant`` is
    non-zero and the datum genuinely reaches the row; with ``gamma = 0`` on both
    tables the row would be vacuous.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    cell = _a_wall_cell(ct, phi)

    def at(datum):
        robin = _robin([[(CONSTANT, datum, 0.0, 0.0, 0.0)]], alpha=0.6, beta=0.4)
        return _row(ct, g, data, robin, geom, cell)

    first_entries, first_c = at(0.3)
    second_entries, second_c = at(-1.25)

    assert first_c != second_c, "vacuous: the datum does not reach this row at all"
    assert [i for i, _a in first_entries] == [i for i, _a in second_entries]
    lhs = np.array([a for _i, a in first_entries])
    rhs = np.array([a for _i, a in second_entries])
    np.testing.assert_array_equal(lhs.view(np.int64), rhs.view(np.int64))


def test_no_entry_of_a_pair_row_ever_names_a_solid_cell(blockamr_session):
    """**F-3 / S3 / Invariant F**, over *every* ``WALL`` cell of the level.

    A ``SOLID`` cell holds the pin and not data, so a row that named one would
    read a pinned value and return a plausible wrong number. Two things make
    that impossible here and both are exercised: each arm is gated on
    ``m(ii, jj, kk) != SOLID``, and every live trilinear donor was validated
    fluid by ``preprocess``'s Invariant-F pass.

    Both body sets, in one row: one cylinder, and two cylinders with a patch
    boundary running between them.

    The marker is read over each box's **fab** box. A ``WALL`` cell on the edge
    of the domain has a face neighbour in the ghost region, and outside a
    non-periodic face that ghost is classified from the analytic body — so its
    arm is a live one, and v1 emits the same entry there (its ``_neighbour``
    evaluates the same analytic test at the same index). A map over valid cells
    only would report a hole that neither side has.
    """
    for bodies in (ONE_BODY, TWO_BODIES):
        g, ct, geom, ba, dm = _classified(bodies)
        phi = _field(ba, dm)
        data = _ghost_data(ct, g, geom, bodies)
        marker = _marker_grown(ct, phi)
        robin = _robin([[(CONSTANT, 0.3, 0.0, 0.0, 0.0)]] * len(bodies), alpha=0.6, beta=0.4)

        cells = _wall_cells(ct, phi)
        assert cells, "vacuous: no WALL cell"
        named = solid_seen = 0
        for cell in cells:
            entries, _c = _row(ct, g, data, robin, geom, cell)
            for index, _a in entries:
                assert index in marker, f"row at {cell} names {index}, outside the fab box"
                assert marker[index] != SOLID, f"row at {cell} names the SOLID cell {index}"
                named += 1
            solid_seen += 6 - _fluid_arm_count(marker, cell)
        assert named > 0
        assert solid_seen > 0, "vacuous: no WALL cell here has a SOLID face neighbour"


def test_a_solid_face_neighbour_is_named_by_the_probe_and_not_by_the_pair(blockamr_session):
    """**F-4** — the pair is not the probe, measured at the same cell.

    ``WallFrameProbe`` emits its ``i +- 1`` donors unconditionally and says so
    in its own docstring; a real pair gates each arm. Asserting that difference
    where it actually bites — a ``WALL`` cell with a ``SOLID`` face neighbour on
    the x axis — is what keeps "B32 must not copy that aspect of it" from being
    a comment nobody can fail.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    straddling = [
        (cell, off)
        for cell in _wall_cells(ct, phi)
        for off in ((-1, 0, 0), (1, 0, 0))
        if marker[_shifted(cell, off)] == SOLID
    ]
    assert straddling, "vacuous: no WALL cell here has a SOLID neighbour on the x axis"

    cell, off = straddling[0]
    neighbour = _shifted(cell, off)
    pair, _c = _row(ct, g, data, robin, geom, cell)
    probe, _pc = _wall_frame_record(g, robin, geom, 0.0, *cell, 0)

    assert neighbour not in [index for index, _a in pair]
    assert neighbour in [(i, j, k) for i, j, k, _a in probe], (
        "vacuous: the probe no longer emits its unconditional arms"
    )


def test_the_pair_row_is_one_diagonal_plus_its_fluid_arms_plus_eight_donors(blockamr_session):
    """**F-5** — the accumulate-then-emit shape, stated as a count.

    ``1 + (6 - #solid arms) + 8`` entries, at most 15, which is exactly v1's
    ``STRIDE``. The rejected alternative — emitting ``+1/h^2`` and ``-1/h^2``
    per arm and eight donors per wall arm — reaches 54 entries *and* forfeits
    bitwise parity with v1 on every wall row, because it accumulates the
    diagonal in a different order (review.md §4 Q49(e)).
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    widths = set()
    for cell in _wall_cells(ct, phi):
        entries, _c = _row(ct, g, data, robin, geom, cell)
        fluid_arms = _fluid_arm_count(marker, cell)
        assert len(entries) == 1 + fluid_arms + blockamr.GHOST_CELL_K, cell
        assert len(entries) <= 15
        widths.add(fluid_arms)
    assert len(widths) > 1, "vacuous: every WALL cell here has the same number of solid arms"


def test_the_diagonal_is_the_accumulated_sum_over_the_fluid_arms(blockamr_session):
    """**F-6** — H-2 and H-3 together.

    The diagonal is ``-sum_{fluid arms} 1/dx_d^2``, accumulated in one register
    over ``d`` ascending and emitted **once**. On this grid ``1/dx^2`` is
    exactly ``256``, so the expected value is exactly representable and the
    assertion is on the bits: a saved divide (``(1/dx)*(1/dx)``), a wrong power,
    or a per-arm emission each shows up.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    for cell in _wall_cells(ct, phi):
        entries, _c = _row(ct, g, data, robin, geom, cell)
        want = 0.0
        for _ in range(_fluid_arm_count(marker, cell)):
            want -= INV_H2
        index, a = entries[0]
        assert index == cell
        assert a == want, f"{cell}: diagonal {a} != {want}"
        arms = entries[1 : 1 + _fluid_arm_count(marker, cell)]
        assert all(value == INV_H2 for _i, value in arms), cell


def test_the_eight_donor_entries_are_the_methods_own_stencil(blockamr_session):
    """**F-7** — the §4 row map lands on the right row.

    The last eight entries must be ``GhostCellData.donor[r]`` for that cell's
    rank ``r``, in slot order, with a dead slot (weight exactly ``0.0``) at the
    row's own cell. ``PLANE_X`` is used because its image point lands on a cell
    face, so half the weights are exactly zero and the dead-slot rule is
    exercised instead of assumed — the wrong rank would name another cell's
    donors, which is the failure mode the map introduces.
    """
    g, ct, geom, ba, dm = _classified(PLANE_X)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, PLANE_X)
    _ip, donor, weight, _distance = _ghost_cell_numpy(ct, g, geom, ["wall"])
    robin = _constant(0.5)

    assert (weight == 0.0).any(), "vacuous: this geometry has no dead donor slot"
    for cell in _wall_cells(ct, phi):
        rank = data.row_at(*cell)
        assert rank >= 0
        entries, _c = _row(ct, g, data, robin, geom, cell)
        got = [index for index, _a in entries[-blockamr.GHOST_CELL_K :]]
        want = [
            cell if weight[rank, q] == 0.0 else tuple(int(v) for v in donor[rank, q])
            for q in range(blockamr.GHOST_CELL_K)
        ]
        assert got == want, f"{cell} (rank {rank})"


def test_a_field_narrower_than_the_pairs_reach_is_refused_naming_the_pair(blockamr_session):
    """**F-8 / S8**, and api §9: the sentence names ``wall_laplacian_ghost_cell``
    and not ``applyWall``, because "applyWall" names nothing a caller can see."""
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    phi = _field(ba, dm, ngrow=0)
    out = _out(ba, dm)
    with pytest.raises(
        RuntimeError, match=r"wall_laplacian_ghost_cell: the functor declares stencil_reach = 1"
    ):
        blockamr.wall_laplacian_ghost_cell(
            out,
            phi,
            ct,
            g,
            data,
            _constant(0.5),
            geom,
            0.0,
            1.0,
            1,
            blockamr.WallMode.Overwrite,
            1.0,
        )


def test_the_pair_reads_the_geometry_at_its_own_cell_and_not_at_a_neighbour(blockamr_session):
    """**F-9 / Q34**, made falsifiable (B30a-R's I-1).

    The functor's only geometry reads are ``patch(i, j, k)`` and
    ``normal(i, j, k, d)``. B30a-R measured that comparing two *builders* cannot
    catch a neighbour read — both fabs agree at every shared index — so this row
    perturbs one fab at one index instead: moving the normal at a **face
    neighbour** must leave the row bitwise identical, and moving it at the
    row's **own cell** must change it. The second half is what stops the first
    from being vacuous.

    ``_slab`` is used so the perturbed neighbour is on the *other* patch, which
    is where a neighbour read would also pick up the wrong ``alpha``/``beta``.
    """
    bodies = _slab(0)
    mesh, geom, ba, dm = _level(bodies)
    g = mesh.ibm.geometry_fab(0, ngrow=1)
    ct = blockamr.CellTypeFab(ba, dm, 1)
    blockamr.classify_default(ct, g, geom)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, bodies)
    patch_of = _patch_of(g, phi)
    robin = _robin([[(CONSTANT, 0.5, 0.0, 0.0, 0.0)], [(CONSTANT, -0.25, 0.0, 0.0, 0.0)]])

    straddling = [
        (cell, _shifted(cell, off))
        for cell in _wall_cells(ct, phi)
        for off in ((1, 0, 0), (-1, 0, 0))
        if patch_of.get(_shifted(cell, off), patch_of[cell]) != patch_of[cell]
    ]
    assert straddling, "vacuous: no WALL cell straddles a patch boundary"
    cell, neighbour = straddling[0]

    base = _row(ct, g, data, robin, geom, cell)
    at_neighbour = _perturbed_geometry(mesh, ba, dm, neighbour, 0.125)
    at_self = _perturbed_geometry(mesh, ba, dm, cell, 0.125)

    assert _row(ct, at_neighbour, data, robin, geom, cell) == base, (
        f"the row at {cell} moved when the geometry at {neighbour} did — Q34 is tripped"
    )
    assert _row(ct, at_self, data, robin, geom, cell) != base, (
        "vacuous: perturbing the geometry at the row's own cell changed nothing"
    )


def test_the_closures_pole_reaches_the_row_as_infinity_and_raises_nothing(blockamr_session):
    """**F-10 / Q46**, ruled at B32: the guard is DEFERRED and the behaviour is
    PINNED.

    ``robin.H``'s ``den = beta - alpha*d`` is exactly zero for the reachable
    ``Mixed(f)`` with ``d = (1 - f)/f``, and v1 divides anyway and returns
    ``+-inf``; nothing in v1 warns, checks or documents it. A raise here would
    be a behaviour change against v1 in the one session whose whole claim is
    that nothing changed, and it would fail the parity bar by design.

    So the configuration is driven through the pair and the ``+-inf`` is
    asserted — in an entry **and** in the constant — together with the fact that
    nothing was raised. A later well-meaning guard turns this green row red and
    is read as the behaviour change it is.

    The consequence is recorded rather than fixed: such a row reaching a real
    sweep makes ``ApplySink::acc`` non-finite and the frame writes it into
    ``out`` for the whole cell, silently. That is the post-G2 note, beside the
    fallback.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    phi = _field(ba, dm)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    _ip, _donor, weight, distance = _ghost_cell_numpy(ct, g, geom, ["cyl"])
    marker = _marker_grown(ct, phi)

    # Exactly ONE solid arm, so there is one wall arm and no `inf - inf`: every
    # value below is then a single signed infinity or a single `inf * 0`, and
    # the expected bits are decided rather than incidental.
    chosen = None
    for cell in _wall_cells(ct, phi):
        if _fluid_arm_count(marker, cell) == 5:
            chosen = (cell, data.row_at(*cell))
            break
    assert chosen is not None, "vacuous: no WALL cell here has exactly one SOLID arm"
    cell, rank = chosen

    # alpha = 1, beta = d  =>  den = beta - alpha*d = 0 exactly, on this row.
    d = float(distance[rank])
    robin = _robin([[(CONSTANT, 0.5, 0.0, 0.0, 0.0)]], alpha=1.0, beta=d)

    entries, c = _row(ct, g, data, robin, geom, cell)  # must not raise

    donors = [a for _i, a in entries[-blockamr.GHOST_CELL_K :]]
    live = [a for a, w in zip(donors, weight[rank]) if w != 0.0]
    dead = [a for a, w in zip(donors, weight[rank]) if w == 0.0]
    assert live and dead, f"vacuous: {cell} has no live or no dead donor slot"
    # a live donor carries `(scale * grad_linear) * w` = a signed infinity, and
    # the SIGN is determined, not incidental (B32-R S-2): den = d - 1.0*d is
    # +0.0 exactly, so grad_linear = -1/den is -inf and the arm's negative scale
    # lands every live donor at +inf, the constant at -inf. Pinned as measured.
    assert all(np.isinf(a) and a > 0.0 for a in live), live
    # ...and a dead one carries `inf * 0.0`, which is a NaN, exactly as v1's
    # numpy `(scale * grad_linear)[:, None] * weight` produces there. NaN stays
    # classification-only: payloads are not contractual.
    assert all(np.isnan(a) for a in dead), dead
    assert np.isinf(c) and c < 0.0, c
    # the finite half of the row is untouched: the closure never reaches it.
    assert all(np.isfinite(a) for _i, a in entries[: -blockamr.GHOST_CELL_K]), entries


def test_assemble_mode_names_the_pair(blockamr_session):
    """**F-11 / S6** — a declared-but-unimplemented capability raises naming
    itself, and the sentence carries the pair's own entry-point name."""
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)
    with pytest.raises(RuntimeError, match=r"wall_laplacian_ghost_cell: WallMode.Assemble"):
        blockamr.wall_laplacian_ghost_cell(
            out,
            phi,
            ct,
            g,
            data,
            _constant(0.5),
            geom,
            0.0,
            1.0,
            1,
            blockamr.WallMode.Assemble,
            1.0,
        )
    assert all(v == SENTINEL for v in _readback(out).values())


def test_each_disagreement_between_the_pairs_arguments_is_refused_by_name(blockamr_session):
    """**F-12** — guard 0 and ``Maker::validate`` (B30a-R's S-5), together.

    Four disagreements, each of which is a silently wrong answer rather than a
    crash in a release build, and each named by the entry point:

    * ``out is phi`` — a row would read cells another row had already written;
    * a mismatched ``BoxArray`` — the sweep pairs fabs by ``MFIter`` local
      index, so a mismatch reads another box's cells;
    * a Robin table narrower than the field — an out-of-bounds ``gammaAt``;
    * **method data preprocessed on other grids** — the one the frame cannot
      see at all, because ``GhostCellData`` lives inside the Maker. That is
      exactly why ``validate`` is a hook on the concept and not four calls
      copied into every pair's binding.
    """
    g, ct, geom, ba, dm = _classified(ONE_BODY)
    data = _ghost_data(ct, g, geom, ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)

    def call(out_mf, phi_mf, ct_fab, data_obj, robin, ncomp=1):
        blockamr.wall_laplacian_ghost_cell(
            out_mf,
            phi_mf,
            ct_fab,
            g,
            data_obj,
            robin,
            geom,
            0.0,
            1.0,
            ncomp,
            blockamr.WallMode.Overwrite,
            1.0,
        )

    with pytest.raises(RuntimeError, match=r"wall_laplacian_ghost_cell: .*different MultiFabs"):
        call(phi, phi, ct, data, _constant(0.5))

    ba8 = blockamr.BoxArray(blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1]))
    ba8.max_size(8)
    dm8 = blockamr.DistributionMapping(ba8)
    with pytest.raises(RuntimeError, match=r"wall_laplacian_ghost_cell: out, phi and the"):
        call(_out(ba8, dm8), phi, ct, data, _constant(0.5))

    phi2 = _field(ba, dm, ncomp=2)
    with pytest.raises(RuntimeError, match=r"the field has 2 but the table has 1"):
        call(_out(ba, dm, ncomp=2), phi2, ct, data, _constant(0.5), ncomp=2)

    g8 = _level(ONE_BODY, max_size=8)[0].ibm.geometry_fab(0, ngrow=1)
    ct8 = blockamr.CellTypeFab(ba8, dm8, 1)
    blockamr.classify_default(ct8, g8, geom)
    other = _ghost_data(ct8, g8, geom, ONE_BODY)
    with pytest.raises(RuntimeError, match=r"wall_laplacian_ghost_cell: the ghostCell data"):
        call(out, phi, ct, other, _constant(0.5))


# ===========================================================================
# 7. `div x ghostCell` — the second real pair, per cell (B33)
#
# **Conformance, not acceptance**, exactly as section 6 is. v1<->v2 bitwise row
# parity over ten configurations, the falsification matrix, the sweep and the
# argument contract live in `test_ibm_div_ghost_cell.py`, which has the heavy
# fixtures and a different vocabulary. What is here is what the shipped frame
# file is the natural home for: which cells a row may name (S3), how the BC datum
# reaches it (S2), where the geometry is read (Q34), what the row's shape is, the
# error surface — and the two things `div` adds to that list, H-6's signed zero
# and the `DivFaceValue` mapping.
# ===========================================================================

_wall_row_dgc = blockamr._blockamr._wall_row_div_ghost_cell

#: `u = omega x r` about the cylinder axis. Rung 8's own velocity shape, and the
#: only thing in the repertoire that produces **exactly zero face fluxes**: `u_z`
#: is identically zero and `u_x`/`u_y` vanish on the two centre lines. Those
#: faces are what H-6 lives on.
DIV_OMEGA = 5.0


def _rotation_velocity(x, y, z, t):
    return -DIV_OMEGA * (y - 0.5), DIV_OMEGA * (x - 0.5), np.zeros_like(z)


def _uniform_velocity(x, y, z, t):
    """Exactly `1.0` on every face, so `f / dx` is exactly `16` on this grid and
    `DivFaceValue`'s two branches have exactly representable answers."""
    return np.ones_like(x), np.ones_like(y), np.ones_like(z)


def _div_case(bodies, velocity=_rotation_velocity, max_size=None, ngrow=1):
    """`(mesh, g, ct, data, geom, ba, dm, mfs)` — a level with face fluxes."""
    from blockamr.operators.div import update_face_fluxes

    mesh, geom, ba, dm = _level(bodies, max_size)
    g = mesh.ibm.geometry_fab(0, ngrow=ngrow)
    ct = blockamr.CellTypeFab(ba, dm, ngrow)
    blockamr.classify_default(ct, g, geom)
    data = _ghost_data(ct, g, geom, bodies)
    from blockamr.field import FaceField

    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    update_face_fluxes(ff[0], velocity, geom, t=0.0)
    return mesh, g, ct, data, geom, ba, dm, tuple(ff[0][d].mf for d in range(3))


def _div_row(ct, g, data, robin, geom, mfs, cell, face_value=None, n=0, t=0.0):
    """The div pair's row at one cell as `([(index, a)], c)`."""
    if face_value is None:
        face_value = blockamr.DivFaceValue.Upwind
    entries, c = _wall_row_dgc(ct, g, data, robin, geom, t, *mfs, face_value, *cell, n)
    return [((i, j, k), a) for i, j, k, a in entries], c


def _raw(value):
    """The raw `int64` of one f64 — `==` on floats cannot see `-0.0`."""
    return np.float64(value).view(np.int64)


def test_the_div_pair_row_is_callable_host_side_on_one_wall_cell(blockamr_session):
    """**F-1** — tasks.md §3's verify column, for the second pair.

    The same `AMREX_GPU_HOST_DEVICE` functor the kernel launches, called from the
    host at one cell against a `RecordSink`, with the three face fluxes staged
    beside the marker and the geometry.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY)
    phi = _field(ba, dm)
    cell = _a_wall_cell(ct, phi)

    entries, c = _div_row(ct, g, data, _constant(0.5), geom, mfs, cell)

    assert data.nrows == len(_wall_cells(ct, phi)) > 0
    assert isinstance(c, float)
    assert entries and all(len(index) == 3 for index, _a in entries)
    assert entries[0][0] == cell, "the first linear entry is the diagonal, at the row's own cell"


def test_the_bc_datum_reaches_the_div_row_through_constant_and_nothing_else(blockamr_session):
    """**F-2 / S2**, and stronger than "the number is not in the list".

    Two Robin tables differing in `gamma` and in nothing else: the linear entries
    must be **bitwise identical** and only `c` may move. `Mixed`-shaped
    `(alpha, beta)` on purpose, so `atConstant` is non-zero and the datum
    genuinely reaches the row.

    A **uniform** flux and `DivFaceValue.Central` are used so the wall face is
    guaranteed to contribute: under `Upwind` a wall face that is an *outflow*
    face has `weight_self = 1`, so `nb_part` is exactly `0.0` and the wall enters
    neither the donors nor the constant of that row. That is the operator's real
    behaviour (v1 agrees cell for cell, and it is pinned as a count by
    `test_ibm_div_ghost_cell.py`'s `DATUM_ROWS`) — but it would make *this* row
    vacuous at the wrong cell, and this row is about the datum's ROUTE.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY, velocity=_uniform_velocity)
    phi = _field(ba, dm)
    cell = _a_wall_cell(ct, phi)

    def at(datum):
        robin = _robin([[(CONSTANT, datum, 0.0, 0.0, 0.0)]], alpha=0.6, beta=0.4)
        return _div_row(ct, g, data, robin, geom, mfs, cell, blockamr.DivFaceValue.Central)

    first_entries, first_c = at(0.3)
    second_entries, second_c = at(-1.25)

    assert first_c != second_c, "vacuous: the datum does not reach this row at all"
    assert [i for i, _a in first_entries] == [i for i, _a in second_entries]
    lhs = np.array([a for _i, a in first_entries])
    rhs = np.array([a for _i, a in second_entries])
    np.testing.assert_array_equal(lhs.view(np.int64), rhs.view(np.int64))


def test_no_entry_of_a_div_row_ever_names_a_solid_cell(blockamr_session):
    """**F-3 / S3 / Invariant F**, over *every* `WALL` cell of the level.

    A `SOLID` cell holds the pin and not data. Each face is gated on
    `m(ii, jj, kk) != SOLID` and every live trilinear donor was validated fluid
    by `preprocess`'s Invariant-F pass. Both body sets, in one row.
    """
    for bodies in (ONE_BODY, TWO_BODIES):
        _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(bodies)
        phi = _field(ba, dm)
        marker = _marker_grown(ct, phi)
        robin = _robin([[(CONSTANT, 0.3, 0.0, 0.0, 0.0)]] * len(bodies), alpha=0.6, beta=0.4)

        cells = _wall_cells(ct, phi)
        assert cells, "vacuous: no WALL cell"
        named = solid_seen = 0
        for cell in cells:
            entries, _c = _div_row(ct, g, data, robin, geom, mfs, cell)
            for index, _a in entries:
                assert index in marker, f"row at {cell} names {index}, outside the fab box"
                assert marker[index] != SOLID, f"row at {cell} names the SOLID cell {index}"
                named += 1
            solid_seen += 6 - _fluid_arm_count(marker, cell)
        assert named > 0
        assert solid_seen > 0, "vacuous: no WALL cell here has a SOLID face neighbour"


def test_a_solid_face_neighbour_is_named_by_the_probe_and_not_by_the_div_pair(blockamr_session):
    """**F-4** — the pair is not the probe, measured at the same cell.

    `WallFrameProbe` emits its `i +- 1` donors unconditionally; a real pair gates
    each face. Asserted where it bites: a `WALL` cell with a `SOLID` face
    neighbour on the x axis.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    straddling = [
        (cell, off)
        for cell in _wall_cells(ct, phi)
        for off in ((-1, 0, 0), (1, 0, 0))
        if marker[_shifted(cell, off)] == SOLID
    ]
    assert straddling, "vacuous: no WALL cell here has a SOLID neighbour on the x axis"

    cell, off = straddling[0]
    neighbour = _shifted(cell, off)
    pair, _c = _div_row(ct, g, data, robin, geom, mfs, cell)
    probe, _pc = _wall_frame_record(g, robin, geom, 0.0, *cell, 0)

    assert neighbour not in [index for index, _a in pair]
    assert neighbour in [(i, j, k) for i, j, k, _a in probe], (
        "vacuous: the probe no longer emits its unconditional arms"
    )


def test_the_div_row_is_one_diagonal_plus_its_fluid_faces_plus_eight_donors(blockamr_session):
    """**F-5** — the accumulate-then-emit shape, stated as a count.

    `1 + (6 - #solid faces) + 8` entries, at most 15, which is exactly v1's
    `STRIDE`. The `arm[6]` register bank H-6 forces (see F-7) does not widen the
    row: it is where the six coefficients are *accumulated*, not a second
    emission.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    widths = set()
    for cell in _wall_cells(ct, phi):
        entries, _c = _div_row(ct, g, data, robin, geom, mfs, cell)
        fluid_faces = _fluid_arm_count(marker, cell)
        assert len(entries) == 1 + fluid_faces + blockamr.GHOST_CELL_K, cell
        assert len(entries) <= 15
        widths.add(fluid_faces)
    assert len(widths) > 1, "vacuous: every WALL cell here has the same number of solid faces"


def test_the_div_diagonal_sums_over_all_six_faces_including_the_solid_ones(blockamr_session):
    """**F-6 / H-3'** — the single most likely copy-paste defect in the pair.

    v1's mask on the diagonal is `ctx.fluid`, a property of the **row** — which
    the frame has already established by calling the functor at a `WALL` cell —
    and *not* of the face. So `scale * weight_self` is accumulated over **all six
    faces**, the ones whose neighbour is SOLID included, unlike
    `laplacian x ghostCell`, whose diagonal really is gated on the arm.

    Measured on a uniform flux, where the expected value is exactly
    representable: `f = 1.0` and `dx = 1/16` make `scale` exactly `+-16`, so the
    `Upwind` diagonal is exactly `3 * 16 = 48` and the `Central` one exactly
    `0.0` — **at every WALL cell, whatever its solid faces are**, which is the
    whole point. Gating on the face would make it cell-dependent.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY, velocity=_uniform_velocity)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    solid_faces = set()
    for cell in _wall_cells(ct, phi):
        for face_value, want in (
            (blockamr.DivFaceValue.Upwind, 48.0),
            (blockamr.DivFaceValue.Central, 0.0),
        ):
            entries, _c = _div_row(ct, g, data, robin, geom, mfs, cell, face_value)
            index, a = entries[0]
            assert index == cell
            assert _raw(a) == _raw(want), f"{cell} ({face_value}): diagonal {a!r} != {want!r}"
        solid_faces.add(6 - _fluid_arm_count(marker, cell))
    assert len(solid_faces) > 1 and max(solid_faces) > 0, (
        "vacuous: the diagonal cannot be shown ungated unless the solid-face count varies"
    )


def test_a_zero_face_flux_reaches_the_row_as_positive_zero_and_not_negative_zero(
    blockamr_session,
):
    """**F-7 / H-6** — the finding this session paid for, pinned where a reader
    will look for it.

    v1's `_blank` allocates `a = np.zeros(...)` and writes each face-neighbour
    slot **once**, so the coefficient it ships is `0.0 + nb_part`, and IEEE says
    `0.0 + (-0.0)` is `+0.0`. A functor that emitted `nb_part` raw would ship
    `-0.0`.

    `nb_part` is `-0.0` exactly when the face flux is `+-0.0` and the face-value
    rule puts the whole weight on the neighbour at `step = -1`: `scale` is
    `(-1) * 0.0 / dx = -0.0` and `scale * (1 - w)` keeps its sign. The rotation
    flux has `u_z` identically zero, so **every** z face is such a face.

    Measured over the whole level and on the raw bits, because `a == 0.0` is true
    of `-0.0` too and would assert nothing. 960 of 3 232 wall rows in
    `test_ibm_div_ghost_cell.py` break if this is got wrong; here it is one cell,
    named, with the mechanism spelled out.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY, velocity=_rotation_velocity)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    # the LOW z face (step = -1) of a WALL cell whose -z neighbour is fluid
    cells = [c for c in _wall_cells(ct, phi) if marker[_shifted(c, (0, 0, -1))] != SOLID]
    assert cells, "vacuous: no WALL cell here has a fluid -z neighbour"

    checked = 0
    for cell in cells:
        for face_value in (blockamr.DivFaceValue.Upwind, blockamr.DivFaceValue.Central):
            entries, _c = _div_row(ct, g, data, robin, geom, mfs, cell, face_value)
            at = dict(entries[1:-blockamr.GHOST_CELL_K])
            below = _shifted(cell, (0, 0, -1))
            assert below in at, f"{cell}: the fluid -z face emitted no entry"
            assert _raw(at[below]) == _raw(0.0), (
                f"{cell} ({face_value}): the -z coefficient is {at[below]!r} with raw bits "
                f"{_raw(at[below])}, but v1 accumulates it into a zero slot and ships +0.0 "
                "(H-6) — the functor emitted the raw value instead of `arm[slot] += nbp`"
            )
            assert _raw(at[below]) != _raw(-0.0), "the two zeros must be distinguishable here"
            checked += 1
    assert checked > 0


def test_the_eight_div_donor_entries_are_the_methods_own_stencil(blockamr_session):
    """**F-8** — the §4 row map lands on the right row.

    The last eight entries must be `GhostCellData.donor[r]` for that cell's rank
    `r`, in slot order, with a dead slot (weight exactly `0.0`) at the row's own
    cell. `PLANE_X` is used because its image point lands on a cell face, so half
    the weights are exactly zero and the dead-slot rule is exercised instead of
    assumed.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(PLANE_X)
    phi = _field(ba, dm)
    _ip, donor, weight, _distance = _ghost_cell_numpy(ct, g, geom, ["wall"])
    robin = _constant(0.5)

    assert (weight == 0.0).any(), "vacuous: this geometry has no dead donor slot"
    for cell in _wall_cells(ct, phi):
        rank = data.row_at(*cell)
        assert rank >= 0
        entries, _c = _div_row(ct, g, data, robin, geom, mfs, cell)
        got = [index for index, _a in entries[-blockamr.GHOST_CELL_K :]]
        want = [
            cell if weight[rank, q] == 0.0 else tuple(int(v) for v in donor[rank, q])
            for q in range(blockamr.GHOST_CELL_K)
        ]
        assert got == want, f"{cell} (rank {rank})"


def test_the_div_pair_reads_the_geometry_at_its_own_cell_and_not_at_a_neighbour(blockamr_session):
    """**F-9 / Q34**, made falsifiable (B30a-R's I-1).

    The functor's only geometry reads are `patch(i, j, k)`, `sdf(i, j, k)` and
    `normal(i, j, k, d)`. Reading `f[dd](i + 1, ...)` is a *face* array at the
    cell's own high face, not a neighbour's geometry — which is why
    `stencil_reach = 1` stays honest.

    B30a-R measured that comparing two *builders* cannot catch a neighbour read,
    so this perturbs one fab at one index instead: moving the normal at a **face
    neighbour** must leave the row bitwise identical, and moving it at the row's
    **own cell** must change it. `_slab` is used so the perturbed neighbour is on
    the *other* patch, which is where a neighbour read would also pick up the
    wrong `alpha`/`beta`.
    """
    bodies = _slab(0)
    mesh, g, ct, data, geom, ba, dm, mfs = _div_case(bodies)
    phi = _field(ba, dm)
    patch_of = _patch_of(g, phi)
    robin = _robin([[(CONSTANT, 0.5, 0.0, 0.0, 0.0)], [(CONSTANT, -0.25, 0.0, 0.0, 0.0)]])

    straddling = [
        (cell, _shifted(cell, off))
        for cell in _wall_cells(ct, phi)
        for off in ((1, 0, 0), (-1, 0, 0))
        if patch_of.get(_shifted(cell, off), patch_of[cell]) != patch_of[cell]
    ]
    assert straddling, "vacuous: no WALL cell straddles a patch boundary"
    cell, neighbour = straddling[0]

    base = _div_row(ct, g, data, robin, geom, mfs, cell)
    at_neighbour = _perturbed_geometry(mesh, ba, dm, neighbour, 0.125)
    at_self = _perturbed_geometry(mesh, ba, dm, cell, 0.125)

    assert _div_row(ct, at_neighbour, data, robin, geom, mfs, cell) == base, (
        f"the row at {cell} moved when the geometry at {neighbour} did — Q34 is tripped"
    )
    assert _div_row(ct, at_self, data, robin, geom, mfs, cell) != base, (
        "vacuous: perturbing the geometry at the row's own cell changed nothing"
    )


def test_the_div_closures_pole_reaches_the_row_as_infinity_and_raises_nothing(blockamr_session):
    """**F-10 / Q46**, inherited and unchanged: the guard is DEFERRED and the
    behaviour is PINNED — **with the signs**, not merely `isinf` (B32-R's S-2).

    `robin.H`'s `den = beta - alpha*d` is exactly zero for the reachable
    `Mixed(f)` with `d = (1 - f)/f`, and v1 divides anyway and returns `+-inf`. A
    raise here would be a behaviour change against v1 in a session whose whole
    claim is that nothing changed, and it would fail the parity bar by design.

    A uniform flux and `DivFaceValue.Central` are used so the sign of `nb_part`
    at the one wall face is decided rather than incidental — and so that it is
    non-zero at all: under `Upwind` an outflow wall face has `nb_part` exactly
    `0.0`, and `0.0 * inf` is a NaN, which would turn the pole into a
    classification question instead of the signed one B32-R's S-2 asks for. A
    cell with exactly one solid face is chosen so there is no `inf - inf`
    anywhere in the row: every value below is a single signed infinity or a
    single `inf * 0`.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY, velocity=_uniform_velocity)
    phi = _field(ba, dm)
    _ip, _donor, weight, distance = _ghost_cell_numpy(ct, g, geom, ["cyl"])
    marker = _marker_grown(ct, phi)

    chosen = None
    for cell in _wall_cells(ct, phi):
        if _fluid_arm_count(marker, cell) == 5:
            chosen = (cell, data.row_at(*cell))
            break
    assert chosen is not None, "vacuous: no WALL cell here has exactly one SOLID face"
    cell, rank = chosen

    # alpha = 1, beta = d  =>  den = beta - alpha*d = 0 exactly, on this row.
    d = float(distance[rank])
    robin = _robin([[(CONSTANT, 0.5, 0.0, 0.0, 0.0)]], alpha=1.0, beta=d)

    entries, c = _div_row(
        ct, g, data, robin, geom, mfs, cell, blockamr.DivFaceValue.Central
    )  # must not raise

    donors = [a for _i, a in entries[-blockamr.GHOST_CELL_K :]]
    live = [a for a, w in zip(donors, weight[rank]) if w != 0.0]
    dead = [a for a, w in zip(donors, weight[rank]) if w == 0.0]
    assert live and dead, f"vacuous: {cell} has no live or no dead donor slot"
    # A live donor carries `(nb_part * atLinear(dG)) * w`, and with one wall face
    # and a fixed flux its sign is determined: all live donors share it, the
    # constant carries the opposite one, and both are pinned rather than merely
    # asserted infinite.
    assert all(np.isinf(a) for a in live), live
    assert len(set(np.sign(live))) == 1, live
    sign = float(np.sign(live[0]))
    assert all(np.isnan(a) for a in dead), dead
    assert np.isinf(c) and np.sign(c) == -sign, (c, sign)
    # the finite half of the row is untouched: the closure never reaches it.
    assert all(np.isfinite(a) for _i, a in entries[: -blockamr.GHOST_CELL_K]), entries


def test_a_field_narrower_than_the_div_pairs_reach_is_refused_naming_the_pair(blockamr_session):
    """**F-11 / S8**, and api §9: the sentence names `wall_div_ghost_cell` and
    not `applyWall`, because "applyWall" names nothing a caller can see.

    `stencil_reach = 1` is the *field* and *marker* reach. The three face fluxes
    are deliberately **not** covered by it and need no ghosts at all: the functor
    reads only the cell's own two faces in each direction, both inside the face
    fab's valid box for every cell of `validbox`.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY)
    phi = _field(ba, dm, ngrow=0)
    out = _out(ba, dm)
    with pytest.raises(
        RuntimeError, match=r"wall_div_ghost_cell: the functor declares stencil_reach = 1"
    ):
        blockamr.wall_div_ghost_cell(
            out,
            phi,
            ct,
            g,
            data,
            _constant(0.5),
            geom,
            0.0,
            1.0,
            1,
            blockamr.WallMode.Overwrite,
            1.0,
            *mfs,
            blockamr.DivFaceValue.Upwind,
        )


def test_each_disagreement_between_the_div_pairs_arguments_is_refused_by_name(blockamr_session):
    """**F-12** — guard 0 and `Maker::validate` (B30a-R's S-5), together, plus
    the guard this pair adds.

    Each of these is a silently wrong answer rather than a crash in a release
    build, and each is named by the entry point:

    * `WallMode.Assemble` — declared and not implemented (S6);
    * `out is phi` — a row would read cells another row had already written;
    * a mismatched `BoxArray` — the sweep pairs fabs by `MFIter` local index;
    * a Robin table narrower than the field — an out-of-bounds `gammaAt`;
    * method data preprocessed on other grids — invisible to the frame;
    * **a face flux on the wrong grids** — the same defect class, and `div`'s
      own: the three face fabs are resolved by local index beside phi/out/ct, so
      they must be the marker's `BoxArray` converted to face centring **in their
      own direction**. Passing `flux_y` where `flux_x` belongs is the cheapest
      real instance of it, and it must name the direction.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)

    def call(out_mf, phi_mf, ct_fab, data_obj, robin, flux=None, ncomp=1, mode=None):
        blockamr.wall_div_ghost_cell(
            out_mf,
            phi_mf,
            ct_fab,
            g,
            data_obj,
            robin,
            geom,
            0.0,
            1.0,
            ncomp,
            blockamr.WallMode.Overwrite if mode is None else mode,
            1.0,
            *(mfs if flux is None else flux),
            blockamr.DivFaceValue.Upwind,
        )

    with pytest.raises(RuntimeError, match=r"wall_div_ghost_cell: WallMode.Assemble"):
        call(out, phi, ct, data, _constant(0.5), mode=blockamr.WallMode.Assemble)
    assert all(v == SENTINEL for v in _readback(out).values())

    with pytest.raises(RuntimeError, match=r"wall_div_ghost_cell: .*different MultiFabs"):
        call(phi, phi, ct, data, _constant(0.5))

    ba8 = blockamr.BoxArray(blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1]))
    ba8.max_size(8)
    dm8 = blockamr.DistributionMapping(ba8)
    with pytest.raises(RuntimeError, match=r"wall_div_ghost_cell: out, phi and the"):
        call(_out(ba8, dm8), phi, ct, data, _constant(0.5))

    phi2 = _field(ba, dm, ncomp=2)
    with pytest.raises(RuntimeError, match=r"the field has 2 but the table has 1"):
        call(_out(ba, dm, ncomp=2), phi2, ct, data, _constant(0.5), ncomp=2)

    g8 = _level(ONE_BODY, max_size=8)[0].ibm.geometry_fab(0, ngrow=1)
    ct8 = blockamr.CellTypeFab(ba8, dm8, 1)
    blockamr.classify_default(ct8, g8, geom)
    other = _ghost_data(ct8, g8, geom, ONE_BODY)
    with pytest.raises(RuntimeError, match=r"wall_div_ghost_cell: the ghostCell data"):
        call(out, phi, ct, other, _constant(0.5))

    with pytest.raises(
        RuntimeError, match=r"wall_div_ghost_cell: the face flux in direction 0"
    ):
        call(out, phi, ct, data, _constant(0.5), flux=(mfs[1], mfs[1], mfs[2]))


def test_the_two_div_face_value_rules_are_exactly_the_v1_face_weights(blockamr_session):
    """**F-13** — the `DivFaceValue` mapping, as a **partition/exactness** row.

    Q44's caution is live for this pair (B35 measured that van Leer's limiter
    *absorbs* a `1e30` solid pin and returns a finite number with no NaN), so
    nothing here is pin-and-watch: the configuration is fixed, the flux is
    exactly `1.0`, and the weights the two branches produce are exactly
    representable and asserted on the bits.

    On this grid `dx = 1/16`, so `scale = step * 1.0 / dx` is exactly `+-16`:

    * `Central` — `w = 0.5` at both faces, so the neighbour's coefficient is
      `+8` at the high face and `-8` at the low one;
    * `Upwind` with `f >= 0` — the whole weight is on the **target** at the high
      face (neighbour coefficient `+0.0`, and `+0.0` and not `-0.0`, which is
      H-6 again) and on the **neighbour** at the low face (coefficient `-16`).

    That is v1's `_face_weights` and the D1 degrade in one statement:
    `linear` maps to `Central`, and `upwind`, `vanLeer` and `quick` all map to
    `Upwind` — the last two because a width-2 stencil reaches through the solid
    inside the band and degrades to first-order upwind there.
    """
    _mesh, g, ct, data, geom, ba, dm, mfs = _div_case(ONE_BODY, velocity=_uniform_velocity)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    want = {
        blockamr.DivFaceValue.Central: {(1, 0, 0): 8.0, (-1, 0, 0): -8.0},
        blockamr.DivFaceValue.Upwind: {(1, 0, 0): 0.0, (-1, 0, 0): -16.0},
    }

    # one named cell with BOTH x neighbours fluid, so the two faces are both
    # observable in the same row.
    cells = [
        c
        for c in _wall_cells(ct, phi)
        if marker[_shifted(c, (1, 0, 0))] != SOLID and marker[_shifted(c, (-1, 0, 0))] != SOLID
    ]
    assert cells, "vacuous: no WALL cell here has both x neighbours fluid"
    cell = cells[0]

    for face_value, expected in want.items():
        entries, _c = _div_row(ct, g, data, robin, geom, mfs, cell, face_value)
        at = dict(entries[1:-blockamr.GHOST_CELL_K])
        for off, value in expected.items():
            index = _shifted(cell, off)
            assert index in at, f"{cell}: no entry at {index}"
            assert _raw(at[index]) == _raw(value), (
                f"{cell} ({face_value}) at {off}: {at[index]!r}, expected {value!r}"
            )


# ===========================================================================
# 8. `grad x ghostCell` — the third real pair, per cell (B34)
#
# **Conformance, not acceptance**, exactly as sections 6 and 7 are. v1<->v2
# bitwise row parity over ten configurations, the falsification matrix, the
# census, the sweep, the argument contract and the `ncomp` refusal end to end
# live in `test_ibm_grad_ghost_cell.py`, which has the heavy fixtures and a
# different vocabulary. What is here is what the shipped frame file is the
# natural home for: which cells a row may name (S3), how the BC datum reaches it
# (S2), where the geometry is read (Q34), what the row's shape is, the error
# surface — and the two things `grad` adds to that list, H-9's one axis and
# H-10's `+0.0` diagonal.
#
# **Q54(a) governs every row below.** Half of a grad wall population has no wall
# arm at all: a `WALL` cell whose solid neighbours are all off the DIFFERENCING
# axis contributes exactly nothing through the closure. Every row here that needs
# the closure therefore names a configuration measured to have x arms
# (`ONE_BODY`, `TWO_BODIES`, `_slab(0)`, `PLANE_X`) and says so, and `PLANE_Y`
# exists precisely to exercise the other half.
# ===========================================================================

_wall_row_ggc = blockamr._blockamr._wall_row_grad_ghost_cell

#: A wall normal to **y**: no `WALL` cell has a `SOLID` x-neighbour, so no row
#: has a wall arm and the closure never enters one. Q54(a)'s extreme, at one
#: cell — `test_ibm_grad_ghost_cell.py` pins it as a census over 576 such rows.
PLANE_Y = {"wall": Plane(point=(0.0, 0.5, 0.0), normal=(0.0, 1.0, 0.0))}

#: The two face offsets a `grad` row can ever name — v1's `axes = (0,)`, in the
#: pair's own loop order (+1 first).
X_ARMS = ((1, 0, 0), (-1, 0, 0))

#: `scale * 0.5` at the high face on this grid: `dx = 1/16`, so
#: `scale = step * 1.0 / dx` is exactly `+-16` and every coefficient below is
#: exactly representable.
GRAD_HALF = 8.0


def _grad_case(bodies, max_size=None, ngrow=1):
    """`(mesh, g, ct, data, geom, ba, dm)` — a level, classified, preprocessed.

    No face field: a `grad` row reads none, which is the whole of §H-9's first
    consequence and why this pair takes exactly the canonical twelve.
    """
    mesh, geom, ba, dm = _level(bodies, max_size)
    g = mesh.ibm.geometry_fab(0, ngrow=ngrow)
    ct = blockamr.CellTypeFab(ba, dm, ngrow)
    blockamr.classify_default(ct, g, geom)
    return mesh, g, ct, _ghost_data(ct, g, geom, bodies), geom, ba, dm


def _grad_row(ct, g, data, robin, geom, cell, n=0, t=0.0):
    """The grad pair's row at one cell as `([(index, a)], c)`."""
    entries, c = _wall_row_ggc(ct, g, data, robin, geom, t, *cell, n)
    return [((i, j, k), a) for i, j, k, a in entries], c


def _fluid_x_faces(marker, cell):
    """How many of the two **x** face neighbours are not `SOLID`."""
    return sum(1 for off in X_ARMS if marker[_shifted(cell, off)] != SOLID)


def test_the_grad_pair_row_is_callable_host_side_on_one_wall_cell(blockamr_session):
    """**F-1** — tasks.md §3's verify column, for the third and last pair.

    The same `AMREX_GPU_HOST_DEVICE` functor the kernel launches, called from the
    host at one cell against a `RecordSink`. Nothing but the marker, the packed
    geometry and the method's rows is staged: there is no face field to stage.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    cell = _a_wall_cell(ct, phi)

    entries, c = _grad_row(ct, g, data, _constant(0.5), geom, cell)

    assert data.nrows == len(_wall_cells(ct, phi)) > 0
    assert isinstance(c, float)
    assert entries and all(len(index) == 3 for index, _a in entries)
    assert entries[0][0] == cell, "the first linear entry is the diagonal, at the row's own cell"


def test_the_bc_datum_reaches_the_grad_row_through_constant_and_nothing_else(blockamr_session):
    """**F-2 / S2**, with Q54(a)'s aggregate guard.

    Two Robin tables differing in `gamma` and in nothing else: the linear entries
    must be **bitwise identical** and only `c` may move. `Mixed`-shaped
    `(alpha, beta)` on purpose, so `atConstant` is non-zero and the datum
    genuinely reaches the row.

    The non-vacuity guard is **aggregate** and it has to be: on a `grad` row
    whose two x neighbours are both fluid the closure never runs at all, so
    `c == 0.0` there is the right answer and not a missing datum. The count of
    rows that *do* carry a datum is asserted, so the two populations cannot
    silently swap — and the per-row assertion runs on every wall cell either way.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    cells = _wall_cells(ct, phi)

    def at(cell, datum):
        robin = _robin([[(CONSTANT, datum, 0.0, 0.0, 0.0)]], alpha=0.6, beta=0.4)
        return _grad_row(ct, g, data, robin, geom, cell)

    with_datum = with_arm = 0
    for cell in cells:
        first_entries, first_c = at(cell, 0.3)
        second_entries, second_c = at(cell, -1.25)
        assert [i for i, _a in first_entries] == [i for i, _a in second_entries]
        lhs = np.array([a for _i, a in first_entries])
        rhs = np.array([a for _i, a in second_entries])
        np.testing.assert_array_equal(lhs.view(np.int64), rhs.view(np.int64))
        with_datum += first_c != second_c
        with_arm += _fluid_x_faces(marker, cell) < 2
    assert with_arm == 192, f"the wall-arm population moved: {with_arm} of {len(cells)}"
    assert with_datum == with_arm, (
        f"vacuous or wrong: {with_datum} rows respond to the datum but {with_arm} have a wall "
        "arm — the datum reaches a grad row through its SOLID x face and through nothing else"
    )


def test_no_entry_of_a_grad_row_ever_names_a_solid_cell(blockamr_session):
    """**F-3 / S3 / Invariant F**, over *every* `WALL` cell of the level.

    A `SOLID` cell holds the pin and not data. Each x face is gated on
    `m(i +- 1, j, k) != SOLID` and every live trilinear donor was validated fluid
    by `preprocess`'s Invariant-F pass. Both body sets, in one row.
    """
    for bodies in (ONE_BODY, TWO_BODIES):
        _mesh, g, ct, data, geom, ba, dm = _grad_case(bodies)
        phi = _field(ba, dm)
        marker = _marker_grown(ct, phi)
        robin = _robin([[(CONSTANT, 0.3, 0.0, 0.0, 0.0)]] * len(bodies), alpha=0.6, beta=0.4)

        cells = _wall_cells(ct, phi)
        assert cells, "vacuous: no WALL cell"
        named = solid_seen = 0
        for cell in cells:
            entries, _c = _grad_row(ct, g, data, robin, geom, cell)
            for index, _a in entries:
                assert index in marker, f"row at {cell} names {index}, outside the fab box"
                assert marker[index] != SOLID, f"row at {cell} names the SOLID cell {index}"
                named += 1
            solid_seen += 2 - _fluid_x_faces(marker, cell)
        assert named > 0
        assert solid_seen > 0, "vacuous: no WALL cell here has a SOLID x neighbour"


def test_a_solid_x_neighbour_is_named_by_the_probe_and_not_by_the_grad_pair(blockamr_session):
    """**F-4** — the pair is not the probe, measured at the same cell.

    `WallFrameProbe` emits its `i +- 1` donors unconditionally; a real pair gates
    each face. Asserted where it bites: a `WALL` cell with a `SOLID` face
    neighbour on the x axis — which for `grad` is the *only* axis a row can name.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    straddling = [
        (cell, off)
        for cell in _wall_cells(ct, phi)
        for off in X_ARMS
        if marker[_shifted(cell, off)] == SOLID
    ]
    assert straddling, "vacuous: no WALL cell here has a SOLID neighbour on the x axis"

    cell, off = straddling[0]
    neighbour = _shifted(cell, off)
    pair, _c = _grad_row(ct, g, data, robin, geom, cell)
    probe, _pc = _wall_frame_record(g, robin, geom, 0.0, *cell, 0)

    assert neighbour not in [index for index, _a in pair]
    assert neighbour in [(i, j, k) for i, j, k, _a in probe], (
        "vacuous: the probe no longer emits its unconditional arms"
    )


def test_the_grad_row_is_one_diagonal_plus_its_fluid_x_faces_plus_eight_donors(blockamr_session):
    """**F-5 / H-9** — the axis collapse, stated as a count.

    `1 + (2 - #solid x faces) + 8` entries, at most **11** and **never 15**. v1's
    slots 3..6 — the `+-y` and `+-z` neighbours — are allocated by `_blank`, left
    pointing at the target and never written by `axes = (0,)`, so v1's own
    liveness rule drops them and the pair emits nothing there. A functor that
    copied `div`'s six-arm emission loop would emit them with `+0.0` and be
    fifteen entries wide; that is the `arms-six` mutant, caught on 10 of 10
    configurations and all 3 136 rows in `test_ibm_grad_ghost_cell.py`.

    The `+-y`/`+-z` neighbours are asserted absent by name, not merely counted:
    on this geometry many of them are fluid, so "the row is narrow" and "the row
    names no y/z neighbour" are different claims.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    widths = set()
    off_axis_fluid = 0
    for cell in _wall_cells(ct, phi):
        entries, _c = _grad_row(ct, g, data, robin, geom, cell)
        fluid_x = _fluid_x_faces(marker, cell)
        assert len(entries) == 1 + fluid_x + blockamr.GHOST_CELL_K, cell
        assert len(entries) <= 11
        named = [index for index, _a in entries]
        for off in ARMS[2:]:
            neighbour = _shifted(cell, off)
            if marker[neighbour] != SOLID:
                off_axis_fluid += 1
                assert neighbour not in named[: 1 + fluid_x], (
                    f"{cell}: the row names the off-axis neighbour {neighbour} — grad "
                    "differences ONE axis (H-9)"
                )
        widths.add(fluid_x)
    assert len(widths) > 1, "vacuous: every WALL cell here has the same number of solid x faces"
    assert off_axis_fluid > 0, "vacuous: no WALL cell here has a fluid y or z neighbour"


def test_the_grad_diagonal_sums_over_both_x_faces_including_a_solid_one(blockamr_session):
    """**F-6 / H-3'** — the single most likely copy-paste defect in the pair.

    v1's mask on the diagonal is `ctx.fluid`, a property of the **row** — which
    the frame has already established by calling the functor at a `WALL` cell —
    and *not* of the face. So `scale * 0.5` is accumulated over **both** x faces,
    the one whose neighbour is SOLID included, unlike `laplacian x ghostCell`,
    whose diagonal really is gated on the arm.

    It is discriminating precisely at a cell with a `SOLID` x neighbour: gated,
    the diagonal would be a single `scale * 0.5 = +-8`; ungated, the two faces
    cancel to `+0.0`. Both values are exactly representable on this grid
    (`dx = 1/16`, so `scale` is exactly `+-16`), so the assertion is on the bits
    and not on a tolerance. `G4`, not `G2`: on a configuration with no wall arm
    every row has both x faces fluid and the gate is invisible.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _constant(0.5)

    with_solid_x = 0
    for cell in _wall_cells(ct, phi):
        entries, _c = _grad_row(ct, g, data, robin, geom, cell)
        index, a = entries[0]
        assert index == cell
        assert _raw(a) == _raw(0.0), (
            f"{cell}: the diagonal is {a!r}; both x faces contribute `scale * 0.5` whatever "
            "the marker says, so it is exactly +0.0 — a value of +-8 means the laplacian's "
            "fluid-arm gate was copied onto it (H-3')"
        )
        if _fluid_x_faces(marker, cell) < 2:
            with_solid_x += 1
            assert _raw(a) != _raw(GRAD_HALF) and _raw(a) != _raw(-GRAD_HALF)
    assert with_solid_x > 0, (
        "vacuous: the diagonal cannot be shown ungated unless some WALL cell has a SOLID x face"
    )


def test_the_grad_diagonal_is_bitwise_positive_zero_and_is_still_emitted(blockamr_session):
    """**F-7 / H-10** — the finding this session paid for, pinned where a reader
    will look for it.

    For `axes = (0,)` slot 0 accumulates exactly twice, and
    `(+1)*1.0/dx * 0.5` and `(-1)*1.0/dx * 0.5` are exact negatives: IEEE
    multiplication and division are sign-symmetric, and `x + (-x)` is `+0.0` in
    round-to-nearest for every finite `x`. So the diagonal of a grad wall row is
    `+0.0` — not `-0.0`, not "approximately zero" — on every row of every
    configuration (measured: 3 136 of 3 136).

    Compared on the **raw bits**, because `a == 0.0` is true of `-0.0` too and
    would assert nothing: a functor that computed the diagonal as
    `-(slf_low + slf_high)`, or initialised its accumulator to `-0.0`, ships a
    different row and v1's is `+0.0`.

    And the entry is **present**. It is not optional and it may not be
    "optimised away" as provably zero: v1's row carries slot 0 with
    `stencil[0] = target` and `a[0] = +0.0`, and a sweep reading it multiplies
    `phi(P)` by `+0.0` and adds it. Both spellings of the defect —
    `diag-dropped` and `diag-neg-zero` — are caught on all 3 136 rows in
    `test_ibm_grad_ghost_cell.py`.
    """
    for bodies in (ONE_BODY, PLANE_Y):
        _mesh, g, ct, data, geom, ba, dm = _grad_case(bodies)
        phi = _field(ba, dm)
        robin = _robin([[(CONSTANT, 0.3, 0.0, 0.0, 0.0)]], alpha=0.6, beta=0.4)

        cells = _wall_cells(ct, phi)
        assert cells, "vacuous: no WALL cell"
        for cell in cells:
            entries, _c = _grad_row(ct, g, data, robin, geom, cell)
            index, a = entries[0]
            assert index == cell, f"{cell}: the diagonal entry is missing — it names {index}"
            assert _raw(a) == _raw(0.0), (
                f"{cell}: the diagonal is {a!r} with raw bits {_raw(a)}, but v1 accumulates "
                f"`0.0 + s + (-s)` and ships +0.0 (raw {_raw(0.0)})"
            )
            assert _raw(a) != _raw(-0.0), "the two zeros must be distinguishable here"


def test_the_eight_grad_donor_entries_are_the_methods_own_stencil(blockamr_session):
    """**F-8** — the §4 row map lands on the right row, and all eight are emitted.

    The last eight entries must be `GhostCellData.donor[r]` for that cell's rank
    `r`, in slot order, with a dead slot (weight exactly `0.0`) at the row's own
    cell. `PLANE_X` is used because its image point lands on a cell face, so half
    the weights are exactly zero and the dead-slot rule is exercised instead of
    assumed.

    `PLANE_Y` is then run for the Q54(a) half: there **no** row has a wall arm,
    so every one of the eight donor coefficients is `+0.0` — and all eight are
    still present, which is what `donors-dropped` (caught on 2 928 of 3 136 rows)
    models the loss of.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(PLANE_X)
    phi = _field(ba, dm)
    _ip, donor, weight, _distance = _ghost_cell_numpy(ct, g, geom, ["wall"])
    robin = _constant(0.5)

    assert (weight == 0.0).any(), "vacuous: this geometry has no dead donor slot"
    for cell in _wall_cells(ct, phi):
        rank = data.row_at(*cell)
        assert rank >= 0
        entries, _c = _grad_row(ct, g, data, robin, geom, cell)
        got = [index for index, _a in entries[-blockamr.GHOST_CELL_K :]]
        want = [
            cell if weight[rank, q] == 0.0 else tuple(int(v) for v in donor[rank, q])
            for q in range(blockamr.GHOST_CELL_K)
        ]
        assert got == want, f"{cell} (rank {rank})"

    _mesh, g, ct, data, geom, ba, dm = _grad_case(PLANE_Y)
    phi = _field(ba, dm)
    marker = _marker_grown(ct, phi)
    robin = _robin([[(CONSTANT, 0.3, 0.0, 0.0, 0.0)]], alpha=0.6, beta=0.4)
    cells = _wall_cells(ct, phi)
    assert cells, "vacuous: no WALL cell"
    for cell in cells:
        assert _fluid_x_faces(marker, cell) == 2, f"{cell}: PLANE_Y is supposed to have no x arm"
        entries, c = _grad_row(ct, g, data, robin, geom, cell)
        assert len(entries) == 1 + 2 + blockamr.GHOST_CELL_K, cell
        assert _raw(c) == _raw(0.0), f"{cell}: a no-arm row has c = {c!r}"
        for index, a in entries[-blockamr.GHOST_CELL_K :]:
            assert _raw(a) == _raw(0.0), f"{cell}: a no-arm row's donor {index} carries {a!r}"


def test_the_grad_pair_reads_the_geometry_at_its_own_cell_and_not_at_a_neighbour(blockamr_session):
    """**F-9 / Q34**, made falsifiable (B30a-R's I-1).

    The functor's only geometry reads are `patch(i, j, k)`, `sdf(i, j, k)` and
    `normal(i, j, k, 0)`. There is no face-centred array here at all, which is
    why `stencil_reach = 1` stays honest and why this pair's `validate` has no
    flux clause.

    B30a-R measured that comparing two *builders* cannot catch a neighbour read,
    so this perturbs one fab at one index instead: moving the normal at a **face
    neighbour** must leave the row bitwise identical, and moving it at the row's
    **own cell** must change it. `_slab(0)` is used so the perturbed neighbour is
    on the *other* patch — where a neighbour read would also pick up the wrong
    `alpha`/`beta` — and so that the perturbed axis is the differencing one: on a
    slab normal to y or z a grad row would not read the normal at all and the
    second half of this row would be vacuous.
    """
    bodies = _slab(0)
    mesh, g, ct, data, geom, ba, dm = _grad_case(bodies)
    phi = _field(ba, dm)
    patch_of = _patch_of(g, phi)
    robin = _robin([[(CONSTANT, 0.5, 0.0, 0.0, 0.0)], [(CONSTANT, -0.25, 0.0, 0.0, 0.0)]])

    straddling = [
        (cell, _shifted(cell, off))
        for cell in _wall_cells(ct, phi)
        for off in X_ARMS
        if patch_of.get(_shifted(cell, off), patch_of[cell]) != patch_of[cell]
    ]
    assert straddling, "vacuous: no WALL cell straddles a patch boundary"
    cell, neighbour = straddling[0]

    base = _grad_row(ct, g, data, robin, geom, cell)
    at_neighbour = _perturbed_geometry(mesh, ba, dm, neighbour, 0.125)
    at_self = _perturbed_geometry(mesh, ba, dm, cell, 0.125)

    assert _grad_row(ct, at_neighbour, data, robin, geom, cell) == base, (
        f"the row at {cell} moved when the geometry at {neighbour} did — Q34 is tripped"
    )
    assert _grad_row(ct, at_self, data, robin, geom, cell) != base, (
        "vacuous: perturbing the geometry at the row's own cell changed nothing"
    )


def test_the_grad_closures_pole_reaches_the_row_as_infinity_and_raises_nothing(blockamr_session):
    """**F-10 / Q46**, inherited and unchanged: the guard is DEFERRED and the
    behaviour is PINNED — **with the signs**, not merely `isinf` (B32-R's S-2).

    `robin.H`'s `den = beta - alpha*d` is exactly zero for the reachable
    `Mixed(f)` with `d = (1 - f)/f`, and v1 divides anyway and returns `+-inf`. A
    raise here would be a behaviour change against v1 in a session whose whole
    claim is that nothing changed, and it would fail the parity bar by design.

    The cell is chosen to have **exactly one** solid x face, for two reasons.
    First, Q54(a): on a row with no wall arm the closure never enters at all and
    the pole would be unaskable — this row would pass while asserting nothing.
    Second, one wall face means there is no `inf - inf` anywhere in the row:
    every value below is a single signed infinity or a single `inf * 0`, and
    `nb_part` is `+-8` exactly, so the sign of each live donor is decided rather
    than incidental.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    _ip, _donor, weight, distance = _ghost_cell_numpy(ct, g, geom, ["cyl"])
    marker = _marker_grown(ct, phi)

    chosen = None
    for cell in _wall_cells(ct, phi):
        if _fluid_x_faces(marker, cell) == 1:
            chosen = (cell, data.row_at(*cell))
            break
    assert chosen is not None, "vacuous: no WALL cell here has exactly one SOLID x face"
    cell, rank = chosen

    # alpha = 1, beta = d  =>  den = beta - alpha*d = 0 exactly, on this row.
    d = float(distance[rank])
    robin = _robin([[(CONSTANT, 0.5, 0.0, 0.0, 0.0)]], alpha=1.0, beta=d)

    entries, c = _grad_row(ct, g, data, robin, geom, cell)  # must not raise

    donors = [a for _i, a in entries[-blockamr.GHOST_CELL_K :]]
    live = [a for a, w in zip(donors, weight[rank]) if w != 0.0]
    dead = [a for a, w in zip(donors, weight[rank]) if w == 0.0]
    assert live and dead, f"vacuous: {cell} has no live or no dead donor slot"
    assert all(np.isinf(a) for a in live), live
    assert len(set(np.sign(live))) == 1, live
    sign = float(np.sign(live[0]))
    assert all(np.isnan(a) for a in dead), dead
    assert np.isinf(c) and np.sign(c) == -sign, (c, sign)
    # the finite half of the row is untouched: the closure never reaches it.
    assert all(np.isfinite(a) for _i, a in entries[: -blockamr.GHOST_CELL_K]), entries


def test_a_field_narrower_than_the_grad_pairs_reach_is_refused_naming_the_pair(blockamr_session):
    """**F-11 / S8**, and api §9: the sentence names `wall_grad_ghost_cell` and
    not `applyWall`, because "applyWall" names nothing a caller can see.

    `stencil_reach = 1` is honest for this pair and cheaply so: the only reads
    outside the target cell are `m(i +- 1, j, k)` and the eight donors, and there
    is no face-centred array to reason about at all.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm, ngrow=0)
    out = _out(ba, dm)
    with pytest.raises(
        RuntimeError, match=r"wall_grad_ghost_cell: the functor declares stencil_reach = 1"
    ):
        blockamr.wall_grad_ghost_cell(
            out,
            phi,
            ct,
            g,
            data,
            _constant(0.5),
            geom,
            0.0,
            1.0,
            1,
            blockamr.WallMode.Overwrite,
            1.0,
        )


def test_each_disagreement_between_the_grad_pairs_arguments_is_refused_by_name(blockamr_session):
    """**F-12** — guard 0 and `Maker::validate` (B30a-R's S-5), together.

    Each of these is a silently wrong answer rather than a crash in a release
    build, and each is named by the entry point:

    * `WallMode.Assemble` — declared and not implemented (S6);
    * `out is phi` — a row would read cells another row had already written;
    * a mismatched `BoxArray` — the sweep pairs fabs by `MFIter` local index;
    * a Robin table narrower than the field — an out-of-bounds `gammaAt`;
    * method data preprocessed on other grids — invisible to the frame.

    There is deliberately **no** face-flux clause here and no sixth case: that is
    `div`'s, and its absence is the argument contract's other half (this pair
    takes exactly the canonical twelve).
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    out = _out(ba, dm)

    def call(out_mf, phi_mf, ct_fab, data_obj, robin, ncomp=1, mode=None):
        blockamr.wall_grad_ghost_cell(
            out_mf,
            phi_mf,
            ct_fab,
            g,
            data_obj,
            robin,
            geom,
            0.0,
            1.0,
            ncomp,
            blockamr.WallMode.Overwrite if mode is None else mode,
            1.0,
        )

    with pytest.raises(RuntimeError, match=r"wall_grad_ghost_cell: WallMode.Assemble"):
        call(out, phi, ct, data, _constant(0.5), mode=blockamr.WallMode.Assemble)
    assert all(v == SENTINEL for v in _readback(out).values())

    with pytest.raises(RuntimeError, match=r"wall_grad_ghost_cell: .*different MultiFabs"):
        call(phi, phi, ct, data, _constant(0.5))

    ba8 = blockamr.BoxArray(blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1]))
    ba8.max_size(8)
    dm8 = blockamr.DistributionMapping(ba8)
    with pytest.raises(RuntimeError, match=r"wall_grad_ghost_cell: out, phi and the"):
        call(_out(ba8, dm8), phi, ct, data, _constant(0.5))

    phi2 = _field(ba, dm, ncomp=2)
    with pytest.raises(RuntimeError, match=r"the field has 2 but the table has 1"):
        call(_out(ba, dm, ncomp=2), phi2, ct, data, _constant(0.5), ncomp=2)

    g8 = _level(ONE_BODY, max_size=8)[0].ibm.geometry_fab(0, ngrow=1)
    ct8 = blockamr.CellTypeFab(ba8, dm8, 1)
    blockamr.classify_default(ct8, g8, geom)
    other = _ghost_data(ct8, g8, geom, ONE_BODY)
    with pytest.raises(RuntimeError, match=r"wall_grad_ghost_cell: the ghostCell data"):
        call(out, phi, ct, other, _constant(0.5))


def test_the_grad_row_hook_refuses_a_robin_table_wider_than_one_component(blockamr_session):
    """**F-13** — v1's `ncomp > 1` refusal, at the row hook (api §9, Q56(c)).

    The row hook has no `ncomp` argument, so it reads the Robin table's own
    width; `Maker::validate` reads the sweep's. The **sentence is the same** in
    both, deliberately, so the two surfaces cannot drift apart — and it is v1's
    own, with the entry point in place of the field name the compiled pair does
    not have.

    Why the refusal exists at all is the row format and not a limitation of this
    transcription: a band row applies **one** coefficient list to every component
    (row-contract §2) while the gradient's component `n` is the difference along
    axis `n`. `test_ibm_grad_ghost_cell.py` binds this sentence to v1's own raise
    end to end; here it is the hook's surface, and that `ncomp == 1` is *not*
    refused — the guard is a refusal, not a wall.
    """
    _mesh, g, ct, data, geom, ba, dm = _grad_case(ONE_BODY)
    phi = _field(ba, dm)
    cell = _a_wall_cell(ct, phi)

    with pytest.raises(
        RuntimeError,
        match=r"_wall_row_grad_ghost_cell: grad x ghostCell needs a one-component field",
    ) as excinfo:
        _grad_row(ct, g, data, _constant(0.5, ncomp=2), geom, cell)
    assert "ncomp = 2" in str(excinfo.value)
    assert "the difference along axis n" in str(excinfo.value)

    entries, _c = _grad_row(ct, g, data, _constant(0.5), geom, cell)
    assert entries[0][0] == cell
