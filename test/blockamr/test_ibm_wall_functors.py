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
