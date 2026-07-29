# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``laplacian x ghostCell`` — the first real ``(operator, method)`` pair (B32).

``src/bindings/blockAMR/schemes/boundary/laplacian_ghost_cell.cpp``: the
compiled peer of v1's
:func:`blockamr.schemes.boundary.ghost_cell._closed_flux_rows`.

**The bar is v1↔v2 row parity, BITWISE.** For every ``WALL`` cell of eight
configurations — 2 560 rows — the compiled row's *ordered* ``(index, a)``
sequence and its constant equal v1's, compared through raw ``int64`` views.
``assert_array_equal`` on f64 cannot see ``-0.0``, and ``grad_linear`` is
``-0.0`` on every Neumann row, so the stricter comparison is not decoration.
review.md §4 Q29(d) refuses the ULP fallback: a residual mismatch stays red and
is escalated.

**Why this bar and not "the rungs, green on v2"** (review.md §4 Q49(a)). When
this file was written there was no driver seam to run rungs through: the v1
registry key ``("laplacian", "ghostCell")`` was taken, and the driver still
called ``rows()`` and uploaded a ``BandTable``. B36 flipped it — the rungs run
on the pair now — and this bar is *kept*, because row parity is strictly
stronger per cell than the rungs, which are aggregate and tolerance-based. It is
what pins the port's correctness after the seam it was written ahead of.

**Why eight configurations and not the two rung geometries** (Q35, discharged by
measurement before the build). The falsification matrix below was measured
against v1's own row builder: the ``donor-assoc`` defect — ``scale * (gl * w)``
for ``(scale * gl) * w`` — is **invisible** on the cylinder-Dirichlet and
cylinder-Neumann geometries, and the ``datum-linear`` defect (an S2 violation)
is invisible on K1, whose datum is ``0.0``. A suite built from the obvious
choice would have shipped two untested hazards.

**The oracle is v1's rows, RECORDED** (:mod:`test.blockamr.v1_golden`). They
were produced by v1's own production code — ``_context`` and
``_closed_flux_rows``, with the term built through ``Equation(exp.laplacian(1.0,
T))`` — on the last tree that had it, and checked in as bits when the band was
deleted. Nothing here re-derives them, which is the strongest form of the oracle
discipline the file always claimed: a re-implementation can drift towards the
code under test, a file of numbers cannot. The mutants are applied to a numpy
model of the functor and never to the oracle — and the model itself is pinned to
the *compiled* row by the parity rows first, which is what makes the matrix a
statement about the shipped code.

**Where the other rows live.** Per-cell functor conformance (S2, S3, Q34, the
error surface) is ``test_ibm_wall_functors.py``, which is where the shipped
frame file says B32's rows belong and which already has the ``RecordSink``
readback shape. None of the four O3 fence files is touched.
"""

import re
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, exp
from blockamr.dsl.solve import _resolve_schemes
from blockamr.field import CellField
from blockamr.ibm.body import Cylinder, Plane
from blockamr.ibm.bc import FixedGradient, FixedValue, Mixed
from blockamr.ibm.classify import _patches, box_grids
from blockamr.ibm.ghost_cell import GhostCell
from blockamr.mesh import Mesh
from blockamr.schemes.boundary import BOUNDARY_SCHEMES

from .v1_golden import load as _load_v1

#: v1's row width — ``self + 6 face neighbours + 8 image donors``. Declared here
#: since the band tree went: it is a property of the RECORDED rows, and
#: ``_v1_row`` slices them with it.
STRIDE = 15


def _wall_row(*args):
    """The underscore-private row hook (api §4). ``from ._blockamr import *``
    skips underscore names, so it is reached on the extension module itself.

    Resolved per call rather than at import: the falsification matrix, its
    control and the FP-flag conformance row are pure numpy and text, and this
    keeps them runnable *before* a rebuild — which is where a defect in the
    plan's own matrix is cheapest to find.
    """
    return blockamr._blockamr._wall_row_laplacian_ghost_cell(*args)


N = 16
SOLID = int(blockamr.CellType.SOLID)
WALL = int(blockamr.CellType.WALL)

#: The canonical twelve (design §4.4), in order. Q39, ruled at B32: a
#: *registered* pair carries all twelve, no defaults, ``t`` included even for a
#: steady datum, because B36's driver calls every pair by keyword from one site.
CANONICAL_TWELVE = (
    "out",
    "phi",
    "cell_type",
    "geom_ibm",
    "method_data",
    "robin",
    "geom",
    "t",
    "coeff",
    "ncomp",
    "mode",
    "constant_scale",
)

#: The unit cube, and K8's deliberately non-dyadic grid. B31-R's Q35 lesson: a
#: power-of-two ``dx`` with ``prob_lo = 0`` only shifts exponents, so most of
#: the arithmetic under test is exact there and cannot tell a correct
#: transcription from a reassociated one.
UNIT_LO, UNIT_HI = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
SKEW_LO, SKEW_HI = (-0.37, 0.11, 0.23), (0.53, 0.81, 1.53)

TILTED = tuple(np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0]))

#: ``(bodies, ibm_bc, prob_lo, prob_hi)`` per configuration.
CONFIGS = {
    # the rung-5 geometry: weights are exactly 0 or 1, and the datum is 0.0 —
    # which is exactly why K1 alone cannot see an S2 violation.
    "K1-plane-x-dirichlet": (
        {"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))},
        {"wall": FixedValue(0.0)},
        UNIT_LO,
        UNIT_HI,
    ),
    # a genuine 3-D interpolation: the best small-grid discriminator for H-4.
    "K2-plane-123-dirichlet": (
        {"wall": Plane(point=(0.5, 0.5, 0.5), normal=TILTED)},
        {"wall": FixedValue(0.3)},
        UNIT_LO,
        UNIT_HI,
    ),
    # the rung-3 geometry.
    "K3-cylinder-dirichlet": (
        {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)},
        {"cyl": FixedValue(0.3)},
        UNIT_LO,
        UNIT_HI,
    ),
    # the beta != 0 arm: `grad_linear` is -0.0 on every row of it.
    "K4-cylinder-neumann": (
        {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)},
        {"cyl": FixedGradient(0.2)},
        UNIT_LO,
        UNIT_HI,
    ),
    # both closure terms non-zero.
    "K5-cylinder-mixed": (
        {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)},
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
    ),
    # two bodies, two BCs: every WALL cell has a face neighbour on the other
    # patch, which is the Q34 discriminator.
    "K6-two-cylinders-two-patches": (
        {
            "a": Cylinder(centre=(0.28, 0.5), radius=0.12, axis=2),
            "b": Cylinder(centre=(0.72, 0.5), radius=0.12, axis=2),
        },
        {"a": FixedValue(0.3), "b": FixedGradient(0.2)},
        UNIT_LO,
        UNIT_HI,
    ),
    # non-dyadic normals and weights on a dyadic grid.
    "K7-nondyadic-cylinder": (
        {"cyl": Cylinder(centre=(0.37, 0.4123), radius=0.1731, axis=2)},
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
    ),
    # the load-bearing grid: prob_lo != 0 and no dx dyadic.
    "K8-nondyadic-grid": (
        {"cyl": Cylinder(centre=(0.13, 0.31), radius=0.1731, axis=2)},
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        SKEW_LO,
        SKEW_HI,
    ),
}

#: WALL rows per configuration — measured, and asserted, so a geometry change
#: that emptied a configuration cannot make its parity row vacuously green.
NWALL = {
    "K1-plane-x-dirichlet": 256,
    "K2-plane-123-dirichlet": 256,
    "K3-cylinder-dirichlet": 320,
    "K4-cylinder-neumann": 320,
    "K5-cylinder-mixed": 320,
    "K6-two-cylinders-two-patches": 448,
    "K7-nondyadic-cylinder": 288,
    "K8-nondyadic-grid": 352,
}

#: The mutants, and the defect each one models.
MUTANTS = (
    "order-step",  # H-1, api §5.3's sketch: -1 first
    "order-axis",  # the axis loop reversed
    "h2",  # a wrong power / a saved divide
    "scale-sign",  # the flux-direction sign lost
    "scale-assoc",  # step*(n/dx) for (step*n)/dx -- a CONTROL, exact
    "donor-assoc",  # H-4: scale*(gl*w) for (scale*gl)*w
    "donor-index",  # the donor rank/index off by one
    "datum-linear",  # S2 violation: the datum through `linear`
    "normal-nb",  # Q34 trip: geometry read at a neighbour
    "arm-ungated",  # S3 violation: WallFrameProbe's unconditional arms
    "diag-perarm",  # H-3, api §5.3's sketch: the per-arm diagonal
)

#: Rows caught, per ``(configuration, mutant)``. Measured against v1's own row
#: builder before the build, and asserted here as an **exact** tuple: a mutant
#: that caught *more* rows than this fails too, because the matrix is a claim
#: about what each configuration can see and not a lower bound.
COVERAGE = {
    #                            ord-step ord-axis   h2  sign assoc donor-a  d-idx datum normal ungated diag
    "K1-plane-x-dirichlet": (256, 256, 256, 256, 0, 0, 256, 0, 256, 256, 256),
    "K2-plane-123-dirichlet": (171, 256, 256, 256, 0, 206, 256, 256, 256, 256, 256),
    "K3-cylinder-dirichlet": (320, 320, 320, 320, 0, 0, 320, 320, 320, 320, 320),
    "K4-cylinder-neumann": (320, 320, 320, 320, 0, 0, 320, 320, 320, 320, 320),
    "K5-cylinder-mixed": (320, 320, 320, 320, 0, 192, 320, 320, 320, 320, 320),
    "K6-two-cylinders-two-patches": (448, 448, 448, 448, 0, 128, 448, 448, 448, 448, 448),
    "K7-nondyadic-cylinder": (288, 288, 288, 288, 0, 144, 288, 288, 288, 288, 288),
    "K8-nondyadic-grid": (352, 352, 352, 352, 0, 303, 352, 352, 352, 352, 352),
}


# ---------------------------------------------------------------------------
# the level, the term, and the two sides of the comparison
# ---------------------------------------------------------------------------


#: Built levels and v1 contexts, memoised for the session. Every one of them is
#: read-only after construction (the rows are rebuilt from the context, never
#: mutated), and building a level plus v1's preprocessing eight times over is
#: most of this file's runtime.
_CASES = {}
_V1 = {}


@pytest.fixture(scope="module", autouse=True)
def _release_the_memoised_levels():
    """Drop the caches while AMReX is still up (B33's pattern, copied).

    The memoised levels own device memory. Left in module globals they are torn
    down at *interpreter* exit, which is after ``blockamr_session`` has
    finalized AMReX, and freeing a device allocation into a destroyed CUDA
    context aborts (``CUDA error 709``). The sibling pair suites
    (``test_ibm_div_ghost_cell.py``, ``test_ibm_grad_ghost_cell.py``) have
    carried this finalizer since B33; this file is the one that did not, and it
    holds the same kind of cache.

    Fixture only: no assertion in this file changes by a bit.
    """
    yield
    _CASES.clear()
    _V1.clear()


def _case(name, max_size=None):
    """``(mesh, term, geom, ba, dm)`` — one configuration, v1 side resolved."""
    key = (name, max_size)
    if key not in _CASES:
        _CASES[key] = _build_case(name, max_size)
    return _CASES[key]


def _v1_side(name):
    """``(ctx, rows, arms)`` of one configuration, at ``coeff = 1.0``.

    **Recorded, not rebuilt** (see :mod:`test.blockamr.v1_golden`). v1's
    ``_context`` / ``_closed_flux_rows`` were deleted with the band; the rows
    they produced are checked in as bits, so every comparison below is against
    the same numbers it was against before, to the last one.
    """
    if name not in _V1:
        ctx, rows, arms, _extra = _load_v1("laplacian", name)
        _V1[name] = (ctx, rows, arms)
    return _V1[name]


def _build_case(name, max_size=None):
    bodies, ibm_bc, lo, hi = CONFIGS[name]
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    geom = blockamr.Geometry(box, blockamr.RealBox(list(lo), list(hi)), 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(N if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
    field = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc=ibm_bc)
    eqn = Equation(exp.laplacian(1.0, field))
    _resolve_schemes(eqn.explicit_terms, eqn.schemes)
    return mesh, eqn.explicit_terms[0], geom, ba, dm


def _v2(mesh, geom, ba, dm, ibm_bc, ngrow=1):
    """``(g, ct, data, robin)`` — the compiled geometry, marker, rows and BCs."""
    names, _bodies = _patches(mesh.bodies)
    g = mesh.ibm.geometry_fab(0, ngrow=ngrow)
    ct = blockamr.CellTypeFab(ba, dm, ngrow)
    blockamr.classify_default(ct, g, geom)
    data = blockamr.ghost_cell_preprocess(ct, g, geom, names)
    return g, ct, data, _robin(names, ibm_bc)


def _robin(names, ibm_bc, ncomp=1):
    """``RobinData`` from the *same* ``(alpha, beta, gamma)`` v1 reads.

    v1's ``_band_closure`` takes ``[ibm_bc[name].robin() for name in names]``
    with ``names = sorted(mesh.bodies)``; the patch index is the position in
    that list, which is what ``IbmGeometry.patch`` carries. Every datum in the
    acceptance set is constant, so the compiled ``gamma(t)`` is the
    ``Constant`` tag and no transcendental is on any path here.
    """
    npatch = len(names)
    alpha = np.zeros(npatch)
    beta = np.zeros(npatch)
    form = np.zeros((npatch, ncomp), dtype=np.int32)  # 0 == GAMMA_CONSTANT
    param = np.zeros((npatch, ncomp, 4), dtype=np.float64)
    for p, name in enumerate(names):
        a, b, gamma = ibm_bc[name].robin()
        alpha[p] = a
        beta[p] = b
        param[p, :, 0] = float(np.asarray(gamma, dtype=float).ravel()[0])
    return blockamr.RobinData(alpha, beta, form, param)


def _bits(value):
    """The raw ``int64`` of one f64. ``==`` on floats cannot see ``-0.0``, and
    ``grad_linear`` is ``-0.0`` on every Neumann row."""
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _v1_row(rows, r):
    """v1's row ``r`` as ``(ordered [(index, a)], c)``, in slot order.

    v1 keeps a **zero slot** at every wall arm's neighbour (the slot is
    allocated by ``_blank`` and never written), and the functor emits nothing
    there. Those, and only those, are dropped — the drop is on the bits being
    exactly ``0.0``, and a fluid arm's ``+1/dx^2`` can never be.
    """
    stencil, a = rows.stencil[r], rows.a[r]
    entries = [(tuple(int(v) for v in stencil[k]), float(a[k])) for k in range(STRIDE)]
    kept = [entries[0]] + [e for e in entries[1:7] if _bits(e[1]) != 0] + entries[7:]
    return kept, float(rows.c[r][0])


def _compiled_row(ct, g, data, robin, geom, cell, n=0, t=0.0):
    """The compiled row at one cell, in the same shape as :func:`_v1_row`."""
    entries, c = _wall_row(ct, g, data, robin, geom, t, *cell, n)
    return [((i, j, k), a) for i, j, k, a in entries], c


def _same(lhs, rhs):
    """Bitwise equality of two ``(entries, c)`` rows, entry sequence included."""
    (le, lc), (re, rc) = lhs, rhs
    if len(le) != len(re):
        return False, f"{len(le)} entries vs {len(re)}"
    for pos, ((li, la), (ri, ra)) in enumerate(zip(le, re)):
        if li != ri:
            return False, f"entry {pos}: index {li} vs {ri}"
        if _bits(la) != _bits(ra):
            return False, f"entry {pos} at {li}: {la!r} vs {ra!r} (raw {_bits(la)} vs {_bits(ra)})"
    if _bits(lc) != _bits(rc):
        return False, f"constant: {lc!r} vs {rc!r} (raw {_bits(lc)} vs {_bits(rc)})"
    return True, ""


# ---------------------------------------------------------------------------
# the numpy model of the functor, and its mutants
# ---------------------------------------------------------------------------


def _model_row(ctx, arms, r, mutant=None):
    """The v2 functor, in numpy: accumulate the six arms, then emit in v1's
    slot order. ``mutant`` injects exactly one defect.

    This is the object the falsification matrix is measured on. It is *not* an
    independent oracle — the parity rows pin it to the compiled row first — so
    a mutant caught here is a defect the compiled pair would also carry.
    """
    cell = tuple(int(v) for v in ctx.target[r])
    dx = ctx.dx
    steps = (-1, 1) if mutant == "order-step" else (1, -1)
    axes = (2, 1, 0) if mutant == "order-axis" else (0, 1, 2)

    diag = 0.0
    fluid_arms = []
    wdon = np.zeros(8)
    cacc = np.zeros(ctx.closure.grad_constant.shape[1])

    for d in axes:
        for step in steps:
            index, nb_fluid_all = arms[(d, step)]
            nb_fluid = bool(nb_fluid_all[r])
            neighbour = tuple(int(v) for v in index[r])
            if nb_fluid or mutant == "arm-ungated":
                inv = 1.0 / dx[d] ** 2
                if mutant == "h2":
                    inv = 1.0 / (dx[d] * dx[d] * dx[d])
                fluid_arms.append((neighbour, inv))
                if mutant == "diag-perarm":
                    fluid_arms.append((cell, -inv))
                else:
                    diag -= inv
            if not nb_fluid:
                normal = ctx.normal[r, d]
                if mutant == "normal-nb":  # Q34 trip: not this cell's normal
                    normal = -normal
                scale = step * normal / dx[d]
                if mutant == "scale-sign":
                    scale = -scale
                if mutant == "scale-assoc":
                    scale = step * (normal / dx[d])
                gl = ctx.closure.grad_linear[r]
                gc = ctx.closure.grad_constant[r]
                sg = scale * gl
                for q in range(8):
                    if mutant == "donor-assoc":
                        wdon[q] += scale * (gl * ctx.weight[r, q])
                    else:
                        wdon[q] += sg * ctx.weight[r, q]
                if mutant == "datum-linear":  # S2 violation
                    diag += float(np.sum(scale * gc))
                else:
                    cacc += scale * gc

    entries = ([] if mutant == "diag-perarm" else [(cell, diag)]) + fluid_arms
    for q in range(8):
        donor = tuple(int(v) for v in ctx.donor[r, q])
        if mutant == "donor-index":
            donor = (donor[0] + 1, donor[1], donor[2])
        entries.append((donor, float(wdon[q])))
    return entries, float(cacc[0])


def _caught(ctx, rows, arms, mutant):
    """WALL rows on which ``mutant`` differs from v1."""
    n = 0
    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        ok, _why = _same(_model_row(ctx, arms, r, mutant), _v1_row(rows, r))
        if not ok:
            n += 1
    return n


# ===========================================================================
# 1. P-1..P-8 — v1 <-> v2 row parity, BITWISE, on all eight configurations
# ===========================================================================


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_compiled_row_is_v1s_row_bitwise(blockamr_session, name):
    """**The acceptance bar** (review.md §4 Q49(a), item ii).

    For every ``WALL`` cell: the ordered ``(index, a)`` sequence the compiled
    functor emits, and the constant it accumulates, equal v1's
    ``_closed_flux_rows`` row on the raw bits. No tolerance — a residual
    difference is a bug, not noise: there is no libm in the closure chain,
    contraction is pinned off by the per-file flags, and the numpy model was
    proven bitwise against v1 before the build.

    The **model** is pinned here too, in the same statement. That is the link
    that makes the falsification matrix below a claim about the shipped code
    rather than about a numpy script beside it.
    """
    mesh, _term, geom, ba, dm = _case(name)
    _bodies, ibm_bc, _lo, _hi = CONFIGS[name]
    ctx, rows, arms = _v1_side(name)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)

    nwall = int(ctx.at_wall.sum())
    assert nwall == NWALL[name], f"{name}: {nwall} wall rows, expected {NWALL[name]}"
    assert data.nrows == nwall, f"the compiled data has {data.nrows} rows, v1's band has {nwall}"

    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        cell = tuple(int(v) for v in ctx.target[r])
        want = _v1_row(rows, r)
        ok, why = _same(_compiled_row(ct, g, data, robin, geom, cell), want)
        assert ok, f"{name}: compiled row at {cell} differs from v1 — {why}"
        ok, why = _same(_model_row(ctx, arms, r), want)
        assert ok, f"{name}: the numpy model at {cell} differs from v1 — {why}"


# ===========================================================================
# 2. P-9, P-10 — the falsification matrix, and its control (Q35)
# ===========================================================================


def test_the_falsification_matrix_is_reproduced_exactly(blockamr_session):
    """**Q35, permanently in-suite.** Eleven defects, counted row by row, over
    all eight configurations and all 2 560 wall rows.

    Asserted as an exact tuple rather than "> 0" (B30b-R's S-6 shape):
    over-coverage fails too, because the matrix is the record of what each
    configuration *can* see. Two entries in it are the reason the suite has
    eight configurations and not two:

    * ``donor-assoc`` (H-4) is **invisible** on K1, K3 and K4;
    * ``datum-linear`` (an S2 violation) is **invisible** on K1, whose datum is
      ``0.0`` — K1 alone would make the S2 claim vacuous.

    Every defect that is caught is caught by at least four configurations, so
    no row of this suite is load-bearing alone.
    """
    measured = {}
    total_wall = 0
    for name in CONFIGS:
        ctx, rows, arms = _v1_side(name)
        total_wall += int(ctx.at_wall.sum())
        assert _caught(ctx, rows, arms, None) == 0, f"{name}: the baseline model is not v1's rows"
        measured[name] = tuple(_caught(ctx, rows, arms, mutant) for mutant in MUTANTS)

    assert total_wall == 2560, f"the acceptance set is {total_wall} wall rows, not 2 560"
    assert measured == COVERAGE, (
        "the falsification matrix moved\n  mutants  "
        + str(MUTANTS)
        + "\n"
        + "\n".join(
            f"  {name}: measured {measured[name]} recorded {COVERAGE[name]}"
            for name in CONFIGS
            if measured[name] != COVERAGE[name]
        )
    )


def test_the_scale_association_control_moves_no_bit_anywhere(blockamr_session):
    """A **control**, and it is labelled one: ``step*(n/dx)`` for ``(step*n)/dx``
    changes nothing, on any configuration, because ``step`` is exactly ``+-1.0``
    and multiplication by a power of two is exact.

    It is in the suite so the matrix is not a list of things that happen to
    differ. A change that made this row red would mean the arithmetic around
    ``scale`` had stopped being exact — which is information, and the opposite
    of what the caught mutants report.
    """
    caught = {}
    for name in CONFIGS:
        ctx, rows, arms = _v1_side(name)
        caught[name] = _caught(ctx, rows, arms, "scale-assoc")
    assert caught == dict.fromkeys(CONFIGS, 0)


# ===========================================================================
# 3. P-11, P-12 — the pair through the frame, over real fabs
# ===========================================================================


def PHI(i, j, k):
    """The sweep rows' field, as a function of the global index.

    **Quadratic on purpose.** On a linear field the interior laplacian is
    exactly zero everywhere, and then "v2 did not write this cell" and "v2 wrote
    zero here" are the same measurement — which would make both the FLUID
    comparison and the ``SOLID`` exclusion below vacuously green. The ``-0.3 k``
    term keeps the values off the dyadic grid so the row arithmetic rounds.
    """
    return 0.375 * i * i - 0.1875 * j * j + 0.0625 * k * k + 0.125 * i * j - 0.3 * k + 1.0


def _phi(ba, dm, ncomp=1, ngrow=1):
    """:func:`PHI` at every index of the grown box, ghosts included."""
    mf = blockamr.MultiFab(ba, dm, ncomp, ngrow)
    for mfi in blockamr.MFIterator(mf):
        vb = mfi.valid_box()
        lo = tuple(v - ngrow for v in vb.small_end())
        hi = tuple(v + ngrow for v in vb.big_end())
        idx = np.meshgrid(
            *(np.arange(lo[d], hi[d] + 1, dtype=float) for d in range(3)), indexing="ij"
        )
        base = PHI(idx[0], idx[1], idx[2])
        mf.copy_grown_from(mfi, np.asfortranarray(np.stack([base] * ncomp, axis=-1)))
    return mf


def _readback(mf):
    out = {}
    for mfi in blockamr.MFIterator(mf):
        lo = tuple(mfi.valid_box().small_end())
        block = mf.copy_to_host(mfi)
        for local in np.ndindex(block.shape):
            out[tuple(lo[d] + local[d] for d in range(3)) + (local[3],)] = block[local]
    return out


def _markers(ct, mf):
    out = {}
    for mfi in blockamr.MFIterator(mf):
        lo = tuple(mfi.valid_box().small_end())
        block = blockamr._blockamr._cell_type_numpy(ct, mfi)
        for local in np.ndindex(block.shape):
            out[tuple(lo[d] + local[d] for d in range(3))] = int(block[local])
    return out


def _dot(row_entries, constant, constant_scale, fused):
    """The sink's own accumulation, in the sink's own order.

    ``constant`` first (the functor emits ``sink.constant`` before any
    ``linear``), then the entries in emission order. ``fused`` selects ONE
    rounding per term instead of two — the exact emulation of a compiler that
    contracted ``acc += a * phi`` into an ``fma``, done in ``Fraction`` so it is
    a statement about IEEE arithmetic and not about this machine.
    """
    acc = 0.0 + constant_scale * constant
    for index, a in row_entries:
        x = PHI(*index)
        acc = float(Fraction(a) * Fraction(x) + Fraction(acc)) if fused else acc + a * x
    return acc


def test_the_sweep_is_the_pairs_own_row_and_v1s_residual_is_its_consumers_fma(
    blockamr_session,
):
    """**The pair through the frame** (Q49(a), item iii) — and the one place
    this session could not deliver "bitwise" as written, with the reason
    measured rather than argued.

    Interior sweep plus wall sweep, over **eight boxes** so the row map's
    cross-box concatenation is what is being exercised, against the interior
    sweep alone and against v1's RECORDED rows. The v1 half used to be v1's own
    interior sweep plus ``apply_band_rows``; that kernel is deleted with the
    band, and what it computed — ``_dot(row, fused=True)``, cell for cell, on
    this exact configuration — was measured against it before the deletion and
    is what the residual below is now counted from.

    What is measured:

    * **FLUID** — bitwise the interior sweep's own value, every cell: the wall
      sweep writes no FLUID cell at all. ``PHI`` is quadratic so this is not the
      equality of two zeros.
    * **WALL** — v2's output is bitwise ``_dot(row, fused=False)``: 320 of 320.
      The rows are identical (the parity rows above), so the entire residual is
      the two consumers' floating-point contraction and nothing else —
      ``band_table.cpp``'s ``acc += t.a[k] * src(...)`` was in a TU with no
      per-file FP flags and nvcc's default ``--fmad=true`` fuses it (PTX: 14
      ``fma.rn.f64`` attributed to that line), while ``ApplySink::linear`` is
      inlined into this pair's ``--fmad=false`` TU and does not (PTX: zero).
      **The flag that buys row parity is the flag that costs sweep parity**, and
      the number of WALL cells where the two land differently is pinned exactly
      (103 of 320, max ``|Δbits|`` 1025 — B32-R's corrected measurement) so the
      finding can neither quietly heal nor quietly grow.
    * **SOLID** — excluded. That is **OPEN-C** (review.md §4 Q49(b)), which B32
      records rather than resolves: v1's ``band(1)`` is ``depth <= 1`` and so
      carries every solid cell as an ``nnz = 0, c = 0`` row, making v1's first
      ``Overwrite`` term write exactly ``0.0`` there, while v2's frame returns
      before the sink at ``m != WALL``. The exclusion is asserted to be
      *load-bearing* — the two sides really do differ at SOLID cells — so it can
      never quietly become the whole comparison.
    """
    name = "K5-cylinder-mixed"
    mesh, _term, geom, ba, dm = _case(name, max_size=8)
    _bodies, ibm_bc, _lo, _hi = CONFIGS[name]
    phi = _phi(ba, dm)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)

    # v1's side, RECORDED: `apply_band_rows` and the rows that fed it went with
    # the band, so the comparison is against v1's rows and `_dot(fused=True)` —
    # the value v1's kernel was measured, cell for cell, to produce (the two
    # assertions in the loop below were both green against the live v1 sweep
    # before the deletion).
    ctx, rows, _arms_unused = _v1_side(name)

    # the interior sweep ALONE, so "the wall sweep writes no FLUID cell" and
    # "a SOLID cell keeps the interior value" are statements this file can make
    # without a second implementation of the wall.
    out_bulk = blockamr.MultiFab(ba, dm, 1, 0)
    out_bulk.set_val(0.0)
    blockamr.laplacian_acc(out_bulk, phi, geom, 1.0, 1)

    # v2: the same interior sweep, then the compiled pair, by keyword.
    out_v2 = blockamr.MultiFab(ba, dm, 1, 0)
    out_v2.set_val(0.0)
    blockamr.laplacian_acc(out_v2, phi, geom, 1.0, 1)
    blockamr.wall_laplacian_ghost_cell(
        out=out_v2,
        phi=phi,
        cell_type=ct,
        geom_ibm=g,
        method_data=data,
        robin=robin,
        geom=geom,
        t=0.0,
        coeff=1.0,
        ncomp=1,
        mode=blockamr.WallMode.Overwrite,
        constant_scale=1.0,
    )

    # v1's row, per WALL cell, in the emission order the sink uses. The parity
    # rows above make this the compiled row too, bitwise.
    by_cell = {
        tuple(int(v) for v in ctx.target[r]): _v1_row(rows, r)
        for r in range(ctx.nrows)
        if ctx.at_wall[r]
    }

    marker = _markers(ct, phi)
    got_bulk, got_v2 = _readback(out_bulk), _readback(out_v2)
    seen = {SOLID: 0, WALL: 0, "fluid": 0}
    solid_differ = wall_differ = 0
    for key, value in got_v2.items():
        m = marker[key[:3]]
        if m == SOLID:
            seen[SOLID] += 1
            # OPEN-C: v1 carried every solid cell as an `nnz = 0, c = 0` row and
            # so wrote exactly 0.0 there; v2's frame returns before the sink at
            # `m != WALL` and the interior sweep's value stands. Both halves are
            # asserted — the value IS the bulk's, bitwise, and it is not 0.0 —
            # so the exclusion stays load-bearing without v1's kernel.
            assert _bits(value) == _bits(got_bulk[key]), (
                f"a SOLID cell is not the interior sweep's value at {key}"
            )
            solid_differ += _bits(value) != _bits(0.0)
            continue
        if m != WALL:
            seen["fluid"] += 1
            assert _bits(value) == _bits(got_bulk[key]), (
                f"the wall sweep wrote a FLUID cell at {key}: {value!r} vs {got_bulk[key]!r}"
            )
            assert value != 0.0, f"vacuous: the interior laplacian is zero at {key}"
            continue
        seen[WALL] += 1
        entries, c = by_cell[key[:3]]
        assert _bits(value) == _bits(_dot(entries, c, 1.0, fused=False)), (
            f"v2's sweep at {key} is not its own row's plain dot product"
        )
        wall_differ += _bits(value) != _bits(_dot(entries, c, 1.0, fused=True))

    assert seen[WALL] == NWALL[name], seen
    assert seen["fluid"] > 0 and seen[SOLID] > 0, seen
    # Measured at B32 and pinned exactly (B32-R S-1). It is now the count of
    # WALL cells where the SAME row's fused and unfused dot products differ —
    # which is what the number always measured, `band_table.cpp` having been
    # shown (cell for cell, before its deletion) to compute the fused one.
    # A toolchain bump that moves it is a real observable change: re-measure,
    # re-pin, and record it in the ledger next to Q50.
    assert wall_differ == 103, (
        f"the contraction residual moved: {wall_differ}/320 WALL cells differ "
        "but the pinned measurement is 103 — this pair's FP flags changed; "
        "re-read Q50 first"
    )
    assert solid_differ > 0, "vacuous: OPEN-C is only a finding where the two sides differ"


def test_overwrite_then_add_composes_and_constant_scale_zero_drops_the_datum(blockamr_session):
    """**R2 and S4**, on a real pair rather than on the frame's probe.

    ``Overwrite`` then ``Add`` doubles the wall cells exactly (the composition
    rule: the first term of an equation writes, every later one adds), and
    ``constant_scale = 0`` drops **exactly** the BC datum — the Krylov matvec of
    an affine operator is the linear part alone, and a residual datum there is a
    wrong operator, not a small error. ``Mixed`` is used so the datum is not
    zero.

    The expected values are the pair's **own recorded row**, dot-produced in
    the sink's order — not ``affine - linear == c``, which is false in floating
    point for any row with cancellation and would have to be relaxed to a
    tolerance to pass. Recomputing both sweeps from the row instead asserts the
    stronger thing: ``constant_scale`` multiplies the constant term and touches
    nothing else, exactly.
    """
    name = "K5-cylinder-mixed"
    mesh, _term, geom, ba, dm = _case(name)
    _bodies, ibm_bc, _lo, _hi = CONFIGS[name]
    phi = _phi(ba, dm)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)
    marker = _markers(ct, phi)

    def sweep(mode, constant_scale):
        out = blockamr.MultiFab(ba, dm, 1, 0)
        out.set_val(0.0)
        blockamr.wall_laplacian_ghost_cell(
            out, phi, ct, g, data, robin, geom, 0.0, 1.0, 1, mode, constant_scale
        )
        return out

    affine = sweep(blockamr.WallMode.Overwrite, 1.0)
    first = _readback(affine)
    blockamr.wall_laplacian_ghost_cell(
        affine, phi, ct, g, data, robin, geom, 0.0, 1.0, 1, blockamr.WallMode.Add, 1.0
    )
    doubled = _readback(affine)
    linear = _readback(sweep(blockamr.WallMode.Overwrite, 0.0))

    moved = 0
    for key, value in first.items():
        if marker[key[:3]] != WALL:
            continue
        moved += 1
        assert doubled[key] == 2.0 * value, f"Add did not compose at {key}"
        entries, c = _compiled_row(ct, g, data, robin, geom, key[:3], key[3])
        assert c != 0.0, f"vacuous: the datum is zero at {key}"
        assert _bits(value) == _bits(_dot(entries, c, 1.0, fused=False)), key
        assert _bits(linear[key]) == _bits(_dot(entries, c, 0.0, fused=False)), key
    assert moved == NWALL[name]


# ===========================================================================
# 4. P-13..P-16 — the contracts the next three pairs inherit
# ===========================================================================


def _signature_args(fn):
    """``[(name, has_default), ...]`` from nanobind's signature in ``__doc__``."""
    head = fn.__doc__.splitlines()[0]
    inside = head[head.index("(") + 1 : head.rindex(")")]
    args, depth, current = [], 0, ""
    for ch in inside:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current)
    return [(a.split(":")[0].strip(), "=" in a) for a in args]


def test_every_registered_wall_pair_carries_the_canonical_twelve(blockamr_session):
    """**Q39**, ruled at B32 and enforced mechanically rather than by review.

    A ``wall_<operator>_<method>`` binding carries all twelve arguments of
    design §4.4, in that order, with no defaults — ``t`` included, even though
    ``laplacian x ghostCell`` has no time-dependent datum in scope. B36's driver
    calls every pair by keyword from **one** call site; a pair that dropped an
    argument would make that site pair-specific, which is exactly the coupling
    the registry exists to remove. Q29(f)'s "the twelve is a minimum" is
    unaffected: B33 may append ``div``'s face fluxes after the twelfth.

    Underscore-private hooks are exempt by construction — they are never
    registered, never resolved and never called by a driver — and the row
    asserts that exemption is real by checking the test hook is not exported.
    """
    module = blockamr._blockamr
    pairs = [n for n in dir(module) if re.fullmatch(r"wall_[a-z_]+", n)]
    assert "wall_laplacian_ghost_cell" in pairs, dir(module)

    for name in pairs:
        args = _signature_args(getattr(module, name))
        assert tuple(a for a, _d in args)[:12] == CANONICAL_TWELVE, f"{name}: {args}"
        assert not any(d for _a, d in args[:12]), f"{name}: an argument of the twelve is defaulted"

    assert hasattr(module, "_wall_row_laplacian_ghost_cell")
    assert not hasattr(blockamr, "_wall_row_laplacian_ghost_cell"), (
        "the row hook is underscore-private and must not reach the package namespace"
    )


def test_the_v1_scheme_names_the_compiled_pair(blockamr_session):
    """The seam B36 flipped (review.md §4 Q49(g)).

    ``register`` raises on a second class for a taken key, and O4 forbids
    removing v1's, so the declaration landed **additively** on the existing
    ``GhostCellLaplacian``: ``rows()`` is untouched and nothing is deregistered.
    B36 made ``WallEvaluation.apply`` call this kernel instead of ``rows()``;
    ``rows()`` stays as the oracle the parity rows above compare against.
    """
    scheme_cls = BOUNDARY_SCHEMES[("laplacian", "ghostCell")]
    kernel = scheme_cls(interior_scheme=None).build_cpp_kernel()
    assert kernel.name == "wall_laplacian_ghost_cell"
    assert hasattr(blockamr, kernel.name)
    assert callable(getattr(blockamr, kernel.name))


# --- Q36: the per-file FP-flag list, mechanically ---------------------------

_BINDINGS = Path(__file__).resolve().parents[2] / "src" / "bindings" / "blockAMR"
#: The two headers whose arithmetic is pinned bitwise against numpy and which
#: are therefore inlined **with the including TU's flags**.
_PINNED_HEADERS = ("schemes/boundary/robin.H", "ibm/ghost_cell.H")


def _flagged_sources():
    """The ``.cpp`` list of the FP-flags ``set_source_files_properties`` call."""
    text = (_BINDINGS / "CMakeLists.txt").read_text()
    match = re.search(
        r"set_source_files_properties\(\s*(.*?)\s*PROPERTIES\s+COMPILE_OPTIONS\s+"
        r'"\$\{_ghost_cell_fp_flags\}"\s*\)',
        text,
        re.S,
    )
    assert match, "the FP-flags set_source_files_properties call is not where this row looks"
    return {s for s in match.group(1).split() if s.endswith(".cpp")}


def _includes(path, seen):
    """Every local header ``path`` reaches, transitively, as repo-relative paths."""
    for line in path.read_text().splitlines():
        m = re.match(r'\s*#include\s+"([^"]+)"', line)
        if not m:
            continue
        target = (path.parent / m.group(1)).resolve()
        if not target.exists():
            continue
        key = target.relative_to(_BINDINGS).as_posix()
        if key in seen:
            continue
        seen.add(key)
        _includes(target, seen)
    return seen


def test_every_includer_of_a_pinned_header_takes_the_fp_flags(blockamr_session):
    """**Q36**, ruled at B32: the per-file opt-in stays, and this is what keeps
    the list from being one CMake edit away from silently incomplete.

    ``robin.H`` and ``ibm/ghost_cell.H`` are headers whose arithmetic is pinned
    bitwise against numpy, so it is inlined into whichever TU includes them,
    **with that TU's flags**. A TU that includes one and is not on the flags
    list ships contracted arithmetic — one rounding where numpy does two.

    The transitive closure is walked, not just the direct includes: a header
    that pulls one of the two in is the same hazard one level down. Pure text,
    no build, milliseconds — B33, B34 and B48 discover a missing flag as a red
    row here rather than as a last-bit mystery in their own parity suite.

    The converse is deliberately **not** asserted. ``ibm/ghost_cell.cpp`` is on
    the list for its own numpy peer and includes neither header of another TU;
    more to the point, ``schemes/stencil_kernels.cpp`` must never be on it
    (Q42), and a "flagged => includer" rule would be the directory posture this
    ruling refuted.
    """
    flagged = _flagged_sources()
    includers = set()
    for source in sorted(_BINDINGS.rglob("*.cpp")):
        closure = _includes(source, set())
        if any(h in closure for h in _PINNED_HEADERS):
            includers.add(source.relative_to(_BINDINGS).as_posix())

    assert len(includers) >= 3, f"vacuous: only {includers} reach a pinned header"
    assert "schemes/boundary/laplacian_ghost_cell.cpp" in includers
    assert includers <= flagged, (
        f"these translation units inline a bitwise-pinned header without the "
        f"contraction flags: {sorted(includers - flagged)}"
    )
    # ...and the TU that shipped without the flags by its own decision (Q36,
    # its GammaExpr::Harmonic is a real fma site) is not dragged onto the list.
    assert "schemes/boundary/wall_frame.cpp" not in flagged
    assert "schemes/stencil_kernels.cpp" not in flagged


def test_the_row_map_agrees_with_the_flat_row_order_on_eight_boxes(blockamr_session):
    """The map B32 added is B31's row order, published per cell.

    ``GhostCellData``'s four arrays are ordered per local box in ``MFIter``
    order and, within a box, by ``i`` then ``j`` then ``k`` — ``np.argwhere``'s
    C order. ``row_at`` must be the rank in exactly that sequence, and on
    **eight boxes** a wrong cross-box concatenation is a total mismatch rather
    than an off-by-one.
    """
    name = "K3-cylinder-dirichlet"
    mesh, _term, geom, ba, dm = _case(name, max_size=8)
    _bodies, ibm_bc, _lo, _hi = CONFIGS[name]
    _g, _ct, data, _robin_ = _v2(mesh, geom, ba, dm, ibm_bc)

    grids = box_grids(mesh, 0)
    assert len(grids) == 8
    expected = np.concatenate(
        [
            np.argwhere(geometry.depth == 1) + np.asarray(grid.lo)
            for grid, geometry in zip(grids, mesh.ibm.geometry(0))
        ]
    )
    assert len(expected) == data.nrows == NWALL[name]

    for rank, cell in enumerate(expected):
        got = data.row_at(*(int(v) for v in cell))
        assert got == rank, f"cell {tuple(cell)} has rank {got}, expected {rank}"

    v1 = GhostCell.preprocess(mesh, 0)
    assert v1.nrows == data.nrows
