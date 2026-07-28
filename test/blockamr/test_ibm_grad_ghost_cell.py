# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``grad x ghostCell`` — the third real ``(operator, method)`` pair, and the
last of the ported three (B34).

``src/bindings/blockAMR/schemes/boundary/grad_ghost_cell.cpp``: the compiled peer
of v1's :func:`blockamr.schemes.boundary.ghost_cell._face_balance_rows` — **the
same function ``div`` calls** — as ``GhostCellGrad`` calls it, with
``axes=(0,)``, ``flux=1`` and ``weight_self=0.5``.

**The bar is v1↔v2 row parity, BITWISE.** For every ``WALL`` cell of ten
configurations — 3 136 rows — the compiled row's *ordered* ``(index, a)``
sequence and its constant equal v1's, compared through raw ``int64`` views.
``assert_array_equal`` on f64 cannot see ``-0.0``, and this operator's diagonal
is a signed zero on **every** row (H-10 below), so the stricter comparison is not
decoration. review.md §4 Q29(d) refuses the ULP fallback: a residual mismatch
stays red and is escalated.

**Why this bar and not "the scheme × method grid green on v2"** (review.md §4
Q56(a), inheriting Q49(a)/Q52(a)). There is no driver seam to run the grid
through yet — the v1 registry key ``("grad", "ghostCell")`` is taken,
``register`` raises on a second class, and ``BandEvaluation.apply`` still calls
``rows()`` and uploads a ``BandTable``; flipping the driver over is B36, at the
grid's own tolerance. Row parity is strictly stronger per cell.

**H-9, the defect this file exists to resist.** ``div``'s functor is
``for (int dd = 0; dd < 3; ++dd)``; ``grad``'s is one axis. The reason is the row
format and not an optimisation: a band row applies **one** coefficient list to
every component (``out(P, n) = Σ_k a_k φ(s_k, n)``, row-contract §2), while the
gradient's component ``n`` is the difference along axis ``n`` — a different
stencil per component, which the format does not have. v1 therefore expresses
only ``n = 0`` and *refuses* ``ncomp > 1``. Copying div's axis loop
(``axes-all``) is caught on 10 of 10 configurations and on all 3 136 rows; so is
the subtler half (``arms-six``), which keeps div's six-arm **emission** loop.

**H-10, the finding this session paid for.** For ``axes = (0,)`` the diagonal
accumulates exactly twice and the two contributions cancel: ``x + (−x)`` is
``+0.0`` in round-to-nearest for every finite ``x``. Measured: the diagonal is
bitwise ``+0.0`` on **3 136 of 3 136** rows — and it is still emitted, because
v1's row carries the slot. Its corollary for this harness is not optional: B32's
``_v1_row`` drops a slot whose bits are ``0``
(``test_ibm_laplacian_ghost_cell.py:335``), and for ``grad`` that rule would
discard **21 688 of 32 939 live entries and the diagonal of every single row**.
The canonicalisation here is therefore **structural** — keep slot 0, keep slots
1..6 whose stencil entry is not the target, keep 7..14 — which is exactly v1's
own liveness rule (``_blank`` points every unwritten slot at the target).

**Q54(a) in its extreme form.** 1 579 of these 3 136 rows (50.4 %) have **no wall
arm at all**: a ``WALL`` cell whose solid neighbours are all off the differencing
axis contributes exactly nothing through the closure — eight donor entries of
``+0.0`` and a constant of ``+0.0``. Two configurations (``G2``, ``G8``) are
100 % such rows, and on them **eleven of the twenty-one mutants are invisible**.
Every row here that wants "the datum reached the row" names a configuration
measured to contribute and guards it in the **aggregate**.

**The oracle is v1's production code**, imported read-only: ``_context`` and
``_face_balance_rows`` with ``axes=(0,)``, ``flux=ones``,
``weight_self=0.5*ones``, with the term built through
``Equation(exp.grad(T), schemes={"Grad": "central"})``, and cross-checked on
every configuration against the production call
``BOUNDARY_SCHEMES[("grad", "ghostCell")](term.scheme).rows(...)``. The mutants
are applied to a numpy model of the functor and never to the oracle — and the
model itself is pinned to the *compiled* row by the parity rows first.

**Where the other rows live.** Per-cell functor conformance (S2, S3, Q34, the
row shape, the pole, the error surface) is ``test_ibm_wall_functors.py``. None of
the four O3 fence files is touched, and ``test_ibm_laplacian_ghost_cell.py`` is
not edited: its Q39 row already asserts the canonical twelve as a *prefix* over
every registered pair, and its Q36 row already requires this TU on the FP-flags
list. Both strengthen for free.
"""

import re
import struct
from fractions import Fraction

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, exp
from blockamr.dsl.solve import _resolve_schemes
from blockamr.field import CellField
from blockamr.ibm.bc import FixedGradient, FixedValue, Mixed
from blockamr.ibm.body import Cylinder, Plane
from blockamr.ibm.classify import _patches
from blockamr.schemes.boundary import BOUNDARY_SCHEMES
from blockamr.schemes.boundary.ghost_cell import (
    STRIDE,
    _context,
    _face_balance_rows,
    _neighbour,
)

N = 16
SOLID = int(blockamr.CellType.SOLID)
WALL = int(blockamr.CellType.WALL)

#: The canonical twelve (design §4.4), in order — B32's Q39 contract. ``grad``
#: takes **exactly** these and appends nothing (review.md §4 Q56(a), R-1).
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

UNIT_LO, UNIT_HI = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
#: B31-R's Q35 lesson: a power-of-two ``dx`` with ``prob_lo = 0`` only shifts
#: exponents, so most of the arithmetic under test is exact there and cannot
#: tell a correct transcription from a reassociated one. For ``grad`` this is
#: sharper than for ``div``: ``nb_part`` is exactly ``±0.5/dx`` on a dyadic grid,
#: a power of two, so ``G10`` is the **only** configuration that can see H-4'.
SKEW_LO, SKEW_HI = (-0.37, 0.11, 0.23), (0.53, 0.81, 1.53)

TILTED = tuple(np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0]))

CYL_Z = {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)}

#: ``(bodies, ibm_bc, prob_lo, prob_hi)`` per configuration. What each one alone
#: contributes is in ``COVERAGE`` below and in the parity row's docstring.
CONFIGS = {
    # the differencing axis IS the normal: every row has a wall arm, the
    # trilinear weights are exactly 0/1, the datum is 0.0 and n̂_0 is 1.
    "G1-plane-x-dirichlet": (
        {"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))},
        {"wall": FixedValue(0.0)},
        UNIT_LO,
        UNIT_HI,
    ),
    # the wall is normal to y: NO row has a wall arm at all — the Q54(a) extreme.
    "G2-plane-y-dirichlet": (
        {"wall": Plane(point=(0.0, 0.5, 0.0), normal=(0.0, 1.0, 0.0))},
        {"wall": FixedValue(0.3)},
        UNIT_LO,
        UNIT_HI,
    ),
    # a genuine 3-D interpolation, and a 1/3 : 2/3 mix of the two row kinds.
    "G3-plane-123-dirichlet": (
        {"wall": Plane(point=(0.5, 0.5, 0.5), normal=TILTED)},
        {"wall": FixedValue(0.3)},
        UNIT_LO,
        UNIT_HI,
    ),
    # the canonical curved Dirichlet wall.
    "G4-cyl-z-dirichlet": (CYL_Z, {"cyl": FixedValue(0.3)}, UNIT_LO, UNIT_HI),
    # the beta != 0 arm (`grad_linear` is -0.0, `atLinear` is exactly 1.0).
    "G5-cyl-z-neumann": (CYL_Z, {"cyl": FixedGradient(0.2)}, UNIT_LO, UNIT_HI),
    # both closure terms non-zero.
    "G6-cyl-z-mixed": (
        CYL_Z,
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
    ),
    # two patches — the Q34 discriminator.
    "G7-two-cyl-two-patches": (
        {
            "a": Cylinder(centre=(0.28, 0.5), radius=0.12, axis=2),
            "b": Cylinder(centre=(0.72, 0.5), radius=0.12, axis=2),
        },
        {"a": FixedValue(0.3), "b": FixedGradient(0.2)},
        UNIT_LO,
        UNIT_HI,
    ),
    # a CURVED wall with n̂_0 identically 0 and no solid x-neighbour anywhere:
    # G2's property without G2's flatness.
    "G8-cyl-x-mixed": (
        {"cyl": Cylinder(centre=(0.0, 0.5, 0.5), radius=0.2, axis=0)},
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
    ),
    # non-dyadic normals and weights on a dyadic grid.
    "G9-nondyadic-cyl": (
        {"cyl": Cylinder(centre=(0.37, 0.4123), radius=0.1731, axis=2)},
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
    ),
    # the load-bearing grid, and the ONLY configuration that sees H-4'.
    "G10-nondyadic-grid": (
        {"cyl": Cylinder(centre=(0.13, 0.31), radius=0.1731, axis=2)},
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        SKEW_LO,
        SKEW_HI,
    ),
}

#: WALL rows per configuration — measured, and asserted, so a geometry change
#: that emptied a configuration cannot make its parity row vacuously green.
NWALL = {
    "G1-plane-x-dirichlet": 256,
    "G2-plane-y-dirichlet": 256,
    "G3-plane-123-dirichlet": 256,
    "G4-cyl-z-dirichlet": 320,
    "G5-cyl-z-neumann": 320,
    "G6-cyl-z-mixed": 320,
    "G7-two-cyl-two-patches": 448,
    "G8-cyl-x-mixed": 320,
    "G9-nondyadic-cyl": 288,
    "G10-nondyadic-grid": 352,
}

TOTAL_WALL = 3136

#: WALL rows with a **wall x-arm** — a solid neighbour on the differencing axis,
#: which is the only way the closure enters a ``grad`` row at all (Q54(a)).
#: ``0`` on G2 and G8 by construction: 1 579 of the 3 136 rows are pure interior
#: arithmetic plus eight ``+0.0`` donors.
ARM_ROWS = {
    "G1-plane-x-dirichlet": 256,
    "G2-plane-y-dirichlet": 0,
    "G3-plane-123-dirichlet": 85,
    "G4-cyl-z-dirichlet": 192,
    "G5-cyl-z-neumann": 192,
    "G6-cyl-z-mixed": 192,
    "G7-two-cyl-two-patches": 256,
    "G8-cyl-x-mixed": 0,
    "G9-nondyadic-cyl": 160,
    "G10-nondyadic-grid": 224,
}

TOTAL_ARM_ROWS = 1557

#: The mutants, and the defect each one models.
MUTANTS = (
    "axes-all",  # H-9: div's `for dd = 0..2` loop copied wholesale
    "order-step",  # H-1, api §5.3's sketch: -1 first
    "face-index",  # `face = 0` for `step = +1` -- a CONTROL, provably
    "scale-sign",  # the difference-direction sign lost
    "scale-assoc",  # step*(f/dx) for (step*f)/dx -- a CONTROL, exact
    "nb-complement",  # H-7: sc - slf for sc*(1 - w) -- a CONTROL
    "self-gated",  # H-3': the laplacian's fluid-arm gate on the diagonal
    "donor-assoc",  # H-4': nb*(lin*w) for (nb*lin)*w
    "donor-index",  # the donor rank/index off by one
    "datum-linear",  # S2 violation: the datum through `linear`
    "normal-nb",  # Q34 trip: geometry read at a neighbour
    "arm-ungated",  # S3 violation: an entry at a SOLID neighbour
    "dG-assoc",  # s_P + step*(dx*n) for s_P + (step*dx)*n -- a CONTROL
    "dG-sign",  # the ghost on the wrong side of the surface
    "dG-no-normal",  # the normal dropped from the extrapolation distance
    "at-wall-value",  # phi_w for phi(d_G) -- "halve the gradient at the wall"
    "arm-raw",  # H-6: the fluid-face coefficient emitted raw -- a CONTROL
    "diag-neg-zero",  # H-10: the diagonal emitted as -0.0
    "diag-dropped",  # H-10: the provably-zero diagonal not emitted at all
    "donors-dropped",  # a donor entry skipped when its coefficient is +0.0
    "arms-six",  # H-9's subtle half: div's SIX-arm emission loop kept
)

#: The five mutants that must move **nothing**, anywhere. Carried so the matrix
#: is not a list of things that happen to differ: three of them
#: (``face-index``, ``nb-complement``, ``arm-raw``) are simplifications a
#: reviewer would otherwise be asked to accept on taste, and they are now
#: permitted on arithmetic instead.
CONTROLS = ("face-index", "scale-assoc", "nb-complement", "dG-assoc", "arm-raw")

#: Rows caught, per ``(configuration, mutant)``. **Measured** against v1's own
#: row builder before the build, and asserted as an *exact* tuple: a mutant that
#: caught more rows than this fails too, because the matrix is a claim about what
#: each configuration can see and not a lower bound.
#:
#: Four entries are why the suite has ten configurations and not two:
#:
#: * ``donor-assoc`` (H-4') is caught by **exactly one** — ``G10``, 194 rows. On
#:   a dyadic grid ``nb_part`` is ``±0.5/dx``, exactly a power of two, so the
#:   association is exact and *every* dyadic configuration is blind. A suite
#:   built from "the plane and the cylinder on the unit cube" would have shipped
#:   the hazard B32 and B33 both had to get right.
#: * ``order-step`` (H-1) is invisible on ``G1``, where every row has exactly one
#:   fluid x-arm and no order is observable.
#: * ``datum-linear`` and ``dG-no-normal`` are invisible on ``G1`` (datum ``0.0``;
#:   ``n̂_0 = 1``) as well as on the two no-arm configurations.
#: * ``G2``/``G8`` make **eleven** of the twenty-one vacuous, and they are half
#:   the row population.
COVERAGE = {
    # axes-all ord-step face-idx sign s-assoc nb-cpl self-gt donor-a d-idx datum
    # normal ungated dG-assoc dG-sign dG-no-n at-wall arm-raw diag-neg0
    # diag-drop donors-drop arms-six
    "G1-plane-x-dirichlet": (
        256, 0, 0, 256, 0, 0, 256, 0, 256, 0, 256, 256, 0, 256, 0, 256, 0, 256, 256, 256, 256,
    ),
    "G2-plane-y-dirichlet": (
        256, 256, 0, 256, 0, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0, 0, 0, 256, 256, 256, 256,
    ),
    "G3-plane-123-dirichlet": (
        256, 171, 0, 256, 0, 0, 85, 0, 256, 85, 85, 85, 0, 85, 85, 82, 0, 256, 256, 174, 256,
    ),
    "G4-cyl-z-dirichlet": (
        320, 128, 0, 320, 0, 0, 192, 0, 320, 192, 192, 192, 0, 192, 192, 192, 0, 320, 320, 320, 320,
    ),
    "G5-cyl-z-neumann": (
        320, 128, 0, 320, 0, 0, 192, 0, 320, 192, 192, 192, 0, 192, 192, 192, 0, 320, 320, 320, 320,
    ),
    "G6-cyl-z-mixed": (
        320, 128, 0, 320, 0, 0, 192, 0, 320, 192, 192, 192, 0, 192, 192, 192, 0, 320, 320, 320, 320,
    ),
    "G7-two-cyl-two-patches": (
        448, 192, 0, 448, 0, 0, 256, 0, 448, 256, 256, 256, 0, 256, 256, 256, 0, 448, 448, 448, 448,
    ),
    "G8-cyl-x-mixed": (
        320, 320, 0, 320, 0, 0, 0, 0, 320, 0, 0, 0, 0, 0, 0, 0, 0, 320, 320, 320, 320,
    ),
    "G9-nondyadic-cyl": (
        288, 128, 0, 288, 0, 0, 160, 0, 288, 160, 160, 160, 0, 160, 160, 160, 0, 288, 288, 288, 288,
    ),
    "G10-nondyadic-grid": (
        352, 128, 0, 352, 0, 0, 224, 194, 352, 224, 224, 224, 0, 224, 224, 224, 0, 352, 352, 226,
        352,
    ),
}

#: The canonicalisation census, measured on the same 3 136 rows. Every row
#: carries at least one **live** entry whose coefficient is bitwise ``±0.0``, and
#: ``ZERO_ENTRIES`` of ``LIVE_ENTRIES`` such entries would be discarded by a
#: bits-are-zero drop rule — including every diagonal.
ZERO_ENTRY_ROWS = 3136
ZERO_ENTRIES = 21688
LIVE_ENTRIES = 32939

#: H-5, measured: rows that move between v1's ``coeff`` placement (folded into
#: ``scale`` before every product) and the frame's (``coeff * sink.value()``,
#: ``wall_apply.H:216``). Identical iff ``coeff`` is a power of two — and
#: ``exp.grad(field)`` exposes no ``coeff`` at all, so H-5 is **inert** on every
#: reachable v1 grad term. Recorded, not fixed: fixing it edits
#: ``wall_apply.H``'s contract for the laplacian and div pairs too.
COEFF_PLACEMENT = {
    "G1-plane-x-dirichlet": (0, 0, 0, 0, 0),
    "G6-cyl-z-mixed": (0, 0, 0, 64, 64),
    "G10-nondyadic-grid": (0, 0, 0, 352, 215),
}
COEFFS = (1.0, 2.0, 0.5, 3.0, 0.1)

#: ``(WALL cells where v1 and v2 land on different bits, max |delta bits|)`` per
#: sweep configuration — **measured post-build at B34 and pinned exactly**
#: (B32-R's S-1; both numbers of a pair come from the same run,
#: ``b34-q50-residual-attempt1.log``, per B32-R's I-1). The whole residual is the
#: two consumers' floating-point contraction and nothing else; see the row's
#: docstring. A toolchain bump that moves either number is a real observable
#: change: re-measure, re-pin, and record it in the ledger next to Q50.
WALL_RESIDUAL = {
    "G4-cyl-z-dirichlet": (34, 26),
    "G10-nondyadic-grid": (227, 380),
}


def _wall_row(*args):
    """The underscore-private row hook (api §4). ``from ._blockamr import *``
    skips underscore names, so it is reached on the extension module itself.

    Resolved per call rather than at import: the falsification matrix, its
    controls and the census are pure numpy, and this keeps them runnable
    *before* a rebuild — which is where a defect in the plan's own matrix is
    cheapest to find.
    """
    return blockamr._blockamr._wall_row_grad_ghost_cell(*args)


# ---------------------------------------------------------------------------
# the level, the term, and the two sides of the comparison
# ---------------------------------------------------------------------------

#: Built levels and v1 contexts, memoised for the session. Every one of them is
#: read-only after construction (the rows are rebuilt from the context, never
#: mutated), and building a level plus v1's preprocessing ten times over is most
#: of this file's runtime.
_CASES = {}
_V1 = {}


@pytest.fixture(scope="module", autouse=True)
def _release_the_memoised_levels():
    """Drop the caches while AMReX is still up (B33's pattern, copied).

    The memoised levels own device memory — ten meshes, each with a
    ``CellField``. Left in module globals they are torn down at *interpreter*
    exit, which is after ``blockamr_session`` has finalized AMReX, and freeing a
    device allocation into a destroyed CUDA context aborts (``CUDA error 709``,
    after a fully green run — this planner reproduced exactly that). A
    module-scoped finalizer runs before the session-scoped one, so this is
    simply the right place to let go.
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


def _build_case(name, max_size=None):
    from blockamr.mesh import Mesh

    bodies, ibm_bc, lo, hi = CONFIGS[name]
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    geom = blockamr.Geometry(box, blockamr.RealBox(list(lo), list(hi)), 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(N if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
    field = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc=ibm_bc)
    eqn = Equation(exp.grad(field), schemes={"Grad": "central"})
    _resolve_schemes(eqn.explicit_terms, eqn.schemes)
    return mesh, eqn.explicit_terms[0], geom, ba, dm


def _v1_rows(ctx, coeff=1.0):
    """v1's ``grad`` rows over ``ctx`` — ``axes=(0,)``, ``flux=1``, ``w=0.5``."""
    ones = np.ones((ctx.nrows, 3, 2))
    return _face_balance_rows(
        ctx,
        axes=(0,),
        flux=ones,
        weight_self=0.5 * ones,
        coeff=coeff,
        ncomp=1,
        stride=STRIDE,
    )


def _v1_side(name):
    """``(ctx, rows, arms, flux)`` of one configuration, ``coeff = 1.0``."""
    if name not in _V1:
        mesh, term, _geom, _ba, _dm = _case(name)
        width = int(getattr(term.scheme, "stencil_width", 1))
        assert width == 1, "grad has exactly one scheme and it is width 1 — Q56(a)"
        ctx = _context(term, mesh.ibm, 0, 1, 0.0, width)
        rows = _v1_rows(ctx, float(term.coeff))
        # The hand assembly above is the PRODUCTION call, checked rather than
        # assumed: `GhostCellGrad.rows` is what an evaluate reaches, and a
        # divergence between the two would make the whole oracle a private
        # re-derivation (oracle discipline, plan §8.7).
        produced = BOUNDARY_SCHEMES[("grad", "ghostCell")](term.scheme).rows(
            term, mesh.ibm, 0, 1, 0.0, width
        )
        np.testing.assert_array_equal(rows.a.view(np.int64), produced.a.view(np.int64))
        np.testing.assert_array_equal(rows.c.view(np.int64), produced.c.view(np.int64))
        np.testing.assert_array_equal(rows.stencil, produced.stencil)
        arms = {(d, s): _neighbour(ctx, d, s) for d in range(3) for s in (1, -1)}
        _V1[name] = (ctx, rows, arms, np.ones((ctx.nrows, 3, 2)))
    return _V1[name]


def _v2(mesh, geom, ba, dm, ibm_bc, ncomp=1, ngrow=1):
    """``(g, ct, data, robin)`` — the compiled geometry, marker, rows and BCs."""
    names, _bodies = _patches(mesh.bodies)
    g = mesh.ibm.geometry_fab(0, ngrow=ngrow)
    ct = blockamr.CellTypeFab(ba, dm, ngrow)
    blockamr.classify_default(ct, g, geom)
    data = blockamr.ghost_cell_preprocess(ct, g, geom, names)
    return g, ct, data, _robin(names, ibm_bc, ncomp)


def _robin(names, ibm_bc, ncomp=1):
    """``RobinData`` from the *same* ``(alpha, beta, gamma)`` v1 reads.

    v1's ``_band_closure`` takes ``[ibm_bc[name].robin() for name in names]``
    with ``names = sorted(mesh.bodies)``; the patch index is the position in that
    list, which is what ``IbmGeometry.patch`` carries. Every datum here is
    constant, so the compiled ``gamma(t)`` is the ``Constant`` tag and no
    transcendental is on any path.
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
    this operator's diagonal is a signed zero on every row (H-10)."""
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _v1_row(rows, r):
    """v1's row ``r`` as ``(ordered [(index, a)], c)``, in slot order.

    The canonicalisation is **structural, not value-based** — see the module
    docstring. v1 keeps a slot at every face whose neighbour is not fluid (the
    slot is allocated by ``_blank``, left pointing at the target and never
    written) and the functor emits nothing there; those, and only those, are
    dropped. For ``axes = (0,)`` that is slots 3..6 always, plus a solid x face.
    A ``bits == 0`` rule would additionally discard 21 688 *live* entries,
    including the diagonal of every single row.
    """
    stencil, a = rows.stencil[r], rows.a[r]
    target = tuple(int(v) for v in stencil[0])
    entries = [(tuple(int(v) for v in stencil[k]), float(a[k])) for k in range(STRIDE)]
    kept = [entries[0]] + [e for e in entries[1:7] if e[0] != target] + entries[7:]
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


def _model_row(ctx, arms, flux, r, mutant=None):
    """The v2 functor, in numpy: accumulate the two x faces, then emit in v1's
    slot order. ``mutant`` injects exactly one defect.

    This is the object the falsification matrix is measured on. It is *not* an
    independent oracle — the parity rows pin it to the compiled row first — so a
    mutant caught here is a defect the compiled pair would also carry.

    ``order-step`` changes the loop order in **both** passes, which is what a
    transcription following api §5.3's sketch would actually do: the C++ writes
    the same ``for s`` loop twice.
    """
    cell = tuple(int(v) for v in ctx.target[r])
    dx = ctx.dx
    s_P = ctx.sdf[r]
    steps = (-1, 1) if mutant == "order-step" else (1, -1)
    axes = (0, 1, 2) if mutant == "axes-all" else (0,)

    diag = 0.0
    cacc = 0.0
    wdon = np.zeros(8)
    arm = {}
    visited = []

    for d in axes:
        for step in steps:
            slot = 2 * d + (0 if step == 1 else 1)
            index, nb_fluid_all = arms[(d, step)]
            nb_fluid = bool(nb_fluid_all[r])
            neighbour = tuple(int(v) for v in index[r])

            # v1's `flux[:, d, face]` with `flux = ones`: the index selects
            # nothing, which is why `face-index` is a control here.
            face = 1 if step == 1 else 0
            if mutant == "face-index":
                face = 1 - face
            fl = flux[r, d, face]
            ws = 0.5
            sc = step * fl / dx[d]
            if mutant == "scale-assoc":
                sc = step * (fl / dx[d])
            if mutant == "scale-sign":
                sc = -sc
            slf = sc * ws
            nbp = (sc - slf) if mutant == "nb-complement" else sc * (1.0 - ws)

            # H-3': v1's mask on the diagonal is the ROW's fluidity, not the
            # face's. `self-gated` is the laplacian's gate copied here.
            if mutant != "self-gated" or nb_fluid:
                diag += slf

            if nb_fluid or mutant == "arm-ungated":
                if slot not in arm:
                    arm[slot] = (neighbour, 0.0)
                    visited.append(slot)
                # H-6: `0.0 + nbp` and not `nbp` -- a control for grad.
                value = nbp if mutant == "arm-raw" else arm[slot][1] + nbp
                arm[slot] = (neighbour, value)
            if not nb_fluid:
                normal = ctx.normal[r, d]
                if mutant == "normal-nb":  # Q34 trip: not this cell's normal
                    normal = -normal
                if mutant == "dG-assoc":
                    dG = s_P + step * (dx[d] * normal)
                elif mutant == "dG-sign":
                    dG = s_P - step * dx[d] * normal
                elif mutant == "dG-no-normal":
                    dG = s_P + step * dx[d]
                else:
                    dG = s_P + step * dx[d] * normal
                if mutant == "at-wall-value":
                    lin = ctx.closure.value_linear[r]
                    con = ctx.closure.value_constant[r, 0]
                else:
                    lin = ctx.closure.value_linear[r] + dG * ctx.closure.grad_linear[r]
                    con = ctx.closure.value_constant[r, 0] + dG * ctx.closure.grad_constant[r, 0]
                sg = nbp * lin  # H-4': once per face
                for q in range(8):
                    if mutant == "donor-assoc":
                        wdon[q] += nbp * (lin * ctx.weight[r, q])
                    else:
                        wdon[q] += sg * ctx.weight[r, q]
                if mutant == "datum-linear":  # S2 violation
                    diag += nbp * con
                else:
                    cacc += nbp * con

    entries = []
    # H-10: the diagonal is bitwise `+0.0` on every row, and it is emitted.
    if mutant == "diag-neg-zero":
        entries.append((cell, -diag))
    elif mutant != "diag-dropped":
        entries.append((cell, diag))

    if mutant == "arms-six":
        # H-9's subtle half: the accumulation restricted to axis 0 but div's SIX
        # -arm emission loop kept, so the ±y/±z neighbours are emitted with +0.0.
        for d in range(3):
            for step in (1, -1):
                slot = 2 * d + (0 if step == 1 else 1)
                index, nb_fluid_all = arms[(d, step)]
                if not bool(nb_fluid_all[r]):
                    continue
                entries.append(
                    arm[slot] if slot in arm else (tuple(int(v) for v in index[r]), 0.0)
                )
    else:
        order = visited if mutant == "order-step" else sorted(visited)
        entries.extend(arm[slot] for slot in order)

    for q in range(8):
        donor = tuple(int(v) for v in ctx.donor[r, q])
        if mutant == "donor-index":
            donor = (donor[0] + 1, donor[1], donor[2])
        if mutant == "donors-dropped" and _bits(wdon[q]) == 0:
            continue
        entries.append((donor, float(wdon[q])))
    return entries, float(cacc)


def _caught(name, mutant):
    """WALL rows on which ``mutant`` differs from v1."""
    ctx, rows, arms, flux = _v1_side(name)
    n = 0
    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        ok, _why = _same(_model_row(ctx, arms, flux, r, mutant), _v1_row(rows, r))
        if not ok:
            n += 1
    return n


def _has_wall_arm(arms, r):
    """Does row ``r`` have a SOLID neighbour on the **differencing** axis?"""
    return not (bool(arms[(0, 1)][1][r]) and bool(arms[(0, -1)][1][r]))


# ===========================================================================
# 1. P-1..P-10 — v1 <-> v2 row parity, BITWISE, on all ten configurations
# ===========================================================================


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_compiled_row_is_v1s_row_bitwise(blockamr_session, name):
    """**The acceptance bar** (review.md §4 Q56(a), item ii).

    For every ``WALL`` cell: the ordered ``(index, a)`` sequence the compiled
    functor emits, and the constant it accumulates, equal v1's
    ``_face_balance_rows`` row on the raw bits. No tolerance — a residual
    difference is a bug, not noise: there is no libm in the closure chain,
    contraction is pinned off by the per-file flags, and the numpy model was
    proven bitwise against v1 before the build on all 3 136 rows.

    The **model** is pinned here too, in the same statement. That is the link
    that makes the falsification matrix below a claim about the shipped code
    rather than about a numpy script beside it.

    The row width is asserted with it: at most ``1 + 2 + 8 = 11`` entries, and
    **never** the fifteen v1 declares — slots 3..6 (the ``±y``/``±z``
    neighbours) are allocated by ``_blank``, never written, and dropped by v1's
    own liveness rule (H-9).
    """
    mesh, _term, geom, ba, dm = _case(name)
    _bodies, ibm_bc, _lo, _hi = CONFIGS[name]
    ctx, rows, arms, flux = _v1_side(name)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)

    nwall = int(ctx.at_wall.sum())
    assert nwall == NWALL[name], f"{name}: {nwall} wall rows, expected {NWALL[name]}"
    assert data.nrows == nwall, f"the compiled data has {data.nrows} rows, v1's band has {nwall}"

    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        cell = tuple(int(v) for v in ctx.target[r])
        want = _v1_row(rows, r)
        assert 9 <= len(want[0]) <= 11, f"{name} at {cell}: {len(want[0])} entries, not 1+(0..2)+8"
        ok, why = _same(_compiled_row(ct, g, data, robin, geom, cell), want)
        assert ok, f"{name}: compiled row at {cell} differs from v1 — {why}"
        ok, why = _same(_model_row(ctx, arms, flux, r), want)
        assert ok, f"{name}: the numpy model at {cell} differs from v1 — {why}"


# ===========================================================================
# 2. P-11..P-14 — the falsification matrix, its controls, the canonicalisation
#    it rests on, and the vacuity trap (Q35, Q53, Q54(a))
# ===========================================================================


def test_the_falsification_matrix_is_reproduced_exactly(blockamr_session):
    """**Q35/Q53, permanently in-suite.** Twenty-one defects, counted row by row,
    over all ten configurations and all 3 136 wall rows.

    Asserted as an exact tuple rather than "> 0" (B30b-R's S-6 shape):
    over-coverage fails too, because the matrix is the record of what each
    configuration *can* see. The entries that pay for the configuration set are
    listed on ``COVERAGE``; the headline is ``donor-assoc`` (H-4'), caught on
    **exactly one** configuration — ``G10``, the non-dyadic grid — because on any
    dyadic grid grad's ``nb_part`` is a power of two and the association is
    exact.

    The two defects most likely to be *written* — copy div's axis loop, skip a
    diagonal you can prove is zero — are the two that cannot ship quietly:
    ``axes-all``, ``arms-six``, ``diag-neg-zero`` and ``diag-dropped`` are each
    caught on 10 of 10 configurations and on **every one** of the 3 136 rows.
    That asymmetry is this session's main safety margin.
    """
    measured = {}
    total_wall = 0
    for name in CONFIGS:
        ctx, _rows, _arms, _flux = _v1_side(name)
        total_wall += int(ctx.at_wall.sum())
        assert _caught(name, None) == 0, f"{name}: the baseline model is not v1's rows"
        measured[name] = tuple(_caught(name, mutant) for mutant in MUTANTS)

    assert total_wall == TOTAL_WALL, f"the acceptance set is {total_wall} wall rows"
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
    columns = dict(zip(MUTANTS, zip(*(COVERAGE[name] for name in CONFIGS))))
    for mutant, column in columns.items():
        seen = sum(1 for v in column if v)
        assert (seen == 0) == (mutant in CONTROLS), mutant
        assert seen == 0 or max(column) >= 194, f"{mutant} is caught by only {max(column)} rows"
    # H-4' rides on ONE configuration and it must stay in the set (plan R7).
    assert sum(1 for v in columns["donor-assoc"] if v) == 1, columns["donor-assoc"]
    assert COVERAGE["G10-nondyadic-grid"][MUTANTS.index("donor-assoc")] == 194
    # H-9 and H-10 are total, in two independent spellings apiece.
    for mutant in ("axes-all", "arms-six", "diag-neg-zero", "diag-dropped"):
        assert sum(columns[mutant]) == TOTAL_WALL, mutant


def test_the_five_controls_move_no_bit_and_the_coeff_placement_is_v1s(blockamr_session):
    """**Controls, and they are labelled as such** (P-12), together with H-5's
    own measurement (P-12b) — both are statements about *reassociations that
    change nothing*, and both are pure numpy.

    Five reassociations that move no bit on any configuration:

    * ``face-index`` — ``face = 0`` for ``step = +1``: v1 passes ``flux = ones``,
      so ``flux[:, d, 0]`` and ``flux[:, d, 1]`` are the same ``1.0`` and the
      index selects nothing. A real defect for ``div``; provably inert here;
    * ``scale-assoc`` — ``step*(f/dx)`` for ``(step*f)/dx``: exact, ``step`` is
      ``±1``;
    * ``nb-complement`` (H-7) — ``sc - slf`` for ``sc*(1 - w)``: exact only
      because ``w`` is exactly ``0.5`` here, which is why this is *recorded*
      rather than acted on — a future non-trivial face weight must not inherit
      the permission;
    * ``dG-assoc`` — ``s_P + step*(dx*n)`` for ``s_P + (step*dx)*n``: exact,
      ``step`` is ``±1``;
    * ``arm-raw`` (H-6) — the fluid-face coefficient emitted raw instead of
      accumulated into a zero slot: for ``grad``, ``nb_part`` is ``±0.5/dx`` and
      never ``±0.0``, so nothing moves. ``div`` needs it on 960 of 3 232 rows,
      and the shape is transcribed here anyway because v1's *shape* is what is
      being ported.

    **H-5** is the one association this port does *not* reproduce: v1 folds
    ``coeff`` into ``scale`` before every product, the frame multiplies the
    finished sum (``wall_apply.H:216``). The two agree bitwise iff ``coeff`` is a
    power of two, and ``exp.grad(field)`` exposes no ``coeff`` at all
    (``Grad.__init__``'s default is ``1.0``), so it is **inert on every
    reachable v1 grad term** — which is what scopes the bar above to
    ``coeff = 1.0``. The exposure is pinned rather than argued: ``64``/``352``
    rows move at ``3.0`` and ``64``/``215`` at ``0.1`` on the two curved
    configurations, and none at all on ``G1``.
    """
    caught = {mutant: {name: _caught(name, mutant) for name in CONFIGS} for mutant in CONTROLS}
    assert caught == {m: dict.fromkeys(CONFIGS, 0) for m in CONTROLS}

    measured = {}
    for name in COEFF_PLACEMENT:
        ctx, _rows, _arms, _flux = _v1_side(name)
        unit = _v1_rows(ctx, 1.0)
        moved = []
        for coeff in COEFFS:
            folded = _v1_rows(ctx, coeff)
            n = 0
            for r in range(ctx.nrows):
                if not ctx.at_wall[r]:
                    continue
                fe, fc = _v1_row(folded, r)
                one_e, one_c = _v1_row(unit, r)
                same = _bits(coeff * one_c) == _bits(fc) and all(
                    fi == oi and _bits(coeff * oa) == _bits(fa)
                    for (fi, fa), (oi, oa) in zip(fe, one_e)
                )
                n += not same
            moved.append(n)
        measured[name] = tuple(moved)
    assert measured == COEFF_PLACEMENT, (
        f"H-5's exposure moved: {measured} against the recorded {COEFF_PLACEMENT} "
        f"at coeff {COEFFS}"
    )


def test_the_diagonal_is_positive_zero_on_every_row_and_is_still_a_live_entry(blockamr_session):
    """**H-10, and the canonicalisation corollary it forces** (P-13).

    For ``axes = (0,)`` slot 0 accumulates exactly twice, and
    ``(coeff·(+1)·1/dx)·0.5`` and ``(coeff·(−1)·1/dx)·0.5`` are exact negatives:
    IEEE multiplication and division are sign-symmetric, and ``x + (−x)`` is
    ``+0.0`` in round-to-nearest for every finite ``x``. So the diagonal of a
    grad wall row is ``+0.0`` — not ``−0.0``, not "approximately zero" — on
    **3 136 of 3 136** rows, and it is compared here on the raw ``int64`` because
    ``== 0.0`` is true of ``−0.0`` too and would assert nothing.

    The slot is nonetheless **live**: v1's row carries ``stencil[0] = target``
    with ``a[0] = +0.0``, and a sweep reading it multiplies ``φ(P)`` by ``+0.0``
    and adds it. ``diag-dropped`` and ``diag-neg-zero`` are each caught on 10/10
    and on every row.

    Hence this file's canonicalisation is **structural** and B32's ``_v1_row``
    bits-are-zero drop rule is *not* reused: it would discard 21 688 of 32 939
    live entries (65.8 %) — every row carries at least one — and the diagonal of
    every single row with them, making the parity claim above weaker than it
    looks. The census is pinned so the rule cannot be "simplified" back.
    """
    rows_with = entries_zero = entries_live = bad_diag = total = 0
    for name in CONFIGS:
        ctx, rows, _arms, _flux = _v1_side(name)
        for r in range(ctx.nrows):
            if not ctx.at_wall[r]:
                continue
            total += 1
            entries, _c = _v1_row(rows, r)
            entries_live += len(entries)
            zero = sum(1 for _i, a in entries if a == 0.0)
            entries_zero += zero
            rows_with += bool(zero)
            bad_diag += _bits(entries[0][1]) != 0

    assert total == TOTAL_WALL, total
    assert bad_diag == 0, f"{bad_diag} of {total} diagonals are not bitwise +0.0"
    assert (rows_with, entries_zero, entries_live) == (
        ZERO_ENTRY_ROWS,
        ZERO_ENTRIES,
        LIVE_ENTRIES,
    ), f"the zero-entry census moved: {(rows_with, entries_zero, entries_live)}"


def test_half_the_wall_rows_have_no_wall_arm_at_all(blockamr_session):
    """**Q54(a) in its extreme form** (P-14) — the vacuity trap, made data.

    A ``WALL`` cell whose solid neighbours are all off the **differencing** axis
    contributes exactly nothing through the closure: both x faces are fluid, so
    the wall branch never runs, all eight donor coefficients stay ``+0.0`` and
    the constant stays ``+0.0``. That is 1 579 of 3 136 rows — half the
    population — and it is 100 % of ``G2`` and of ``G8``, on which **eleven of
    the twenty-one mutants are invisible**.

    ``G8`` is in the set for exactly that reason and is not a duplicate of
    ``G2``: it is a *curved* wall with ``n̂_0 ≡ 0`` and no solid x-neighbour
    anywhere, so it has G2's property without G2's flatness.

    Every conformance row elsewhere that wants "the datum reached the row" must
    therefore name a configuration from ``{G1, G3..G7, G9, G10}`` and guard in
    the aggregate. ``ARM_ROWS`` is pinned so a geometry change cannot quietly
    move a row into the blind half.
    """
    measured = {}
    for name in CONFIGS:
        ctx, rows, arms, _flux = _v1_side(name)
        with_arm = 0
        for r in range(ctx.nrows):
            if not ctx.at_wall[r]:
                continue
            if _has_wall_arm(arms, r):
                with_arm += 1
                continue
            # a no-arm row: the closure never enters it.
            entries, c = _v1_row(rows, r)
            assert _bits(c) == 0, f"{name}: a no-arm row at {entries[0][0]} has c = {c!r}"
            for index, a in entries[-8:]:
                assert _bits(a) == 0, f"{name}: a no-arm row's donor {index} carries {a!r}"
        measured[name] = with_arm

    assert measured == ARM_ROWS, f"the wall-arm census moved: {measured}"
    assert sum(measured.values()) == TOTAL_ARM_ROWS
    assert measured["G2-plane-y-dirichlet"] == 0 and measured["G8-cyl-x-mixed"] == 0


# ===========================================================================
# 3. P-15, P-16 — the pair through the frame, over real fabs
# ===========================================================================


def PHI(i, j, k):
    """The sweep rows' field, as a function of the global index.

    **Quadratic on purpose** (B32's deviation 6): the gradient of a linear field
    is a constant, and then "v2 did not write this cell" and "v2 wrote the same
    number here" stop being distinguishable, which would make both the FLUID
    comparison and the ``SOLID`` exclusion below vacuously green.
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


@pytest.mark.parametrize("name", list(WALL_RESIDUAL))
def test_the_sweep_is_the_pairs_own_row_and_v1s_residual_is_its_consumers_fma(
    blockamr_session, name
):
    """**The pair through the frame** (Q56(a), item iii) — Q50's attributed
    sentence, written for ``grad``, with ``grad``'s own number.

    Interior sweep plus wall sweep, against v1's interior sweep plus
    ``apply_band_rows``, on the *same* ``phi``, over **eight boxes** so the row
    map's cross-box concatenation is what is being exercised.

    **No width scoping is needed, and that is a measured simplification**
    (contrast ``div``'s Q52(c)). ``SCHEME_REGISTRY["grad"]`` has exactly one
    entry — ``central``, at width 1 — so the equation's band is ``depth <= 1``,
    which is exactly the WALL layer: B33's wide-scheme obstruction (a
    ``depth == 2`` band row against v2's degraded interior kernel) cannot arise
    here at all.

    ``out`` carries **three** components because that is what ``grad_acc``
    writes (scalar → 3-vector); the wall sweep and ``apply_band_rows`` both run
    at ``ncomp = 1`` and own component 0, which is the axis-0 derivative the wall
    row expresses. Deviation from the B34 plan's "1-component ``out``", and
    deliberate: v1's own driver allocates its accumulator at the *field's*
    ``ncomp`` (``dsl/solve.py:189``), so an ``exp.grad`` evaluate on the cpp
    backend has ``grad_acc`` writing components 1 and 2 past the end of the fab.
    That is a v1 defect, out of this task's scope, and a bitwise parity claim may
    not be built on undefined behaviour.

    What is measured:

    * **FLUID** — bitwise equal, every cell and every component. Both sides run
      the *same* ``grad_acc`` and neither writes a FLUID cell afterwards.
      ``PHI`` is quadratic so this is not the equality of two constants.
    * **WALL** — v2's component 0 is bitwise ``_dot(row, fused=False)`` and v1's
      is bitwise ``_dot(row, fused=True)``, from **the same row**. The rows are
      identical (the parity rows above), so the entire residual is the two
      consumers' floating-point contraction and nothing else:
      ``band_table.cpp:688`` carries no per-file FP flags and takes nvcc's
      default ``--fmad=true``, while ``ApplySink::linear`` is inlined into this
      pair's ``--fmad=false`` TU and does not. **The flag that buys row parity is
      the flag that costs sweep parity**, and the count of WALL cells where the
      two land differently is pinned exactly, so the finding can neither quietly
      heal nor quietly grow.
    * **SOLID** — excluded, and the exclusion is asserted **load-bearing**
      (OPEN-C, review.md §4 Q49(b)): v1's ``band(1)`` is ``depth <= 1`` and so
      carries every solid cell as an ``nnz = 0, c = 0`` row, making v1's first
      ``Overwrite`` term write exactly ``0.0`` there, while v2's frame returns
      before the sink at ``m != WALL``. Recorded rather than resolved.
    """
    from blockamr.ibm.band_rows import band_table

    mesh, term, geom, ba, dm = _case(name, max_size=8)
    _bodies, ibm_bc, _lo, _hi = CONFIGS[name]

    nbox = sum(1 for _ in blockamr.MFIterator(blockamr.MultiFab(ba, dm, 1, 0)))
    assert nbox == 8, f"vacuous: the level did not decompose ({nbox} boxes)"

    phi = _phi(ba, dm)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)

    # v1: the untouched interior sweep, then the band rows in Overwrite mode.
    ctx = _context(term, mesh.ibm, 0, 1, 0.0, 1)
    rows = _v1_rows(ctx, 1.0)
    out_v1 = blockamr.MultiFab(ba, dm, 3, 0)
    out_v1.set_val(0.0)
    blockamr.grad_acc(out_v1, phi, geom, 1.0)
    version = mesh.ibm.grid_version
    blockamr.apply_band_rows(
        out_v1, phi, band_table(rows, version), 1, blockamr.BandMode.Overwrite, 1.0, version
    )

    # v2: the same interior sweep, then the compiled pair, by keyword.
    out_v2 = blockamr.MultiFab(ba, dm, 3, 0)
    out_v2.set_val(0.0)
    blockamr.grad_acc(out_v2, phi, geom, 1.0)
    blockamr.wall_grad_ghost_cell(
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

    by_cell = {
        tuple(int(v) for v in ctx.target[r]): _v1_row(rows, r)
        for r in range(ctx.nrows)
        if ctx.at_wall[r]
    }

    marker = _markers(ct, phi)
    got_v1, got_v2 = _readback(out_v1), _readback(out_v2)
    seen = {SOLID: 0, WALL: 0, "fluid": 0}
    solid_differ = wall_differ = max_delta = 0
    for key, value in got_v2.items():
        m = marker[key[:3]]
        if m == SOLID:
            seen[SOLID] += 1
            solid_differ += key[3] == 0 and _bits(value) != _bits(got_v1[key])
            continue
        if m != WALL:
            seen["fluid"] += 1
            assert _bits(value) == _bits(got_v1[key]), (
                f"a FLUID cell moved at {key}: v2 {value!r} vs v1 {got_v1[key]!r}"
            )
            continue
        seen[WALL] += 1
        if key[3] != 0:
            # components 1 and 2 are grad_acc's, untouched by either wall sweep.
            assert _bits(value) == _bits(got_v1[key]), key
            continue
        entries, c = by_cell[key[:3]]
        assert _bits(value) == _bits(_dot(entries, c, 1.0, fused=False)), (
            f"v2's sweep at {key} is not its own row's plain dot product"
        )
        assert _bits(got_v1[key]) == _bits(_dot(entries, c, 1.0, fused=True)), (
            f"v1's sweep at {key} is not the same row's FUSED dot product"
        )
        delta = abs(_bits(value) - _bits(got_v1[key]))
        wall_differ += delta != 0
        max_delta = max(max_delta, delta)

    assert seen[WALL] == 3 * NWALL[name], seen
    assert seen["fluid"] > 0 and seen[SOLID] > 0, seen
    assert wall_differ > 0, "vacuous: Q50 is only a finding where the two sides differ"
    assert (wall_differ, max_delta) == WALL_RESIDUAL[name], (
        f"the contraction residual moved: {wall_differ}/{NWALL[name]} WALL cells differ, "
        f"max |delta bits| {max_delta}, but the pinned measurement is "
        f"{WALL_RESIDUAL[name]} — either band_table.cpp's contraction changed or this "
        "pair's flags did; re-read Q50 first"
    )
    assert solid_differ > 0, "vacuous: OPEN-C is only a finding where the two sides differ"


# ===========================================================================
# 4. P-17..P-19 — the extension contract, the declaration, and the refusal
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


def test_the_grad_pair_takes_exactly_the_canonical_twelve(blockamr_session):
    """**R-1** — Q29(f)'s minimum, closed from the other side (review.md §4
    Q56(a)).

    ``test_ibm_laplacian_ghost_cell`` already asserts the twelve as a *prefix*
    over **every** registered ``wall_*`` attribute, so this pair entered that
    contract without an edit. What no existing row can say is that there is no
    thirteenth: ``div`` appends four (its three face fluxes and its face-value
    selector), and ``grad`` appends **none**.

    That is measured rather than preferred. ``GhostCellGrad.rows`` passes
    ``flux = ones`` and ``weight_self = 0.5 * ones`` into the same
    ``_face_balance_rows`` ``div`` uses, so no face field and no face-value rule
    is reachable; ``__init__`` stores ``interior_scheme`` and never reads it; and
    ``SCHEME_REGISTRY["grad"]`` has exactly one entry, so there is no scheme bit
    to transport even in principle.
    """
    module = blockamr._blockamr
    args = _signature_args(module.wall_grad_ghost_cell)
    names = tuple(a for a, _d in args)

    assert names == CANONICAL_TWELVE, names
    assert len(names) == 12, names
    assert not any(d for _a, d in args), "no argument of a registered pair is defaulted"

    assert re.fullmatch(r"wall_[a-z_]+", "wall_grad_ghost_cell")
    assert hasattr(module, "_wall_row_grad_ghost_cell")
    assert not hasattr(blockamr, "_wall_row_grad_ghost_cell"), (
        "the row hook is underscore-private and must not reach the package namespace"
    )


def test_the_v1_scheme_names_the_compiled_pair(blockamr_session):
    """The seam B36 flips (review.md §4 Q49(g)).

    ``register`` raises on a second class for a taken key, and O4 forbids
    removing v1's, so the declaration lands **additively** on the existing
    ``GhostCellGrad``: ``rows()`` is untouched, ``_check_grad_ncomp`` is
    untouched, nothing is deregistered, and B36 changes ``BandEvaluation.apply``
    from the one to the other.
    """
    from blockamr.schemes.grad_schemes import CentralDiffGrad

    scheme_cls = BOUNDARY_SCHEMES[("grad", "ghostCell")]
    kernel = scheme_cls(interior_scheme=CentralDiffGrad()).build_cpp_kernel()
    assert kernel.name == "wall_grad_ghost_cell"
    assert hasattr(blockamr, kernel.name)
    assert callable(getattr(blockamr, kernel.name))


def test_a_multi_component_sweep_is_refused_by_v1_and_by_the_compiled_pair(blockamr_session):
    """**The ``ncomp > 1`` refusal, on both surfaces and bound together**
    (review.md §4 Q56(c), api §9).

    The band row applies **one** coefficient list to every component
    (row-contract §2) while the gradient's component ``n`` is the difference
    along axis ``n`` — a different stencil per component, which the row format
    does not have. v1 refuses rather than returning a plausible field, in
    ``_check_grad_ncomp``, as the **first statement** of ``rows()``; the compiled
    pair refuses in ``Maker::validate``, before any launch, which is the same
    place in this architecture.

    One ``pytest.raises`` covers both, and that is not a coincidence worth
    hiding: v1 raises ``NotImplementedError`` naming ``term.field.name``, the
    compiled pair raises ``std::runtime_error`` naming the entry point (it has no
    field name), and ``NotImplementedError`` **is a** ``RuntimeError``. The type
    gap is owed to B36 beside B31's Invariant-F precedent; the *sentence* is
    v1's, verbatim from "the band row applies" onwards, and that shared tail is
    what this row pins.
    """
    name = "G6-cyl-z-mixed"
    mesh, term, geom, ba, dm = _case(name)
    _bodies, ibm_bc, _lo, _hi = CONFIGS[name]

    tail = (
        "the band row applies one coefficient list to every component, while the "
        "gradient's component n is the difference along axis n. Expressing that needs "
        "a per-component row, which the "
    )

    # v1, through the production call.
    scheme = BOUNDARY_SCHEMES[("grad", "ghostCell")](term.scheme)
    with pytest.raises(RuntimeError, match="grad x ghostCell needs a one-component field") as v1:
        scheme.rows(term, mesh.ibm, 0, 2, 0.0, 1)
    assert isinstance(v1.value, NotImplementedError), type(v1.value)
    assert tail in str(v1.value), str(v1.value)
    assert "ncomp = 2" in str(v1.value)

    # v2, through the registered entry point, and it must raise before any
    # launch: `out` is left exactly as it was.
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc, ncomp=2)
    phi = _phi(ba, dm, ncomp=2)
    out = blockamr.MultiFab(ba, dm, 2, 0)
    out.set_val(-7.0)
    with pytest.raises(
        RuntimeError, match=r"wall_grad_ghost_cell: grad x ghostCell needs a one-component field"
    ) as v2:
        blockamr.wall_grad_ghost_cell(
            out, phi, ct, g, data, robin, geom, 0.0, 1.0, 2, blockamr.WallMode.Overwrite, 1.0
        )
    assert tail in str(v2.value), str(v2.value)
    assert "ncomp = 2" in str(v2.value)
    assert all(v == -7.0 for v in _readback(out).values()), "the sweep launched before refusing"

    # ...and one component is not refused: the guard is a refusal, not a wall.
    g1, ct1, data1, robin1 = _v2(mesh, geom, ba, dm, ibm_bc)
    out1 = blockamr.MultiFab(ba, dm, 1, 0)
    out1.set_val(0.0)
    blockamr.wall_grad_ghost_cell(
        out1,
        _phi(ba, dm),
        ct1,
        g1,
        data1,
        robin1,
        geom,
        0.0,
        1.0,
        1,
        blockamr.WallMode.Overwrite,
        1.0,
    )
