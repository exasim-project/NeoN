# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``div x ghostCell`` — the second real ``(operator, method)`` pair (B33).

``src/bindings/blockAMR/schemes/boundary/div_ghost_cell.cpp``: the compiled peer
of v1's :func:`blockamr.schemes.boundary.ghost_cell._face_balance_rows`, as
``GhostCellDiv`` calls it.

**The bar is v1↔v2 row parity, BITWISE.** For every ``WALL`` cell of ten
configurations — 3 232 rows — the compiled row's *ordered* ``(index, a)``
sequence and its constant equal v1's, compared through raw ``int64`` views.
``assert_array_equal`` on f64 cannot see ``-0.0``, and this operator produces
signed zeros by the plane (see H-6 below), so the stricter comparison is not
decoration. review.md §4 Q29(d) refuses the ULP fallback: a residual mismatch
stays red and is escalated.

**Why this bar and not "rung 8 green on v2"** (review.md §4 Q52(a)). When this
file was written there was no driver seam to run rungs through — the v1 registry
key ``("div", "ghostCell")`` was taken and the driver still called ``rows()``.
B36 flipped it, and this bar is kept. And rung 8 is a **strict xfail**
(``RECONSTRUCTION_ORDER``/T14: the reconstruction is trilinear and the exact
discrete cancellation breaks in the band), so a *correct* port keeps it red —
"rung 8 green on v2" would mean the port had changed the numerics, which is the
one thing it forbids. Row parity is strictly stronger per cell than the rungs,
which are aggregate and tolerance-based.

**H-6, the hazard this file exists to pin.** v1's ``_blank`` allocates
``a = np.zeros(...)`` and writes each face-neighbour slot exactly once, so its
shipped coefficient is ``0.0 + nb_part`` — and IEEE says ``0.0 + (-0.0)`` is
``+0.0``. A functor that emitted ``nb_part`` raw would ship ``-0.0`` there.
``nb_part`` is ``-0.0`` exactly when the face flux is ``+-0.0`` and the
face-value rule puts the whole weight on the neighbour at ``step = -1``, which a
rigid-rotation velocity produces by the plane (``u_z == 0`` identically, and
``u_x``/``u_y`` vanish on the two centre lines). Measured: **960 of 3 232 rows**
break without the accumulation, on D3/D4/D9 and on **none** of the other seven —
a suite built from uniform or merely "non-dyadic" fluxes would have shipped it.

**Its corollary for this harness, and it is not optional.** B32's ``_v1_row``
drops a slot whose bits are ``0`` (``test_ibm_laplacian_ghost_cell.py:335``). For
``div`` that rule is *wrong*: 1 920 of these 3 232 rows carry at least one
**live** face slot whose coefficient is exactly ``+-0.0``, because an upwind face
on the wrong side has ``nb_part = scale * 0.0``. The canonicalisation here is
therefore **structural** — keep slot 0, keep slots 1..6 whose stencil entry is
not the target, keep 7..14 — which is exactly v1's own liveness rule
(``_blank`` points every unwritten slot at the target). ``test_the_zero_valued_
face_slots_are_live`` measures the census so the rule cannot be "simplified"
back.

**The oracle is v1's production code**, imported read-only: ``_context``,
``_band_face_flux``, ``_face_weights`` and ``_face_balance_rows``, with the term
built through ``Equation(exp.div(FaceField, T), schemes={"Div": name})``, and
cross-checked on every configuration against the production call
``BOUNDARY_SCHEMES[("div", "ghostCell")](term.scheme).rows(...)``. The mutants
are applied to a numpy model of the functor and never to the oracle — and the
model itself is pinned to the *compiled* row by the parity rows first.

**Where the other rows live.** Per-cell functor conformance (S2, S3, Q34, the
degrade mapping, the error surface) is ``test_ibm_wall_functors.py``, which is
where the shipped frame file says a pair's rows belong and which already has the
``RecordSink`` readback shape. None of the four O3 fence files is touched, and
``test_ibm_laplacian_ghost_cell.py`` is not edited: its Q39 row already asserts
the canonical twelve as a *prefix* over every registered pair, and its Q36 row
already requires this TU on the FP-flags list. Both strengthen for free.
"""

import re
import struct
from fractions import Fraction

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, exp
from blockamr.dsl.solve import _resolve_schemes
from blockamr.field import CellField, FaceField
from blockamr.ibm.bc import FixedGradient, FixedValue, Mixed
from blockamr.ibm.body import Cylinder, Plane
from blockamr.ibm.classify import _patches
from blockamr.operators.div import update_face_fluxes
from blockamr.schemes.boundary import BOUNDARY_SCHEMES
from blockamr.schemes.boundary.ghost_cell import (
    STRIDE,
    _band_face_flux,
    _context,
    _face_balance_rows,
    _face_weights,
    _neighbour,
)

N = 16
SOLID = int(blockamr.CellType.SOLID)
WALL = int(blockamr.CellType.WALL)

#: The canonical twelve (design §4.4), in order — B32's Q39 contract.
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

#: What ``div`` appends past the twelfth (review.md §4 Q52(b), extending
#: Q29(f)). The three fluxes come first and in this order because
#: ``div_*_acc(out, phi, fx, fy, fz, geom, coeff, ncomp)`` already takes them so.
APPENDED_FOUR = ("flux_x", "flux_y", "flux_z", "face_value")

UNIT_LO, UNIT_HI = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
#: B31-R's Q35 lesson: a power-of-two ``dx`` with ``prob_lo = 0`` only shifts
#: exponents, so most of the arithmetic under test is exact there and cannot
#: tell a correct transcription from a reassociated one.
SKEW_LO, SKEW_HI = (-0.37, 0.11, 0.23), (0.53, 0.81, 1.53)

TILTED = tuple(np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0]))

#: Rung 8's own angular velocity shape. ``u = omega x r`` is exactly zero on the
#: two centre lines, sign-changing, and — the load-bearing part — **independent
#: of its own coordinate in every component**, so a cell's two faces along an
#: axis carry bitwise the same number.
OMEGA = 5.0


def _uniform(x, y, z, t):
    return np.ones_like(x), np.ones_like(y), np.ones_like(z)


def _rotation(x, y, z, t):
    return -OMEGA * (y - 0.5), OMEGA * (x - 0.5), np.zeros_like(z)


def _skew(x, y, z, t):
    """Non-dyadic, sign-changing, and different on the two faces of a cell."""
    return 0.37 + 1.3 * x - 0.9, 0.21 - 1.7 * y + 0.6, 0.13 + 0.7 * z - 0.4


CYL = {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)}
NONDYADIC_CYL = {"cyl": Cylinder(centre=(0.13, 0.31), radius=0.1731, axis=2)}

#: ``(bodies, ibm_bc, prob_lo, prob_hi, velocity, Div scheme)`` per
#: configuration. What each one alone contributes is in the parity row's
#: docstring and in ``COVERAGE`` below.
CONFIGS = {
    # the D1 acceptance geometry: weights exactly 0/1, datum 0.0 — which is
    # exactly why D1 alone is vacuous for four of the eighteen mutants.
    "D1-plane-x-dir-uniform-linear": (
        {"wall": Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))},
        {"wall": FixedValue(0.0)},
        UNIT_LO,
        UNIT_HI,
        _uniform,
        "linear",
    ),
    # a genuine 3-D interpolation.
    "D2-plane-123-dir-uniform-linear": (
        {"wall": Plane(point=(0.5, 0.5, 0.5), normal=TILTED)},
        {"wall": FixedValue(0.3)},
        UNIT_LO,
        UNIT_HI,
        _uniform,
        "linear",
    ),
    # rung 8's own geometry AND flux: exactly-zero faces, which is the only
    # thing in the repertoire that can see H-6.
    "D3-cyl-dir-rotation-linear": (
        CYL,
        {"cyl": FixedValue(0.3)},
        UNIT_LO,
        UNIT_HI,
        _rotation,
        "linear",
    ),
    # the beta != 0 arm (`grad_linear` is -0.0) x the upwind branch x zero faces.
    "D4-cyl-neu-rotation-upwind": (
        CYL,
        {"cyl": FixedGradient(0.2)},
        UNIT_LO,
        UNIT_HI,
        _rotation,
        "upwind",
    ),
    # both closure terms non-zero, upwind.
    "D5-cyl-mixed-skew-upwind": (
        CYL,
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
        _skew,
        "upwind",
    ),
    # every WALL cell has a face neighbour on the other patch — the Q34
    # discriminator.
    "D6-two-cyl-two-patches-skew-linear": (
        {
            "a": Cylinder(centre=(0.28, 0.5), radius=0.12, axis=2),
            "b": Cylinder(centre=(0.72, 0.5), radius=0.12, axis=2),
        },
        {"a": FixedValue(0.3), "b": FixedGradient(0.2)},
        UNIT_LO,
        UNIT_HI,
        _skew,
        "linear",
    ),
    # non-dyadic normals and weights on a dyadic grid.
    "D7-nondyadic-cyl-skew-upwind": (
        {"cyl": Cylinder(centre=(0.37, 0.4123), radius=0.1731, axis=2)},
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
        _skew,
        "upwind",
    ),
    # the load-bearing grid, and the best H-4' discriminator (284 rows).
    "D8-nondyadic-grid-skew-linear": (
        NONDYADIC_CYL,
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        SKEW_LO,
        SKEW_HI,
        _skew,
        "linear",
    ),
    # a WIDTH-2 scheme: the D1 degrade mapping, measured rather than asserted.
    "D9-cyl-mixed-rotation-vanleer": (
        CYL,
        {"cyl": Mixed(value=0.3, gradient=0.2, fraction=0.6)},
        UNIT_LO,
        UNIT_HI,
        _rotation,
        "vanLeer",
    ),
    # the second width-2 scheme, on the hard grid.
    "D10-nondyadic-grid-quick": (
        NONDYADIC_CYL,
        {"cyl": FixedValue(0.3)},
        SKEW_LO,
        SKEW_HI,
        _skew,
        "quick",
    ),
}

#: WALL rows per configuration — measured, and asserted, so a geometry change
#: that emptied a configuration cannot make its parity row vacuously green.
NWALL = {
    "D1-plane-x-dir-uniform-linear": 256,
    "D2-plane-123-dir-uniform-linear": 256,
    "D3-cyl-dir-rotation-linear": 320,
    "D4-cyl-neu-rotation-upwind": 320,
    "D5-cyl-mixed-skew-upwind": 320,
    "D6-two-cyl-two-patches-skew-linear": 448,
    "D7-nondyadic-cyl-skew-upwind": 288,
    "D8-nondyadic-grid-skew-linear": 352,
    "D9-cyl-mixed-rotation-vanleer": 320,
    "D10-nondyadic-grid-quick": 352,
}

TOTAL_WALL = 3232

#: The mutants, and the defect each one models.
MUTANTS = (
    "order-step",  # H-1, api §5.3's sketch: -1 first
    "order-axis",  # H-2', the axis loop reversed
    "face-index",  # the cell's low and high face swapped
    "scale-sign",  # the flux-direction sign lost
    "scale-assoc",  # step*(f/dx) for (step*f)/dx -- a CONTROL, exact
    "nb-complement",  # H-7: scale - self_part for scale*(1-w) -- a CONTROL
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
    "weight-gt",  # H-8: f > 0 for f >= 0 -- a CONTROL, provably exact
    "arm-raw",  # H-6: the fluid-face coefficient emitted raw
)

#: The four mutants that must move **nothing**, anywhere. Carried so the matrix
#: is not a list of things that happen to differ: two of them
#: (``nb-complement``, ``weight-gt``) are simplifications a reviewer would
#: otherwise be asked to reject on taste, and they are now permitted on
#: arithmetic instead.
CONTROLS = ("scale-assoc", "nb-complement", "dG-assoc", "weight-gt")

#: Rows caught, per ``(configuration, mutant)``. **Measured** against v1's own
#: row builder before the build, and asserted as an *exact* tuple: a mutant that
#: caught more rows than this fails too, because the matrix is a claim about
#: what each configuration can see and not a lower bound.
#:
#: Three entries are why the suite has ten configurations and not two:
#:
#: * ``arm-raw`` (H-6) is caught on **exactly three** — D3, D4, D9, the
#:   rotation-flux ones — and is invisible on every uniform-flux and every
#:   skew-flux configuration;
#: * ``donor-assoc`` (H-4') is invisible on D1 (weights 0/1), D2 (a uniform flux
#:   on a dyadic grid makes ``nb_part`` a power of two) and D4 (Neumann, where
#:   ``atLinear`` is exactly 1.0);
#: * ``datum-linear`` (an S2 violation) and ``dG-no-normal`` are invisible on D1,
#:   whose datum is ``0.0`` and whose only solid faces lie along its own normal.
#:
#: ``face-index`` is invisible on **all three rotation configurations** and not
#: only on D3: for ``u = omega x r`` every component is independent of its own
#: coordinate (``u_z`` is identically zero), so a cell's two faces along an axis
#: carry bitwise the same number and swapping them is a no-op whatever the
#: face-value rule is. That is measured — ``flux[:, d, 0]`` and ``flux[:, d, 1]``
#: are bitwise equal on all three axes of D3, D4 and D9 — and it corrects the
#: B33 plan's §8.3, which predicted 320 rows on D4 and D9.
COVERAGE = {
    # ord-step ord-axis face-idx sign s-assoc nb-cpl self-gt donor-a d-idx datum
    # normal ungated dG-assoc dG-sign dG-no-n at-wall w-gt arm-raw
    "D1-plane-x-dir-uniform-linear": (
        256,
        256,
        0,
        256,
        0,
        0,
        256,
        0,
        256,
        0,
        256,
        256,
        0,
        256,
        0,
        256,
        0,
        0,
    ),
    "D2-plane-123-dir-uniform-linear": (
        171,
        256,
        0,
        256,
        0,
        0,
        256,
        0,
        256,
        256,
        256,
        256,
        0,
        256,
        256,
        254,
        0,
        0,
    ),
    "D3-cyl-dir-rotation-linear": (
        320,
        320,
        0,
        320,
        0,
        0,
        256,
        256,
        320,
        304,
        256,
        320,
        0,
        256,
        288,
        256,
        0,
        320,
    ),
    "D4-cyl-neu-rotation-upwind": (
        320,
        320,
        0,
        320,
        0,
        0,
        192,
        0,
        320,
        192,
        192,
        320,
        0,
        192,
        192,
        192,
        0,
        320,
    ),
    "D5-cyl-mixed-skew-upwind": (
        320,
        320,
        320,
        320,
        0,
        0,
        192,
        128,
        320,
        192,
        192,
        320,
        0,
        192,
        192,
        192,
        0,
        0,
    ),
    "D6-two-cyl-two-patches-skew-linear": (
        448,
        448,
        448,
        448,
        0,
        0,
        448,
        48,
        448,
        448,
        448,
        448,
        0,
        448,
        448,
        448,
        0,
        0,
    ),
    "D7-nondyadic-cyl-skew-upwind": (
        288,
        288,
        288,
        288,
        0,
        0,
        192,
        64,
        288,
        160,
        160,
        288,
        0,
        160,
        160,
        160,
        0,
        0,
    ),
    "D8-nondyadic-grid-skew-linear": (
        352,
        352,
        352,
        352,
        0,
        0,
        352,
        284,
        352,
        352,
        352,
        352,
        0,
        352,
        352,
        352,
        0,
        0,
    ),
    "D9-cyl-mixed-rotation-vanleer": (
        320,
        320,
        0,
        320,
        0,
        0,
        192,
        64,
        320,
        192,
        192,
        320,
        0,
        192,
        192,
        192,
        0,
        320,
    ),
    "D10-nondyadic-grid-quick": (
        352,
        352,
        352,
        352,
        0,
        0,
        192,
        131,
        352,
        192,
        192,
        352,
        0,
        192,
        192,
        192,
        0,
        0,
    ),
}

#: The zero-slot census (§2.4 of the B33 plan), measured on the same 3 232 rows.
#: ``ZERO_SLOT_ROWS`` carry at least one **live** face slot whose coefficient is
#: exactly ``+-0.0``; ``ZERO_SLOT_FACES`` of ``LIVE_FACES`` such entries would be
#: discarded by a bits-are-zero drop rule.
ZERO_SLOT_ROWS = 1920
ZERO_SLOT_FACES = 5100
LIVE_FACES = 15457


def _wall_row(*args):
    """The underscore-private row hook (api §4). ``from ._blockamr import *``
    skips underscore names, so it is reached on the extension module itself.

    Resolved per call rather than at import: the falsification matrix, its
    controls and the census are pure numpy, and this keeps them runnable
    *before* a rebuild — which is where a defect in the plan's own matrix is
    cheapest to find. It was: see ``COVERAGE``'s note on ``face-index``.
    """
    return blockamr._blockamr._wall_row_div_ghost_cell(*args)


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
    """Drop the caches while AMReX is still up.

    The memoised levels own device memory — ten meshes, each with a ``CellField``
    and a three-component ``FaceField``. Left in module globals they are torn
    down at *interpreter* exit, which is after ``blockamr_session`` has finalized
    AMReX, and freeing a device allocation into a destroyed CUDA context aborts
    (measured: ``CUDA error 709``, after a fully green run). A module-scoped
    finalizer runs before the session-scoped one, so this is simply the right
    place to let go.
    """
    yield
    _CASES.clear()
    _V1.clear()


def _case(name, max_size=None):
    """``(mesh, term, geom, ba, dm, flux)`` — one configuration, v1 side resolved."""
    key = (name, max_size)
    if key not in _CASES:
        _CASES[key] = _build_case(name, max_size)
    return _CASES[key]


def _build_case(name, max_size=None):
    from blockamr.mesh import Mesh

    bodies, ibm_bc, lo, hi, velocity, scheme = CONFIGS[name]
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    geom = blockamr.Geometry(box, blockamr.RealBox(list(lo), list(hi)), 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(N if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
    field = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc=ibm_bc)
    face = FaceField(mesh, ncomp=1, ngrow=1, name="phi")
    update_face_fluxes(face[0], velocity, geom, t=0.0)
    eqn = Equation(exp.div(face, field), schemes={"Div": scheme})
    _resolve_schemes(eqn.explicit_terms, eqn.schemes)
    return mesh, eqn.explicit_terms[0], geom, ba, dm, face


def _width(term):
    """The equation's band width — the driver's own rule (``driver.py:148``)."""
    return int(getattr(term.scheme, "stencil_width", 1))


def _face_value(term):
    """§4's mapping, in one line: ``Linear`` is central, everything else — the
    first-order ``Upwind`` and the width-2 ``vanLeer``/``quick`` that DEGRADE to
    it inside the band (D1) — takes the upwind cell.

    This is v1's ``_face_weights`` bit, and B36 made exactly this call at the
    driver — see ``GhostCellDiv.wall_extras``.
    """
    if getattr(term.scheme, "type", None) == "Linear":
        return blockamr.DivFaceValue.Central
    return blockamr.DivFaceValue.Upwind


def _v1_side(name):
    """``(ctx, rows, arms, flux, central)`` of one configuration, ``coeff = 1.0``."""
    if name not in _V1:
        mesh, term, _geom, _ba, _dm, _face = _case(name)
        width = _width(term)
        ctx = _context(term, mesh.ibm, 0, 1, 0.0, width)
        flux = _band_face_flux(term.coefficient, 0, ctx.band)
        rows = _face_balance_rows(
            ctx,
            axes=(0, 1, 2),
            flux=flux,
            weight_self=_face_weights(term.scheme, flux),
            coeff=float(term.coeff),
            ncomp=1,
            stride=STRIDE,
        )
        # The hand assembly above is the PRODUCTION call, checked rather than
        # assumed: `GhostCellDiv.rows` is what an evaluate reaches, and a
        # divergence between the two would make the whole oracle a private
        # re-derivation (oracle discipline, plan §8.5).
        produced = BOUNDARY_SCHEMES[("div", "ghostCell")](term.scheme).rows(
            term, mesh.ibm, 0, 1, 0.0, width
        )
        np.testing.assert_array_equal(rows.a.view(np.int64), produced.a.view(np.int64))
        np.testing.assert_array_equal(rows.c.view(np.int64), produced.c.view(np.int64))
        np.testing.assert_array_equal(rows.stencil, produced.stencil)
        arms = {(d, s): _neighbour(ctx, d, s) for d in range(3) for s in (1, -1)}
        central = getattr(term.scheme, "type", None) == "Linear"
        _V1[name] = (ctx, rows, arms, flux, central)
    return _V1[name]


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
    with ``names = sorted(mesh.bodies)``; the patch index is the position in that
    list, which is what ``IbmGeometry.patch`` carries. Every datum in the
    acceptance set is constant, so the compiled ``gamma(t)`` is the ``Constant``
    tag and no transcendental is on any path here.
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


def _mfs(face):
    """The three face-flux ``MultiFab``s of a level, in ``x, y, z`` order."""
    return tuple(face[0][d].mf for d in range(3))


def _bits(value):
    """The raw ``int64`` of one f64. ``==`` on floats cannot see ``-0.0``, and
    this operator produces signed zeros by the plane (H-6)."""
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _v1_row(rows, r):
    """v1's row ``r`` as ``(ordered [(index, a)], c)``, in slot order.

    The canonicalisation is **structural, not value-based** — see the module
    docstring. v1 keeps a slot at every face whose neighbour is not fluid (the
    slot is allocated by ``_blank``, left pointing at the target and never
    written) and the functor emits nothing there; those, and only those, are
    dropped. A ``bits == 0`` rule would additionally discard 5 100 *live*
    entries whose coefficient happens to be ``+-0.0``.
    """
    stencil, a = rows.stencil[r], rows.a[r]
    target = tuple(int(v) for v in stencil[0])
    entries = [(tuple(int(v) for v in stencil[k]), float(a[k])) for k in range(STRIDE)]
    kept = [entries[0]] + [e for e in entries[1:7] if e[0] != target] + entries[7:]
    return kept, float(rows.c[r][0])


def _compiled_row(ct, g, data, robin, geom, mfs, face_value, cell, n=0, t=0.0):
    """The compiled row at one cell, in the same shape as :func:`_v1_row`."""
    entries, c = _wall_row(ct, g, data, robin, geom, t, *mfs, face_value, *cell, n)
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


def _weight_self(flux, step, central, gt=False):
    """v1's ``_face_weights`` at one face. ``>=``, never ``>`` (H-8)."""
    if central:
        return 0.5
    positive = (flux > 0.0) if gt else (flux >= 0.0)
    if step == 1:
        return 1.0 if positive else 0.0
    return 0.0 if positive else 1.0


def _model_row(ctx, arms, flux, central, r, mutant=None):
    """The v2 functor, in numpy: accumulate the six faces, then emit in v1's slot
    order. ``mutant`` injects exactly one defect.

    This is the object the falsification matrix is measured on. It is *not* an
    independent oracle — the parity rows pin it to the compiled row first — so a
    mutant caught here is a defect the compiled pair would also carry.

    The two ordering mutants change the loop order in **both** passes, which is
    what a transcription following api §5.3's sketch would actually do: the C++
    writes the same ``for dd / for s`` twice.
    """
    cell = tuple(int(v) for v in ctx.target[r])
    dx = ctx.dx
    steps = (-1, 1) if mutant == "order-step" else (1, -1)
    axes = (2, 1, 0) if mutant == "order-axis" else (0, 1, 2)
    s_P = ctx.sdf[r]

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

            face = 1 if step == 1 else 0
            if mutant == "face-index":
                face = 1 - face
            fl = flux[r, d, face]
            ws = _weight_self(fl, step, central, gt=(mutant == "weight-gt"))
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
                # H-6: `0.0 + nbp` and not `nbp`.
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

    entries = [(cell, diag)]
    order = visited if mutant in ("order-step", "order-axis") else sorted(visited)
    entries.extend(arm[slot] for slot in order)
    for q in range(8):
        donor = tuple(int(v) for v in ctx.donor[r, q])
        if mutant == "donor-index":
            donor = (donor[0] + 1, donor[1], donor[2])
        entries.append((donor, float(wdon[q])))
    return entries, float(cacc)


def _caught(name, mutant):
    """WALL rows on which ``mutant`` differs from v1."""
    ctx, rows, arms, flux, central = _v1_side(name)
    n = 0
    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        ok, _why = _same(_model_row(ctx, arms, flux, central, r, mutant), _v1_row(rows, r))
        if not ok:
            n += 1
    return n


# ===========================================================================
# 1. P-1..P-10 — v1 <-> v2 row parity, BITWISE, on all ten configurations
# ===========================================================================


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_compiled_row_is_v1s_row_bitwise(blockamr_session, name):
    """**The acceptance bar** (review.md §4 Q52(a), item ii).

    For every ``WALL`` cell: the ordered ``(index, a)`` sequence the compiled
    functor emits, and the constant it accumulates, equal v1's
    ``_face_balance_rows`` row on the raw bits. No tolerance — a residual
    difference is a bug, not noise: there is no libm in the closure chain,
    contraction is pinned off by the per-file flags, and the numpy model was
    proven bitwise against v1 before the build.

    The **model** is pinned here too, in the same statement. That is the link
    that makes the falsification matrix below a claim about the shipped code
    rather than about a numpy script beside it.

    D9 and D10 are ``vanLeer`` and ``QUICK`` — **width-2** schemes, whose wall
    row is nonetheless the width-1 upwind one. That is the D1 degrade measured
    rather than asserted, and it is what ``DivFaceValue.Upwind`` means for them.
    """
    mesh, term, geom, ba, dm, face = _case(name)
    _bodies, ibm_bc, _lo, _hi, _vel, _scheme = CONFIGS[name]
    ctx, rows, arms, flux, central = _v1_side(name)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)
    mfs = _mfs(face)
    face_value = _face_value(term)

    nwall = int(ctx.at_wall.sum())
    assert nwall == NWALL[name], f"{name}: {nwall} wall rows, expected {NWALL[name]}"
    assert data.nrows == nwall, f"the compiled data has {data.nrows} rows, v1's band has {nwall}"

    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        cell = tuple(int(v) for v in ctx.target[r])
        want = _v1_row(rows, r)
        ok, why = _same(_compiled_row(ct, g, data, robin, geom, mfs, face_value, cell), want)
        assert ok, f"{name}: compiled row at {cell} differs from v1 — {why}"
        ok, why = _same(_model_row(ctx, arms, flux, central, r), want)
        assert ok, f"{name}: the numpy model at {cell} differs from v1 — {why}"


def test_the_row_hook_agrees_with_v1_across_box_seams(blockamr_session):
    """**Rider D-1's own discriminating row** (B33-R I-2).

    Every other row-hook row runs on a single-box level, where B32's fab-box
    selection and the shipped valid-box-first ``localBoxOf`` agree — so nothing
    in the suite would notice a regression of D-1. Here the same D5
    configuration is decomposed at ``max_size = 8`` (eight boxes, asserted):
    B33-R measured that under the old fab-box-first selection **80 of these 320
    WALL cells resolve a non-owner box at the seams** and the hook goes wrong,
    while the shipped selection reproduces v1 on all 320
    (``b33r-d1-probe-attempt1.log``).

    The v1 oracle is the single-box case: rows are compared **by cell**, so the
    compiled side's decomposition is exactly and only what varies.
    """
    name = "D5-cyl-mixed-skew-upwind"
    mesh, term, geom, ba, dm, face = _case(name, max_size=8)
    _bodies, ibm_bc, _lo, _hi, _vel, _scheme = CONFIGS[name]
    ctx, rows, _arms, _flux, _central = _v1_side(name)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)
    mfs = _mfs(face)
    face_value = _face_value(term)

    nbox = sum(1 for _ in blockamr.MFIterator(blockamr.MultiFab(ba, dm, 1, 0)))
    assert nbox == 8, f"vacuous: the level did not decompose ({nbox} boxes)"

    checked = 0
    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        cell = tuple(int(v) for v in ctx.target[r])
        ok, why = _same(
            _compiled_row(ct, g, data, robin, geom, mfs, face_value, cell), _v1_row(rows, r)
        )
        assert ok, f"{name}: compiled row at {cell} differs from v1 across box seams — {why}"
        checked += 1
    assert checked == NWALL[name], checked


# ===========================================================================
# 2. P-11, P-12, P-13 — the falsification matrix, its controls, and the
#    canonicalisation the matrix rests on (Q35, and §2.4's corollary)
# ===========================================================================


def test_the_falsification_matrix_is_reproduced_exactly(blockamr_session):
    """**Q35, permanently in-suite.** Eighteen defects, counted row by row, over
    all ten configurations and all 3 232 wall rows.

    Asserted as an exact tuple rather than "> 0" (B30b-R's S-6 shape):
    over-coverage fails too, because the matrix is the record of what each
    configuration *can* see. The entries that pay for the configuration set are
    listed on ``COVERAGE``; the headline is ``arm-raw`` (H-6), caught on exactly
    **three** configurations and invisible on the seven a first guess would have
    built the suite from.

    Every defect that is caught is caught by at least three configurations, so
    no row of this suite is load-bearing alone.
    """
    measured = {}
    total_wall = 0
    for name in CONFIGS:
        ctx, _rows, _arms, _flux, _central = _v1_side(name)
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
    for mutant, column in zip(MUTANTS, zip(*(COVERAGE[name] for name in CONFIGS))):
        seen = sum(1 for v in column if v)
        assert (seen == 0) == (mutant in CONTROLS), mutant
        assert seen == 0 or seen >= 3, f"{mutant} is caught by only {seen} configurations"


def test_the_four_controls_move_no_bit_anywhere(blockamr_session):
    """**Controls, and they are labelled as such.** Four reassociations that
    change nothing on any configuration:

    * ``scale-assoc`` — ``step*(f/dx)`` for ``(step*f)/dx``: exact, ``step`` is
      ``+-1``;
    * ``nb-complement`` (H-7) — ``scale - self_part`` for ``scale*(1 - w)``:
      exact only because ``w`` is one of ``{0, 0.5, 1}`` here, which is why this
      is *recorded* rather than acted on — a future non-trivial face weight must
      not inherit the permission;
    * ``dG-assoc`` — ``s_P + step*(dx*n)`` for ``s_P + (step*dx)*n``: exact,
      ``step`` is ``+-1``;
    * ``weight-gt`` (H-8) — ``f > 0`` for ``f >= 0``: the two differ only at
      ``f = +-0.0``, where ``scale`` is ``+-0.0`` and both ``scale*w`` and
      ``scale*(1-w)`` carry ``scale``'s own sign whichever weight is chosen.

    A change that made this row red would mean the arithmetic around those sites
    had stopped being exact — which is information, and the opposite of what the
    caught mutants report.
    """
    caught = {mutant: {name: _caught(name, mutant) for name in CONFIGS} for mutant in CONTROLS}
    assert caught == {m: dict.fromkeys(CONFIGS, 0) for m in CONTROLS}


def test_a_bits_are_zero_drop_rule_would_discard_live_face_entries(blockamr_session):
    """**§2.4's corollary, measured** — why this file's canonicalisation is
    structural and B32's is not reused.

    An upwind face on the wrong side has ``nb_part = scale * 0.0``, so its
    coefficient is exactly ``+-0.0`` while the slot is perfectly **live**: v1
    wrote it, the stencil entry names the neighbour, and the compiled functor
    emits it. B32's ``_v1_row`` drops a slot on its bits being zero, which for
    ``div`` would silently delete those entries from *both* sides and make the
    parity claim weaker than it looks — and, where only one side had them, wrong.

    The census is pinned so the rule cannot be "simplified" back: 1 920 of 3 232
    rows carry at least one such entry, 5 100 of 15 457 live face entries in
    total.
    """
    rows_with = faces_zero = faces_live = 0
    for name in CONFIGS:
        ctx, rows, _arms, _flux, _central = _v1_side(name)
        for r in range(ctx.nrows):
            if not ctx.at_wall[r]:
                continue
            entries, _c = _v1_row(rows, r)
            faces = entries[1:-8]
            faces_live += len(faces)
            zero = sum(1 for _i, a in faces if a == 0.0)
            faces_zero += zero
            rows_with += bool(zero)

    assert (rows_with, faces_zero, faces_live) == (
        ZERO_SLOT_ROWS,
        ZERO_SLOT_FACES,
        LIVE_FACES,
    ), f"the zero-slot census moved: {(rows_with, faces_zero, faces_live)}"


# ===========================================================================
# 3. P-14..P-16 — the pair through the frame, over real fabs
# ===========================================================================


def PHI(i, j, k):
    """The sweep rows' field, as a function of the global index.

    **Quadratic on purpose** (B32's deviation 6). A linear field on a
    divergence-free flux makes ``div(u phi)`` exactly ``u . grad phi`` — a
    constant — and then "v2 did not write this cell" and "v2 wrote the same
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


#: The interior kernel each width-1 configuration's ``Div`` scheme names.
_INTERIOR = {"linear": "div_linear_acc", "upwind": "div_upwind_acc"}

#: WALL rows whose constant is non-zero, per configuration — measured. Under
#: ``Upwind`` a wall face that is an *outflow* face has ``weight_self = 1``, so
#: ``nb_part`` is exactly ``0.0`` and the wall enters neither the donors nor the
#: constant of that row. v1 agrees cell for cell (the parity rows), so this is
#: the operator's behaviour and not a missing datum.
DATUM_ROWS = {"D5-cyl-mixed-skew-upwind": 192}

#: ``(WALL cells where v1 and v2 land on different bits, max |delta bits|)`` per
#: sweep configuration — **measured post-build at B33 and pinned exactly**
#: (B32-R's S-1, offered to B33 verbatim; both numbers of a pair come from the
#: same run — ``b33-q50-residual-attempt2.log`` — per B32-R's I-1). The whole
#: residual is the two consumers' floating-point contraction and nothing else;
#: see the row's docstring. A toolchain bump that moves either number is a real
#: observable change: re-measure, re-pin, and record it in the ledger next to
#: Q50.
WALL_RESIDUAL = {
    "D3-cyl-dir-rotation-linear": (144, 17),
    "D5-cyl-mixed-skew-upwind": (213, 133),
}


@pytest.mark.parametrize("name", list(WALL_RESIDUAL))
def test_the_sweep_is_the_pairs_own_row_and_v1s_residual_is_its_consumers_fma(
    blockamr_session, name
):
    """**The pair through the frame** (Q52(a), item iii) — Q50's attributed
    sentence, written for ``div``, with ``div``'s own number.

    Interior sweep plus wall sweep, against v1's interior sweep plus
    ``apply_band_rows``, on the *same* ``phi`` and the *same* face fluxes, over
    **eight boxes** so the row map's cross-box concatenation is what is being
    exercised.

    What is measured:

    * **FLUID** — bitwise equal, every cell. Both sides run the *same*
      ``div_*_acc`` and neither writes a FLUID cell afterwards. ``PHI`` is
      quadratic so this is not the equality of two zeros.
    * **WALL** — v2's output is bitwise ``_dot(row, fused=False)`` and v1's is
      bitwise ``_dot(row, fused=True)``, from **the same row**. The rows are
      identical (the parity rows above), so the entire residual is the two
      consumers' floating-point contraction and nothing else:
      ``band_table.cpp:688`` carries no per-file FP flags and takes nvcc's
      default ``--fmad=true``, while ``ApplySink::linear`` is inlined into this
      pair's ``--fmad=false`` TU and does not. **The flag that buys row parity is
      the flag that costs sweep parity**, and the count of WALL cells where the
      two land differently is asserted non-zero *and* pinned exactly, so the
      finding can neither quietly heal nor quietly grow.
    * **SOLID** — excluded, and the exclusion is asserted **load-bearing**
      (OPEN-C, review.md §4 Q49(b)): v1's ``band(1)`` is ``depth <= 1`` and so
      carries every solid cell as an ``nnz = 0, c = 0`` row, making v1's first
      ``Overwrite`` term write exactly ``0.0`` there, while v2's frame returns
      before the sink at ``m != WALL``. B33 records it rather than resolving it.

    **Width-1 schemes only** (review.md §4 Q52(c)). For a width-2 ``Div`` the
    equation's band is ``depth <= 2``: v1 writes a *band row* — a width-1 upwind
    row — at every ``depth == 2`` cell, while v2 leaves the **degraded interior
    kernel's** value there (``div_vanleer_acc_ibm``'s per-cell degrade, Q42(a)).
    Same mathematics, different association: not bitwise, and not B33's to
    reconcile. The wide-scheme composition is owed at B36, where the bar is D1's
    own ``atol = 1e-12`` and not a bitwise one.
    """
    from blockamr.ibm.band_rows import band_table

    mesh, term, geom, ba, dm, face = _case(name, max_size=8)
    _bodies, ibm_bc, _lo, _hi, _vel, scheme = CONFIGS[name]
    assert _width(term) == 1, "the sweep row is scoped to width-1 schemes — Q52(c)"

    phi = _phi(ba, dm)
    mfs = _mfs(face)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)
    interior = getattr(blockamr, _INTERIOR[scheme])

    # v1: the untouched interior sweep, then the band rows in Overwrite mode.
    ctx = _context(term, mesh.ibm, 0, 1, 0.0, 1)
    flux = _band_face_flux(term.coefficient, 0, ctx.band)
    rows = _face_balance_rows(
        ctx,
        axes=(0, 1, 2),
        flux=flux,
        weight_self=_face_weights(term.scheme, flux),
        coeff=1.0,
        ncomp=1,
        stride=STRIDE,
    )
    out_v1 = blockamr.MultiFab(ba, dm, 1, 0)
    out_v1.set_val(0.0)
    interior(out_v1, phi, *mfs, geom, 1.0, 1)
    version = mesh.ibm.grid_version
    blockamr.apply_band_rows(
        out_v1, phi, band_table(rows, version), 1, blockamr.BandMode.Overwrite, 1.0, version
    )

    # v2: the same interior sweep, then the compiled pair, by keyword.
    out_v2 = blockamr.MultiFab(ba, dm, 1, 0)
    out_v2.set_val(0.0)
    interior(out_v2, phi, *mfs, geom, 1.0, 1)
    blockamr.wall_div_ghost_cell(
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
        flux_x=mfs[0],
        flux_y=mfs[1],
        flux_z=mfs[2],
        face_value=_face_value(term),
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
            solid_differ += _bits(value) != _bits(got_v1[key])
            continue
        if m != WALL:
            seen["fluid"] += 1
            assert _bits(value) == _bits(got_v1[key]), (
                f"a FLUID cell moved at {key}: v2 {value!r} vs v1 {got_v1[key]!r}"
            )
            continue
        seen[WALL] += 1
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

    assert seen[WALL] == NWALL[name], seen
    assert seen["fluid"] > 0 and seen[SOLID] > 0, seen
    assert wall_differ > 0, "vacuous: Q50 is only a finding where the two sides differ"
    assert (wall_differ, max_delta) == WALL_RESIDUAL[name], (
        f"the contraction residual moved: {wall_differ}/{NWALL[name]} WALL cells differ, "
        f"max |delta bits| {max_delta}, but the pinned measurement is "
        f"{WALL_RESIDUAL[name]} — either band_table.cpp's contraction changed or this "
        "pair's flags did; re-read Q50 first"
    )
    assert solid_differ > 0, "vacuous: OPEN-C is only a finding where the two sides differ"


def test_overwrite_then_add_composes_and_constant_scale_zero_drops_the_datum(blockamr_session):
    """**R2 and S4**, on this pair rather than on the frame's probe.

    ``Overwrite`` then ``Add`` doubles the wall cells exactly (the composition
    rule: the first term of an equation writes, every later one adds), and
    ``constant_scale = 0`` drops **exactly** the BC datum — the Krylov matvec of
    an affine operator is the linear part alone, and a residual datum there is a
    wrong operator, not a small error. ``Mixed`` is used so the datum is not
    zero.

    The expected values are the pair's **own recorded row**, dot-produced in the
    sink's order — not ``affine - linear == c``, which is false in floating point
    for any row with cancellation and would have to be relaxed to a tolerance to
    pass.

    The non-vacuity guard is deliberately **aggregate** ("some WALL cell has a
    non-zero datum") and not per cell, and that is a measured property of this
    operator rather than a weakening. Under ``Upwind`` a wall face that is an
    *outflow* face has ``weight_self = 1``, so ``nb_part`` is exactly ``0.0`` and
    the wall contributes nothing at all to that row — neither a donor term nor a
    constant. v1 does the same thing at the same cells (the parity rows prove it
    row for row), so ``c == 0.0`` there is the right answer and not a missing
    datum. The counts below are asserted so the two populations cannot silently
    swap.
    """
    name = "D5-cyl-mixed-skew-upwind"
    mesh, term, geom, ba, dm, face = _case(name)
    _bodies, ibm_bc, _lo, _hi, _vel, _scheme = CONFIGS[name]
    phi = _phi(ba, dm)
    mfs = _mfs(face)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)
    marker = _markers(ct, phi)
    face_value = _face_value(term)

    def sweep(mode, constant_scale):
        out = blockamr.MultiFab(ba, dm, 1, 0)
        out.set_val(0.0)
        blockamr.wall_div_ghost_cell(
            out, phi, ct, g, data, robin, geom, 0.0, 1.0, 1, mode, constant_scale, *mfs, face_value
        )
        return out

    affine = sweep(blockamr.WallMode.Overwrite, 1.0)
    first = _readback(affine)
    blockamr.wall_div_ghost_cell(
        affine,
        phi,
        ct,
        g,
        data,
        robin,
        geom,
        0.0,
        1.0,
        1,
        blockamr.WallMode.Add,
        1.0,
        *mfs,
        face_value,
    )
    doubled = _readback(affine)
    linear = _readback(sweep(blockamr.WallMode.Overwrite, 0.0))

    moved = with_datum = 0
    for key, value in first.items():
        if marker[key[:3]] != WALL:
            continue
        moved += 1
        assert doubled[key] == 2.0 * value, f"Add did not compose at {key}"
        entries, c = _compiled_row(ct, g, data, robin, geom, mfs, face_value, key[:3], key[3])
        with_datum += c != 0.0
        assert _bits(value) == _bits(_dot(entries, c, 1.0, fused=False)), key
        assert _bits(linear[key]) == _bits(_dot(entries, c, 0.0, fused=False)), key
        if c != 0.0:
            assert _bits(linear[key]) != _bits(value), (
                f"vacuous: constant_scale dropped nothing at {key}, whose datum is {c!r}"
            )
    assert moved == NWALL[name]
    assert with_datum == DATUM_ROWS[name], (
        f"{with_datum} of {moved} WALL rows carry a non-zero datum, pinned at "
        f"{DATUM_ROWS[name]}; under Upwind an OUTFLOW wall face has nb_part exactly "
        "0.0 and contributes neither a donor term nor a constant"
    )


# ===========================================================================
# 4. P-17, P-18 — the extension contract, and the declaration
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


def test_the_div_pair_appends_exactly_four_arguments_past_the_canonical_twelve(blockamr_session):
    """**Q29(f)'s extension, made a contract** (review.md §4 Q52(b)).

    The twelve are a *minimum*, and ``div`` is the first pair to need more: its
    row is a face balance, so it needs the three face-flux ``MultiFab``s and it
    needs to know which face-value rule the interior scheme uses. This row pins
    *what* was appended and *in which order*, with no defaults, so B36's single
    keyword call site has something to compile against.

    The prefix itself is not re-asserted here — ``test_ibm_laplacian_ghost_cell``
    already checks ``[:12] == CANONICAL_TWELVE`` over **every** registered
    ``wall_*`` attribute, so this pair entered that contract without an edit. The
    twelve are spelled out again only so a reader can see the boundary.

    The rejected alternatives, so they are not re-litigated: two registered entry
    points (the registry key is ``(operator, method)`` and B36 resolves one name
    per pair, so two names put scheme-specific dispatch back at the driver); a
    bare ``bool central`` (a bool at a sixteen-argument call site); and passing
    ``weight_self`` as a precomputed field (that is v1's host-side array and the
    per-evaluate device-to-host read the design removes).
    """
    fn = blockamr._blockamr.wall_div_ghost_cell
    args = _signature_args(fn)
    names = tuple(a for a, _d in args)

    assert names[:12] == CANONICAL_TWELVE, names
    assert names[12:] == APPENDED_FOUR, names
    assert len(names) == 16, names
    assert not any(d for _a, d in args), "no argument of a registered pair is defaulted"

    assert set(blockamr.DivFaceValue.__members__) == {"Central", "Upwind"}, (
        blockamr.DivFaceValue.__members__
    )
    assert int(blockamr.DivFaceValue.Central) == 0
    assert int(blockamr.DivFaceValue.Upwind) == 1

    module = blockamr._blockamr
    assert re.fullmatch(r"wall_[a-z_]+", "wall_div_ghost_cell")
    assert hasattr(module, "_wall_row_div_ghost_cell")
    assert not hasattr(blockamr, "_wall_row_div_ghost_cell"), (
        "the row hook is underscore-private and must not reach the package namespace"
    )


def test_the_v1_scheme_names_the_compiled_pair(blockamr_session):
    """The seam B36 flipped (review.md §4 Q49(g)).

    ``register`` raises on a second class for a taken key, and O4 forbids
    removing v1's, so the declaration landed **additively** on the existing
    ``GhostCellDiv``: ``rows()`` is untouched and nothing is deregistered. B36
    made ``WallEvaluation.apply`` call this kernel instead of ``rows()``.

    B33 deliberately shipped **no** scheme-to-``DivFaceValue`` helper, because a
    helper with no caller is exactly the speculative code CLAUDE.md §2 refuses.
    B36 gave it its caller: the mapping is ``GhostCellDiv.wall_extras``, whose
    one-line rule is :func:`_face_value` above, made at the driver's call site.
    """
    from blockamr.schemes.div_schemes import Linear

    scheme_cls = BOUNDARY_SCHEMES[("div", "ghostCell")]
    kernel = scheme_cls(interior_scheme=Linear()).build_cpp_kernel()
    assert kernel.name == "wall_div_ghost_cell"
    assert hasattr(blockamr, kernel.name)
    assert callable(getattr(blockamr, kernel.name))


# ===========================================================================
# 5. D-1, D-2 — the I-1/S-3 rider (B33-R, executed at B34)
#
# `wall_stage.H::stageFaceBox` used to resolve its CELL index through
# `localBoxOf`, which compares the index against the container's OWN boxes. For a
# cell-centred container the valid boxes TILE the level, so the first match owns
# the cell. For a FACE-centred one they OVERLAP on the shared face: a cell on a
# box seam lies in the face valid box of its own box AND of the box below, and
# the pass returned whichever `IndexArray()` reached first — measured at 32/32/20
# of D5's 320 WALL cells per axis at `max_size = 8`.
#
# It was latent only because every shipped row builds its `FaceField` with
# `ngrow = 1`: the wrong box's *fab* box still contained the cell's high face, so
# the staged values happened to be right. At `ngrow = 0` it stops being latent —
# 80 of 96 seam WALL cells raised. `localCellBoxOf` maps each candidate back to
# its CELL box first, and cell boxes tile.
#
# The rider lands here and not in `test_ibm_grad_ghost_cell.py` because
# `stageFaceBox` has **no grad caller** — a `grad` row reads no face field at all
# — so div's row hook is the fix's only observable surface. Both rows below are
# additive; no existing row in this file is touched.
#
# S-3's corrected sentence (the second throw named `stageMiss`'s "lies in no
# local box", but at that point a box *was* found) gets no dedicated row, and
# that is stated rather than papered over: after the I-1 fix the branch is
# unreachable for a legal cell. D-1's **zero raises**, where 80 of 96 seam cells
# raise today, is the honest non-vacuity statement.
# ===========================================================================


def _seam_case(name, ngrow, max_size=8):
    """The same configuration, decomposed, with the face flux at `ngrow`."""
    from blockamr.field import FaceField
    from blockamr.mesh import Mesh

    bodies, ibm_bc, lo, hi, velocity, scheme = CONFIGS[name]
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    geom = blockamr.Geometry(box, blockamr.RealBox(list(lo), list(hi)), 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
    field = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc=ibm_bc)
    face = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    update_face_fluxes(face[0], velocity, geom, t=0.0)
    eqn = Equation(exp.div(face, field), schemes={"Div": scheme})
    _resolve_schemes(eqn.explicit_terms, eqn.schemes)
    return mesh, eqn.explicit_terms[0], geom, ba, dm, face


@pytest.mark.parametrize("ngrow", [0, 1])
def test_the_row_hook_stages_the_owning_face_box_across_box_seams(blockamr_session, ngrow):
    """**D-1 / D-2** — B33-R's I-1, fixed and pinned.

    Every `WALL` cell of D5 at `max_size = 8`, with the face flux carried at
    `ngrow = 0` and at `ngrow = 1`, reproduces v1's row bitwise — 320 of 320,
    both times, and **raising nowhere**.

    `ngrow = 0` is the discriminating half: under the old fab-box selection a
    seam cell resolved a box whose fab box does not reach its high face, and 80
    of the 96 seam `WALL` cells raised outright. `ngrow = 1` is the control: the
    shipped configuration, which was already green, and which must stay bit-for-
    bit unmoved — the fix changes which box is *selected*, never an answer.

    The v1 oracle is the single-box case, compared **by cell**, so the compiled
    side's decomposition and its flux ghost width are exactly and only what
    varies.
    """
    name = "D5-cyl-mixed-skew-upwind"
    mesh, term, geom, ba, dm, face = _seam_case(name, ngrow)
    _bodies, ibm_bc, _lo, _hi, _vel, _scheme = CONFIGS[name]
    ctx, rows, _arms, _flux, _central = _v1_side(name)
    g, ct, data, robin = _v2(mesh, geom, ba, dm, ibm_bc)
    mfs = _mfs(face)
    face_value = _face_value(term)

    nbox = sum(1 for _ in blockamr.MFIterator(blockamr.MultiFab(ba, dm, 1, 0)))
    assert nbox == 8, f"vacuous: the level did not decompose ({nbox} boxes)"
    assert all(mf.n_grow() == ngrow for mf in mfs), "the flux was not built at this ghost width"

    seam = checked = 0
    for r in range(ctx.nrows):
        if not ctx.at_wall[r]:
            continue
        cell = tuple(int(v) for v in ctx.target[r])
        # a seam cell: its high face in some direction is a box boundary.
        seam += any((cell[d] + 1) % 8 == 0 for d in range(3))
        ok, why = _same(
            _compiled_row(ct, g, data, robin, geom, mfs, face_value, cell), _v1_row(rows, r)
        )
        assert ok, f"{name} (flux ngrow = {ngrow}): compiled row at {cell} differs from v1 — {why}"
        checked += 1
    assert checked == NWALL[name], checked
    assert seam > 0, "vacuous: no WALL cell of this level sits on a box seam"
