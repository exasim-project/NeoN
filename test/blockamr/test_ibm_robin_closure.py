# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The compiled wall CLOSURE — ``schemes/boundary/robin.H`` and
``robin_closure.cpp`` (B30b).

**Conformance, not acceptance**, exactly like ``test_ibm_ghost_cell_cpp.py`` and
``test_ibm_wall_functors.py``: no row of the equation suite may read this hook,
now or later. What this file asserts is tasks.md §3's verify column for B30b —

    the closure reproduces v1's arithmetic **bitwise** on a ``RecordSink``
    cell — **both branches**, Dirichlet and ``beta != 0`` alike

— against v1's own production ``wall_closure``, imported read-only and never
copied. Every expected number here comes from calling it; no literal is
transcribed by hand.

**Bitwise means raw ``int64`` views.** Both branches carry a load-bearing
``-0.0`` — ``value_linear`` on every Dirichlet row and ``grad_linear`` on every
Neumann row — and ``np.testing.assert_array_equal`` cannot see the difference
between ``-0.0`` and ``+0.0``. The ``-0.0`` in ``grad_linear`` is the whole of
B44's structural finding (at ``alpha = 0`` the field coupling vanishes
identically and the row keeps only its constant), so it is pinned rather than
tolerated.

**This file makes no accuracy claim, and B30b changed no formula.** Q41 (user
decision, 2026-07-28) ports v1's ``beta != 0`` arm verbatim, staircase metric
and all; the three Neumann accuracy rows keep their
``B18_NEUMANN_WALL_ACCURACY`` strict-xfail markers and no recorded digit moves.
A red row here means the C++ transcription drifted, never that the formula is
wrong.

The configuration set, and why it is this one
---------------------------------------------
The closure has no grid; its "grid" is the tuple ``(alpha, beta, gamma, d,
dg)``, and on convenient inputs every transcription hazard is invisible. That is
B31's dyadic-vacuity trap in scalar form and it was **measured** before this
file was written: of eight plausible, algebraically-exact mutants, **six were
bit-invisible on all seven first-guess configurations**. ``K1``'s dyadic
Dirichlet numbers discriminate **nothing** — it is kept for the structure, the
signs, the shapes and the readable hex, and it is *not* parity evidence.

The tree can only pose three ``(alpha, beta)`` shapes (``ibm/bc.py``):
``FixedValue -> (1, 0)``, ``FixedGradient -> (0, 1)``, ``Mixed(f) -> (f, 1-f)``;
``d`` is an image-point distance and ``dg`` a ghost-centre distance, negative.
Every configuration in ``K`` below is inside that box; ``K_POLE`` alone is not,
and its own note says so and why. The coverage, measured, is::

    mutant                                        caught by
    M1  value_constant := -d*(gamma/den)          K2, K5
    M2  den := fma(-alpha, d, beta)               K5, K6      <- the FMA hazard
    M3  value_linear := beta*(1/den)              K4, K6
    M8  value_constant := gamma when beta == 0    K2
    M9  grad_linear := (0.0 - alpha)/den          K3, K7
    A1  atLinear := fma(dg, grad_linear, vl)      K2
    M12 grad_constant := gamma*(1/den)            K2, K6

``M2`` is the hazard the ``--fmad=false`` flag exists for, and it **cannot fire
on the Dirichlet branch at all** (``beta = 0`` makes fused and unfused agree
exactly). It is testable only under cancellation, which is why the near-pole
``Mixed(0.95)`` configuration ``K6`` is mandatory rather than decorative.
``test_each_hazard_is_discriminated_by_a_configuration_in_this_set`` keeps that
matrix a permanent, in-suite assertion: an edit that swaps in a "nicer" constant
and quietly makes this file vacuous turns it red.
"""

from fractions import Fraction

import numpy as np
import pytest

import blockamr
from blockamr.schemes.boundary.ghost_cell import wall_closure

# Underscore-private test bindings (api §4). `from ._blockamr import *` skips
# underscore names, so they are reached on the extension module itself.
_wall_closure_record = blockamr._blockamr._wall_closure_record
_wall_closure_device = blockamr._blockamr._wall_closure_device

CONSTANT = blockamr.GAMMA_CONSTANT
HARMONIC = blockamr.GAMMA_HARMONIC

#: ``read`` selector of both hooks: which of the closure's three readings the
#: probe emits into the sink.
VALUE, GRAD, AT = 0, 1, 2

#: ``(alpha, beta, gamma, d, dg)``. See the module docstring for the coverage
#: each one buys; ``K1`` buys none and says so in its own row.
K = {
    "K1-dirichlet-dyadic": (1.0, 0.0, 0.5, 0.25, -0.125),
    "K2-dirichlet": (1.0, 0.0, 0.3, 0.053, -0.029),
    "K3-neumann": (0.0, 1.0, 0.3, 0.031, -0.029),
    "K4-mixed-0.3": (0.3, 0.7, 0.3, 0.043, -0.029),
    "K5-mixed-0.6": (0.6, 0.4, 0.3, 0.097, -0.029),
    "K6-mixed-0.95-near-pole": (0.95, 0.05, 1.3, 0.053, -0.0625),
    "K7-neumann": (0.0, 1.0, 1.3, 0.097, -0.091),
}

#: ``den = beta - alpha*d`` is exactly ``0.0`` here: ``0.125 - 0.5*0.25``. v1
#: divides anyway and returns ``+-inf`` with no warning, no check and no
#: documentation, and B30b adds no guard (review.md §4 Q43(c)).
#:
#: ``(alpha, beta) = (0.5, 0.125)`` is NOT a shape the BC tree can pose (none of
#: ``(1,0)``, ``(0,1)``, ``(f, 1-f)``) — these are the plan's numbers, kept for
#: the arithmetic claim, which does not need reachability. The pole IS reachable
#: in-box: ``Mixed(0.8) -> (0.8, 0.2)`` with ``d = 0.25`` has ``den == 0.0``
#: exactly (review.md §4 Q46 records the ``d = (1-f)/f`` family).
K_POLE = (0.5, 0.125, 1.3, 0.25, -0.0625)


# ---------------------------------------------------------------------------
# the oracle: v1's production wall_closure, called on a one-row batch
# ---------------------------------------------------------------------------


def _v1(alpha, beta, gamma, d):
    """v1's four numbers for one (row, component), as plain floats.

    v1 is vectorised — ``alpha``, ``beta``, ``distance`` are ``(n,)`` and
    ``gamma`` is ``(n, ncomp)`` — but every operation in it is elementwise
    float64, so a one-row, one-component batch *is* the scalar arithmetic the
    C++ must reproduce, in exactly this order.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = wall_closure(
            np.array([alpha]),
            np.array([beta]),
            np.array([[gamma]]),
            np.array([d]),
        )
    return (
        float(w.value_linear[0]),
        float(w.value_constant[0, 0]),
        float(w.grad_linear[0]),
        float(w.grad_constant[0, 0]),
    )


def _v1_at(alpha, beta, gamma, d, dg):
    """v1's third read, through its own ``WallClosure.at``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        w = wall_closure(
            np.array([alpha]),
            np.array([beta]),
            np.array([[gamma]]),
            np.array([d]),
        )
        linear, constant = w.at(np.array([dg]))
    return float(linear[0]), float(constant[0, 0])


def _v1_read(cfg, read):
    """``(linear, constant)`` — the pair the probe's ``read`` selector emits."""
    alpha, beta, gamma, d, dg = cfg
    vl, vc, gl, gc = _v1(alpha, beta, gamma, d)
    if read == VALUE:
        return vl, vc
    if read == GRAD:
        return gl, gc
    return _v1_at(alpha, beta, gamma, d, dg)


# ---------------------------------------------------------------------------
# the comparison, and the compiled side
# ---------------------------------------------------------------------------


def _bits(x):
    """The raw bit pattern — the only comparison that sees ``-0.0``."""
    return int(np.float64(x).view(np.int64))


def _assert_bitwise(got, want, name):
    if _bits(got) == _bits(want):
        return
    raise AssertionError(
        f"{name}: compiled {got!r} vs v1 {want!r} "
        f"(raw {_bits(got)} vs {_bits(want)}; hex {float(got).hex()} vs {float(want).hex()})"
    )


def _robin(alpha, beta, gamma, ncomp=1, npatch=1, form=CONSTANT, param=None):
    """A one-patch ``RobinData`` carrying ``(alpha, beta)`` and a constant gamma."""
    gform = np.full((npatch, ncomp), form, dtype=np.int32)
    gparam = np.zeros((npatch, ncomp, 4), dtype=np.float64)
    if param is None:
        gparam[..., 0] = gamma
    else:
        gparam[...] = param
    return blockamr.RobinData(
        np.full(npatch, alpha, dtype=np.float64),
        np.full(npatch, beta, dtype=np.float64),
        gform,
        gparam,
    )


def _record(cfg, read, i=3, j=5, k=7, t=0.0):
    """``(entries, c)`` from the host hook on one ``RecordSink`` cell."""
    alpha, beta, gamma, d, dg = cfg
    return _wall_closure_record(_robin(alpha, beta, gamma), 0, 0, t, d, dg, read, i, j, k)


def _compiled(cfg, read):
    """``(linear, constant)`` — the one row the closure emitted."""
    entries, c = _record(cfg, read)
    assert len(entries) == 1, f"the closure emitted {len(entries)} linear entries, not one"
    return entries[0][3], c


# ---------------------------------------------------------------------------
# 1. the parity core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(K))
@pytest.mark.parametrize("read", [VALUE, GRAD, AT])
def test_the_closure_reproduces_v1_bitwise(blockamr_session, key, read):
    """The verify column, for both branches and all three reads.

    Nothing here is a tolerance in disguise: with no libm in the chain and
    contraction pinned off for this TU, a residual difference is a bug and stays
    red (review.md §4 Q29(d) refuses the ULP fallback).
    """
    got_a, got_c = _compiled(K[key], read)
    want_a, want_c = _v1_read(K[key], read)
    _assert_bitwise(got_a, want_a, f"{key} read={read} linear")
    _assert_bitwise(got_c, want_c, f"{key} read={read} constant")


def test_the_dyadic_dirichlet_configuration_tests_structure_and_not_parity(blockamr_session):
    """``K1`` discriminates **no** mutant and is not offered as parity evidence.

    Every value in it is exactly representable, so each of the eight
    algebraically-exact rewrites this file guards against reproduces it to the
    bit. What it *does* test is the plumbing a human can check by hand: the
    field order, the sign of the Dirichlet zero, the shapes, the hook. The
    numbers below are readable on purpose — ``den = -0.25``, so
    ``grad_linear = 4``, ``grad_constant = -2``, ``value_constant = 0.5``.
    """
    alpha, beta, gamma, d, _dg = K["K1-dirichlet-dyadic"]
    assert (alpha, beta, gamma, d) == (1.0, 0.0, 0.5, 0.25)
    vl, vc, gl, gc = _v1(alpha, beta, gamma, d)
    assert (vc, gl, gc) == (0.5, 4.0, -2.0)
    assert _bits(vl) == _bits(-0.0)

    value_a, value_c = _compiled(K["K1-dirichlet-dyadic"], VALUE)
    grad_a, grad_c = _compiled(K["K1-dirichlet-dyadic"], GRAD)
    assert (value_c, grad_a, grad_c) == (0.5, 4.0, -2.0)
    assert _bits(value_a) == _bits(-0.0)


def test_the_dirichlet_arm_carries_a_negative_zero_linear_part(blockamr_session):
    """``value_linear = 0.0 / -d`` is ``-0.0``, and the sink stores what it is
    handed. A negation computed by subtraction (``0.0 - alpha``) would give
    ``+0.0`` here and pass ``assert_array_equal``."""
    for key in ("K1-dirichlet-dyadic", "K2-dirichlet"):
        a, _c = _compiled(K[key], VALUE)
        assert _bits(a) == _bits(-0.0), f"{key}: value_linear is {a!r}, not -0.0"
        assert _bits(_v1_read(K[key], VALUE)[0]) == _bits(-0.0)


def test_the_neumann_arm_has_no_field_coupling_and_says_so_in_the_sign(blockamr_session):
    """B44's structural finding, pinned: at ``alpha = 0`` the ``a``-write
    vanishes identically (``grad_linear = -0.0 / 1``) and the row keeps only its
    constant. B30b reproduces it; it does not repair it (Q41)."""
    for key in ("K3-neumann", "K7-neumann"):
        a, c = _compiled(K[key], GRAD)
        assert _bits(a) == _bits(-0.0), f"{key}: grad_linear is {a!r}, not -0.0"
        # `grad_constant = gamma / 1.0` is gamma exactly on this branch.
        assert c == K[key][2]


def test_the_dirichlet_datum_is_not_simplified_to_the_datum_itself(blockamr_session):
    """H-e. The docstring says Dirichlet gives ``phi_w = value``, and in exact
    arithmetic it does — ``value_constant = -d*gamma/-d``. In float64 it does
    **not**: two roundings sit between them. Returning ``gamma`` there is the
    single most plausible "simplification" of this function, so the difference
    is asserted rather than assumed."""
    _linear, value_constant = _compiled(K["K2-dirichlet"], VALUE)
    gamma = K["K2-dirichlet"][2]
    assert _bits(value_constant) != _bits(gamma), (
        "vacuous: this configuration's value_constant happens to equal gamma exactly, so the "
        "shortcut it is meant to catch would pass"
    )
    _assert_bitwise(value_constant, _v1_read(K["K2-dirichlet"], VALUE)[1], "value_constant")


def test_the_third_read_is_the_closure_read_again_and_not_a_new_approximation(blockamr_session):
    """``at(dg) = value + dg * grad`` on the compiled side, bitwise, and equal to
    v1's own ``WallClosure.at``. The addition is itself a contraction site, which
    is what makes this row an FMA detector as well as a parity one."""
    for key, cfg in K.items():
        dg = cfg[4]
        vl, vc = _compiled(cfg, VALUE)
        gl, gc = _compiled(cfg, GRAD)
        al, ac = _compiled(cfg, AT)
        _assert_bitwise(al, vl + dg * gl, f"{key} atLinear")
        _assert_bitwise(ac, vc + dg * gc, f"{key} atConstant")
        want_l, want_c = _v1_read(cfg, AT)
        _assert_bitwise(al, want_l, f"{key} atLinear vs v1")
        _assert_bitwise(ac, want_c, f"{key} atConstant vs v1")


# ---------------------------------------------------------------------------
# 2. the seam: where alpha, beta and gamma come from
# ---------------------------------------------------------------------------


def test_the_closure_reads_alpha_and_beta_at_its_own_patch(blockamr_session):
    """The first production read anywhere of ``RobinView::alpha`` / ``beta``
    (B30a shipped them in place and unread).

    Two patches with different ``(alpha, beta)`` in one table: each patch's row
    must match its own v1 closure, so a hook that ignored ``patch`` — or read
    ``alpha`` from patch 0 and ``beta`` from patch 1 — turns red.
    """
    alpha = np.array([1.0, 0.6], dtype=np.float64)
    beta = np.array([0.0, 0.4], dtype=np.float64)
    gammas = (0.3, 0.3)
    d, dg = 0.097, -0.029
    gform = np.full((2, 1), CONSTANT, dtype=np.int32)
    gparam = np.zeros((2, 1, 4), dtype=np.float64)
    gparam[0, 0, 0], gparam[1, 0, 0] = gammas
    robin = blockamr.RobinData(alpha, beta, gform, gparam)

    for patch in (0, 1):
        entries, c = _wall_closure_record(robin, patch, 0, 0.0, d, dg, VALUE, 0, 0, 0)
        want_a, want_c = _v1(alpha[patch], beta[patch], gammas[patch], d)[:2]
        _assert_bitwise(entries[0][3], want_a, f"patch {patch} value_linear")
        _assert_bitwise(c, want_c, f"patch {patch} value_constant")


def test_the_datum_reaches_the_closure_through_gamma_at(blockamr_session):
    """Component ``n``'s constants use component ``n``'s gamma, bitwise. The
    linear parts are gamma-free and must be identical across components — which
    is v1's shape fact (one linear number per row, constants per component)
    restated as an assertion."""
    ncomp = 3
    gammas = [0.3, 1.3, -0.7]
    alpha, beta, d, dg = 0.6, 0.4, 0.097, -0.029
    gform = np.full((1, ncomp), CONSTANT, dtype=np.int32)
    gparam = np.zeros((1, ncomp, 4), dtype=np.float64)
    for n, g in enumerate(gammas):
        gparam[0, n, 0] = g
    robin = blockamr.RobinData(np.array([alpha]), np.array([beta]), gform, gparam)

    linear_parts = set()
    for n, gamma in enumerate(gammas):
        entries, c = _wall_closure_record(robin, 0, n, 0.0, d, dg, GRAD, 0, 0, 0)
        want_a, want_c = _v1(alpha, beta, gamma, d)[2:]
        _assert_bitwise(entries[0][3], want_a, f"component {n} grad_linear")
        _assert_bitwise(c, want_c, f"component {n} grad_constant")
        linear_parts.add(_bits(entries[0][3]))
    assert len(linear_parts) == 1, "grad_linear is gamma-free and may not vary by component"


def test_a_harmonic_datum_moves_the_closure_constants_with_time(blockamr_session):
    """The compiled ``GammaExpr`` reaches the closure, and the constants track
    ``gamma(t)``.

    **Not bitwise, and it says why**: B30a's D8 — the expected value would be a
    ``cos``/``sin`` of the *test's* libm compared against the *build's*, and two
    correctly-rounded libms need not agree in the last bit. Every other row in
    this file uses a ``Constant``-form datum for exactly that reason.
    """
    alpha, beta, d, dg = 0.6, 0.4, 0.097, -0.029
    a0, ac, asin, omega = 0.1, 0.7, -0.25, 3.0
    robin = _robin(alpha, beta, 0.0, param=np.array([a0, ac, asin, omega]), form=HARMONIC)
    for t in (0.0, 0.37):
        entries, c = _wall_closure_record(robin, 0, 0, t, d, dg, GRAD, 0, 0, 0)
        gamma = a0 + ac * np.cos(omega * t) + asin * np.sin(omega * t)
        want_a, want_c = _v1(alpha, beta, gamma, d)[2:]
        _assert_bitwise(entries[0][3], want_a, f"t={t} grad_linear")
        assert c == pytest.approx(want_c, rel=1e-14)


# ---------------------------------------------------------------------------
# 3. the device path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(K))
@pytest.mark.parametrize("read", [VALUE, GRAD, AT])
def test_the_closure_is_bitwise_the_same_on_the_device_path(blockamr_session, key, read):
    """Host, device and v1, all three.

    A ``__host__ __device__`` function that is never called from a kernel is
    never codegen'd for the device, so without this row B32 would be the first
    place a device compile of ``robin.H`` is attempted — and a device/host
    disagreement is a statement about the build's flags, not about the code.
    """
    alpha, beta, gamma, d, dg = K[key]
    got_a, got_c = _wall_closure_device(_robin(alpha, beta, gamma), 0, 0, 0.0, d, dg, read)
    host_a, host_c = _compiled(K[key], read)
    want_a, want_c = _v1_read(K[key], read)
    _assert_bitwise(got_a, host_a, f"{key} read={read} device vs host linear")
    _assert_bitwise(got_c, host_c, f"{key} read={read} device vs host constant")
    _assert_bitwise(got_a, want_a, f"{key} read={read} device vs v1 linear")
    _assert_bitwise(got_c, want_c, f"{key} read={read} device vs v1 constant")


# ---------------------------------------------------------------------------
# 4. the pole, unguarded
# ---------------------------------------------------------------------------


def test_the_singular_configuration_is_not_guarded(blockamr_session):
    """``den = beta - alpha*d`` is exactly ``0.0`` and **neither side raises**.

    ``Mixed(f)`` with ``d = (1-f)/f`` reaches it; v1 divides anyway and returns
    ``+-inf``, warning about nothing and documenting nothing. Q41's charter is
    verbatim and Q43(c) rules that verbatim includes unguarded: a raise where v1
    returns a number is a behavioural change in the one session whose entire
    claim is that nothing changed. This row exists so that a later well-meaning
    guard turns green to red and gets read as the change it is.

    ``gamma != 0`` and ``d != 0``, so no ``0/0`` arises and every result is a
    signed infinity — the comparison stays bitwise like every other row. **If a
    configuration is ever added here that does produce a NaN, its comparison
    must degrade to classification** (``isnan``/``isinf``/sign): NaN payloads are
    not contractual across host and device.
    """
    alpha, beta, gamma, d, dg = K_POLE
    assert beta - alpha * d == 0.0, "this configuration is meant to sit exactly on the pole"

    for read in (VALUE, GRAD, AT):
        got_a, got_c = _compiled(K_POLE, read)
        want_a, want_c = _v1_read(K_POLE, read)
        assert not np.isnan([want_a, want_c]).any(), "no 0/0 is expected at this configuration"
        assert np.isinf([want_a, want_c]).all(), "v1 returns +-inf at the pole"
        _assert_bitwise(got_a, want_a, f"pole read={read} linear")
        _assert_bitwise(got_c, want_c, f"pole read={read} constant")

        # and the device path agrees, infinities included, on every read
        dev_a, dev_c = _wall_closure_device(_robin(alpha, beta, gamma), 0, 0, 0.0, d, dg, read)
        _assert_bitwise(dev_a, got_a, f"pole device read={read} linear")
        _assert_bitwise(dev_c, got_c, f"pole device read={read} constant")


# ---------------------------------------------------------------------------
# 5. non-vacuity — the permanent half of Q35's falsify-before-trust
# ---------------------------------------------------------------------------
#
# These rows call no compiled hook. They assert properties of the CONFIGURATION
# SET itself, so that an edit which swaps in a rounder constant and quietly
# makes every parity row above unable to fail turns red here instead of passing
# silently. Each mutant is an algebraically-exact rewrite a competent author
# would write; the discrimination is measured on the numpy oracle, which is the
# same test as mutating the shipped source and needs no second build (O1).


def _fma(a, b, c):
    """Exactly-rounded ``a*b + c``; python 3.12 has no ``math.fma``, and
    ``float(Fraction)`` rounds correctly, so the emulation is exact."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _verbatim(alpha, beta, gamma, d, dg):
    """The six reads, exactly as ``robin.H`` spells them."""
    den = beta - alpha * d
    vl = beta / den
    vc = -d * gamma / den
    gl = -alpha / den
    gc = gamma / den
    return (vl, vc, gl, gc, vl + dg * gl, vc + dg * gc)


def _mutate(name, alpha, beta, gamma, d, dg):
    den = beta - alpha * d
    vl, vc, gl, gc, al, ac = _verbatim(alpha, beta, gamma, d, dg)
    if name == "M1":  # H-c: reuse grad_constant, saving a divide
        vc = -d * gc
    elif name == "M2":  # H-a: the fused denominator
        den = _fma(-alpha, d, beta)
        vl, vc, gl, gc = beta / den, -d * gamma / den, -alpha / den, gamma / den
    elif name == "M3":  # H-d: reciprocal multiply
        vl = beta * (1.0 / den)
    elif name == "M8":  # H-e: "Dirichlet is exact"
        vc = gamma if beta == 0.0 else vc
    elif name == "M9":  # H-f: negation by subtraction
        gl = (0.0 - alpha) / den
    elif name == "M12":  # H-d': reciprocal multiply, the other one
        gc = gamma * (1.0 / den)
    elif name == "A1":  # H-b: the fused third read
        return (vl, vc, gl, gc, _fma(dg, gl, vl), _fma(dg, gc, vc))
    elif name == "M4":  # control: sign fold
        gl = -(alpha / den)
    elif name == "M10":  # control: commuted product
        vc = (gamma * -d) / den
    elif name == "M11":  # control: antisymmetric subtraction
        den = -(alpha * d - beta)
        vl, vc, gl, gc = beta / den, -d * gamma / den, -alpha / den, gamma / den
    else:
        raise AssertionError(f"unknown mutant {name}")
    return (vl, vc, gl, gc, vl + dg * gl, vc + dg * gc)


def _differing_reads(name, cfg):
    return sum(1 for a, b in zip(_verbatim(*cfg), _mutate(name, *cfg)) if _bits(a) != _bits(b))


#: mutant -> the configurations measured to catch it. This IS the falsification
#: design (review.md §4 Q35): drop any of K2..K7 and a named hazard goes
#: untested.
COVERAGE = {
    "M1": ("K2-dirichlet", "K5-mixed-0.6"),
    "M2": ("K5-mixed-0.6", "K6-mixed-0.95-near-pole"),
    "M3": ("K4-mixed-0.3", "K6-mixed-0.95-near-pole"),
    "M8": ("K2-dirichlet",),
    "M9": ("K3-neumann", "K7-neumann"),
    "A1": ("K2-dirichlet",),
    "M12": ("K2-dirichlet", "K6-mixed-0.95-near-pole"),
}

#: Rewrites that IEEE sign symmetry and antisymmetric subtraction make exact.
#: Recorded as discriminating nothing so that nobody adds a row for them that
#: can never fail — and so that a build which somehow *did* move them is loud.
CONTROLS = ("M4", "M10", "M11")


def test_the_mutation_oracle_is_itself_the_v1_formula():
    """``_verbatim`` is a *third* transcription of the closure, and the whole
    ``COVERAGE`` matrix is measured against it. If it drifted, the coverage
    rows would keep passing while measuring a formula nobody ships — so it is
    pinned to v1 bitwise on every configuration (B30b-R S-1)."""
    for key, cfg in K.items():
        alpha, beta, gamma, d, dg = cfg
        want = (*_v1(alpha, beta, gamma, d), *_v1_at(alpha, beta, gamma, d, dg))
        names = (
            "value_linear",
            "value_constant",
            "grad_linear",
            "grad_constant",
            "at_linear",
            "at_constant",
        )
        for got, ref, name in zip(_verbatim(*cfg), want, names, strict=True):
            _assert_bitwise(got, ref, f"{key} oracle {name}")


@pytest.mark.parametrize("mutant", list(COVERAGE))
def test_each_hazard_is_discriminated_by_a_configuration_in_this_set(mutant):
    """Every named transcription hazard changes bits on at least one
    configuration this file drives — and on exactly the ones measured."""
    caught = tuple(key for key in K if _differing_reads(mutant, K[key]) > 0)
    assert caught == COVERAGE[mutant], (
        f"{mutant}: caught by {caught}, but the recorded coverage is {COVERAGE[mutant]} — "
        "the configuration set moved and the parity rows above may no longer be able to fail"
    )


def test_the_fma_hazard_needs_the_near_pole_configuration(blockamr_session):
    """Why ``K6`` is mandatory, as an assertion rather than a comment.

    ``M2`` is the hazard the per-file ``--fmad=false`` / ``-ffp-contract=off``
    exists for. It cannot fire anywhere on the Dirichlet branch — ``beta = 0``
    makes ``fma(-alpha, d, 0)`` and ``0 - alpha*d`` agree exactly — and on the
    Neumann branch ``alpha = 0`` does the same. Only a ``Mixed`` configuration
    under cancellation moves it, and it moves **all six** reads there.
    """
    for key in ("K1-dirichlet-dyadic", "K2-dirichlet", "K3-neumann", "K7-neumann"):
        assert _differing_reads("M2", K[key]) == 0, f"{key} unexpectedly discriminates M2"
    assert _differing_reads("M2", K["K6-mixed-0.95-near-pole"]) == 6
    assert _differing_reads("M2", K["K5-mixed-0.6"]) == 6


def test_the_dyadic_configuration_discriminates_nothing(blockamr_session):
    """``K1`` is the dyadic-vacuity trap, kept and labelled. Every mutant — the
    controls and the real hazards alike — reproduces it to the bit, which is
    why it is documented as structure/sign/shape evidence and not as parity
    evidence."""
    for mutant in list(COVERAGE) + list(CONTROLS):
        assert _differing_reads(mutant, K["K1-dirichlet-dyadic"]) == 0


@pytest.mark.parametrize("mutant", CONTROLS)
def test_the_safe_rewrites_are_bit_identical_everywhere(mutant):
    """The three rewrites that are provably safe, recorded so they are not
    re-litigated: ``-(alpha/den)``, ``(gamma*-d)/den`` and
    ``-(alpha*d - beta)``. They are still not written that way in ``robin.H`` —
    "provably equal" is a claim someone has to re-derive at every reading."""
    for key, cfg in K.items():
        assert _differing_reads(mutant, cfg) == 0, f"{mutant} moved bits on {key}"


# ---------------------------------------------------------------------------
# 6. the hooks' own error surface (api §9)
# ---------------------------------------------------------------------------


def test_the_hooks_refuse_a_patch_the_table_lacks(blockamr_session):
    robin = _robin(1.0, 0.0, 0.5)
    with pytest.raises(RuntimeError, match=r"patch 4 is outside the Robin table's 1"):
        _wall_closure_record(robin, 4, 0, 0.0, 0.25, -0.125, VALUE, 0, 0, 0)
    with pytest.raises(RuntimeError, match=r"patch 4 is outside the Robin table's 1"):
        _wall_closure_device(robin, 4, 0, 0.0, 0.25, -0.125, VALUE)


def test_the_hooks_refuse_a_component_the_table_lacks(blockamr_session):
    robin = _robin(1.0, 0.0, 0.5)
    with pytest.raises(RuntimeError, match=r"component 2 is outside the Robin table's 1"):
        _wall_closure_record(robin, 0, 2, 0.0, 0.25, -0.125, VALUE, 0, 0, 0)


def test_the_hooks_refuse_a_reading_the_closure_does_not_have(blockamr_session):
    """There are exactly three readings and the refusal names all three."""
    robin = _robin(1.0, 0.0, 0.5)
    with pytest.raises(RuntimeError, match=r"read 3 is not one of 0 .*1 .*2"):
        _wall_closure_record(robin, 0, 0, 0.0, 0.25, -0.125, 3, 0, 0, 0)


def test_the_recorded_row_names_the_cell_it_was_called_on(blockamr_session):
    """The closure is a function of its four arguments, but the row it emits is
    a row *at a cell* — the sink's entry carries the index the probe was called
    on, which is what makes this the ``RecordSink`` cell tasks.md §3 asks for."""
    entries, _c = _record(K["K2-dirichlet"], VALUE, i=3, j=5, k=7)
    assert [e[:3] for e in entries] == [(3, 5, 7)]
