# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Rung 9 — operators in combination (verification plan §6).

Every rung below this one evaluates **one** term. This file asks the next
question: does an equation still mean the sum of its parts once the IBM path
is in it?

    §6.1  E(div + lap)        == E(div) + E(lap)
    §6.2  E(div)              == E(div + lap(0, T))     (a zero term is inert)
    §6.2  E(div + lap)        == E(lap + div)           (order is not data)
    §6.4  two fields, two methods, in one session       (cache keying)
    §6.3  E(a + b)            == E(a) + E(b) - E(source alone)  (penalization)

The wall reconstruction is *operator-independent* — it rewrites the field the
operators read, it does not change what an operator is — so linear terms must
superpose under IBM exactly as they do without it. A failure here is a shared
scratch buffer that was not zeroed, a reconstruction that depends on which term
asked for it first, or a table cached against the wrong key.

**§6.3 is why this file parametrizes over a scheme *list* rather than the whole
registry.** Superposition is a statement about linear operators, and
``vanLeer`` is not linear in ``T`` — its limiter divides one difference of
``T`` by another — so ``E(aT) != aE(T)`` for it. ``DIV_SCHEME_IS_LINEAR``
below records the split, read off the kernels rather than guessed from the
names: ``upwind`` *is* linear here because the flux field is fixed data, so
its branch is a constant per face.

Be precise about what the exclusion buys, though. The three identities below
sum *terms* evaluated on **one common field**; each term is the same function
of the same ``T`` on both sides, so they would hold for a nonlinear scheme
too, and measuring ``vanLeer`` against them confirms that. The exclusion is
kept because it is the plan's rule and because the moment a variant splits or
scales the operand (``E(a*T) == a*E(T)``, or ``E(T1 + T2) == E(T1) + E(T2)``)
it becomes load-bearing — not because ``vanLeer`` is expected to fail here.
``vanLeer``'s accuracy under IBM is rung 7's job (a convergence study),
not this file's.

The second §6.3 exception is the source-type method: if the wall correction is
*added* rather than written over, one ``evaluate`` carries one source and two
``evaluate``s carry two, so superposition holds only up to exactly one source
term. That is the only construction in the plan that makes "the correction ran
once per evaluate" observable through the public API at all — for the
overwrite-type methods (``ghostCell``) both spellings produce identical
numbers and the property is genuinely invisible from outside.

Pre-merge tier (verification plan §10): small meshes, a handful of seconds.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp
from blockamr.field import CellField, FaceField
from blockamr.ibm import Cylinder, FixedValue, Plane
from blockamr.mesh import Mesh
from blockamr.operators.div import update_face_fluxes
from blockamr.schemes.registry import SCHEME_REGISTRY

from .ibm_gaps import PENALIZATION, T6_DIRECT_FORCING_ROWS

BACKEND = "cpp"

N = 16  # superposition is an algebraic identity — it needs no resolution
NZ = 4  # thin in the cylinder axis direction; every probe is z-invariant

# ``quick`` and ``vanLeer`` declare stencil_width 2, so every field here is
# built wide enough for the widest scheme in the registry.
NGROW = 2

R = 0.2
CENTRE = (0.5, 0.5)
AXIS = 2

# The MMS quadratic of rungs 6-7: T(r) = A + B*(r^2 - R^2), so
# T|_R = A exactly and the scalar FixedValue datum is consistent with the
# interior data. laplacian(T) = 4B and div(u T) = u . grad T are both nonzero,
# which is what makes the sum of two terms a real probe rather than 0 == 0 + 0.
A_MMS = 0.3
B_MMS = 0.5

# The laplacian coefficient. Any value works; this one is the natural
# viscosity-sized number and keeps the two terms comparable in magnitude, so
# neither one can hide inside the other's rounding.
NU = 0.01

# Linearity in ``T`` for a **fixed** flux field, read off the cell-level
# kernels in ``src/bindings/blockAMR/stencil_kernels.cpp`` rather than
# inferred from the scheme name (verification plan §6.3):
#
#   divLinearCell  0.5*(s_l + s_r)                       - constant weights
#   divUpwindCell  branch on the *flux* sign, then s     - constant weights
#                  per face; linear because the flux is data here, not T
#   divQuickCell   0.375/0.75/-0.125 blend, branch on the flux sign again
#                  - constant weights, so linear (the plan calls quick
#                    nonlinear; the implementation is the unlimited QUICK)
#   divVanLeerCell vanleerCorr(d_up, d_down) = 2*d_up*d_down/(d_up + d_down)
#                  - a ratio of differences of T, and gated on their product's
#                    sign: NOT linear.
#
# Written as a table over the whole registry, not a filtered list, so adding a
# scheme without classifying it breaks ``test_every_div_scheme_is_classified``
# below instead of silently dropping out of the superposition grid
# (verification plan §10: both axes enter as data).
DIV_SCHEME_IS_LINEAR = {
    "linear": True,
    "upwind": True,
    "quick": True,
    "vanLeer": False,
}
LINEAR_DIV_SCHEMES = sorted(name for name, linear in DIV_SCHEME_IS_LINEAR.items() if linear)

# Operator -> the key its scheme is looked up under in ``Equation(schemes=...)``
# (``lookup_scheme`` tries the term's scheme_key, then its class name).
DIV_KEY = "Div"


# ---------------------------------------------------------------------------
# Helpers — mesh/field construction, analytic fills, result extraction
# ---------------------------------------------------------------------------


def _cylinder():
    return {"cyl": Cylinder(centre=CENTRE, radius=R, axis=AXIS)}


def _make_mesh(n=N, nz=NZ, bodies=None, periodic=(0, 0, 0)):
    """Mesh on the unit cube, ``n x n x nz`` cells, one box.

    Non-periodic by default: the MMS quadratic is not periodic, so a wrapped
    halo would contaminate the domain-edge cells for reasons that have nothing
    to do with the IBM. The exterior ghosts are seeded analytically instead
    (:func:`_fill_halo`).
    """
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, list(periodic))
    ba = blockamr.BoxArray(box)
    ba.max_size(max(n, nz))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {} if bodies is None else bodies
    return mesh


def _coords(mesh, lo, shape):
    """Cell-centre coordinate meshgrid for a box starting at index ``lo``."""
    geom = mesh.geom(0)
    dx = geom.cell_size()
    plo = geom.prob_lo()
    axes = [
        np.array([plo[d] + (lo[d] + i + 0.5) * dx[d] for i in range(shape[d])]) for d in range(3)
    ]
    return np.meshgrid(*axes, indexing="ij")


def _fill(field, mesh, func):
    """Fill every component of ``field`` from ``func(X, Y, Z, comp)``.

    Solid cells are seeded too: the IBM must reconstruct its near-surface
    stencil from its own BC, never lean on values found in the body.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3])
        for comp in range(field.ncomp):
            arr[:, :, :, comp] = func(X, Y, Z, comp)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)


def _fill_halo(field, mesh, func):
    """Seed the *ghost* cells from the same analytic ``func``, after
    ``fill_patch``.

    ``fill_boundary`` fills inter-box and periodic halos but leaves
    domain-exterior ghosts untouched, and an unfilled halo would contaminate
    every edge cell. Filling them analytically is an exact Dirichlet halo and
    lets the assertions cover the whole domain instead of an eroded interior.
    """
    mf = field.mf[0]
    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        for comp in range(field.ncomp):
            arr[:, :, :, comp] = func(X, Y, Z, comp)
        mf.copy_grown_from(mfi, arr)


def _r2(X, Y):
    return (X - CENTRE[0]) ** 2 + (Y - CENTRE[1]) ** 2


def _mms_scalar(X, Y, Z, comp):
    """T(r) = A + B*(r^2 - R^2): equals ``A_MMS`` on the cylinder surface."""
    return A_MMS + B_MMS * (_r2(X, Y) - R**2)


def _mms_vector(X, Y, Z, comp):
    """U_c(r) = (c + 1) * B * (r^2 - R^2): **zero** on the cylinder surface for
    every component, so it is consistent with a no-slip ``FixedValue(0.0)``.

    The per-component factor makes the three components genuinely different —
    a term that mixed components would otherwise be invisible.
    """
    return (comp + 1.0) * B_MMS * (_r2(X, Y) - R**2)


def _unit_velocity(x, y, z, t):
    return np.ones_like(x), np.ones_like(x), np.ones_like(x)


def _uniform_flux(mesh, ngrow=NGROW):
    """Face flux field for the uniform, divergence-free velocity (1, 1, 1)."""
    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    update_face_fluxes(ff[0], _unit_velocity, mesh.geom(0), t=0.0)
    return ff


def _field(mesh, name, ncomp, func, ibm_bc):
    field = CellField(mesh, ncomp=ncomp, ngrow=NGROW, name=name, ibm_bc=ibm_bc)
    _fill(field, mesh, func)
    _fill_halo(field, mesh, func)
    return field


def _scalar_case():
    """Cylinder mesh + the scalar MMS field + the uniform flux."""
    mesh = _make_mesh(bodies=_cylinder())
    T = _field(mesh, "T", 1, _mms_scalar, {"cyl": FixedValue(A_MMS)})
    return mesh, T, _uniform_flux(mesh)


def _vector_case():
    """Cylinder mesh + the vector MMS field (no-slip) + the uniform flux."""
    mesh = _make_mesh(bodies=_cylinder())
    U = _field(mesh, "U", 3, _mms_vector, {"cyl": FixedValue(0.0)})
    return mesh, U, _uniform_flux(mesh)


def _flat(results):
    """All valid cells of level 0, every component, as one flat array."""
    return np.concatenate([np.asarray(a).ravel() for a in results[0]])


def _sol(method=None, backend=BACKEND):
    """The fvSolution block: no ``"ibm"`` key at all means no IBM."""
    return {"backend": backend} if method is None else {"ibm": method, "backend": backend}


def _div(phi, field, scheme):
    return Equation(exp.div(phi, field), schemes={DIV_KEY: scheme})


def _lap(field, nu=NU):
    return Equation(exp.laplacian(nu, field))


def _both(phi, field, scheme, nu=NU):
    return Equation(exp.div(phi, field) + exp.laplacian(nu, field), schemes={DIV_KEY: scheme})


def _assert_nondegenerate(*arrays):
    """Guard: a probe made of zeros proves superposition trivially."""
    for i, arr in enumerate(arrays):
        assert np.max(np.abs(arr)) > 0.0, f"term {i} is identically zero — the probe is vacuous"


# ---------------------------------------------------------------------------
# The linear/nonlinear split is data, and it must cover the registry
# ---------------------------------------------------------------------------


def test_every_div_scheme_is_classified_linear_or_not():
    """A new div scheme must be classified before the superposition tests can
    say anything about it — so it enters this file as one dict entry, never as
    a new test body (verification plan §10)."""
    assert set(DIV_SCHEME_IS_LINEAR) == set(SCHEME_REGISTRY["div"])


# ---------------------------------------------------------------------------
# §6.1 — superposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", LINEAR_DIV_SCHEMES)
def test_equation_equals_the_sum_of_its_linear_terms(blockamr_session, scheme):
    """§6.1. The wall reconstruction is operator-independent, so linear terms
    superpose under IBM exactly as they do without it.

    One field, three evaluations — ``evaluate`` does not mutate its input (the
    rung-5 tier asserts that), so "the same data" is literally the same data
    and any difference is the IBM path leaking across terms.

    ``vanLeer`` is excluded because it is not linear in ``T`` (verification
    plan §6.3), not because it is expected to fail here — see the module
    docstring and ``DIV_SCHEME_IS_LINEAR``.
    """
    _mesh, T, phi = _scalar_case()
    sol = _sol("ghostCell")

    both = _flat(evaluate(_both(phi, T, scheme), t=0.0, solution=sol))
    div = _flat(evaluate(_div(phi, T, scheme), t=0.0, solution=sol))
    lap = _flat(evaluate(_lap(T), t=0.0, solution=sol))

    _assert_nondegenerate(div, lap)
    np.testing.assert_allclose(both, div + lap, rtol=1e-13)


# ---------------------------------------------------------------------------
# §6.1, the mixed-width case — hand-computed, one cell ring at a time
# ---------------------------------------------------------------------------

# A grid-aligned plane at x = X_PLANE with the fluid on its right, so every
# quantity below is one-dimensional and every number is a dyadic rational this
# module can compute by hand. ``T = x^2`` on ``dx = 1/16``: the cell centres,
# their squares and every difference of them are exact in binary64.
X_PLANE = 0.5
DX = 1.0 / N
# The first fluid column (its left face is the plane) and the depth of a column:
# column i has depth i - FIRST_FLUID + 1 for the cross stencil, because the
# plane is normal to x and nothing else in the mesh is solid.
FIRST_FLUID = N // 2


def _square(X, Y, Z, comp):
    """``T = x^2``: exact at every cell centre and every halo cell of this mesh."""
    return X * X


def _plane_case():
    """Plane mesh + ``T = x^2`` + the uniform flux, wide enough for ``quick``."""
    mesh = _make_mesh(bodies={"wall": Plane(point=(X_PLANE, 0.0, 0.0), normal=(1.0, 0.0, 0.0))})
    T = _field(mesh, "T", 1, _square, {"wall": FixedValue(X_PLANE**2)})
    return mesh, T, _uniform_flux(mesh)


def _one_box(results):
    """The single box's level-0 result as an ``(N, N, NZ)`` array."""
    arr = np.asarray(results[0][0])
    return arr.reshape(arr.shape[:3])


def test_terms_of_different_widths_compose_on_the_equations_band(blockamr_session):
    """§6.1 for an equation whose terms want **different** bands.

    ``div(quick)`` is width 2 and ``laplacian`` is width 1, so the two terms'
    own bands differ and the composition has to say what happens in the ring
    between them. The rule (design §6) is that the band is the equation's — the
    widest term's — and every term's rows cover it; outside its own band a
    term's row is its plain interior formula. Three rings, three hand-computed
    numbers, and each one falsifies a different way of getting it wrong:

    ============================  =====================  ===========================
    ring                          expected               what a wrong rule gives
    ============================  =====================  ===========================
    ``depth >= 3`` (the bulk)     ``2x + 2``             a band that leaked outward
    ``depth == 2``                ``2x - dx + 2``        ``2x + 2`` (no degrade) or
                                                         ``2x - dx`` (laplacian lost)
    ``depth == 1``                the wall row           — not this test's claim
    ============================  =====================  ===========================

    The numbers, on ``T = x^2`` with ``u = (1, 1, 1)`` and a plane normal to x
    (so the y and z contributions of both operators are identically zero):

    * ``laplacian(T) = (T(x+h) - 2T(x) + T(x-h))/h^2 = 2`` — exact for a
      quadratic at *any* width-1 cell, band ring included, which is why the
      laplacian's ring-2 row must reproduce it and not the wall;
    * ``div(u T)`` under **quick** is ``(pr - pl)/h = 2x`` exactly (the
      0.375/0.75/-0.125 blend is third-order and a quadratic is in its kernel);
    * ``div(u T)`` under the width-1 **upwind** it degrades to (D1) is
      ``(T(x) - T(x-h))/h = 2x - h`` exactly.

    So the ring at ``depth == 2`` is the one that can only come out right if
    the div degrades there *and* the laplacian still contributes its interior
    value there. ``==``, not a tolerance: every value is a dyadic rational.
    """
    _mesh, T, phi = _plane_case()
    eqn = Equation(exp.div(phi, T) + exp.laplacian(1.0, T), schemes={DIV_KEY: "quick"})

    out = _one_box(evaluate(eqn, t=0.0, solution=_sol("ghostCell")))

    x = (np.arange(N) + 0.5) * DX
    for i in range(FIRST_FLUID + 1, N):
        expected = 2.0 * x[i] + 2.0 - (DX if i == FIRST_FLUID + 1 else 0.0)
        assert (out[i] == expected).all(), (
            f"column {i} (depth {i - FIRST_FLUID + 1}): {out[i].ravel()[0]!r} != {expected!r}"
        )


# ---------------------------------------------------------------------------
# §6.2 — independence of a term from its neighbours
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", LINEAR_DIV_SCHEMES)
def test_a_term_result_is_unchanged_by_adding_a_zero_coefficient_term(blockamr_session, scheme):
    """§6.2. ``div(phi, T)`` must be the same number whether or not a laplacian
    sits beside it.

    ``laplacian(0.0, T)`` contributes an exact zero, so the two equations are
    the same equation — but only if the accumulation buffer is zeroed per
    evaluate and the reconstruction does not depend on which term asked for it
    first. Scratch reused across terms without zeroing, or an order-dependent
    reconstruction, shows up here and nowhere above.
    """
    _mesh, T, phi = _scalar_case()
    sol = _sol("ghostCell")

    alone = _flat(evaluate(_div(phi, T, scheme), t=0.0, solution=sol))
    beside = _flat(
        evaluate(
            Equation(
                exp.div(phi, T) + exp.laplacian(0.0, T),
                schemes={DIV_KEY: scheme},
            ),
            t=0.0,
            solution=sol,
        )
    )

    _assert_nondegenerate(alone)
    np.testing.assert_allclose(beside, alone, rtol=1e-14)


@pytest.mark.parametrize("scheme", LINEAR_DIV_SCHEMES)
def test_term_order_does_not_change_the_result(blockamr_session, scheme):
    """§6.2. ``div + lap`` and ``lap + div`` are the same equation.

    The tolerance is not bitwise on purpose, and this is the one place in the
    IBM suite where that is the right call: the accumulate kernels compute
    ``out += coeff * op(phi)`` and the compiler contracts that into an FMA, so
    swapping the two terms swaps which product is rounded once and which is
    rounded twice. That is a ~1 ulp property of the C++ backend's arithmetic,
    not of the wall treatment — an IBM defect (a reconstruction that saw a
    different term first) is an O(1) effect and would blow through ``rtol``
    by ten orders of magnitude.

    At ``NU = 0.01`` the two orders happen to agree bitwise; do **not** tighten
    this to ``assert_array_equal`` on the strength of that. Measured at
    ``NU = 0.3`` and ``NU = 0.137`` the same comparison differs by 2.6e-16 and
    6.0e-16 relative, on every scheme.
    """
    _mesh, T, phi = _scalar_case()
    sol = _sol("ghostCell")

    forward = _flat(evaluate(_both(phi, T, scheme), t=0.0, solution=sol))
    reversed_ = _flat(
        evaluate(
            Equation(
                exp.laplacian(NU, T) + exp.div(phi, T),
                schemes={DIV_KEY: scheme},
            ),
            t=0.0,
            solution=sol,
        )
    )

    _assert_nondegenerate(forward)
    np.testing.assert_allclose(reversed_, forward, rtol=1e-14)


# ---------------------------------------------------------------------------
# §6.4 — two fields in one session
# ---------------------------------------------------------------------------


@T6_DIRECT_FORCING_ROWS
def test_two_fields_with_different_methods_do_not_interfere(blockamr_session):
    """§6.4. ``solution["ibm"]`` is per **field**, so one simulation carries two
    methods at once — momentum on one, a transported scalar on another.

    The interleaving is the test. ``U`` is evaluated on a clean session, then
    ``T`` runs and builds its own tables, then ``U`` runs again: a table cached
    by mesh (or by grid generation) alone passes every single-field test in
    this suite and fails right here, because ``T``'s evaluation overwrote the
    entry ``U`` re-reads.

    ``assert_array_equal``, never ``allclose``: a tolerance here would permit
    exactly the coupling the test exists to forbid (verification plan §10).

    **Red today**, and the reason is a design question, not a bug: the only
    two operator methods in the registry are ``ghostCell`` and the degenerate
    ``noIbm``. ``directForcing`` still carries ``kind = "step"`` (it runs the
    jnp-mask path over the field between steps) and ``evaluate`` refuses it
    with the §8 sentence. Once T6 expresses direct forcing as wall rows it
    becomes an operator method, and this is the test that catches the shared
    cache the second method would expose.
    """
    mesh = _make_mesh(bodies=_cylinder())
    phi = _uniform_flux(mesh)
    U = _field(mesh, "U", 3, _mms_vector, {"cyl": FixedValue(0.0)})
    T = _field(mesh, "T", 1, _mms_scalar, {"cyl": FixedValue(A_MMS)})

    eqn_U = _both(phi, U, "linear")
    eqn_T = _both(phi, T, "linear")
    sol_U = _sol("directForcing")
    sol_T = _sol("ghostCell")

    u_alone = _flat(evaluate(eqn_U, t=0.0, solution=sol_U))
    t_alone = _flat(evaluate(eqn_T, t=0.0, solution=sol_T))
    u_both = _flat(evaluate(eqn_U, t=0.0, solution=sol_U))
    t_both = _flat(evaluate(eqn_T, t=0.0, solution=sol_T))

    _assert_nondegenerate(u_alone, t_alone)
    np.testing.assert_array_equal(u_both, u_alone)
    np.testing.assert_array_equal(t_both, t_alone)


def test_two_fields_with_different_wall_data_do_not_interfere(blockamr_session):
    """§6.4, the half that is reachable today — same property, one method.

    Two scalar fields carry **identical interior data** and different
    ``ibm_bc`` data on the same patch of the same mesh. Everything a table
    caches is then shared between them except the one thing that must not be:
    ``gamma``, the field's own wall datum. A cache keyed by mesh or by grid
    generation would hand the second field the first field's wall value, and
    the interleaved re-evaluation below is what sees it.

    The two results are asserted **different** first: if the wall datum did not
    reach the numbers at all, the equalities below would hold for the wrong
    reason and the test would be worthless.

    Green today (the rows are rebuilt per evaluate and read the wall datum off
    the term's own field, so nothing between two fields is shared but the
    geometry), and it must stay green — this is the constraint on the T6 work
    above, not a prediction about it.
    """
    mesh = _make_mesh(bodies=_cylinder())
    phi = _uniform_flux(mesh)
    T = _field(mesh, "T", 1, _mms_scalar, {"cyl": FixedValue(A_MMS)})
    S = _field(mesh, "S", 1, _mms_scalar, {"cyl": FixedValue(A_MMS + 1.0)})

    eqn_T = _both(phi, T, "linear")
    eqn_S = _both(phi, S, "linear")
    sol = _sol("ghostCell")

    t_alone = _flat(evaluate(eqn_T, t=0.0, solution=sol))
    s_alone = _flat(evaluate(eqn_S, t=0.0, solution=sol))
    t_both = _flat(evaluate(eqn_T, t=0.0, solution=sol))
    s_both = _flat(evaluate(eqn_S, t=0.0, solution=sol))

    _assert_nondegenerate(t_alone, s_alone)
    assert not np.array_equal(t_alone, s_alone), (
        "the two fields' wall data did not reach the result — the probe cannot "
        "distinguish a shared table from a per-field one"
    )
    np.testing.assert_array_equal(t_both, t_alone)
    np.testing.assert_array_equal(s_both, s_alone)


# ---------------------------------------------------------------------------
# §6.3 — the source-type exception: superposition up to exactly one source
# ---------------------------------------------------------------------------


@PENALIZATION
def test_penalization_source_is_added_once_per_evaluate(blockamr_session):
    """§6.3. A source-type method *adds* its correction instead of writing over
    the result, so superposition holds only up to exactly one source term::

        E(a + b)    = a + b + S     one evaluate, one source
        E(a) + E(b) = a + b + 2S    two evaluates, two sources

    and the assertion is therefore ``E(a+b) == E(a) + E(b) - E(S alone)``, not
    ``== E(a) + E(b)``. ``E(laplacian(0.0, U))`` is the source alone: the bulk
    term contributes an exact zero, so whatever comes back is the correction
    and nothing else.

    This is the whole reason the test exists. For the overwrite-type methods
    (``ghostCell``: reconstruct, then zero the band) applying the correction
    per term and applying it per evaluate produce **identical** numbers,
    because both halves are idempotent — so "the correction ran once" is
    genuinely unobservable through the Equation API and belongs in a design
    review. Under a source-type method it becomes a number, and this is the
    only construction in the plan that turns it into one.

    ``src`` is asserted nonzero for the same reason: a method that added
    nothing would satisfy the identity trivially and prove nothing.

    **Red**: ``penalization`` is not in the method registry, so
    ``IBM.lookup`` refuses the name. The infrastructure is already shaped for
    it — ``BandMode.Add`` is the kernel mode a source-type method's rows are
    applied with, next to ``Overwrite`` — so what is missing is the method
    class and its boundary schemes, not the schedule.
    """
    _mesh, U, phi = _vector_case()
    sol = _sol("penalization")

    both = _flat(evaluate(_both(phi, U, "linear"), t=0.0, solution=sol))
    div = _flat(evaluate(_div(phi, U, "linear"), t=0.0, solution=sol))
    lap = _flat(evaluate(_lap(U), t=0.0, solution=sol))
    src = _flat(evaluate(_lap(U, nu=0.0), t=0.0, solution=sol))

    _assert_nondegenerate(div, lap, src)
    np.testing.assert_allclose(both, div + lap - src, rtol=1e-12)
