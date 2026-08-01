# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``solution["projection"]``: the CELL-centred pressure route, opt-in beside nodal.

The nodal route is the default and the one the cavity Ghia profiles and the cylinder
Cd/Cl/St numbers are acceptance oracles for, so the first thing checked here is that
nothing selects the new route by accident.

What is checked:

* the DEFAULT is nodal, by assertion rather than by reading the code -- the nodal cache
  is built, the cell cache is not, and the cell entry point is monkeypatched to abort if
  it is reached at all;
* selecting ``"cell"`` builds the cell cache and NOT the nodal one, with
  ``MLNodeLaplacian`` monkeypatched to abort, so the two routes are exclusive rather
  than merely both-present;
* an unrecognised value RAISES, naming the accepted ones;
* the MAC face flux is divergence-free after a step on the cell route -- the shipped
  gate's own measure (``test_verification_projection.py::_max_face_divergence``);
* the divergence the cell route itself controls, ``div_f(interpolate(U))``, drops by
  orders of magnitude across the correction on a SMOOTH field, and the residual it
  leaves falls at second order. That residual is not zero by construction and cannot
  be: see the module note below;
* the two routes' corrected velocities agree, and the disagreement falls at ~2nd order
  under refinement;
* a manufactured pressure is reproduced at ~2nd order.

Why the cell route leaves a residual cell divergence at all (amendment A9). The face
gradient is the exact adjoint of the face divergence, so a corrected FACE flux is
exactly divergence-free -- that is what ``mac_project`` exploits. The pressure
projection instead corrects the CELL velocity with the face gradient AVERAGED to cells,
and ``div_f(interpolate(.))`` of that average is the ``2dx`` Laplacian while the operator
inverted is the compact one. The two differ by ``O(dx**2)`` on a smooth pressure and by
``O(1)`` on a rough one, so the cell velocity is divergence-free only to truncation.
The nodal route is exact for the cell field instead. That trade is the reason both exist
in AMReX and the reason this route ships opt-in.
"""

from importlib import import_module

import numpy as np
import pytest

import blockamr
from blockamr.dsl import exp, imp
from blockamr.dsl.equation import Equation
from blockamr.field import CellField, FaceField
from blockamr.fillpatch import FillPatchCellConservative
from blockamr.incompressible import build_incompressible, step
from blockamr.mesh import Mesh
from blockamr.operators.interpolate import interpolate

# ``blockamr.dsl.solve`` the ATTRIBUTE is the re-exported ``solve`` FUNCTION (see
# ``dsl/__init__.py``), which shadows the submodule of the same name, so neither
# ``from blockamr.dsl import solve`` nor ``import blockamr.dsl.solve as ...`` reaches the
# module the monkeypatching below has to target.
solve_module = import_module("blockamr.dsl.solve")

# Tight enough that "at solve tolerance" means the discretisation, not the Krylov stop.
_SOL_P = {"rtol": 1e-12, "atol": 1e-14, "maxIter": 400, "verbose": 0}
_NU = 0.01
_TWO_PI = 2.0 * np.pi


def _periodic_mesh(n, nz=None, max_size=None):
    """Single-level fully periodic mesh on ``[0,1]x[0,1]x[0,nz/n]``."""
    nz = n if nz is None else nz
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, float(nz) / n])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size if max_size is not None else max(n, nz))
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _dirichlet_mesh(n):
    """Single-level NON-periodic unit cube, for the pinned (Dirichlet) configuration."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _cell_coords(mf, mfi, geom):
    """Cell-centre coordinates of one box's VALID region, as (x, y, z) meshgrids."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    lo, hi = mfi.valid_box().small_end(), mfi.valid_box().big_end()
    coords = [
        prob_lo[ax] + (lo[ax] + np.arange(hi[ax] - lo[ax] + 1) + 0.5) * dx[ax] for ax in range(3)
    ]
    return np.meshgrid(*coords, indexing="ij")


def _fill_velocity(field, geom, func):
    """Write ``func(x, y, z) -> (u, v, w)`` into a ncomp=3 field and fill its ghosts."""
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        x, y, z = _cell_coords(mf, mfi, geom)
        u, v, w = func(x, y, z)
        arr[:, :, :, 0] = u
        arr[:, :, :, 1] = v
        arr[:, :, :, 2] = w
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)


def _smooth_divergent_velocity(x, y, z):
    """Smooth, periodic, and NOT solenoidal -- so the projection has work to do."""
    k = _TWO_PI
    return (
        np.sin(k * x) * np.cos(k * y),
        np.sin(k * y) * np.cos(k * z),
        0.5 * np.sin(k * z) * np.cos(k * x),
    )


def _set_random_velocity(solver, seed):
    """Verbatim from ``test_verification_projection.py`` -- the shipped gate's input."""
    rng = np.random.default_rng(seed)
    mf = solver.U.mf[0]
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        shape = tuple(hi[i] - lo[i] + 1 for i in range(3)) + (3,)
        mf.copy_from(mfi, np.asarray(rng.standard_normal(shape), dtype=float))
    solver.U.fill_patch(0, 0.0)


def _max_face_divergence(phi, mesh):
    """Verbatim from ``test_verification_projection.py::_max_face_divergence``."""
    dx = mesh.geom(0).cell_size()
    max_abs = 0.0
    face_arrs = [phi[0][d].mf.arrays() for d in range(3)]
    for bi in range(len(face_arrs[0])):
        div_val = None
        for d in range(3):
            f = face_arrs[d][bi][:, :, :, 0]
            ng = phi[0][d].mf.n_grow()
            nc = [int(f.shape[ax]) - 2 * ng - (1 if ax == d else 0) for ax in range(3)]
            sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_hi[d] = slice(ng + 1, ng + 1 + nc[d])
            sl_lo[d] = slice(ng, ng + nc[d])
            contrib = (f[tuple(sl_hi)] - f[tuple(sl_lo)]) / dx[d]
            div_val = contrib if div_val is None else div_val + contrib
        max_abs = max(max_abs, float(np.max(np.abs(div_val))))
    return max_abs


def _max_interpolated_divergence(U_field):
    """``max|div_f(interpolate(U))|`` -- the divergence the CELL route's rhs is."""
    mesh = U_field.mesh
    probe = FaceField(mesh, ncomp=1, ngrow=0, name="probe")
    interpolate(U_field, probe)
    return _max_face_divergence(probe, mesh)


def _gather_cells(mf, ncomp):
    """Valid regions of every box concatenated -- shape-agnostic, for norms only."""
    ng = mf.n_grow()
    out = []
    for arr in mf.arrays():
        n = [int(arr.shape[ax]) - 2 * ng for ax in range(3)]
        sl = tuple(slice(ng, ng + n[ax]) for ax in range(3))
        out.append(np.asarray(arr[sl[0], sl[1], sl[2], :ncomp]).reshape(-1, ncomp))
    return np.concatenate(out, axis=0)


def _l2(values, cell_vol):
    """Discrete L2 norm, so errors at different resolutions are comparable."""
    return float(np.sqrt(np.sum(np.asarray(values) ** 2) * cell_vol))


def _observed_orders(ns, errors):
    return [float(np.log2(c / f)) for c, f in zip(errors[:-1], errors[1:])]


def _order_detail(ns, errors):
    orders = _observed_orders(ns, errors)
    return (
        ", ".join(f"N={n}: {e:.6e}" for n, e in zip(ns, errors))
        + " | orders "
        + ", ".join(f"{c}->{f}: {o:.4f}" for c, f, o in zip(ns[:-1], ns[1:], orders))
    )


def _build(mesh, dt, projection=None):
    sol_p = dict(_SOL_P)
    if projection is not None:
        sol_p["projection"] = projection
    return build_incompressible(
        mesh,
        _NU,
        dt,
        fill_patch=FillPatchCellConservative(),
        sol_p=sol_p,
    )


# --- 1. the default, and the exclusivity of the two routes -------------------


def test_the_default_route_is_nodal_and_never_enters_the_cell_path(blockamr_session, monkeypatch):
    """No ``projection`` key -> the nodal cache is built and the cell path is unreached.

    Three independent statements, because "the default did not change" is the claim the
    whole opt-in rests on: the cell entry point aborts if called at all, the nodal cache
    object exists afterwards, and the cell cache attribute was never created.
    """

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("the default config entered the CELL projection route")

    monkeypatch.setattr(solve_module, "_solve_implicit_cell", _must_not_be_called)

    mesh, geom = _periodic_mesh(8, nz=4)
    solver = _build(mesh, 0.2 / 8)
    _fill_velocity(solver.U, geom, _smooth_divergent_velocity)

    step(solver)

    assert isinstance(getattr(solver.p, "_imp_cache", None), solve_module.ImplicitSolveCache)
    assert getattr(solver.p, "_cell_imp_cache", None) is None
    # Anti-vacuity: the nodal route really produced a correction to be applied.
    assert max(float(np.max(np.abs(g))) for g in solver.p.grad[0]) > 1e-6


def test_the_cell_route_builds_its_own_cache_and_not_the_nodal_operator(
    blockamr_session, monkeypatch
):
    """``projection="cell"`` -> no ``MLNodeLaplacian`` is constructed anywhere."""

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("the cell projection route built a nodal operator")

    monkeypatch.setattr(blockamr, "MLNodeLaplacian", _must_not_be_called)

    mesh, geom = _periodic_mesh(8, nz=4)
    solver = _build(mesh, 0.2 / 8, projection="cell")
    _fill_velocity(solver.U, geom, _smooth_divergent_velocity)

    step(solver)

    assert isinstance(getattr(solver.p, "_cell_imp_cache", None), solve_module.CellSolveCache)
    assert getattr(solver.p, "_imp_cache", None) is None
    assert max(float(np.max(np.abs(g))) for g in solver.p.grad[0]) > 1e-6
    # The cell route's unknowns ARE the cells, so ``p`` itself is filled (nodal leaves
    # it at zero, having nowhere to put nodal values in a cell-centred field).
    assert float(np.max(np.abs(_gather_cells(solver.p.mf[0], 1)))) > 1e-6


@pytest.mark.parametrize("bad", ["Nodal", "cell-centred", "mac", ""])
def test_an_unknown_projection_value_raises_naming_the_accepted_ones(blockamr_session, bad):
    """Never a silent fall-through to the default: a typo must be loud."""
    mesh, geom = _periodic_mesh(8, nz=4)
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")
    eqn = Equation(imp.laplacian(0.1, p) == exp.div(U))

    with pytest.raises(ValueError) as excinfo:
        eqn.solve(dt=0.1, solution={"projection": bad})

    message = str(excinfo.value)
    assert "'nodal'" in message and "'cell'" in message, message
    assert repr(bad) in message, message


# --- 2. the cell route projects ---------------------------------------------


def test_the_cell_route_leaves_the_mac_flux_divergence_free(blockamr_session):
    """The shipped gate's measure, on the shipped gate's input, on the cell route.

    ``test_verification_projection.py`` measures ``max|div(phi)|`` on the MAC face flux.
    Note what this does and does not establish: ``phi`` is projected by ``mac_project``
    inside the momentum predictor, which this task did not touch, so the row is a
    regression guard on the cell route not DISTURBING that -- the sharper measure of
    what the pressure route itself removes is the next test.
    """
    n, nz = 16, 4
    mesh, _ = _periodic_mesh(n, nz=nz)
    solver = _build(mesh, 0.2 / n, projection="cell")
    _set_random_velocity(solver, seed=3)

    div_before = _max_face_divergence(solver.phi, mesh)
    step(solver)
    div_after = _max_face_divergence(solver.phi, mesh)

    print(f"\nmax|div phi|  after one cell-route step: {div_after:.3e}")
    assert div_after < 1e-6, f"projection left max|div phi| = {div_after:.3e}"
    assert div_before == 0.0 or div_after < div_before


def _predicted_and_corrected_divergence(n, dt):
    """One step, hand-unrolled, measuring ``max|div_f(interp(U))|`` across the correction.

    ``step()`` cannot be used: the quantity of interest is U* BETWEEN the momentum
    predictor and the pressure correction. The sequence below is ``incompressible.step``
    verbatim up to the point where it diverges.
    """
    from blockamr.operators.correct import correct
    from blockamr.operators.mac_project import mac_project

    mesh, geom = _periodic_mesh(n)
    solver = _build(mesh, dt, projection="cell")
    _fill_velocity(solver.U, geom, _smooth_divergent_velocity)

    solver.U.fill_patch(0, 0.0)
    interpolate(solver.U, solver.phi)
    mac_project(solver.phi, solver.sol_p)
    solver.UEqn.solve(dt=dt, t=0.0, solution=solver.sol_U)
    solver.U.fill_patch(0, 0.0)

    before = _max_interpolated_divergence(solver.U)

    solver.pEqn.implicit_lhs.sigma = dt
    solver.pEqn.implicit_lhs.coefficient = dt
    solver.pEqn.solve(dt=dt, t=0.0, solution=solver.sol_p)
    correct(solver.U, -dt * exp.grad(solver.p))
    solver.U.fill_patch(0, 0.0)

    after = _max_interpolated_divergence(solver.U)
    return before, after


def test_the_cell_route_removes_the_divergence_it_solved_for(blockamr_session):
    """``div_f(interp(U))`` collapses across the correction, and its remainder is O(dx^2).

    This is what the cell pressure solve actually controls. It is not driven to zero and
    cannot be -- the correction uses the face gradient AVERAGED to cells, whose
    ``div_f(interp(.))`` is the ``2dx`` Laplacian rather than the compact one that was
    inverted -- so the claim tested is the honest one: a large drop, and a remainder that
    is second order in dx.
    """
    ns = [16, 32, 64]
    dt = 0.002
    pairs = [_predicted_and_corrected_divergence(n, dt) for n in ns]
    predicted = [p for p, _ in pairs]
    corrected = [c for _, c in pairs]

    print("\nmax|div_f(interp(U))| before -> after the cell correction")
    for n, pre, post in zip(ns, predicted, corrected):
        print(f"  N={n:3d}  {pre:.6e} -> {post:.6e}   ratio {post / pre:.3e}")
    orders = _observed_orders(ns, corrected)
    print("  remainder orders " + ", ".join(f"{o:.4f}" for o in orders))

    for n, pre, post in zip(ns, predicted, corrected):
        assert pre > 1.0, f"N={n}: nothing to remove, predicted = {pre:.3e}"
        # A 20x drop in the MAX norm on the coarsest grid; the rate below is the real
        # claim, and the coarse-grid factor is only the guard that says the solve did
        # something rather than nothing.
        assert post < 0.05 * pre, f"N={n}: {post:.3e} vs {pre:.3e}"
    assert min(orders) > 1.8, f"remainder orders {orders}, values {corrected}"


# --- 3. the two routes agree, at second order -------------------------------


def _one_step_velocity(n, dt, projection):
    mesh, geom = _periodic_mesh(n)
    solver = _build(mesh, dt, projection=projection)
    _fill_velocity(solver.U, geom, _smooth_divergent_velocity)
    step(solver)
    dx = mesh.geom(0).cell_size()
    return _gather_cells(solver.U.mf[0], 3), float(dx[0] * dx[1] * dx[2])


def test_the_two_routes_agree_and_the_gap_falls_at_second_order(blockamr_session):
    """The nodal and cell corrected velocities differ by a DISCRETISATION error.

    Both routes are consistent approximations of the same continuous projection of the
    same predicted velocity (the momentum predictor is bit-identical between them), so
    their difference must vanish with dx -- and at the design rate. A gap that did not
    shrink under refinement would mean the two are solving different problems, which is
    the failure this test exists to catch; a tolerance on a single grid would not see it.
    """
    ns = [16, 32, 64]
    dt = 0.002
    errors = []
    scales = []
    for n in ns:
        nodal, cell_vol = _one_step_velocity(n, dt, projection=None)
        cell, _ = _one_step_velocity(n, dt, projection="cell")
        errors.append(_l2(cell - nodal, cell_vol))
        scales.append(_l2(nodal, cell_vol))

    print("\n|U_cell - U_nodal|_L2  " + _order_detail(ns, errors))
    print(
        "  relative to |U_nodal|_L2: " + ", ".join(f"{e / s:.3e}" for e, s in zip(errors, scales))
    )

    orders = _observed_orders(ns, errors)
    # Anti-vacuity: a gap of exactly zero would pass any order test by accident, and
    # would mean the two routes were not actually different code paths here.
    assert errors[0] > 1e-9, f"the two routes produced the same field: {errors}"
    assert min(orders) > 1.8, f"observed orders {orders}, L2 gaps {errors}"


# --- 4. manufactured solution ----------------------------------------------


def _fill_velocity_over_ghosts(field, geom, func):
    """``func`` evaluated over the GROWN region of a ncomp=3 field, ghosts included.

    The rhs the cell route forms is a FACE interpolation of ``U``, so the ghost layer
    feeds the boundary rows of the linear system directly. Writing analytic ghosts keeps
    a manufactured-solution rate a statement about the pressure operator's boundary
    closure rather than about whatever a fill-patch happened to leave outside the domain.
    """
    mf = field.mf[0]
    ng = mf.n_grow()
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    # Consumed as a full comprehension: an MFIterator left alive aborts the next one.
    los = [mfi.valid_box().small_end() for mfi in blockamr.MFIterator(mf)]

    grown = mf.grown_arrays()
    results = []
    for bi, lo in enumerate(los):
        shape = grown[bi].shape[:3]
        coords = [
            prob_lo[ax] + (lo[ax] - ng + np.arange(shape[ax]) + 0.5) * dx[ax] for ax in range(3)
        ]
        x, y, z = np.meshgrid(*coords, indexing="ij")
        results.append(np.stack(func(x, y, z), axis=-1))
    mf.copy_grown_arrays(results)


def _manufactured_pressure_error(n, bc_kind):
    """Solve with ``U = sigma grad(p*)``, so the exact discrete answer is ``p*``.

    Mirrors ``test_verification_poisson.py``'s manufactured field, but fed to the route as
    the solver feeds it -- a velocity, whose divergence the route forms itself with its own
    interpolate-then-face-divergence -- rather than as an injected Laplacian.

    Two boundary configurations, because ``project_nullspace`` is derived from them and
    setting it wrongly solves a different problem in silence:

    * ``periodic`` -- ``p* = sin(2 pi x) sin(2 pi y) sin(2 pi z)``, operator SINGULAR, so
      ``project_nullspace`` is on and the comparison is MEAN-FREE (only ``grad(p)``
      reaches the answer);
    * ``dirichlet`` -- ``p* = sin(pi x) sin(pi y) sin(pi z)``, which vanishes ON every
      domain face, so homogeneous Dirichlet is exact; the operator is PINNED,
      ``project_nullspace`` must be off, and the pressure itself is compared.
    """
    sigma = 0.1
    if bc_kind == "periodic":
        mesh, geom = _periodic_mesh(n)
        p_bc = None
        k = _TWO_PI
    else:
        mesh, geom = _dirichlet_mesh(n)
        p_bc = ([blockamr.LinOpBCType.Dirichlet] * 3, [blockamr.LinOpBCType.Dirichlet] * 3)
        k = np.pi

    def p_exact(x, y, z):
        return np.sin(k * x) * np.sin(k * y) * np.sin(k * z)

    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    _fill_velocity_over_ghosts(
        U,
        geom,
        lambda x, y, z: (
            sigma * k * np.cos(k * x) * np.sin(k * y) * np.sin(k * z),
            sigma * k * np.sin(k * x) * np.cos(k * y) * np.sin(k * z),
            sigma * k * np.sin(k * x) * np.sin(k * y) * np.cos(k * z),
        ),
    )

    p = CellField(mesh, ncomp=1, ngrow=0, name="p")
    p.pressure_bc = p_bc
    eqn = Equation(imp.laplacian(sigma, p) == exp.div(U))
    eqn.solve(dt=sigma, solution={**_SOL_P, "projection": "cell"})

    got = _gather_cells(p.mf[0], 1)[:, 0]
    exact = []
    for mfi in blockamr.MFIterator(p.mf[0]):
        x, y, z = _cell_coords(p.mf[0], mfi, geom)
        exact.append(p_exact(x, y, z).reshape(-1))
    exact = np.concatenate(exact)
    if p_bc is None:
        got = got - got.mean()
        exact = exact - exact.mean()

    dx = geom.cell_size()
    cell_vol = float(dx[0] * dx[1] * dx[2])
    return _l2(got - exact, cell_vol), float(np.max(np.abs(got)))


@pytest.mark.parametrize("bc_kind", ["periodic", "dirichlet"])
def test_the_cell_route_reproduces_a_manufactured_pressure_at_second_order(
    blockamr_session, bc_kind
):
    ns = [16, 32, 64]
    results = [_manufactured_pressure_error(n, bc_kind) for n in ns]
    errors = [e for e, _ in results]
    peaks = [p for _, p in results]

    print(f"\nmanufactured p L2 error ({bc_kind})  " + _order_detail(ns, errors))

    # Anti-vacuity: a solve that returned zero would give a constant error, not an order.
    assert min(peaks) > 0.1, f"the solved pressure is ~zero, peaks {peaks}"
    orders = _observed_orders(ns, errors)
    assert min(orders) > 1.8, f"observed orders {orders}, L2 errors {errors}"
    assert errors[0] < 1e-1, f"coarse-grid L2 error too large: {errors[0]:.3e}"
