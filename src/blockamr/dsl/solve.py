# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
from .. import backends
from ..schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from ..schemes.registry import lookup_scheme, resolve as resolve_scheme

# Backward-compat re-exports of the explicit machinery moved to
# ``blockamr.backends.jax_backend`` (removed in plan 06). ``BF`` is a
# mutable module global there — proxied live via ``__getattr__`` below so
# callers reading ``dsl.solve.BF`` see ``set_tile_size`` updates.
from ..backends.jax_backend import forward_euler, parallel_for, set_tile_size  # noqa: F401


def __getattr__(name):
    # PEP 562: keep ``from blockamr.dsl.solve import BF`` live-tracking the
    # single source of truth in jax_backend (a plain re-export would freeze a
    # copy that ``set_tile_size`` could not update).
    if name == "BF":
        from ..backends import jax_backend

        return jax_backend.BF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_schemes(spatial_ops, schemes):
    """Resolve operator schemes from the schemes dict (names or objects, keyed
    by scheme_key or class name).

    A scheme object pinned via direct operator construction
    (``Div(..., scheme=obj)``) wins over the dict; the ``exp.*`` DSL surface has
    no ``scheme=`` kwarg and always resolves by name. Shared by ``solve`` and
    ``evaluate`` — an operator must discretise the same way whether it is
    stepped or merely evaluated.
    """
    for sp_op in spatial_ops:
        if sp_op._scheme_explicit or sp_op._scheme_operator is None:
            continue
        keys = [sp_op._scheme_key_or_none(), type(sp_op).__name__]
        sp_op.scheme = lookup_scheme(schemes, keys, sp_op._scheme_operator, sp_op.scheme)


def solve(equation, *, dt=None, t=None, solution=None):
    """Discretise and solve an Equation.

    Two forms, dispatched on the equation's terms:

      solve(Equation(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U),
                     schemes=schemes), dt=dt, t=t)
        → explicit time integration (Euler / RK2 / RK4). ``solution`` may
          carry ``"backend"`` (dispatch backend, default ``"cpp"``), ``"ibm"``
          (the field's immersed boundary method — absent means the IBM path is
          not entered at all and the result is bitwise the plain operator's)
          and ``"ddt"`` (the time scheme name; when present it wins over the
          equation's ``schemes["ddt"]``, which stays the default route).

      solve(Equation(imp.laplacian(sigma, p) == exp.div(U)), dt=dt,
            solution=sol_p)
        → implicit MLMG solve (AMReX C++), configured by ``solution``
          (solver/rtol/atol/maxIter/bottomSolver/verbose/bottomVerbose).
    """
    from .equation import Equation
    from ..ibm import IBM, evaluation

    if not isinstance(equation, Equation):
        raise TypeError(f"solve() expects an Equation, got {type(equation).__name__}")

    if equation.implicit_lhs is not None:
        _solve_implicit(equation, solution=solution)
        return

    if len(equation.temporal_ops) != 1:
        raise ValueError(
            "solve() can only dispatch an equation with either an implicit_lhs "
            "(imp.laplacian(...) == ...) or exactly one explicit ddt term "
            f"(momentum predictor); got {len(equation.temporal_ops)} ddt term(s) "
            "and no implicit_lhs."
        )

    schemes = equation.schemes
    cell_field = equation.temporal_ops[0].field  # CellField
    mesh = cell_field.mesh
    cfg = solution or {}

    if "ddt" in cfg:
        # Decision Q8 (plans/IBM/review.md §4): the solution key wins when both
        # are present; the equation route stays the default when it is absent.
        ddt_scheme = resolve_scheme("ddt", cfg["ddt"])()
    else:
        ddt_scheme = lookup_scheme(schemes, ["ddt", "Ddt"], "ddt", ForwardEuler())

    _resolve_schemes(equation.spatial_ops, schemes)

    # Validate that the field has enough ghost cells for the widest stencil
    required = equation.required_ngrow
    actual = cell_field.ngrow
    if actual < required:
        raise ValueError(
            f"Field '{cell_field.name}' has ngrow={actual} but the expression "
            f"requires ngrow>={required} (from operator stencil widths). "
            f"Create the field with ngrow>={required}."
        )

    # solution["ibm"] (api §1): validate the name, then branch on the method's
    # kind. Operator methods become the band driver (None when the IBM path is
    # not entered at all — no key, noIbm, empty band — keeping those results
    # bitwise the plain operator's); step methods fire on the field after each
    # stage update.
    band = None
    step_method = None
    ibm_name = cfg.get("ibm")
    if ibm_name is not None:
        method = IBM.lookup(ibm_name)  # unknown name -> ValueError + names()
        if method.kind == "step":
            step_method = method
        else:
            band = evaluation(ibm_name, cell_field, equation.spatial_ops)

    impl = backends.get(cfg.get("backend", "cpp"))
    ddt_coeff = equation.temporal_ops[0].coeff

    if isinstance(ddt_scheme, ForwardEuler):
        for lev in range(mesh.n_levels()):
            cell_field.fill_patch(lev, t)
            if band is None:
                # The exact pre-IBM call chain — bitwise the plain operator.
                impl.euler_step(equation, cell_field, lev, t, dt)
            else:
                # Accumulating source (interior sweep + band rows; the pin ran
                # at classification time, Q3/B25) plus the generic update,
                # never a fused step kernel (R4).
                src = band.source_level(impl, equation.spatial_ops, cell_field, lev, t)
                blockamr.euler_update(
                    cell_field.mf[lev], src, dt / ddt_coeff, cell_field.ncomp
                )
        if step_method is not None:
            _apply_step_method(step_method, cell_field, dt, t + dt)
    elif isinstance(ddt_scheme, RungeKutta2):
        _rk2_step(impl, equation, band, step_method, cell_field, dt, t, ddt_coeff)
    elif isinstance(ddt_scheme, RungeKutta4):
        _rk4_step(impl, equation, band, step_method, cell_field, dt, t, ddt_coeff)
    else:
        raise ValueError(f"Unknown ddt scheme: {ddt_scheme}")

    # restrict fine -> coarse
    for lev in reversed(range(mesh.n_levels() - 1)):
        blockamr.average_down(
            cell_field.mf[lev + 1],
            cell_field.mf[lev],
            mesh.geom(lev + 1),
            mesh.geom(lev),
            0,
            cell_field.ncomp,
            mesh.ref_ratio(lev),
        )


def _stage_source(impl, equation, band, cell_field, lev, t_s):
    """One stage's accumulated source on one level, at stage time ``t_s``.

    The band route (interior sweep + band rows; the pin ran at classification
    time, Q3/B25) when a band driver is active, the plain accumulating kernels
    otherwise — never a fused step kernel (R4). The returned MultiFab may be a
    backend scratch reused by the next call: consume it before the next
    ``source``.
    """
    if band is None:
        return impl.source(equation.spatial_ops, cell_field, lev, t_s)
    return band.source_level(impl, equation.spatial_ops, cell_field, lev, t_s)


def _snapshot(cell_field, lev):
    """A fresh copy of the field's level-``lev`` MultiFab (ghosts included)."""
    mesh = cell_field.mesh
    mf = cell_field.mf[lev]
    ng = mf.n_grow()
    snap = blockamr.MultiFab(
        mesh.box_array(lev), mesh.dm(lev), cell_field.ncomp, ng, memory=cell_field._memory
    )
    blockamr.copy_multifab(snap, mf, cell_field.ncomp, ng)
    return snap


def _accumulator(cell_field, lev):
    """A zeroed ngrow=0 MultiFab on the field's level-``lev`` box array."""
    mesh = cell_field.mesh
    acc = blockamr.MultiFab(
        mesh.box_array(lev), mesh.dm(lev), cell_field.ncomp, 0, memory=cell_field._memory
    )
    acc.set_val(0.0)
    return acc


def _rk2_step(impl, equation, band, step_method, cell_field, dt, t, ddt_coeff):
    """Explicit midpoint (two-stage, second order), built from ``source`` +
    ``copy_multifab`` + ``euler_update`` only (R4: no fused step kernel).

    ``phi_half = phi0 - (dt/2c)·src(phi0, t)``;
    ``phi_new  = phi0 - (dt/c)·src(phi_half, t + dt/2)``.
    A step method (directForcing) fires after every stage update, so the solid
    value is held between stages, not just between steps.

    Single-level per-stage schedule: a fine level's stage sees the already
    advanced coarse level and no ``average_down`` runs between stages —
    multi-level per-stage AMR is out of scope (rung 10 is single-level).
    """
    mesh = cell_field.mesh
    ncomp = cell_field.ncomp
    phi0 = [_snapshot(cell_field, lev) for lev in range(mesh.n_levels())]

    for a_stage, t_stage, t_state in ((0.5, t, t + 0.5 * dt), (1.0, t + 0.5 * dt, t + dt)):
        for lev in range(mesh.n_levels()):
            cell_field.fill_patch(lev, t_stage)
            src = _stage_source(impl, equation, band, cell_field, lev, t_stage)
            ng = cell_field.mf[lev].n_grow()
            blockamr.copy_multifab(cell_field.mf[lev], phi0[lev], ncomp, ng)
            blockamr.euler_update(cell_field.mf[lev], src, a_stage * dt / ddt_coeff, ncomp)
        if step_method is not None:
            _apply_step_method(step_method, cell_field, dt, t_state)


def _rk4_step(impl, equation, band, step_method, cell_field, dt, t, ddt_coeff):
    """Classical RK4, built from ``source`` + ``copy_multifab`` +
    ``euler_update`` only (R4: no fused step kernel).

    Each stage's source is folded into a zeroed accumulator with its classical
    weight (``acc += w·src`` via ``euler_update(acc, src, -w)``) and, for the
    first three stages, also forms the next stage state from the ``phi0``
    snapshot. The final state is ``phi0 - (dt/6c)·acc``. A step method fires
    after every stage update and after the final update.

    Single-level per-stage schedule: a fine level's stage sees the already
    advanced coarse level and no ``average_down`` runs between stages —
    multi-level per-stage AMR is out of scope (rung 10 is single-level).
    """
    mesh = cell_field.mesh
    ncomp = cell_field.ncomp
    phi0 = [_snapshot(cell_field, lev) for lev in range(mesh.n_levels())]
    acc = [_accumulator(cell_field, lev) for lev in range(mesh.n_levels())]

    # (stage time, classical weight, next-state coefficient a in
    #  phi <- phi0 - a*(dt/c)*src; None for the last stage) — the state formed
    # by stage s lives at time t + c_{s+1}*dt, which is what a step method sees.
    stages = (
        (t, 1.0, 0.5, t + 0.5 * dt),
        (t + 0.5 * dt, 2.0, 0.5, t + 0.5 * dt),
        (t + 0.5 * dt, 2.0, 1.0, t + dt),
        (t + dt, 1.0, None, None),
    )
    for t_stage, weight, a_next, t_state in stages:
        for lev in range(mesh.n_levels()):
            cell_field.fill_patch(lev, t_stage)
            src = _stage_source(impl, equation, band, cell_field, lev, t_stage)
            blockamr.euler_update(acc[lev], src, -weight, ncomp)  # acc += w*src
            if a_next is not None:
                ng = cell_field.mf[lev].n_grow()
                blockamr.copy_multifab(cell_field.mf[lev], phi0[lev], ncomp, ng)
                blockamr.euler_update(cell_field.mf[lev], src, a_next * dt / ddt_coeff, ncomp)
        if a_next is not None and step_method is not None:
            _apply_step_method(step_method, cell_field, dt, t_state)

    for lev in range(mesh.n_levels()):
        ng = cell_field.mf[lev].n_grow()
        blockamr.copy_multifab(cell_field.mf[lev], phi0[lev], ncomp, ng)
        blockamr.euler_update(cell_field.mf[lev], acc[lev], dt / (6.0 * ddt_coeff), ncomp)
    if step_method is not None:
        _apply_step_method(step_method, cell_field, dt, t + dt)


def _apply_step_method(method, cell_field, dt, t):
    """Apply a step-kind IBM method (directForcing) to the field.

    Builds the method's mesh data lazily when ``mesh.build_ibm`` was never
    called, and derives the pin datum from the field's own ``ibm_bc``: a single
    patch carrying ``FixedValue`` broadcasts to the field's ncomp. Anything
    else raises naming the missing capability (S6: loud, never a silent
    interior value).
    """
    from ..ibm.bc import FixedValue, broadcast_gamma

    mesh = cell_field.mesh
    try:
        data = mesh.ibm_data(method)
    except RuntimeError:
        built = list(getattr(mesh, "_ibm_methods", []))
        if method not in built:
            built.append(method)
        mesh.build_ibm(built)
        data = mesh.ibm_data(method)

    ibm_bc = cell_field.ibm_bc
    if len(ibm_bc) != 1:
        raise NotImplementedError(
            f"step-kind IBM '{method.__name__}' through solve() supports exactly "
            f"one immersed patch; field '{cell_field.name}' has "
            f"{sorted(ibm_bc) or 'none'}."
        )
    ((patch, bc),) = ibm_bc.items()
    if not isinstance(bc, FixedValue):
        raise NotImplementedError(
            f"step-kind IBM '{method.__name__}' through solve() supports a "
            f"FixedValue ibm_bc datum only; patch {patch!r} carries "
            f"{type(bc).__name__}."
        )
    u_body = broadcast_gamma(bc.value, cell_field.ncomp)
    method.apply(cell_field, dt, t, data, u_body=u_body)


def evaluate(expr, t=0.0, solution=None):
    """Evaluate spatial operators and return the source term.

    Unlike solve(), does NOT update the field — just computes and returns
    the sum of spatial operator contributions as per-box arrays.

    Parameters
    ----------
    expr : Equation or single spatial operator
        e.g. ``Div(phi, U, scheme=VanLeer())`` or
        ``exp.div(phi, U) - exp.laplacian(nu, U)``.
    t : float
        Current time (for time-dependent coefficients).
    solution : dict, optional
        ``fvSolution.solvers[field]`` entries. ``"backend"`` picks the dispatch
        backend (default ``"cpp"``); ``"ibm"`` names the field's immersed
        boundary method. With no ``"ibm"`` key — and with ``"noIbm"``, and with
        a body that has no boundary cell on this mesh — the IBM path is not
        entered at all, so the result is bitwise the plain operator's.

    Returns
    -------
    list[list[ndarray]]
        Outer list: per level. Inner list: per box.
        Each array has shape (vNx, vNy, vNz) for ncomp=1
        or (vNx, vNy, vNz, ncomp) for ncomp>1.
    """
    from ..ibm import evaluation
    from .equation import Equation

    # Wrap a bare operator in an equation if needed
    if not isinstance(expr, Equation):
        op = expr
        cell_field = op.field
        spatial_ops = [op]
    else:
        spatial_ops = expr.spatial_ops
        cell_field = spatial_ops[0].field
        _resolve_schemes(spatial_ops, expr.schemes)

    cfg = solution or {}
    mesh = cell_field.mesh
    impl = backends.get(cfg.get("backend", "cpp"))
    # None when the IBM path is not entered at all: no "ibm" key, the noIbm
    # opt-out, or an empty band. Then the level loop below is the plain
    # operator's, call for call (plans/IBM/design.md §6).
    ibm = evaluation(cfg.get("ibm"), cell_field, spatial_ops)
    all_levels = []

    for lev in range(mesh.n_levels()):
        # Once per level, before the terms: nothing writes to phi during them.
        cell_field.fill_patch(lev, t)
        if ibm is None:
            all_levels.append(impl.evaluate(spatial_ops, cell_field, lev, t))
        else:
            all_levels.append(ibm.evaluate_level(impl, spatial_ops, cell_field, lev, t))

    return all_levels


# ---------------------------------------------------------------------------
# Implicit equation solver (AMReX MLMG — unchanged)
# ---------------------------------------------------------------------------

# Old snake_case `solution` keys, renamed to the fvSolution camelCase
# spellings (API doc §5). Passing an old key is a clear migration error
# rather than a silently-ignored setting.
_DEPRECATED_SOLUTION_KEYS = {
    "max_iter": "maxIter",
    "bottom_solver": "bottomSolver",
    "bottom_verbose": "bottomVerbose",
}


def _check_solution_keys(solution):
    """Raise a clear error for renamed (dropped) snake_case solution keys."""
    if not solution:
        return
    for key in solution:
        new_key = _DEPRECATED_SOLUTION_KEYS.get(key)
        if new_key is not None:
            raise ValueError(
                f"solution key '{key}' was renamed to '{new_key}' "
                "(fvSolution.solvers[field] key spellings changed) — "
                f"use solution={{'{new_key}': ...}}."
            )


class ImplicitSolveCache:
    """Cached AMReX MLMG solver objects for one field's implicit solve.

    Stored on the field (``p_field._imp_cache``); rebuilt whenever *key*
    (n_levels, sigma, bottomSolver) changes — see ``_solve_implicit``.
    """

    def __init__(self, key, lp, mlmg, phi_mfs, rhs_mfs, vel3_mfs, fluxes_mfs, has_dirichlet):
        self.key = key
        self.lp = lp
        self.mlmg = mlmg
        self.phi_mfs = phi_mfs
        self.rhs_mfs = rhs_mfs
        self.vel3_mfs = vel3_mfs
        self.fluxes_mfs = fluxes_mfs
        self.has_dirichlet = has_dirichlet


def _solve_implicit(eqn, solution=None):
    """Solve imp.laplacian(sigma, p) == exp.div(U).

    Supports single-level and multi-level AMR meshes:
    1. Pack U into ncomp=3 MultiFab with ghost cells (per level)
    2. compDivergence → nodal RHS (per level)
    3. MLMG.solve → nodal p (all levels simultaneously)
    4. getFluxes → store cell-centred gradient on p.grad (per level)
    """
    imp_op = eqn.implicit_lhs  # ImplicitLaplacian
    rhs_op = eqn.rhs  # CellDivergence

    _check_solution_keys(solution)
    cfg = solution or {}
    solver_name = cfg.get("solver", "MLMG")
    if solver_name != "MLMG":
        raise ValueError(f"Unknown solution['solver']='{solver_name}': only 'MLMG' is supported.")
    rtol = cfg.get("rtol", 1e-10)
    atol = cfg.get("atol", 1e-12)
    max_iter = cfg.get("maxIter", 200)
    verbose = cfg.get("verbose", 0)
    # Optional explicit nodal bottom solver: one of "cg", "bicgstab", "smoother",
    # "cgbicg", "bicgcg", "default". None → let AMReX pick its default (a Krylov
    # solver, which converges this system in ~5 V-cycles).
    bottom_solver = cfg.get("bottomSolver", None)

    p_field = imp_op.field
    U_field = rhs_op.vel_field
    sigma = imp_op.sigma
    mesh = U_field.mesh
    n_levels = mesh.n_levels()

    # The rebuild key includes the `solution` values that affect the built
    # AMReX objects (bottomSolver changes take effect only via a rebuild —
    # `set_bottom_solver` is otherwise sticky across calls that omit it).
    cache_key = (n_levels, sigma, bottom_solver)
    cache = getattr(p_field, "_imp_cache", None)
    needs_rebuild = cache is None or cache.key != cache_key

    if needs_rebuild:
        geoms = [mesh.geom(lev) for lev in range(n_levels)]
        bas = [mesh.box_array(lev) for lev in range(n_levels)]
        dms = [mesh.dm(lev) for lev in range(n_levels)]

        phi_mfs = []
        rhs_mfs = []
        vel3_mfs = []
        fluxes_mfs = []
        nodal_type = blockamr.node_type()

        for lev in range(n_levels):
            ba_lev = bas[lev]
            dm_lev = dms[lev]

            nodal_ba = blockamr.convert_ba(ba_lev, nodal_type)

            phi_mf = blockamr.MultiFab(nodal_ba, dm_lev, 1, 1)
            phi_mf.set_val(0.0)

            phi_mfs.append(phi_mf)
            rhs_mfs.append(blockamr.MultiFab(nodal_ba, dm_lev, 1, 0))
            vel3_mfs.append(blockamr.MultiFab(ba_lev, dm_lev, 3, U_field.ngrow))
            fluxes_mfs.append(blockamr.MultiFab(ba_lev, dm_lev, 3, 0))

        is_per = geoms[0].is_periodic()

        # Per-face pressure BC: use the solver-derived spec stashed on the
        # pressure field (outflow face → Dirichlet, inlet/wall → Neumann) when
        # present; otherwise fall back to the periodic/all-Neumann default.
        p_bc = getattr(p_field, "pressure_bc", None)
        if p_bc is not None:
            lo_bc, hi_bc = p_bc
        else:
            lo_bc = [
                blockamr.LinOpBCType.Periodic if is_per[d] else blockamr.LinOpBCType.Neumann
                for d in range(3)
            ]
            hi_bc = lo_bc[:]

        # A lone outflow-Dirichlet face anchoring an otherwise-Neumann domain is
        # badly conditioned for plain nodal multigrid (coarse-grid correction is
        # ineffective → convergence stalls). Agglomeration + consolidation let
        # AMReX coarsen far enough for an effective bottom solve — the standard
        # incflo nodal-projection setup. Only enabled when a Dirichlet face is
        # present, to leave the periodic/closed (all-Neumann) path untouched.
        has_dirichlet = any(bc == blockamr.LinOpBCType.Dirichlet for bc in (*lo_bc, *hi_bc))
        info = blockamr.LPInfo()
        if has_dirichlet:
            info.set_agglomeration(True)
            info.set_consolidation(True)
        if n_levels == 1:
            lp = blockamr.MLNodeLaplacian(geoms[0], bas[0], dms[0], info, sigma)
        else:
            lp = blockamr.MLNodeLaplacian(geoms, bas, dms, info, sigma)

        lp.set_domain_bc(lo_bc, hi_bc)

        cache = ImplicitSolveCache(
            key=cache_key,
            lp=lp,
            mlmg=blockamr.MLMG(lp),
            phi_mfs=phi_mfs,
            rhs_mfs=rhs_mfs,
            vel3_mfs=vel3_mfs,
            fluxes_mfs=fluxes_mfs,
            has_dirichlet=has_dirichlet,
        )
        p_field._imp_cache = cache

    cache.mlmg.set_verbose(verbose)
    cache.mlmg.set_max_iter(max_iter)
    cache.mlmg.set_bottom_verbose(cfg.get("bottomVerbose", 0))
    # Bottom solver: default (None) lets AMReX use its Krylov default, which —
    # with the agglomeration+consolidation enabled above for the has_dirichlet
    # (outflow) case — converges the nodal projection in ~5 V-cycles. Override
    # via solution["bottomSolver"] if needed. (Do NOT force "smoother" here: it
    # cost ~600 iters/solve, ~100x the Krylov default, and dominated runtime.)
    if bottom_solver is not None:
        cache.mlmg.set_bottom_solver(bottom_solver)

    # 1. Pack velocity with ghost cells into ncomp=3 MultiFab (per level)
    for lev in range(n_levels):
        mf = U_field.mf[lev]
        grown = mf.grown_arrays()
        for bi, mfi in enumerate(blockamr.MFIterator(cache.vel3_mfs[lev])):
            cache.vel3_mfs[lev].copy_grown_from(mfi, grown[bi])

    # 2. compDivergence → nodal RHS
    if n_levels == 1:
        cache.lp.comp_divergence(cache.rhs_mfs[0], cache.vel3_mfs[0])
    else:
        cache.lp.comp_divergence(cache.rhs_mfs, cache.vel3_mfs)

    # 3. MLMG.solve (warm-start from previous phi)
    if n_levels == 1:
        cache.mlmg.solve(cache.phi_mfs[0], cache.rhs_mfs[0], rtol, atol)
    else:
        cache.mlmg.solve(cache.phi_mfs, cache.rhs_mfs, rtol, atol)

    if verbose:
        print(
            f"  MLMG  iters={cache.mlmg.get_num_iters()}  "
            f"init_res={cache.mlmg.get_init_residual():.6e}  "
            f"final_res={cache.mlmg.get_final_residual():.6e}"
        )

    # 4. getFluxes → store gradient on p_field for correct()
    if n_levels == 1:
        cache.mlmg.get_fluxes(cache.fluxes_mfs[0])
    else:
        cache.mlmg.get_fluxes(cache.fluxes_mfs)

    p_field.grad = []
    for lev in range(n_levels):
        box_grads = [-arr / sigma for arr in cache.fluxes_mfs[lev].arrays()]
        p_field.grad.append(box_grads)
