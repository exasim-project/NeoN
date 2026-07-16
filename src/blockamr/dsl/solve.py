# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
from .. import backends
from ..schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from ..schemes.registry import lookup_scheme

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


def solve(equation, *, dt=None, t=None, solution=None):
    """Discretise and solve an Equation.

    Two forms, dispatched on the equation's terms:

      solve(Equation(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U),
                     schemes=schemes), dt=dt, t=t)
        → explicit Forward Euler (JAX/Pallas). Schemes are resolved from the
          equation's own ``schemes`` (bound at construction); ``solution``
          may carry the field's IBM method.

      solve(Equation(imp.laplacian(sigma, p) == exp.div(U)), dt=dt,
            solution=sol_p)
        → implicit MLMG solve (AMReX C++), configured by ``solution``
          (solver/rtol/atol/maxIter/bottomSolver/verbose/bottomVerbose).
    """
    from .equation import Equation

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

    ddt_scheme = lookup_scheme(schemes, ["ddt", "Ddt"], "ddt", ForwardEuler())

    # Resolve operator schemes from the schemes dict (names or objects,
    # keyed by scheme_key or class name). A scheme object pinned via direct
    # operator construction (Div(..., scheme=obj)) wins over the dict; the
    # exp.* DSL surface has no scheme= kwarg and always resolves by name.
    for sp_op in equation.spatial_ops:
        if sp_op._scheme_explicit or sp_op._scheme_operator is None:
            continue
        keys = [sp_op._scheme_key_or_none(), type(sp_op).__name__]
        sp_op.scheme = lookup_scheme(schemes, keys, sp_op._scheme_operator, sp_op.scheme)

    # Validate that the field has enough ghost cells for the widest stencil
    required = equation.required_ngrow
    actual = cell_field.ngrow
    if actual < required:
        raise ValueError(
            f"Field '{cell_field.name}' has ngrow={actual} but the expression "
            f"requires ngrow>={required} (from operator stencil widths). "
            f"Create the field with ngrow>={required}."
        )

    if isinstance(ddt_scheme, ForwardEuler):
        impl = backends.get((solution or {}).get("backend", "jax"))
        for lev in range(mesh.n_levels()):
            cell_field.fill_patch(lev, t)
            impl.euler_step(equation, cell_field, lev, t, dt)

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
    elif isinstance(ddt_scheme, (RungeKutta2, RungeKutta4)):
        raise NotImplementedError(f"{ddt_scheme.type} is not yet implemented")
    else:
        raise ValueError(f"Unknown ddt scheme: {ddt_scheme}")


def evaluate(expr, t=0.0):
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

    Returns
    -------
    list[list[ndarray]]
        Outer list: per level. Inner list: per box.
        Each array has shape (vNx, vNy, vNz) for ncomp=1
        or (vNx, vNy, vNz, ncomp) for ncomp>1.
    """
    from .equation import Equation

    # Wrap a bare operator in an equation if needed
    if not isinstance(expr, Equation):
        op = expr
        cell_field = op.field
        spatial_ops = [op]
    else:
        spatial_ops = expr.spatial_ops
        cell_field = spatial_ops[0].field

    mesh = cell_field.mesh
    impl = backends.get("jax")
    all_levels = []

    for lev in range(mesh.n_levels()):
        cell_field.fill_patch(lev, t)
        all_levels.append(impl.evaluate(spatial_ops, cell_field, lev, t))

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
