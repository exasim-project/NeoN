# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
from .. import backends
from ..field import FaceField
from ..linear_algebra import LinearSystem, MFFaceCoeffs, Solver, SolverConfig, laplacian
from ..operators.face_gradient import cell_gradient
from ..operators.interpolate import interpolate
from ..operators.mac_project import _face_divergence, _la_bc_from_pressure_bc
from ..schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from ..schemes.registry import lookup_scheme

# Backward-compat re-exports; ``BF`` is proxied via ``__getattr__`` below instead.
from ..backends.jax_backend import forward_euler, parallel_for, set_tile_size  # noqa: F401


def __getattr__(name):
    # ``BF`` is a mutable global in jax_backend; a plain re-export would freeze a copy
    # that ``set_tile_size`` could not update.
    if name == "BF":
        from ..backends import jax_backend

        return jax_backend.BF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def solve(equation, *, dt=None, t=None, solution=None):
    """Discretise and solve an Equation.

    Two forms, dispatched on the equation's terms:

      solve(Equation(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U),
                     schemes=schemes), dt=dt, t=t)
        → explicit CELL-CENTRED Forward Euler (JAX/Pallas). Schemes come from the
          equation's own ``schemes``; ``solution`` may carry the field's IBM method.

      solve(Equation(imp.laplacian(sigma, p) == exp.div(U)), dt=dt,
            solution=sol_p)
        → implicit NODAL MLMG solve (AMReX C++), configured by ``solution``
          (solver/rtol/atol/maxIter/bottomSolver/verbose/bottomVerbose).
          ``solution["projection"] = "cell"`` selects the cell-centred route
          instead; see :func:`_solve_implicit`.
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

    # A scheme pinned at construction (Div(..., scheme=obj)) wins over the dict; the
    # exp.* surface has no scheme= kwarg and always resolves by name.
    for sp_op in equation.spatial_ops:
        if sp_op._scheme_explicit or sp_op._scheme_operator is None:
            continue
        keys = [sp_op._scheme_key_or_none(), type(sp_op).__name__]
        sp_op.scheme = lookup_scheme(schemes, keys, sp_op._scheme_operator, sp_op.scheme)

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
    """Sum the spatial operator contributions. Does NOT update the field.

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
        Outer list per level, inner list per box. Each array has shape
        (vNx, vNy, vNz) for ncomp=1 or (vNx, vNy, vNz, ncomp) for ncomp>1.
    """
    from .equation import Equation

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


# Renamed to the fvSolution camelCase spellings (API doc §5); an old key must raise
# rather than be silently ignored.
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


# The pressure discretisations ``solution["projection"]`` selects. Nodal is FIRST
# because it is the default and the one the physical validation oracles were measured
# against; cell is opt-in until that measurement is repeated for it.
_PROJECTION_ROUTES = ("nodal", "cell")


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
    """Solve imp.laplacian(sigma, p) == exp.div(U). Single- or multi-level.

    ``solution["solver"]`` accepts only ``"MLMG"``. ``solution["projection"]`` picks the
    DISCRETISATION, defaulting to ``"nodal"`` — the validated route the cavity Ghia
    profiles and the cylinder Cd/Cl/St numbers are the acceptance oracles for:

    * ``"nodal"`` — 27-point ``MLNodeLaplacian``, nodal unknowns, nodal rhs from
      ``compDivergence``. Steps below.
    * ``"cell"`` — compact 7-point face-coefficient operator through
      ``blockamr.linear_algebra``, cell unknowns; see :func:`_solve_implicit_cell`.

    Both fill ``p.grad`` with the same CELL-CENTRED shape, so ``correct()`` and the whole
    DSL surface are identical either way.

    The nodal route:

    1. Pack U into ncomp=3 MultiFab with ghost cells (per level)
    2. compDivergence → nodal RHS (per level)
    3. MLMG.solve → nodal p (all levels simultaneously)
    4. getFluxes → store cell-centred gradient on p.grad (per level)

    BCs, agglomeration and the bottom solver: see
    report/blockamr-python-notes.md#nodal-pressure-projection-bcs-agglomeration-and-the-bottom-solver
    """
    imp_op = eqn.implicit_lhs  # ImplicitLaplacian
    rhs_op = eqn.rhs  # CellDivergence

    _check_solution_keys(solution)
    cfg = solution or {}
    solver_name = cfg.get("solver", "MLMG")
    if solver_name != "MLMG":
        raise ValueError(f"Unknown solution['solver']='{solver_name}': only 'MLMG' is supported.")

    projection = cfg.get("projection", "nodal")
    if projection not in _PROJECTION_ROUTES:
        raise ValueError(
            f"Unknown solution['projection']='{projection}': accepted values are "
            + ", ".join(repr(route) for route in _PROJECTION_ROUTES)
            + "."
        )
    if projection == "cell":
        _solve_implicit_cell(imp_op, rhs_op, cfg)
        return

    rtol = cfg.get("rtol", 1e-10)
    atol = cfg.get("atol", 1e-12)
    max_iter = cfg.get("maxIter", 200)
    verbose = cfg.get("verbose", 0)
    # One of "cg", "bicgstab", "smoother", "cgbicg", "bicgcg", "default"; None → AMReX's.
    bottom_solver = cfg.get("bottomSolver", None)

    p_field = imp_op.field
    U_field = rhs_op.vel_field
    sigma = imp_op.sigma
    mesh = U_field.mesh
    n_levels = mesh.n_levels()

    # bottomSolver is in the key because `set_bottom_solver` is sticky across calls that
    # omit it, so a change can only take effect through a rebuild.
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

        # Per-face pressure BC stashed on the field, else the periodic/all-Neumann default.
        p_bc = getattr(p_field, "pressure_bc", None)
        if p_bc is not None:
            lo_bc, hi_bc = p_bc
        else:
            lo_bc = [
                blockamr.LinOpBCType.Periodic if is_per[d] else blockamr.LinOpBCType.Neumann
                for d in range(3)
            ]
            hi_bc = lo_bc[:]

        # Agglomeration only for the outflow-Dirichlet case, which plain nodal multigrid
        # coarsens badly; see the notes linked from this function's docstring.
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
    # None → AMReX's Krylov default, ~5 V-cycles here. Do NOT force "smoother": measured
    # ~600 iters/solve, ~100x the Krylov default.
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


class CellSolveCache:
    """Cached ``blockamr.linear_algebra`` objects for one field's CELL-centred solve.

    Stored on the field (``p_field._cell_imp_cache``), the same regrid-invalidation route
    as ``ImplicitSolveCache``. As in ``mac_project.MacProjectLaCache``, the matrix and the
    rhs must OUTLIVE the non-owning system and ``ops::Laplacian`` holds gamma by POINTER,
    so all three are kept here rather than let go of.
    """

    def __init__(self, lev, bc, flux, gamma_mf, matrix, system, p_mf, rhs_mf, solver):
        self.lev = lev
        self.bc = bc
        self.flux = flux
        self.gamma_mf = gamma_mf
        self.matrix = matrix
        self.system = system
        self.p_mf = p_mf
        self.rhs_mf = rhs_mf
        self.solver = solver


def _solve_implicit_cell(imp_op, rhs_op, cfg):
    """``div(sigma grad p) == div(U)`` with CELL unknowns, through the la layer.

    The MAC pattern of ``operators/mac_project.py``, moved from the face flux to the
    pressure projection: a compact 7-point operator differencing between ADJACENT cells,
    solved by ``cg``+``gmg``, with T08's face-based ``cell_gradient`` reconstructing the
    ``p.grad`` that ``correct()`` reads.

    The rhs is the face divergence of a FRESHLY interpolated ``U``, mirroring what the
    nodal route asks ``compDivergence`` for. Reusing the ``phi`` that ``momentum`` left
    behind would not be a projection at all: ``mac_project`` has already made that flux
    divergence-free, so its divergence is at solve tolerance and the correction would be
    the zero field.

    What the correction removes is therefore the divergence of the interpolated
    PREDICTED velocity — exactly the quantity ``mac_project`` will re-project at the top
    of the next step, and the cell-centred counterpart of the nodal ``compDivergence(U)``.
    """
    p_field = imp_op.field
    U_field = rhs_op.vel_field
    sigma = imp_op.sigma
    mesh = U_field.mesh
    n_levels = mesh.n_levels()
    if n_levels != 1:
        raise NotImplementedError(
            f"solution['projection']='cell' is single-level only, got {n_levels} levels. "
            "The nodal route solves every level simultaneously; a per-level cell solve is "
            "a different problem, not a cheaper spelling of the same one."
        )

    lev = 0
    geom = mesh.geom(lev)
    cache = _ensure_cell_cache(p_field, U_field, lev, cfg)

    interpolate(U_field, cache.flux)

    # ``laplacian`` writes each face coefficient as -gamma/dx**2, so the assembled system
    # is -div(grad(p)) and the rhs is negated to match — the same sign pairing
    # ``mac_project`` uses. sigma rides on the rhs instead of on gamma, which leaves the
    # matrix (and its multigrid hierarchy) valid when dt changes.
    cache.rhs_mf.copy_arrays([-arr / sigma for arr in _face_divergence(cache.flux, lev)])

    # Cold start, as the MAC solve does: the pressure written back below is then a
    # function of this step's rhs alone, and not of how many steps preceded it.
    cache.p_mf.set_val(0.0)
    cache.solver.solve(cache.system, cache.p_mf)

    # The nodal route cannot write its solution to ``p`` (nodal values, cell-centred
    # field), so ``p`` stays zero there. This route's unknowns ARE the cells, so the
    # field that is named p and marked for output can finally hold p.
    ng = cache.p_mf.n_grow()
    valid = []
    for arr in cache.p_mf.arrays():
        n = [int(arr.shape[ax]) - 2 * ng for ax in range(3)]
        sl = tuple(slice(ng, ng + n[ax]) for ax in range(3))
        valid.append(arr[sl[0], sl[1], sl[2], 0])
    p_field.mf[lev].copy_arrays(valid)

    # Last, because it OVERWRITES p_mf's ghost layer — that layer is what carries the
    # boundary condition into the outermost face difference.
    p_field.grad = [cell_gradient(cache.p_mf, geom, cache.bc)]


def _ensure_cell_cache(p_field, U_field, lev, cfg):
    """Build or return the cached linear-algebra objects for the cell-centred solve."""
    cache = getattr(p_field, "_cell_imp_cache", None)
    if cache is not None and cache.lev == lev:
        return cache

    mesh = U_field.mesh
    geom = mesh.geom(lev)
    ba = mesh.box_array(lev)
    dm = mesh.dm(lev)
    bc = _la_bc_from_pressure_bc(getattr(p_field, "pressure_bc", None), geom)

    matrix = MFFaceCoeffs.symmetric(blockamr.MeshLevel(ba, dm, geom), bc=bc)

    gamma_mf = blockamr.MultiFab(ba, dm, 1, 0)
    gamma_mf.set_val(1.0)

    p_mf = blockamr.MultiFab(ba, dm, 1, 1)
    p_mf.set_val(0.0)
    rhs_mf = blockamr.MultiFab(ba, dm, 1, 0)
    rhs_mf.set_val(0.0)

    system = LinearSystem(matrix, rhs_mf)
    # alpha=0: NO diagonal source. The coefficients are constant on a fixed level and
    # operators ACCUMULATE, so the Laplacian is assembled once here, not per step.
    system += laplacian(gamma_mf, geom, bc=bc)

    solver = Solver(
        SolverConfig(
            solver="cg",
            precond="gmg",
            max_iter=cfg.get("maxIter", 200),
            rtol=cfg.get("rtol", 1e-10),
            atol=cfg.get("atol", 1e-12),
            # Without a Dirichlet side the operator is singular; MLMG absorbs the
            # constant nullspace internally, the Krylov route has to be told. Setting
            # this on a PINNED system would silently solve a different problem.
            project_nullspace="dirichlet" not in bc,
        )
    )

    cache = CellSolveCache(
        lev=lev,
        bc=bc,
        # The rhs the nodal route gets from compDivergence(U): a face flux interpolated
        # from U, freshly, every step. Kept on the cache so it is allocated once.
        flux=FaceField(mesh, ncomp=1, ngrow=0, name=f"{p_field.name}_predicted_flux"),
        gamma_mf=gamma_mf,
        matrix=matrix,
        system=system,
        p_mf=p_mf,
        rhs_mf=rhs_mf,
        solver=solver,
    )
    p_field._cell_imp_cache = cache
    return cache
