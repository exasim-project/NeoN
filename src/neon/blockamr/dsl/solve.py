# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

import neon.blockamr as blockamr
import equinox as eqx
from ..cell_kernels_3d import FusedEulerKernel, CombinedSource
from ..flat_refs import FlatCellRef
from ..tiled_context import TiledContext
from ..schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from ..schemes.registry import lookup_scheme


def forward_euler(spatial_kernels, dt_over_coeff):
    """Build a fused forward Euler kernel: phi_new = phi - dt * sum(spatial_kernels)."""
    return FusedEulerKernel(spatial_kernels=spatial_kernels, dt_over_coeff=dt_over_coeff)


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
    ddt_coeff = equation.temporal_ops[0].coeff

    ddt_scheme = lookup_scheme(schemes, ["ddt", "Ddt"], "ddt", ForwardEuler())

    # Resolve operator schemes from the schemes dict (names or objects,
    # keyed by scheme_key or class name). A scheme object passed at the
    # call site (exp.div(..., scheme=obj)) wins over the dict.
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
        for lev in range(mesh.n_levels()):
            cell_field.fill_patch(lev, t)
            _forward_euler_level(equation, cell_field, lev, t, dt, ddt_coeff)

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
    expr : Expression or single spatial operator
        e.g. ``exp.div(phi, U, scheme=VanLeer())`` or
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
    all_levels = []

    for lev in range(mesh.n_levels()):
        cell_field.fill_patch(lev, t)
        mf = cell_field.mf[lev]
        dh = tuple(float(d) for d in mesh.geom(lev).cell_size())
        ng = mf.n_grow()

        ctx = TiledContext(dh=dh, ng=ng, lev=lev)
        spatial_kernels = tuple(op.build_kernel_3d(ctx, t) for op in spatial_ops)
        kernel = CombinedSource(spatial_kernels)

        # Create temp MultiFab for output
        out_mf = blockamr.MultiFab(
            mesh.box_array(lev), mesh.dm(lev), cell_field.ncomp, ng, memory="default"
        )
        out_mf.set_val(0.0)

        parallel_for(kernel, cell_field, lev, out_mf=out_mf)

        # Extract per-box valid results
        meta = mf.fab_metadata()
        lev_results = []
        for arr, m in zip(out_mf.arrays(), meta):
            Nx, Ny, Nz = m[1], m[2], m[3]
            vNx, vNy, vNz = Nx - 2 * ng, Ny - 2 * ng, Nz - 2 * ng
            lev_results.append(arr[ng : ng + vNx, ng : ng + vNy, ng : ng + vNz, : cell_field.ncomp])
        all_levels.append(lev_results)

    return all_levels


# ---------------------------------------------------------------------------
# Tiled Pallas dispatch using TileLayout
# ---------------------------------------------------------------------------

BF = 8
NUM_WARPS = 8
NUM_STAGES = 1


def set_tile_size(bf):
    """Set the Pallas tile size (default 8). Must be a power of 2."""
    global BF
    BF = bf


def _forward_euler_level(expr, cell_field, lev, t, dt, ddt_coeff):
    """One forward Euler step for all boxes on one AMR level."""
    dh = tuple(float(d) for d in cell_field.mesh.geom(lev).cell_size())
    ng = cell_field.mf[lev].n_grow()
    dt_over_coeff = dt / ddt_coeff

    ctx = TiledContext(dh=dh, ng=ng, lev=lev)
    spatial_kernels = tuple(op.build_kernel_3d(ctx, t) for op in expr.spatial_ops)
    kernel = FusedEulerKernel(spatial_kernels, dt_over_coeff)

    parallel_for(kernel, cell_field, lev)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _run_pallas(k_treedef, n_tiles, n_padded, total_phi, bf, phi_flat, tiles, cvs, *k_leaves):
    """JIT'd Pallas dispatch. Static args define the compilation key.

    cvs: per-box cell-valid-start offsets (int32, n_boxes_padded).
    """
    n_k = k_treedef.num_leaves
    tile_vol = bf**3

    def pallas_kernel(*refs):
        from ..flat_refs import _FaceAxisBoxed

        fn = jax.tree.unflatten(k_treedef, refs[:n_k])
        phi_ref = refs[n_k]
        tiles_ref = refs[n_k + 1]
        cvs_ref = refs[n_k + 2]
        out_ref = refs[n_k + 3]

        tid = pl.program_id(0)
        base = tid * 5
        cell_off = plt.load(tiles_ref.at[base + 0])
        c_sx = plt.load(tiles_ref.at[base + 1])
        c_sy = plt.load(tiles_ref.at[base + 2])
        c_sz = plt.load(tiles_ref.at[base + 3])
        bid = plt.load(tiles_ref.at[base + 4])

        @pl.when(tid < n_tiles)
        def _():
            li = jnp.arange(bf)[:, None, None]
            lj = jnp.arange(bf)[None, :, None]
            lk = jnp.arange(bf)[None, None, :]

            # Compute box-local valid indices from tile offset
            cell_vs = plt.load(cvs_ref.at[bid])
            delta = cell_off - cell_vs
            vi0 = delta % c_sy
            vj0 = (delta // c_sy) % (c_sz // c_sy)
            vk0 = delta // c_sz
            gi = vi0 + li
            gj = vj0 + lj
            gk = vk0 + lk

            phi = FlatCellRef(phi_ref, cell_vs, c_sx, c_sy, c_sz)

            # Bind real box_id on face refs (replaces dummy box_id=0)
            fn_bound = fn
            face_axes = [
                l
                for l in jax.tree.leaves(fn, is_leaf=lambda x: isinstance(x, _FaceAxisBoxed))
                if isinstance(l, _FaceAxisBoxed)
            ]
            if face_axes:
                fn_bound = eqx.tree_at(
                    lambda k: tuple(
                        l.box_id
                        for l in jax.tree.leaves(k, is_leaf=lambda x: isinstance(x, _FaceAxisBoxed))
                        if isinstance(l, _FaceAxisBoxed)
                    ),
                    fn,
                    tuple(bid for _ in face_axes),
                )

            val = fn_bound(bid, gi, gj, gk, phi)
            oi = cell_vs + gi * c_sx + gj * c_sy + gk * c_sz
            plt.store(out_ref.at[oi.reshape(tile_vol)], val=val.reshape(tile_vol))

    k_leaf_shapes = tuple(l.shape for l in k_leaves)
    n_cvs = cvs.shape[0]
    in_specs = [pl.BlockSpec(s, lambda i, _n=len(s): (0,) * _n) for s in k_leaf_shapes] + [
        pl.BlockSpec((total_phi,), lambda i: (0,)),
        pl.BlockSpec((n_padded * 5,), lambda i: (0,)),
        pl.BlockSpec((n_cvs,), lambda i: (0,)),
    ]

    return pl.pallas_call(
        pallas_kernel,
        out_shape=jax.ShapeDtypeStruct((total_phi,), phi_flat.dtype),
        grid=(n_padded,),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((total_phi,), lambda i: (0,)),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
    )(*k_leaves, phi_flat, tiles, cvs)


def _gather_valid(flat, box_off, Nx, Ny, Nz, ng):
    """Extract valid (non-ghost) cells from a flat Fortran-ordered buffer."""
    vNx, vNy, vNz = Nx - 2 * ng, Ny - 2 * ng, Nz - 2 * ng
    ix = jnp.arange(vNx) + ng
    it = jnp.arange(vNy) + ng
    iz = jnp.arange(vNz) + ng
    idx = ix[:, None, None] + Nx * it[None, :, None] + Nx * Ny * iz[None, None, :]
    return flat[box_off + idx.reshape(-1)].reshape(vNx, vNy, vNz)


def _extract_valid_boxes(flat, meta, ng, ncomp):
    """Extract valid cells from flat output for all boxes. Returns list of arrays."""
    results = []
    for m in meta:
        off, Nx, Ny, Nz = int(m[0]), m[1], m[2], m[3]
        if ncomp == 1:
            results.append(_gather_valid(flat, off, Nx, Ny, Nz, ng))
        else:
            bM = Nx * Ny * Nz
            comps = [_gather_valid(flat, off + c * bM, Nx, Ny, Nz, ng) for c in range(ncomp)]
            results.append(jnp.stack(comps, axis=-1))
    return results


def parallel_for(kernel, cell_field, lev, bf=BF, out_mf=None):
    """Tiled Pallas dispatch. Kernel holds all data as equinox leaves.

    For ncomp>1, processes each component separately — the kernel always
    operates on phi[i,j,k,0] (single component). The tile offsets point
    to the same flat buffer; for comp c we shift by c * plane_size per box.
    With uniform boxes (all same size), plane_size is total_phi / ncomp.
    """
    mf = cell_field.mf[lev]
    phi_flat = mf.contiguous_array()

    # Reduce bf so it evenly divides all valid dimensions across all boxes
    ng = mf.n_grow()
    meta = mf.fab_metadata()
    while bf > 1:
        if all(
            (m[1] - 2 * ng) % bf == 0 and (m[2] - 2 * ng) % bf == 0 and (m[3] - 2 * ng) % bf == 0
            for m in meta
        ):
            break
        bf //= 2
    layout = blockamr.build_tile_layout(mf, bf)
    ncomp = cell_field.ncomp
    total_phi = phi_flat.shape[0]

    k_leaves, k_treedef = jax.tree.flatten(kernel)
    k_leaves = [jnp.asarray(l) if not hasattr(l, "shape") else l for l in k_leaves]

    # Per-box cell-valid-start: offset to first valid cell (ng, ng, ng)
    n_boxes = len(meta)
    mb = 1
    while mb < n_boxes:
        mb <<= 1
    cvs = jnp.array(
        [int(m[0]) + ng + m[1] * ng + m[1] * m[2] * ng for m in meta],
        dtype=jnp.int32,
    )
    cvs = jnp.pad(cvs, (0, mb - n_boxes), constant_values=int(cvs[0]))

    target_mf = out_mf if out_mf is not None else mf

    if ncomp == 1:
        out_flat = _run_pallas(
            k_treedef,
            layout.n_tiles,
            layout.n_tiles_padded,
            total_phi,
            bf,
            phi_flat,
            layout.tiles,
            cvs,
            *k_leaves,
        )
        # Extract valid cells only — ghost cells must not be overwritten
        results = _extract_valid_boxes(out_flat, meta, ng, ncomp=1)
        target_mf.copy_arrays(results)
    else:
        # For ncomp>1: run the kernel once per component.
        # Shift tile offsets so phi[i,j,k,0] reads component c's data.
        # Fortran order: comp c starts at c*M within each box (M=Nx*Ny*Nz).
        n_boxes = len(meta)
        n_t = layout.n_tiles_padded
        offset_idx = jnp.arange(n_t) * 5

        # Per-tile component stride (M = Nx*Ny*Nz per box)
        box_Ms = [m[1] * m[2] * m[3] for m in meta]
        M0 = box_Ms[0]
        uniform = all(bM == M0 for bM in box_Ms)
        if uniform:
            per_tile_M = jnp.full(n_t, M0, dtype=jnp.int32)
        else:
            comp_strides = jnp.array(box_Ms, dtype=jnp.int32)
            mb = 1
            while mb < n_boxes:
                mb <<= 1
            padded_strides = jnp.pad(
                comp_strides, (0, mb - n_boxes), constant_values=int(comp_strides[0])
            )
            box_ids = layout.tiles[4::5][:n_t]
            per_tile_M = padded_strides[box_ids]

        # Run kernel per component, extract valid cells
        comp_valid = []  # comp_valid[c][bi] = (vNx, vNy, vNz) array
        for c in range(ncomp):
            shifted_tiles = layout.tiles.at[offset_idx].add(per_tile_M * c)
            shifted_cvs = (
                cvs
                + jnp.array(
                    [m[1] * m[2] * m[3] * c for m in meta] + [0] * (mb - n_boxes), dtype=jnp.int32
                )[:mb]
            )
            out_c = _run_pallas(
                k_treedef,
                layout.n_tiles,
                layout.n_tiles_padded,
                total_phi,
                bf,
                phi_flat,
                shifted_tiles,
                shifted_cvs,
                *k_leaves,
            )
            # Extract valid cells for comp c from each box
            per_box = []
            for bi, m in enumerate(meta):
                off, Nx, Ny, Nz = int(m[0]), m[1], m[2], m[3]
                bM = Nx * Ny * Nz
                per_box.append(_gather_valid(out_c, off + c * bM, Nx, Ny, Nz, ng))
            comp_valid.append(per_box)

        # Assemble per-box (vNx, vNy, vNz, ncomp) and write back
        results = []
        for bi in range(n_boxes):
            results.append(jnp.stack([comp_valid[c][bi] for c in range(ncomp)], axis=-1))
        target_mf.copy_arrays(results)


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
