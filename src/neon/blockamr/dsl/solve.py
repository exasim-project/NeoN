# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon.blockamr as blockamr
from ..schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from ..schemes.schemes_dict import SchemesDict


def solve(expr, t=None, dt=None, schemes=None):
    """Solve an expression or equation.

    Two forms:
      solve(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U), t, dt)
        → explicit Forward Euler (JAX)

      solve(imp.laplacian(sigma, p) == exp.div(U), schemes=schemes_p)
        → implicit MLMG solve (AMReX C++)
    """
    from .equation import Equation
    if isinstance(expr, Equation):
        _solve_implicit(expr, schemes=schemes)
        return

    assert len(expr.temporal_ops) == 1
    cell_field = expr.temporal_ops[0].field  # CellField
    mesh = cell_field.mesh
    ddt_coeff = expr.temporal_ops[0].coeff

    sd = SchemesDict(schemes)
    ddt_scheme = sd.lookup("Ddt", ForwardEuler())

    # Override operator schemes from schemes dict
    for sp_op in expr.spatial_ops:
        resolved = sd.lookup(sp_op._name, sp_op.scheme)
        sp_op.scheme = resolved

    # Validate that the field has enough ghost cells for the widest stencil
    required = expr.required_ngrow
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
            _forward_euler_level(expr, cell_field, lev, t, dt, ddt_coeff)

        # restrict fine -> coarse
        for lev in reversed(range(mesh.n_levels() - 1)):
            blockamr.average_down(
                cell_field.mf[lev + 1], cell_field.mf[lev],
                mesh.geom(lev + 1), mesh.geom(lev),
                0, cell_field.ncomp, mesh.ref_ratio(lev),
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
    from ..flattened_boxes import flattened_boxes_from_mf, build_buckets
    from ..bucket_dispatch import evaluate_bucket
    from .expression import Expression

    # Wrap a bare operator in an expression if needed
    if not isinstance(expr, Expression):
        from . import exp as _exp
        # Single operator — need to find its field for fill_patch
        op = expr
        cell_field = op.field
        spatial_ops = [op]
    else:
        spatial_ops = expr.spatial_ops
        # Get field from first spatial op
        cell_field = spatial_ops[0].field

    mesh = cell_field.mesh
    all_levels = []

    for lev in range(mesh.n_levels()):
        cell_field.fill_patch(lev, t)
        mf = cell_field.mf[lev]
        fb = flattened_boxes_from_mf(mf)
        dh = tuple(float(d) for d in mesh.geom(lev).cell_size())
        buckets = build_buckets(fb, dh, lev=lev)

        lev_results = [None] * fb.n_boxes
        for bucket in buckets:
            if bucket.n_valid == 0:
                continue
            kernels = tuple(op.build_kernel(bucket, t) for op in spatial_ops)
            result = evaluate_bucket(bucket, kernels)
            _scatter_results(lev_results, result, bucket)

        all_levels.append(lev_results)

    return all_levels


def _forward_euler_level(expr, cell_field, lev, t, dt, ddt_coeff):
    """One forward Euler step for all boxes on one AMR level."""
    from ..flattened_boxes import flattened_boxes_from_mf, build_buckets
    from ..bucket_dispatch import process_bucket

    mf = cell_field.mf[lev]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in cell_field.mesh.geom(lev).cell_size())
    buckets = build_buckets(fb, dh, lev=lev)
    dt_over_coeff = dt / ddt_coeff

    all_results = [None] * fb.n_boxes

    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        kernels = tuple(op.build_kernel(bucket, t) for op in expr.spatial_ops)
        result = process_bucket(bucket, dt_over_coeff, kernels)
        _scatter_results(all_results, result, bucket)

    mf.copy_arrays(all_results)


def _scatter_results(all_results, result, bucket):
    """Unpack process_bucket output into per-box 4D arrays for copy_arrays.

    ncomp=1: result is (max_boxes, n_cells_padded) → (vNx, vNy, vNz, 1)
    ncomp>1: result is (max_boxes, n_cells_padded, ncomp) → (vNx, vNy, vNz, ncomp)

    Uses per-box Nx_arr/Ny_arr/Nz_arr for reshaping since boxes in the
    same bucket can have different shapes.
    """
    import jax.numpy as jnp

    ng = bucket.ng
    for bi, mf_idx in enumerate(bucket.box_indices[:bucket.n_valid]):
        Nx = int(bucket.Nx_arr[bi])
        Ny = int(bucket.Ny_arr[bi])
        Nz = int(bucket.Nz_arr[bi])
        vNx = Nx - 2 * ng
        vNy = Ny - 2 * ng
        vNz = Nz - 2 * ng
        actual_n_cells = vNx * vNy * vNz

        cell_data = result[bi]
        if cell_data.ndim == 1:
            # ncomp=1: take only actual valid cells, then reshape
            valid = cell_data[:actual_n_cells]
            valid_3d = valid.reshape(vNz, vNy, vNx).transpose(2, 1, 0)
            all_results[mf_idx] = valid_3d[:, :, :, None]
        else:
            # ncomp>1: (n_cells_padded, ncomp)
            ncomp = cell_data.shape[1]
            comps = []
            for c in range(ncomp):
                comp_valid = cell_data[:actual_n_cells, c]
                comp_3d = comp_valid.reshape(vNz, vNy, vNx).transpose(2, 1, 0)
                comps.append(comp_3d)
            all_results[mf_idx] = jnp.stack(comps, axis=-1)


# ---------------------------------------------------------------------------
# Implicit equation solver
# ---------------------------------------------------------------------------

def _solve_implicit(eqn, schemes=None):
    """Solve imp.laplacian(sigma, p) == exp.div(U).

    Supports single-level and multi-level AMR meshes:
    1. Pack U into ncomp=3 MultiFab with ghost cells (per level)
    2. compDivergence → nodal RHS (per level)
    3. MLMG.solve → nodal p (all levels simultaneously)
    4. getFluxes → store cell-centred gradient on p.grad (per level)
    """
    import jax.numpy as jnp

    imp_op = eqn.lhs  # ImplicitLaplacian
    rhs_op = eqn.rhs  # CellDivergence

    cfg = schemes or {}
    rtol = cfg.get("rtol", 1e-10)
    atol = cfg.get("atol", 1e-12)
    max_iter = cfg.get("max_iter", 200)
    verbose = cfg.get("verbose", 0)

    p_field = imp_op.field
    U_field = rhs_op.vel_field
    sigma = imp_op.sigma
    mesh = U_field.mesh
    n_levels = mesh.n_levels()

    # Rebuild when n_levels or sigma changes.
    # Safe to rebuild any time — convert_ba creates independent nodal BAs
    # without mutating the mesh's cell BAs (no shared-ownership issue).
    needs_rebuild = not hasattr(p_field, '_imp_solver')
    if not needs_rebuild:
        s_old = p_field._imp_solver
        if s_old['n_levels'] != n_levels or s_old['sigma'] != sigma:
            needs_rebuild = True

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

            # Independent nodal BA — does not mutate the mesh's cell BA
            nodal_ba = blockamr.convert_ba(ba_lev, nodal_type)

            phi_mf = blockamr.MultiFab(nodal_ba, dm_lev, 1, 1)
            phi_mf.set_val(0.0)

            phi_mfs.append(phi_mf)
            rhs_mfs.append(blockamr.MultiFab(nodal_ba, dm_lev, 1, 0))
            vel3_mfs.append(blockamr.MultiFab(ba_lev, dm_lev, 3, U_field.ngrow))
            fluxes_mfs.append(blockamr.MultiFab(ba_lev, dm_lev, 3, 0))

        is_per = geoms[0].is_periodic()
        if n_levels == 1:
            lp = blockamr.MLNodeLaplacian(geoms[0], bas[0], dms[0],
                                          blockamr.LPInfo(), sigma)
        else:
            lp = blockamr.MLNodeLaplacian(geoms, bas, dms,
                                          blockamr.LPInfo(), sigma)

        lo_bc = [blockamr.LinOpBCType.Periodic if is_per[d]
                 else blockamr.LinOpBCType.Neumann for d in range(3)]
        lp.set_domain_bc(lo_bc, lo_bc[:])

        p_field._imp_solver = {
            'lp': lp,
            'mlmg': blockamr.MLMG(lp),
            'phi_mfs': phi_mfs,
            'rhs_mfs': rhs_mfs,
            'vel3_mfs': vel3_mfs,
            'fluxes_mfs': fluxes_mfs,
            'n_levels': n_levels,
            'sigma': sigma,
        }

    s = p_field._imp_solver
    s['mlmg'].set_verbose(verbose)
    s['mlmg'].set_max_iter(max_iter)
    s['mlmg'].set_bottom_verbose(0)

    # 1. Pack velocity with ghost cells into ncomp=3 MultiFab (per level)
    for lev in range(n_levels):
        mf = U_field.mf[lev]
        grown = mf.grown_arrays()
        for bi, mfi in enumerate(blockamr.MFIterator(s['vel3_mfs'][lev])):
            s['vel3_mfs'][lev].copy_grown_from(mfi, grown[bi])

    # 2. compDivergence → nodal RHS
    if n_levels == 1:
        s['lp'].comp_divergence(s['rhs_mfs'][0], s['vel3_mfs'][0])
    else:
        s['lp'].comp_divergence(s['rhs_mfs'], s['vel3_mfs'])

    # 3. MLMG.solve (warm-start from previous phi)
    if n_levels == 1:
        s['mlmg'].solve(s['phi_mfs'][0], s['rhs_mfs'][0], rtol, atol)
    else:
        s['mlmg'].solve(s['phi_mfs'], s['rhs_mfs'], rtol, atol)

    print(f"  MLMG  iters={s['mlmg'].get_num_iters()}  "
          f"init_res={s['mlmg'].get_init_residual():.6e}  "
          f"final_res={s['mlmg'].get_final_residual():.6e}")

    # 4. getFluxes → store gradient on p_field for correct()
    # getFluxes returns -sigma * grad(phi). Negate and divide by sigma
    # so p_field.grad[lev] = list of per-box +grad(phi) arrays.
    if n_levels == 1:
        s['mlmg'].get_fluxes(s['fluxes_mfs'][0])
    else:
        s['mlmg'].get_fluxes(s['fluxes_mfs'])

    p_field.grad = []
    for lev in range(n_levels):
        box_grads = [-arr / sigma for arr in s['fluxes_mfs'][lev].arrays()]
        p_field.grad.append(box_grads)
