# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax

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


@jax.jit(static_argnames=["ng"])
def _fused_euler_step(phi_4d, kernels, dt_over_coeff, ng):
    """Fuse operator evaluation + slicing + Euler update into one kernel."""
    total = 0.0
    for k in kernels:
        total = total + k(phi_4d)
    phi = phi_4d[:, :, :, 0]
    s = slice(ng, -ng if ng else None)
    phi_old = phi[s, s, s]
    return phi_old - dt_over_coeff * total


def _forward_euler_level(expr, cell_field, lev, t, dt, ddt_coeff):
    mf = cell_field.mf[lev]
    ng = mf.n_grow()
    ncomp = cell_field.ncomp
    dt_over_coeff = dt / ddt_coeff
    res = []
    for mfi in blockamr.MFIterator(mf):
        phi_4d = mf.array(mfi)
        kernels = [op.build_kernel(mfi, t, lev=lev) for op in expr.spatial_ops]
        if ncomp == 1:
            phi_new = _fused_euler_step(phi_4d, kernels, dt_over_coeff, ng)
            res.append(phi_new)
        else:
            # Process each component: create a (nx,ny,nz,1) view per component
            # so existing kernels (which read [:,:,:,0]) work unchanged.
            comps = []
            for c in range(ncomp):
                phi_1c = phi_4d[:, :, :, c:c+1]  # (nx, ny, nz, 1)
                comp_new = _fused_euler_step(phi_1c, kernels, dt_over_coeff, ng)
                comps.append(comp_new)
            res.append(jax.numpy.stack(comps, axis=-1))

    mf.copy_arrays(res)


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
