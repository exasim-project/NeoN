# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Free MAC projection helper: mac_project(phi, sol_p).

Not modeled as an ``Equation`` (the API doc's worked ``step`` calls it
directly). Logic moved verbatim from the old
``DSLIncompressibleSolver._mac_project`` family; tolerances come from
``sol_p`` (the field's ``fvSolution.solvers['p']`` block). The MAC cache
lives on the ``phi`` field (same regrid-invalidation route as the MLMG
implicit-solve cache on the pressure field).
"""

import jax.numpy as jnp

import neon.blockamr as blockamr


class MacProjectCache:
    """Cached MLABecLaplacian/MLMG objects for MAC projection on one level."""

    def __init__(self, lev, lp, mlmg, phi_mf, rhs_mf, b_mfs, flux_x, flux_y, flux_z):
        self.lev = lev
        self.lp = lp
        self.mlmg = mlmg
        self.phi_mf = phi_mf
        self.rhs_mf = rhs_mf
        self.b_mfs = b_mfs
        self.flux_x = flux_x
        self.flux_y = flux_y
        self.flux_z = flux_z


def mac_project(phi, sol_p):
    """Project face fluxes to be divergence-free using MLABecLaplacian.

    Solves: div(beta * grad(p_mac)) = div(phi)
    Corrects: phi_f -= beta * grad_f(p_mac)

    Uses MLABecLaplacian with alpha=0, beta=1 (reduces to Laplacian
    for constant density). getFluxes returns face-centred gradients,
    which are the exact adjoint of the face divergence operator.

    Parameters
    ----------
    phi : FaceField
        Face fluxes to project (mutated in place).
    sol_p : dict, optional
        The pressure field's fvSolution.solvers['p'] block — reads
        ``rtol``/``atol`` for the MAC MLMG solve.
    """
    mesh = phi.mesh
    for lev in range(mesh.n_levels()):
        _mac_project_level(phi, lev, sol_p)


def _mac_project_level(phi, lev, sol_p):
    """MAC projection for one level."""
    mesh = phi.mesh
    geom = mesh.geom(lev)

    cache = _ensure_mac_cache(phi, lev)

    p_bc = getattr(phi, "pressure_bc", None)
    has_dirichlet = p_bc is not None and any(
        bc == blockamr.LinOpBCType.Dirichlet for side in p_bc for bc in side
    )

    # 1. Compute RHS = -div(phi) from face fluxes (JAX)
    # MLABecLaplacian solves (alpha*a - beta*div(b*grad))phi = rhs
    # with alpha=0, beta=1: -div(grad(p)) = rhs
    # We want div(grad(p)) = div(phi), so rhs = -div(phi)
    rhs_arrs = [-arr for arr in _face_divergence(phi, lev)]
    cache.rhs_mf.copy_arrays(rhs_arrs)

    # 2. Zero initial guess (incl. ghost cells when a Dirichlet outlet face
    #    reads its boundary value from them)
    cache.phi_mf.set_val(0.0)
    if has_dirichlet:
        pm = cache.phi_mf
        pm.copy_grown_arrays([jnp.zeros_like(a) for a in pm.grown_arrays()])

    # 3. Solve: div(beta * grad(p_mac)) = div(phi)
    cfg = sol_p or {}
    cache.mlmg.solve(
        cache.phi_mf,
        cache.rhs_mf,
        cfg.get("rtol", 1e-10),
        cfg.get("atol", 1e-12),
    )

    # 4. Get face-centred fluxes = -beta * grad(p_mac)
    cache.mlmg.get_fluxes(cache.flux_x, cache.flux_y, cache.flux_z)

    # 5. Correct phi: phi_f += flux_f (flux = -beta*grad, so phi -= beta*grad)
    flux_by_axis = (cache.flux_x, cache.flux_y, cache.flux_z)
    for d in range(3):
        face_mf = phi[lev][d].mf
        face_ng = face_mf.n_grow()
        face_arrs = face_mf.arrays()
        flux_arrs = flux_by_axis[d].arrays()

        results = []
        for bi in range(len(face_arrs)):
            f = face_arrs[bi][:, :, :, 0]
            fl = flux_arrs[bi][:, :, :, 0]
            # flux has ngrow=0, face has ngrow=face_ng
            if face_ng > 0:
                nf = [int(face_arrs[bi].shape[ax]) - 2 * face_ng for ax in range(3)]
                sl = tuple(slice(face_ng, face_ng + nf[ax]) for ax in range(3))
                results.append(f[sl] + fl)
            else:
                results.append(f + fl)

        face_mf.copy_arrays(results)
        face_mf.fill_boundary(geom)


def _face_divergence(phi, lev):
    """Compute cell-centred divergence from face fluxes (JAX).

    Returns list of per-box arrays (nx, ny, nz).
    """
    dx = phi.mesh.geom(lev).cell_size()
    face_arrs = [phi[lev][d].mf.arrays() for d in range(3)]
    face_ngs = [phi[lev][d].mf.n_grow() for d in range(3)]
    n_boxes = len(face_arrs[0])

    results = []
    for bi in range(n_boxes):
        div_val = None
        for d in range(3):
            f = face_arrs[d][bi][:, :, :, 0]
            ng = face_ngs[d]
            # Valid cell count: face valid count - 1 in normal dir, same in others
            nf = [int(f.shape[ax]) - 2 * ng for ax in range(3)]
            nc = list(nf)
            nc[d] -= 1  # one fewer cell than faces in normal direction

            sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_hi[d] = slice(ng + 1, ng + 1 + nc[d])
            sl_lo[d] = slice(ng, ng + nc[d])
            contrib = (f[tuple(sl_hi)] - f[tuple(sl_lo)]) / dx[d]
            div_val = contrib if div_val is None else div_val + contrib
        results.append(div_val)
    return results


def _ensure_mac_cache(phi, lev):
    """Build or return the cached MAC solver objects for one level."""
    cache = getattr(phi, "_mac_cache", None)
    if cache is not None and cache.lev == lev:
        # Rebind face data but reuse operator/solver
        return cache

    mesh = phi.mesh
    geom = mesh.geom(lev)
    ba = mesh.box_array(lev)
    dm = mesh.dm(lev)
    is_per = geom.is_periodic()

    # MLABecLaplacian: (alpha*a - beta*div(b*grad)) phi
    # For MAC: alpha=0, beta=1, b=1 → -div(grad(phi)) = RHS
    lp = blockamr.MLABecLaplacian(geom, ba, dm, blockamr.LPInfo())
    p_bc = getattr(phi, "pressure_bc", None)
    if p_bc is not None:
        lo_bc, hi_bc = p_bc
    else:
        lo_bc = [
            blockamr.LinOpBCType.Periodic if is_per[d] else blockamr.LinOpBCType.Neumann
            for d in range(3)
        ]
        hi_bc = lo_bc[:]
    lp.set_domain_bc(lo_bc, hi_bc)
    lp.set_level_bc(0, None)
    lp.set_scalars(0.0, 1.0)  # alpha=0, beta=1

    # b-coefficients = 1 on all faces
    b_mfs = []
    for d in range(3):
        ba_copy = blockamr.BoxArray(ba)
        ba_copy.surrounding_nodes(d)
        b_mf = blockamr.MultiFab(ba_copy, dm, 1, 0)
        b_mf.set_val(1.0)
        ba_copy.enclosed_cells(d)
        b_mfs.append(b_mf)
    lp.set_b_coeffs(0, b_mfs[0], b_mfs[1], b_mfs[2])

    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.set_bottom_verbose(0)

    # Scratch MultiFabs
    phi_mf = blockamr.MultiFab(ba, dm, 1, 1)
    rhs_mf = blockamr.MultiFab(ba, dm, 1, 0)

    # Face-centred flux MultiFabs for getFluxes output
    flux_mfs = []
    for d in range(3):
        ba_face = blockamr.BoxArray(ba)
        ba_face.surrounding_nodes(d)
        flux_mf = blockamr.MultiFab(ba_face, dm, 1, 0)
        ba_face.enclosed_cells(d)
        flux_mfs.append(flux_mf)

    cache = MacProjectCache(
        lev=lev,
        lp=lp,
        mlmg=mlmg,
        phi_mf=phi_mf,
        rhs_mf=rhs_mf,
        b_mfs=b_mfs,
        flux_x=flux_mfs[0],
        flux_y=flux_mfs[1],
        flux_z=flux_mfs[2],
    )
    phi._mac_cache = cache
    return cache
