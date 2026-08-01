# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Free MAC projection helper: mac_project(phi, sol_p).

Deliberately not an ``Equation`` — the worked ``step`` calls it directly. Tolerances
come from ``sol_p`` (the field's ``fvSolution.solvers['p']`` block). The MAC cache
lives on the ``phi`` field, on the same regrid-invalidation route as the MLMG
implicit-solve cache on the pressure field.

Two interchangeable solve routes, ``backend="la"`` (the default) and
``backend="mlmg"``, assemble the SAME cell-centred operator: ``MFFaceCoeffs`` plus
``linear_algebra.laplacian`` at gamma=1 is ``MLABecLaplacian`` at alpha=0/beta=1/b=1.
MLMG stays reachable as the A/B oracle the substitution is measured against
(``test/blockAmr/test_mac_project_la.py``).

The two agree to the solve tolerance on a periodic, a multi-box and an all-Neumann
configuration, and the face-gradient correction below is BITWISE ``MLMG.get_fluxes``
there. They differ in exactly one place: a DIRICHLET domain face, where AMReX defaults
to a third-order ghost extrapolation (``MLLinOp::setMaxOrder``, 3) while the
linear-algebra layer reflects the boundary cell — which is AMReX's ``max_order=2``.
Both closures are second-order accurate, so this is a boundary-closure difference and
not a correction; it is not bitwise.
"""

import jax.numpy as jnp

import blockamr

from ..linear_algebra import LinearSystem, MFFaceCoeffs, Solver, SolverConfig, laplacian

# LinOpBCType (what ``pressure_bc`` carries) -> the linear-algebra layer's spelling.
_LA_BC_NAMES = {
    blockamr.LinOpBCType.Periodic: "periodic",
    blockamr.LinOpBCType.Dirichlet: "dirichlet",
    blockamr.LinOpBCType.Neumann: "neumann",
}

# The same three kinds as MultiFab.fill_domain_boundary's integer codes.
_FILL_BC_CODES = {"periodic": 0, "dirichlet": 1, "neumann": 2}


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


class MacProjectLaCache:
    """Cached ``blockamr.linear_algebra`` objects for MAC projection on one level."""

    def __init__(self, lev, bc, gamma_mf, matrix, system, p_mf, rhs_mf, solver):
        self.lev = lev
        self.bc = bc
        # The matrix and the rhs must OUTLIVE the non-owning system, and ops::Laplacian
        # holds gamma by POINTER, so all three are kept here rather than let go of.
        self.gamma_mf = gamma_mf
        self.matrix = matrix
        self.system = system
        self.p_mf = p_mf
        self.rhs_mf = rhs_mf
        self.solver = solver


def mac_project(phi, sol_p, backend="la"):
    """Project face fluxes to be divergence-free (cell-centred Poisson solve).

    Solves ``div(beta grad(p_mac)) = div(phi)`` with alpha=0, beta=1, then corrects
    ``phi_f -= beta * grad_f(p_mac)`` with the FACE-centred gradient, the exact
    adjoint of the face divergence operator — which is what makes the corrected face
    flux exactly divergence-free rather than divergence-free to truncation.

    Parameters
    ----------
    phi : FaceField
        Face fluxes to project — MUTATED IN PLACE.
    sol_p : dict, optional
        The pressure field's fvSolution.solvers['p'] block — reads
        ``rtol``/``atol``/``maxIter`` for the MAC solve.
    backend : {"la", "mlmg"}, optional
        ``"la"`` assembles the operator through ``blockamr.linear_algebra``;
        ``"mlmg"`` is the MLABecLaplacian/MLMG route it replaced, kept as the A/B
        oracle (see the module docstring for where the two are not bitwise).
    """
    mesh = phi.mesh
    for lev in range(mesh.n_levels()):
        if backend == "la":
            _mac_project_level_la(phi, lev, sol_p)
        elif backend == "mlmg":
            _mac_project_level_mlmg(phi, lev, sol_p)
        else:
            raise ValueError(f"Unknown mac_project backend '{backend}': use 'la' or 'mlmg'.")


def _mac_project_level_la(phi, lev, sol_p):
    """MAC projection for one level, through the linear-algebra layer."""
    geom = phi.mesh.geom(lev)
    cache = _ensure_mac_la_cache(phi, lev, sol_p)

    # 1. laplacian() writes each face coefficient as -gamma/dx**2, so the assembled
    #    system is -div(grad(p)) — the same sign as MLABecLaplacian at alpha=0/beta=1,
    #    and the same negated divergence on the rhs.
    cache.rhs_mf.copy_arrays([-arr for arr in _face_divergence(phi, lev)])

    # 2. Cold start each step, as the MLMG route does. Only the valid region seeds the
    #    solve; the ghosts are (re)filled below, by the boundary condition.
    cache.p_mf.set_val(0.0)

    # 3. Solve -div(grad(p_mac)) = -div(phi).
    cache.solver.solve(cache.system, cache.p_mf)

    # 4. flux = -grad_f(p_mac), face-centred: what MLMG.get_fluxes returns.
    flux_by_axis = _face_gradient_flux(cache.p_mf, geom, cache.bc)

    # 5. phi_f += flux_f, i.e. phi -= grad since flux carries the minus sign.
    _correct_faces(phi, lev, flux_by_axis)


def _face_gradient_flux(p_mf, geom, bc):
    """``-grad_f(p)`` as per-box FACE arrays per direction, one per axis.

    The two-point difference between adjacent cell values, which is all the compact
    operator's gradient is. The GHOST layer is what carries the boundary condition, so
    it is filled first: FillBoundary for the periodic and inter-box neighbours, then
    the homogeneous domain sides with the reflection the matrix itself applies —
    Dirichlet ghost = -interior, Neumann ghost = interior (``core/bc.hpp``).
    """
    p_mf.fill_boundary(geom)
    p_mf.fill_domain_boundary(geom, [_FILL_BC_CODES[kind] for kind in bc], [[0.0]] * 6)

    dx = geom.cell_size()
    ng = p_mf.n_grow()
    results = [[] for _ in range(3)]
    for arr in p_mf.arrays():
        p = arr[:, :, :, 0]
        nc = [int(p.shape[ax]) - 2 * ng for ax in range(3)]
        for d in range(3):
            # One more face than cells in the normal direction; the low face reads the
            # ghost below the box, the high face the ghost above it.
            sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_lo = list(sl_hi)
            sl_hi[d] = slice(ng, ng + nc[d] + 1)
            sl_lo[d] = slice(ng - 1, ng + nc[d])
            results[d].append(-(p[tuple(sl_hi)] - p[tuple(sl_lo)]) / dx[d])
    return results


def _la_bc_from_pressure_bc(p_bc, geom):
    """``pressure_bc`` as the layer's 6-element (xlo, xhi, ylo, yhi, zlo, zhi) list.

    ``pressure_bc`` is ``(lo_bc, hi_bc)``, two per-AXIS ``LinOpBCType`` lists, so side
    ``2*d`` is ``lo_bc[d]`` and side ``2*d+1`` is ``hi_bc[d]`` — the side order
    ``la::parseBc`` documents. ``None`` is the periodic/all-Neumann default the MLMG
    route builds for itself.
    """
    is_per = geom.is_periodic()
    if p_bc is None:
        return ["periodic" if is_per[d] else "neumann" for d in range(3) for _ in range(2)]

    halves = p_bc
    bc = []
    for d in range(3):
        for half in (0, 1):
            kind = _LA_BC_NAMES.get(halves[half][d])
            if kind is None:
                raise ValueError(
                    f"pressure_bc side {2 * d + half} is {halves[half][d]}, which the "
                    "linear-algebra layer does not model"
                )
            bc.append(kind)
    return bc


def _ensure_mac_la_cache(phi, lev, sol_p):
    """Build or return the cached linear-algebra objects for one level."""
    cache = getattr(phi, "_mac_la_cache", None)
    if cache is not None and cache.lev == lev:
        return cache

    mesh = phi.mesh
    geom = mesh.geom(lev)
    ba = mesh.box_array(lev)
    dm = mesh.dm(lev)
    bc = _la_bc_from_pressure_bc(getattr(phi, "pressure_bc", None), geom)

    matrix = MFFaceCoeffs.symmetric(blockamr.MeshLevel(ba, dm, geom), bc=bc)

    # beta=1 with b=1 on every face is gamma=1 in the cell field the operator averages.
    gamma_mf = blockamr.MultiFab(ba, dm, 1, 0)
    gamma_mf.set_val(1.0)

    p_mf = blockamr.MultiFab(ba, dm, 1, 1)
    p_mf.set_val(0.0)
    rhs_mf = blockamr.MultiFab(ba, dm, 1, 0)
    rhs_mf.set_val(0.0)

    system = LinearSystem(matrix, rhs_mf)
    # alpha=0: NO diagonal source, so MFFaceCoeffs.diagonal_source is never called. The
    # coefficients do not change from step to step (gamma=1 on a fixed level) and
    # operators ACCUMULATE, so the Laplacian is assembled once here, not per step.
    system += laplacian(gamma_mf)

    cfg = sol_p or {}
    solver = Solver(
        SolverConfig(
            solver="cg",
            # gmg_kokkos runs the same V-cycle under one fused Kokkos launch per level
            # instead of one AMReX launch per box, which is where its margin comes from:
            # identical iteration counts, ~1.4-1.7x MLMG's wall clock against "gmg"'s
            # 0.6-1.2x (benchmarks/blockAmr/bench_solvers2.py, 32-256 cubed). Its three
            # refusals are all satisfied by the defaults this config leaves alone --
            # smoother bottom, symmetric operator, red-black smoother.
            precond=cfg.get("precond", "gmg_kokkos"),
            max_iter=cfg.get("maxIter", 200),
            rtol=cfg.get("rtol", 1e-10),
            atol=cfg.get("atol", 1e-12),
            # With no Dirichlet side the pure-Neumann/periodic Poisson operator is
            # singular; MLMG handles the constant nullspace internally, the Krylov
            # route has to be told. The projection only reads the GRADIENT of p_mac,
            # so which representative of p_mac comes back does not reach the answer.
            project_nullspace="dirichlet" not in bc,
        )
    )

    cache = MacProjectLaCache(
        lev=lev,
        bc=bc,
        gamma_mf=gamma_mf,
        matrix=matrix,
        system=system,
        p_mf=p_mf,
        rhs_mf=rhs_mf,
        solver=solver,
    )
    phi._mac_la_cache = cache
    return cache


def _mac_project_level_mlmg(phi, lev, sol_p):
    """MAC projection for one level, through MLABecLaplacian/MLMG."""
    cache = _ensure_mac_cache(phi, lev)

    p_bc = getattr(phi, "pressure_bc", None)
    has_dirichlet = p_bc is not None and any(
        bc == blockamr.LinOpBCType.Dirichlet for side in p_bc for bc in side
    )

    # 1. MLABecLaplacian solves -div(grad(p)) = rhs at alpha=0/beta=1, and we want
    #    div(grad(p)) = div(phi), hence the sign.
    rhs_arrs = [-arr for arr in _face_divergence(phi, lev)]
    cache.rhs_mf.copy_arrays(rhs_arrs)

    # 2. Ghost cells too: a Dirichlet outlet face reads its boundary value from them.
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

    # 4. flux = -beta * grad(p_mac), face-centred.
    cache.mlmg.get_fluxes(cache.flux_x, cache.flux_y, cache.flux_z)

    # 5. phi_f += flux_f, i.e. phi -= beta*grad since flux carries the minus sign.
    flux_by_axis = (cache.flux_x, cache.flux_y, cache.flux_z)
    _correct_faces(phi, lev, [[arr[:, :, :, 0] for arr in flux.arrays()] for flux in flux_by_axis])


def _correct_faces(phi, lev, flux_by_axis):
    """``phi_f += flux_f`` per direction, with ``flux_f = -grad_f(p_mac)``.

    ``flux_by_axis[d][box]`` is an (i, j, k) face array over the valid region (no
    ghosts), whether it came from ``MLMG.get_fluxes`` or from the face-difference
    kernel.
    """
    geom = phi.mesh.geom(lev)
    for d in range(3):
        face_mf = phi[lev][d].mf
        face_ng = face_mf.n_grow()
        face_arrs = face_mf.arrays()
        flux_arrs = flux_by_axis[d]

        results = []
        for bi in range(len(face_arrs)):
            f = face_arrs[bi][:, :, :, 0]
            fl = flux_arrs[bi]
            # flux has ngrow=0, face has ngrow=face_ng.
            if face_ng > 0:
                nf = [int(face_arrs[bi].shape[ax]) - 2 * face_ng for ax in range(3)]
                sl = tuple(slice(face_ng, face_ng + nf[ax]) for ax in range(3))
                results.append(f[sl] + fl)
            else:
                results.append(f + fl)

        face_mf.copy_arrays(results)
        face_mf.fill_boundary(geom)


def _face_divergence(phi, lev):
    """Cell-centred divergence of the face fluxes, as per-box (nx, ny, nz) arrays."""
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
        return cache

    mesh = phi.mesh
    geom = mesh.geom(lev)
    ba = mesh.box_array(lev)
    dm = mesh.dm(lev)
    is_per = geom.is_periodic()

    # (alpha*a - beta*div(b*grad)) phi, so alpha=0/beta=1/b=1 gives -div(grad(phi)).
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

    phi_mf = blockamr.MultiFab(ba, dm, 1, 1)
    rhs_mf = blockamr.MultiFab(ba, dm, 1, 0)

    # getFluxes output, face-centred.
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
