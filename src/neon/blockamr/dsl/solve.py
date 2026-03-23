# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax

import neon.blockamr as blockamr
from ..schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from ..schemes.schemes_dict import SchemesDict


def solve(expr, t, dt, schemes=None):
    """Explicit solve for one timestep.

    Convention: ddt(phi) + div(U, phi) = 0
    -> phi_new = phi_old - dt * sum(coeff_i * spatial_op_i)

    Works with both single-level Mesh and multi-level AmrMesh.
    """
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


@jax.jit
def _fused_step(phi, kernels):
    """Apply all operator kernels and sum the results."""
    total = 0.0
    for k in kernels:
        total = total + k(phi)
    return total


def _forward_euler_level(expr, cell_field, lev, t, dt, ddt_coeff):
    mf = cell_field.mf[lev]
    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        phi_4d = mf.array(mfi)

        kernels = [op.build_kernel(mfi, t, lev=lev) for op in expr.spatial_ops]
        source = _fused_step(phi_4d, kernels)

        phi = phi_4d[:, :, :, 0]
        s = slice(ng, -ng if ng else None)
        phi_old = phi[s, s, s]
        phi_new = phi_old - (dt / ddt_coeff) * source

        mf.copy_from(mfi, phi_new)
