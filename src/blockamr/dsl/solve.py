# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import numpy as np

from blockamr.schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from blockamr.schemes.schemes_dict import SchemesDict


def solve(expr, t, dt, schemes=None):
    """Explicit solve for one timestep.

    Convention: ddt(phi) + div(U, phi) = 0
    -> phi_new = phi_old - dt * sum(coeff_i * spatial_op_i)
    """
    assert len(expr.temporal_ops) == 1
    field = expr.temporal_ops[0].field
    ddt_coeff = expr.temporal_ops[0].coeff

    sd = SchemesDict(schemes)
    ddt_scheme = sd.lookup("Ddt", ForwardEuler())

    # Override operator schemes from schemes dict
    for sp_op in expr.spatial_ops:
        resolved = sd.lookup(sp_op._name, sp_op.scheme)
        sp_op.scheme = resolved

    if isinstance(ddt_scheme, ForwardEuler):
        _forward_euler(expr, field, t, dt, ddt_coeff)
    elif isinstance(ddt_scheme, (RungeKutta2, RungeKutta4)):
        raise NotImplementedError(f"{ddt_scheme.type} is not yet implemented")
    else:
        raise ValueError(f"Unknown ddt scheme: {ddt_scheme}")


def _forward_euler(expr, field, t, dt, ddt_coeff):
    field.fill_boundary()

    for patch in field.patches():
        source = jnp.zeros(patch.valid_arr.shape[:3])

        for sp_op in expr.spatial_ops:
            source = source + sp_op.coeff * sp_op.compute(patch, t)

        phi_old = jnp.asarray(patch.valid_arr[:, :, :, 0])
        phi_new = phi_old - (dt / ddt_coeff) * source
        patch.valid_arr[:, :, :, 0] = np.asarray(phi_new)
