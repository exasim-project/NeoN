# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
from .. import explicit
from ..schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from ..schemes.registry import lookup_scheme


def solve(equation, *, dt=None, t=None, solution=None):
    """Discretise and solve an Equation.

      solve(Equation(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U),
                     schemes=schemes), dt=dt, t=t)
        → explicit CELL-CENTRED Forward Euler (AMReX C++ kernels). Schemes come from the
          equation's own ``schemes``.
    """
    from .equation import Equation

    if not isinstance(equation, Equation):
        raise TypeError(f"solve() expects an Equation, got {type(equation).__name__}")

    if len(equation.temporal_ops) != 1:
        raise ValueError(
            "solve() can only dispatch an equation with exactly one explicit ddt term; "
            f"got {len(equation.temporal_ops)} ddt term(s)."
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
        for lev in range(mesh.n_levels()):
            cell_field.fill_patch(lev, t)
            explicit.euler_step(equation, cell_field, lev, t, dt)

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
    all_levels = []

    for lev in range(mesh.n_levels()):
        cell_field.fill_patch(lev, t)
        all_levels.append(explicit.evaluate(spatial_ops, cell_field, lev, t))

    return all_levels

