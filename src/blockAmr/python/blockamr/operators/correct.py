# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Velocity correction: ``correct(U, -dt * exp.grad(p))`` applies ``U -= dt*grad(p)``."""

import jax.numpy as jnp

import blockamr


def correct(cell_field, expr):
    """Apply a correction to a CellField, IN PLACE.

    Parameters
    ----------
    cell_field : CellField
        Field to correct (ncomp=1 or 3).
    expr : evaluable object
        Must have an .evaluate(lev) method returning a list of per-box
        JAX arrays with shape (nx, ny, nz, ncomp).
    """
    mesh = cell_field.mesh
    n_levels = mesh.n_levels()
    ncomp = cell_field.ncomp

    for lev in range(n_levels):
        corrections = expr.evaluate(lev=lev)  # per-box arrays
        mf = cell_field.mf[lev]
        ng = mf.n_grow()
        arrs = mf.arrays()

        results = []
        for bi in range(len(arrs)):
            n = [int(arrs[bi].shape[ax]) - 2 * ng for ax in range(3)]
            sl = tuple(slice(ng, ng + n[ax]) for ax in range(3))

            if ncomp == 1:
                valid = arrs[bi][sl[0], sl[1], sl[2], 0]
                results.append(valid + corrections[bi])
            else:
                valid = arrs[bi][sl[0], sl[1], sl[2], :]
                corrected = jnp.stack(
                    [valid[:, :, :, c] + corrections[bi][:, :, :, c]
                     for c in range(ncomp)],
                    axis=-1,
                )
                results.append(corrected)

        mf.copy_arrays(results)

    # Restrict fine -> coarse.
    for lev in reversed(range(n_levels - 1)):
        blockamr.average_down(
            cell_field.mf[lev + 1], cell_field.mf[lev],
            mesh.geom(lev + 1), mesh.geom(lev),
            0, ncomp, mesh.ref_ratio(lev),
        )
