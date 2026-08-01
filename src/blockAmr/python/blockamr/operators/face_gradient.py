# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cell-centred ``grad(p)`` built from COMPACT face differences.

The shape ``dsl/solve.py`` stores on ``p_field.grad`` and ``exp.grad`` reads back
through ``PressureGradient``: per-box ``(nx, ny, nz, 3)`` cell-centred arrays. So a
cell-centred pressure solve can feed ``correct(U, -dt * exp.grad(p))`` with no DSL
change at all, which is the whole reason this helper is separate from the solve.

Why the FACE route rather than a wide stencil: the gradient the projection corrects
with must be the one the solved operator is the divergence of. ``linear_algebra``'s
``laplacian`` is assembled from face coefficients ``-gamma/dx**2``, i.e. it is
``-div_f(grad_f(.))`` over ADJACENT cells, so its partner gradient is the two-point
face difference and nothing else. Building the correction from a wide difference
instead would pair the compact operator that was inverted with the ``2h`` operator's
gradient, and the ``2h`` Laplacian is the one that decouples odd from even points and
carries a checkerboard pressure null space.

The face values come from ``mac_project._face_gradient_flux``, unchanged, rather than
from a second two-point kernel written here: it is already pinned BITWISE against
``MLMG.get_fluxes`` (``test_mac_project_la.py``), and a boundary closure that agreed
with the matrix in one module and not the other is exactly how a projection loses its
adjointness.
"""

import jax.numpy as jnp

from .mac_project import _face_gradient_flux


def cell_gradient(p_mf, geom, bc):
    """``grad(p)`` cell-centred, as per-box ``(nx, ny, nz, 3)`` arrays for one level.

    The cell value is the mean of the two faces bounding the cell in each direction.
    That average is a RECONSTRUCTION of the face gradient, not part of the operator
    that was inverted: the face gradient is the exact adjoint of the face divergence
    and corrects a face flux to be divergence-free exactly, while its cell average
    carries the ``O(dx**2)`` of the interpolation.

    The outermost face on a domain side is the boundary closure, and it is the
    MATRIX's, because ``_face_gradient_flux`` fills the ghost layer the way the
    matrix does: periodic wraparound, Neumann ghost = interior (so the wall face
    gradient is exactly zero, the discrete ``dp/dn = 0``), Dirichlet ghost =
    -interior (so the wall face gradient is ``-2 p/dx``, the one-sided difference to
    a zero value ON the face — AMReX's ``max_order=2``, not its third-order default).

    Parameters
    ----------
    p_mf : blockamr.MultiFab
        Cell-centred pressure, at least one ghost cell. Its GHOST layer is
        overwritten, since that is what carries the boundary condition.
    geom : blockamr.Geometry
    bc : list of str
        Six sides (xlo, xhi, ylo, yhi, zlo, zhi), as ``la::parseBc`` spells them.
    """
    # _face_gradient_flux returns the FLUX, -grad_f(p), so undo the sign it carries.
    flux_by_axis = _face_gradient_flux(p_mf, geom, bc)

    results = []
    for bi in range(len(flux_by_axis[0])):
        components = []
        for d in range(3):
            face_grad = -flux_by_axis[d][bi]
            sl_hi = [slice(None)] * 3
            sl_lo = [slice(None)] * 3
            sl_hi[d] = slice(1, None)
            sl_lo[d] = slice(0, -1)
            components.append(0.5 * (face_grad[tuple(sl_hi)] + face_grad[tuple(sl_lo)]))
        results.append(jnp.stack(components, axis=-1))
    return results
