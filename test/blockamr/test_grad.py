# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
from blockamr.field import Field
from blockamr.operators.grad import Grad


def _make_field(n_cell=64, max_size=32, ngrow=1):
    """Create a periodic Field on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom)


def _init_sin3d(field):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = field.dx
    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = (lo[2] + k + 0.5) * dx[2]
                    arr[i, j, k, 0] = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
    field.fill_boundary()


def _compute_grad_error(n_cell):
    """Compute max error of grad(sin3d) vs analytical."""
    field = _make_field(n_cell=n_cell, max_size=n_cell, ngrow=1)
    _init_sin3d(field)

    grad_op = Grad(field)
    pi = math.pi

    max_err = 0.0
    for patch in field.patches():
        result = grad_op.compute(patch, t=0.0)
        lo = patch.box.small_end()
        dx = patch.geom.cell_size()
        prob_lo = patch.geom.prob_lo()
        nx, ny, nz = patch.valid_arr.shape[:3]
        for i in range(nx):
            x = prob_lo[0] + (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = prob_lo[1] + (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = prob_lo[2] + (lo[2] + k + 0.5) * dx[2]
                    sx = math.sin(2 * pi * x)
                    sy = math.sin(2 * pi * y)
                    sz = math.sin(2 * pi * z)
                    cx = math.cos(2 * pi * x)
                    cy = math.cos(2 * pi * y)
                    cz = math.cos(2 * pi * z)
                    exact = [
                        2 * pi * cx * sy * sz,
                        2 * pi * sx * cy * sz,
                        2 * pi * sx * sy * cz,
                    ]
                    for d in range(3):
                        err = abs(float(result[i, j, k, d]) - exact[d])
                        if err > max_err:
                            max_err = err
    return max_err


def test_gradient_convergence():
    """Central-difference gradient converges at O(dx^2) on sin3d."""
    errors = []
    for n in [16, 32, 64]:
        err = _compute_grad_error(n)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 3.5, f"Ratio 16->32: {ratio_1:.2f}, expected ~4"
    assert ratio_2 > 3.5, f"Ratio 32->64: {ratio_2:.2f}, expected ~4"
