# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math
import os
import shutil

import blockamr
import jax.numpy as jnp
import numpy as np


def test_create_box():
    lo = [0, 0, 0]
    hi = [63, 63, 63]
    box = blockamr.Box(lo, hi)
    assert box.num_pts() == 64**3


def test_create_multifab():
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    assert mf.num_comp() == 1


def test_sin_wave_to_plotfile():
    n_cell = 64
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [2 * math.pi, 2 * math.pi, 2 * math.pi])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])

    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, 0)

    dx = geom.cell_size()

    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        arr = mf.copy_to_host(mfi)

        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        x = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        y = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        z = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])

        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        vals = jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)

        arr[:, :, :, 0] = np.asarray(vals)
        mf.copy_from(mfi, arr)

    plotdir = "plt_sin"
    if os.path.exists(plotdir):
        shutil.rmtree(plotdir)

    blockamr.write_single_level_plotfile(plotdir, mf, ["sinxyz"], geom, 0.0, 0)

    assert os.path.isdir(plotdir)
    assert os.path.isfile(os.path.join(plotdir, "Header"))

    shutil.rmtree(plotdir)
