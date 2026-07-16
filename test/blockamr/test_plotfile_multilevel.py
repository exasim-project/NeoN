# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import os
import shutil

import blockamr


def test_write_multilevel_plotfile(tmp_path):
    """Write a 2-level plotfile and verify the directory structure."""
    ncell = 16
    ratio = 2

    # Level 0
    cbox = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cgeom = blockamr.Geometry(cbox, rb, 0, [1, 1, 1])
    cba = blockamr.BoxArray(cbox)
    cba.max_size(16)
    cdm = blockamr.DistributionMapping(cba)
    cmf = blockamr.MultiFab(cba, cdm, 1, 0, memory="pinned")

    # Level 1
    fbox = blockamr.Box(
        [0, 0, 0], [ncell * ratio - 1, ncell * ratio - 1, ncell * ratio - 1]
    )
    fgeom = blockamr.Geometry(fbox, rb, 0, [1, 1, 1])
    fba = blockamr.BoxArray(fbox)
    fba.max_size(16)
    fdm = blockamr.DistributionMapping(fba)
    fmf = blockamr.MultiFab(fba, fdm, 1, 0, memory="pinned")

    plotdir = str(tmp_path / "plt_multilevel")

    blockamr.write_multilevel_plotfile(
        plotdir,
        2,
        [cmf, fmf],
        ["phi"],
        [cgeom, fgeom],
        0.0,
        [0, 0],
        [blockamr.IntVect(ratio, ratio, ratio)],
    )

    # Verify directory structure
    assert os.path.isdir(plotdir)
    assert os.path.isfile(os.path.join(plotdir, "Header"))
    assert os.path.isdir(os.path.join(plotdir, "Level_0"))
    assert os.path.isdir(os.path.join(plotdir, "Level_1"))
