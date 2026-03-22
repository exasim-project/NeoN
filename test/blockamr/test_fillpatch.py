# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import blockamr


def test_bcrec_periodic():
    bc = blockamr.periodic_bcrec()
    assert bc is not None


def test_interpolater_singletons():
    interp = blockamr.cell_cons_interp()
    assert interp is not None
    pc = blockamr.pc_interp()
    assert pc is not None


def test_fill_patch_single_level():
    """FillPatchSingleLevel fills ghost cells on a single level."""
    ncell = 32
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    ba = blockamr.BoxArray(box)
    ba.max_size(16)
    dm = blockamr.DistributionMapping(ba)
    ngrow = 1
    mf = blockamr.MultiFab(ba, dm, 1, ngrow, memory="pinned")

    # Fill with constant value
    for mfi in blockamr.MFIterator(mf):
        arr = mf.host_grown_array(mfi)
        arr[:] = 42.0

    blockamr.fill_patch_single_level(mf, 0.0, [mf], [0.0], geom, 0, 1)

    # After fill, ghost cells at periodic boundaries should also be 42
    for mfi in blockamr.MFIterator(mf):
        arr = mf.host_grown_array(mfi)
        assert np.allclose(arr, 42.0)


def test_fill_patch_two_levels():
    """FillPatchTwoLevels fills fine MF from coarse + fine sources."""
    ncell = 16
    ratio = 2

    # Coarse level
    cbox = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cgeom = blockamr.Geometry(cbox, rb, 0, [1, 1, 1])
    cba = blockamr.BoxArray(cbox)
    cba.max_size(16)
    cdm = blockamr.DistributionMapping(cba)
    cmf = blockamr.MultiFab(cba, cdm, 1, 1, memory="pinned")

    # Fill coarse with constant 7
    for mfi in blockamr.MFIterator(cmf):
        arr = cmf.host_grown_array(mfi)
        arr[:] = 7.0
    cmf.fill_boundary(cgeom)

    # Fine level — same domain, fully refined
    fbox = blockamr.Box(
        [0, 0, 0], [ncell * ratio - 1, ncell * ratio - 1, ncell * ratio - 1]
    )
    fgeom = blockamr.Geometry(fbox, rb, 0, [1, 1, 1])
    fba = blockamr.BoxArray(fbox)
    fba.max_size(16)
    fdm = blockamr.DistributionMapping(fba)
    ngrow = 1
    # Source fine MF — fill valid with 7.0
    fmf_src = blockamr.MultiFab(fba, fdm, 1, ngrow, memory="pinned")
    for mfi in blockamr.MFIterator(fmf_src):
        arr = fmf_src.host_grown_array(mfi)
        arr[:] = 7.0

    # Target fine MF — starts at 0
    fmf = blockamr.MultiFab(fba, fdm, 1, ngrow, memory="pinned")
    for mfi in blockamr.MFIterator(fmf):
        arr = fmf.host_grown_array(mfi)
        arr[:] = 0.0

    bcs = [blockamr.periodic_bcrec()]
    rr = blockamr.IntVect(ratio, ratio, ratio)
    mapper = blockamr.cell_cons_interp()

    blockamr.fill_patch_two_levels(
        fmf, 0.0,
        [cmf], [0.0],
        [fmf_src], [0.0],
        0, 0, 1,
        cgeom, fgeom,
        rr, mapper, bcs,
    )

    # Both valid and ghost cells should be 7.0
    for mfi in blockamr.MFIterator(fmf):
        arr = fmf.host_grown_array(mfi)
        assert np.allclose(arr, 7.0), f"Expected 7.0, got {arr.mean()}"
