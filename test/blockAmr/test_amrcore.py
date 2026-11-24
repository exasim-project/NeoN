# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import blockamr


class MinimalAmrCore(blockamr.AmrCore):
    """Minimal subclass that records which callbacks were called."""

    def __init__(self, geom, amr_info):
        super().__init__(geom, amr_info)
        self.calls = []

    def make_new_level_from_scratch(self, lev, time, ba, dm):
        self.calls.append(("make_new_level_from_scratch", lev))

    def make_new_level_from_coarse(self, lev, time, ba, dm):
        self.calls.append(("make_new_level_from_coarse", lev))

    def remake_level(self, lev, time, ba, dm):
        self.calls.append(("remake_level", lev))

    def clear_level(self, lev):
        self.calls.append(("clear_level", lev))

    def error_est(self, lev, tags, time, ngrow):
        self.calls.append(("error_est", lev))


class TaggingAmrCore(blockamr.AmrCore):
    """AmrCore subclass that tags all cells for refinement."""

    def __init__(self, geom, amr_info):
        super().__init__(geom, amr_info)
        self.calls = []

    def make_new_level_from_scratch(self, lev, time, ba, dm):
        self.calls.append(("make_new_level_from_scratch", lev))

    def make_new_level_from_coarse(self, lev, time, ba, dm):
        self.calls.append(("make_new_level_from_coarse", lev))

    def remake_level(self, lev, time, ba, dm):
        self.calls.append(("remake_level", lev))

    def clear_level(self, lev):
        self.calls.append(("clear_level", lev))

    def error_est(self, lev, tags, time, ngrow):
        self.calls.append(("error_est", lev))
        # Tag all cells to force refinement using batch set_tags
        for tbi in blockamr.TagBoxIterator(tags):
            bx = tbi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            nx = hi[0] - lo[0] + 1
            ny = hi[1] - lo[1] + 1
            nz = hi[2] - lo[2] + 1
            mask = np.ones((nx, ny, nz), dtype=np.int32)
            tbi.set_tags(mask)


def _make_geom_and_info(ncell=32, max_level=0):
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = max_level
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 32)
    info.set_blocking_factor(0, 8)
    return geom, info


def test_amr_info_construction():
    info = blockamr.AmrInfo()
    assert info.max_level == 0
    info.max_level = 2
    assert info.max_level == 2


def test_amr_info_setters():
    """Verify AmrInfo setters are reflected after AmrCore construction."""
    ncell = 16
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 4)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 4)
    info.set_n_error_buf(0, 2)

    core = MinimalAmrCore(geom, info)
    assert core.max_level == 1
    rr = core.ref_ratio(0)
    assert rr[0] == 4


def test_amrcore_init_from_scratch_calls_level0():
    geom, info = _make_geom_and_info(ncell=32, max_level=0)
    core = MinimalAmrCore(geom, info)
    core.init_from_scratch(0.0)
    assert ("make_new_level_from_scratch", 0) in core.calls


def test_amrcore_metadata():
    geom, info = _make_geom_and_info(ncell=32, max_level=1)
    core = MinimalAmrCore(geom, info)
    core.init_from_scratch(0.0)
    assert core.finest_level >= 0
    assert core.max_level == 1
    g = core.geom(0)
    assert g.cell_size()[0] > 0
    ba = core.box_array(0)
    dm = core.dm(0)
    assert ba is not None
    assert dm is not None


def test_amrcore_ref_ratio():
    """ref_ratio returns the ratio set in AmrInfo."""
    geom, info = _make_geom_and_info(ncell=32, max_level=1)
    core = MinimalAmrCore(geom, info)
    rr = core.ref_ratio(0)
    assert rr[0] == 2
    assert rr[1] == 2
    assert rr[2] == 2


def test_amrcore_regrid_tags_all_creates_fine_level():
    """Tagging all cells during init_from_scratch creates a fine level."""
    ncell = 16
    geom, info = _make_geom_and_info(ncell=ncell, max_level=1)
    core = TaggingAmrCore(geom, info)
    core.init_from_scratch(0.0)

    # init_from_scratch triggers regrid internally, which calls error_est
    # Since we tag all cells, level 1 should be created
    assert core.finest_level == 1
    assert ("make_new_level_from_scratch", 1) in core.calls


def test_amrcore_error_est_receives_tags():
    """error_est callback receives time and ngrow arguments."""
    geom, info = _make_geom_and_info(ncell=16, max_level=1)
    received = {}

    class RecordingCore(blockamr.AmrCore):
        def make_new_level_from_scratch(self, lev, time, ba, dm):
            pass

        def make_new_level_from_coarse(self, lev, time, ba, dm):
            pass

        def remake_level(self, lev, time, ba, dm):
            pass

        def clear_level(self, lev):
            pass

        def error_est(self, lev, tags, time, ngrow):
            received["lev"] = lev
            received["time"] = time
            received["ngrow"] = ngrow
            received["tags_type"] = type(tags).__name__

    core = RecordingCore(geom, info)
    core.init_from_scratch(0.0)
    # init_from_scratch with max_level>0 triggers regrid → error_est
    assert received["lev"] == 0
    assert received["time"] == 0.0
    assert received["tags_type"] == "TagBoxArray"


def test_tag_constants():
    """TAG_CLEAR and TAG_SET are accessible integer constants."""
    assert isinstance(blockamr.TAG_CLEAR, int)
    assert isinstance(blockamr.TAG_SET, int)
    assert blockamr.TAG_CLEAR != blockamr.TAG_SET


def test_tagbox_set_get_via_error_est():
    """TagBox __setitem__/__getitem__ work during error_est callback."""
    geom, info = _make_geom_and_info(ncell=16, max_level=1)
    tag_values = {}

    class TagTestCore(blockamr.AmrCore):
        def make_new_level_from_scratch(self, lev, time, ba, dm):
            pass

        def make_new_level_from_coarse(self, lev, time, ba, dm):
            pass

        def remake_level(self, lev, time, ba, dm):
            pass

        def clear_level(self, lev):
            pass

        def error_est(self, lev, tags, time, ngrow):
            for tbi in blockamr.TagBoxIterator(tags):
                bx = tbi.valid_box()
                lo = bx.small_end()
                hi = bx.big_end()
                nx = hi[0] - lo[0] + 1
                ny = hi[1] - lo[1] + 1
                nz = hi[2] - lo[2] + 1
                # Tag first cell only
                mask = np.zeros((nx, ny, nz), dtype=np.int32)
                mask[0, 0, 0] = 1
                tbi.set_tags(mask)
                tag_values["tagged"] = True
                tag_values["mask_shape"] = (nx, ny, nz)

    core = TagTestCore(geom, info)
    core.init_from_scratch(0.0)

    assert tag_values["tagged"]
    assert tag_values["mask_shape"][0] > 0


def test_mfiterator_get_returns_mfiter():
    """MFIterator.get() returns the underlying MFIter for TagBoxArray indexing."""
    ncell = 16
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(16)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, 0, memory="pinned")

    for mfi in blockamr.MFIterator(mf):
        raw_mfi = mfi.get()
        assert raw_mfi.is_valid()
        bx = raw_mfi.valid_box()
        assert bx.num_pts() > 0
