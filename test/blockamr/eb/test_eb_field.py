# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Stage A — Mesh / CellField runtime EB switching.

Verifies M4: a single ``Mesh`` class supports both EB and non-EB modes
via the optional ``eb_factory=`` argument, and ``CellField.fill_patch``
zeros covered cells when (and only when) ``mesh.has_eb`` is True. The
non-EB code path must remain a strict no-op so existing examples are
unperturbed.
"""

import numpy as np
import pytest

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh
from neon.blockamr.field import CellField
from neon.blockamr.fillpatch import FillPatchCellConservative


def _make_unit_cube_mesh(ncell=32, max_size=16, eb_factory=None):
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom, eb_factory=eb_factory), geom, ba, dm


# ---------------------------------------------------------------------------
# A1 — has_eb / vol_frac wiring
# ---------------------------------------------------------------------------


def test_mesh_no_eb_has_eb_false():
    """A Mesh constructed without ``eb_factory`` reports ``has_eb=False``
    and ``vol_frac()`` returns ``None`` (the no-op signal)."""
    mesh, _, _, _ = _make_unit_cube_mesh()
    assert mesh.has_eb is False
    assert mesh.eb_factory(0) is None
    assert mesh.vol_frac(0) is None


def test_mesh_with_eb_has_eb_true():
    """A Mesh constructed with ``eb_factory=ebf`` reports ``has_eb=True``
    and ``vol_frac()`` returns one jnp array per local fab with the
    valid-cell shape."""
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(16)
    dm = blockamr.DistributionMapping(ba)

    cyl = blockamr.EB2_CylinderIF(0.15, 2, [0.5, 0.5, 0.0], False)
    blockamr.eb2_build_cylinder(cyl, geom, 0, 100)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    mesh = Mesh(ba, dm, geom, eb_factory=ebf)
    assert mesh.has_eb is True
    assert mesh.eb_factory(0) is ebf

    vf = mesh.vol_frac(0)
    assert vf is not None
    assert len(vf) > 0
    for vfb in vf:
        a = np.asarray(vfb)
        # 16^3 max-size means valid shape ≤ 16 in each dimension
        assert a.ndim == 3
        assert a.shape == (16, 16, 16)
        assert float(a.min()) >= 0.0
        assert float(a.max()) <= 1.0


# ---------------------------------------------------------------------------
# A2 — CellField on a non-EB mesh: fill_patch must NOT touch values
# ---------------------------------------------------------------------------


def test_cellfield_no_eb_fill_patch_does_not_zero_anything():
    """On a non-EB mesh, ``CellField.fill_patch`` must not call
    ``eb_set_covered`` — every valid cell stays at 7.0 after the call."""
    mesh, _, _, _ = _make_unit_cube_mesh()
    fld = CellField(
        mesh, ncomp=1, ngrow=1, name="u",
        fill_patch=FillPatchCellConservative(),
    )
    fld.mf[0].set_val(7.0)
    fld.fill_patch(0, 0.0)

    arr = np.asarray(fld.mf[0].contiguous_array())
    # Every value should still be 7.0 — periodic ghosts also see 7.0.
    assert (arr == 7.0).all(), (
        f"non-EB CellField.fill_patch perturbed values: "
        f"min={arr.min()}, max={arr.max()}, n_seven={int((arr==7.0).sum())}/{arr.size}"
    )


# ---------------------------------------------------------------------------
# A2 — CellField on EB mesh: fill_patch zeros covered cells
# ---------------------------------------------------------------------------


def test_cellfield_eb_fill_patch_zeros_covered_cells():
    """On an EB mesh, ``CellField.fill_patch`` calls ``eb_set_covered`` so
    that all covered (vol_frac == 0) cells become 0 while fluid cells
    keep their assigned value."""
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(16)
    dm = blockamr.DistributionMapping(ba)
    cyl = blockamr.EB2_CylinderIF(0.15, 2, [0.5, 0.5, 0.0], False)
    blockamr.eb2_build_cylinder(cyl, geom, 0, 100)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    mesh = Mesh(ba, dm, geom, eb_factory=ebf)
    fld = CellField(
        mesh, ncomp=1, ngrow=1, name="u",
        fill_patch=FillPatchCellConservative(),
    )
    fld.mf[0].set_val(7.0)
    fld.fill_patch(0, 0.0)

    arr = np.asarray(fld.mf[0].contiguous_array())
    n_zero = int((arr == 0.0).sum())
    n_seven = int((arr == 7.0).sum())
    assert n_zero > 0, "expected some covered cells to be zeroed"
    assert n_seven > 0, "expected some fluid cells to keep value 7.0"
    assert n_zero + n_seven == arr.size, (
        f"intermediate values present after fill_patch: "
        f"zero={n_zero}, seven={n_seven}, total={arr.size}"
    )


def test_cellfield_eb_fill_patch_preserves_zero_buffer_size():
    """``CellField`` on an EB mesh still uses Option A — its MultiFab is
    one zero-copy contiguous buffer of the right size."""
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(16)
    dm = blockamr.DistributionMapping(ba)
    cyl = blockamr.EB2_CylinderIF(0.15, 2, [0.5, 0.5, 0.0], False)
    blockamr.eb2_build_cylinder(cyl, geom, 0, 100)
    ebf = blockamr.make_eb_factory(geom, ba, dm)
    mesh = Mesh(ba, dm, geom, eb_factory=ebf)

    fld = CellField(
        mesh, ncomp=1, ngrow=1, name="u",
        fill_patch=FillPatchCellConservative(),
    )
    expected = blockamr.MultiFab.required_buffer_size(ba, dm, 1, 1)
    buf = fld.mf[0].contiguous_array()
    assert buf.size == expected, (
        f"EB CellField is no longer single-chunk: contiguous size {buf.size} "
        f"!= required {expected}"
    )
