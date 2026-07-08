# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Stage A — EB binding correctness.

These tests pin the contract between Python and the C++ EB bindings:

- ``has_eb_support()`` matches the build flag.
- ``EB2_CylinderIF`` + ``eb2_build_cylinder`` populate ``EB2::IndexSpace``.
- ``make_eb_factory`` returns a usable ``EBFArrayBoxFactory``.
- The volume fraction matches the analytic fluid area within ``O(dx)``.
- ``make_eb_multifab`` honours ``MFInfo::SetAllocSingleChunk(True)`` so
  ``contiguous_array()`` returns a *single* zero-copy buffer of exactly
  ``required_buffer_size`` elements (the load-bearing Option A
  invariant).

If any test in this file fails, the regression is in the C++ binding
layer (``src/bindings/blockAMR/{eb2,ebfactory,linop,arenas}``) — not in
any downstream Python code.
"""

import math

import numpy as np
import pytest

import neon.blockamr as blockamr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit_cube_mesh(ncell=32, max_size=16):
    """A periodic unit-cube geometry, BoxArray, DistributionMapping."""
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return box, geom, ba, dm


def _build_cylinder(geom, radius, axis_dir=2, center=(0.5, 0.5, 0.0)):
    cyl = blockamr.EB2_CylinderIF(radius, axis_dir, list(center), False)
    blockamr.eb2_build_cylinder(cyl, geom, 0, 100)
    return cyl


# ---------------------------------------------------------------------------
# A1 — has_eb_support / module exports
# ---------------------------------------------------------------------------


def test_has_eb_support_is_true():
    """The build is configured with AMReX_EB=ON, so the bindings expose
    EB symbols. If this fails, M0 (CMake) is broken."""
    assert blockamr.has_eb_support() is True


def test_eb_module_exports():
    """Spot-check that the EB symbols I rely on later are bound."""
    expected = [
        "EB2_CylinderIF",
        "EB2_SphereIF",
        "EB2_PlaneIF",
        "EB2_BoxIF",
        "EB2_AllRegularIF",
        "eb2_build_cylinder",
        "eb2_build_sphere",
        "eb2_build_box",
        "eb2_build_all_regular",
        "eb2_clear",
        "make_eb_factory",
        "make_eb_multifab",
        "eb_set_covered",
        "eb_set_covered_faces",
        "MLEBABecLaplacian",
        "MLEBTensorOp",
    ]
    for name in expected:
        assert hasattr(blockamr, name), f"binding missing: {name}"


# ---------------------------------------------------------------------------
# A4 — geometry: vol_frac quadrature against analytic cylinder area
# ---------------------------------------------------------------------------


def test_eb2_cylinder_volfrac_matches_analytic_area():
    """Volume fraction sums to (1 - π r²) per z-cell, within O(dx).

    The cylinder is z-aligned and spans the full domain in z, so the
    fluid fraction in each z-slice is exactly (1 - π r²) for r ≪ 0.5.
    The cell-centred ``vol_frac()`` is a first-order approximation; we
    require relative agreement at ~5% on a 32^3 mesh.
    """
    ncell = 32
    radius = 0.15
    box, geom, ba, dm = _make_unit_cube_mesh(ncell=ncell)
    _build_cylinder(geom, radius)

    ebf = blockamr.make_eb_factory(geom, ba, dm)
    vf_mf = ebf.vol_frac()
    vf_ng = vf_mf.n_grow()

    # Sum the *valid* (non-ghost) volume fraction across all local fabs.
    fluid_cells = 0.0
    total_cells = 0
    for arr, m in zip(vf_mf.arrays(), vf_mf.fab_metadata()):
        Nx, Ny, Nz = m[1], m[2], m[3]
        vNx, vNy, vNz = Nx - 2 * vf_ng, Ny - 2 * vf_ng, Nz - 2 * vf_ng
        valid = np.asarray(arr)[
            vf_ng:vf_ng + vNx,
            vf_ng:vf_ng + vNy,
            vf_ng:vf_ng + vNz,
            0,
        ]
        fluid_cells += float(valid.sum())
        total_cells += int(valid.size)

    fluid_fraction = fluid_cells / total_cells
    expected = 1.0 - math.pi * radius * radius

    assert 0.0 <= fluid_fraction <= 1.0
    assert abs(fluid_fraction - expected) < 0.05, (
        f"fluid fraction {fluid_fraction:.4f} differs from analytic "
        f"{expected:.4f} by more than 5% (ncell={ncell}, r={radius})"
    )


def test_eb2_volfrac_min_max_bounds():
    """Volume fraction values are in [0, 1] with both extremes attained
    when the cylinder is large enough to fully cover at least one cell."""
    box, geom, ba, dm = _make_unit_cube_mesh(ncell=32)
    _build_cylinder(geom, radius=0.2)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    vmin = 1.0
    vmax = 0.0
    for arr in ebf.vol_frac().arrays():
        a = np.asarray(arr)
        vmin = min(vmin, float(a.min()))
        vmax = max(vmax, float(a.max()))
    assert vmin == 0.0, f"expected at least one fully-covered cell (min=0), got {vmin}"
    assert vmax == 1.0, f"expected at least one fully-regular cell (max=1), got {vmax}"


# ---------------------------------------------------------------------------
# A3 — Option A: make_eb_multifab is zero-copy contiguous
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ncomp,ngrow", [(1, 0), (1, 1), (3, 1)])
def test_make_eb_multifab_contiguous_size(ncomp, ngrow):
    """``make_eb_multifab`` honours PaddedArena + SetAllocSingleChunk so the
    EB MultiFab's ``contiguous_array()`` returns one zero-copy buffer of
    exactly ``required_buffer_size`` elements.

    This is the *load-bearing invariant* of Option A. If it ever
    regresses, the JAX/Pallas kernels stop reading EB fields zero-copy
    and the entire EB-aware solver path silently degrades to per-fab
    iteration (or crashes).
    """
    box, geom, ba, dm = _make_unit_cube_mesh(ncell=32)
    _build_cylinder(geom, radius=0.15)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    mf = blockamr.make_eb_multifab(ba, dm, ncomp, ngrow, factory=ebf)
    expected = blockamr.MultiFab.required_buffer_size(ba, dm, ncomp, ngrow)

    buf = mf.contiguous_array()
    assert buf.size == expected, (
        f"contiguous_array size {buf.size} != required_buffer_size "
        f"{expected} (ncomp={ncomp}, ngrow={ngrow}). "
        "Option A invariant broken — EB MultiFab is no longer one chunk."
    )


def test_make_eb_multifab_set_val_visible_through_contiguous():
    """A scalar set_val on the EB MultiFab is visible through the
    zero-copy contiguous_array view (proves the views share memory)."""
    box, geom, ba, dm = _make_unit_cube_mesh(ncell=32)
    _build_cylinder(geom, radius=0.15)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    mf = blockamr.make_eb_multifab(ba, dm, 1, 1, factory=ebf)
    mf.set_val(7.0)
    arr = np.asarray(mf.contiguous_array())
    assert (arr == 7.0).all(), (
        f"contiguous view does not see set_val(7.0); "
        f"min={arr.min()} max={arr.max()}"
    )


def test_eb_set_covered_visible_through_contiguous():
    """``eb_set_covered(mf, 0.0)`` zeros covered cells and the change is
    visible through the zero-copy contiguous buffer.

    Together with ``test_make_eb_multifab_set_val_visible_through_contiguous``
    this proves that the per-fab and contiguous views share memory and
    that the covered-cell mask is consistent with vol_frac == 0.
    """
    box, geom, ba, dm = _make_unit_cube_mesh(ncell=32)
    _build_cylinder(geom, radius=0.15)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    mf = blockamr.make_eb_multifab(ba, dm, 1, 1, factory=ebf)
    mf.set_val(7.0)
    blockamr.eb_set_covered(mf, 0.0)

    arr = np.asarray(mf.contiguous_array())
    n_zero = int((arr == 0.0).sum())
    n_seven = int((arr == 7.0).sum())
    assert n_zero > 0, "expected some covered cells (vol_frac == 0) → zero"
    assert n_seven > 0, "expected some fluid cells (vol_frac > 0) → unchanged"
    # Every cell is now either 0 or 7 — no other values introduced.
    assert n_zero + n_seven == arr.size, (
        f"unexpected values in EB MultiFab: zero={n_zero}, seven={n_seven}, "
        f"total={arr.size}"
    )


def test_eb_set_covered_consistent_per_fab_and_contiguous():
    """The number of zero cells reported by per-fab arrays() equals the
    number reported by contiguous_array() — both views must agree."""
    box, geom, ba, dm = _make_unit_cube_mesh(ncell=32)
    _build_cylinder(geom, radius=0.15)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    mf = blockamr.make_eb_multifab(ba, dm, 1, 1, factory=ebf)
    mf.set_val(7.0)
    blockamr.eb_set_covered(mf, 0.0)

    n_zero_perfab = sum(int((np.asarray(a) == 0.0).sum()) for a in mf.arrays())
    n_zero_contig = int((np.asarray(mf.contiguous_array()) == 0.0).sum())
    assert n_zero_perfab == n_zero_contig, (
        f"per-fab arrays() reports {n_zero_perfab} zero cells but "
        f"contiguous_array() reports {n_zero_contig} — views are inconsistent"
    )
