# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""AMR regrid with the explicit C++ DSL solve: scratch-cache tracking and correctness.

Uses AmrMesh with regrid and the DSL solve (AMReX ``ParallelFor`` kernels). Checks
that the per-level scratch source MultiFab follows the box array across regrids
and that a uniform field stays uniform through level removal and restoration.
"""

import numpy as np

import blockamr
from blockamr.mesh import AmrMesh
from blockamr.field import CellField, FaceField
from blockamr.fillpatch import FillPatchCellConservative
from blockamr.operators.div import Div, update_face_fluxes
from blockamr.dsl import exp, solve
from blockamr.schemes.div_schemes import Upwind


def _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16):
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = max_level
    for lev in range(max_level):
        info.set_ref_ratio(lev, 2)
    info.set_max_grid_size(0, max_size)
    info.set_blocking_factor(0, 4)
    return AmrMesh(geom, info)


def _make_tag_func(mesh, width, center_x=0.5):
    """Tag cells within `width` of (center_x, 0.5)."""

    def _tag(lev, tags, time, ngrow):
        dx = mesh.geom(lev).cell_size()
        prob_lo = mesh.geom(lev).prob_lo()
        for tbi in blockamr.TagBoxIterator(tags):
            bx = tbi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            nx = hi[0] - lo[0] + 1
            ny = hi[1] - lo[1] + 1
            nz = hi[2] - lo[2] + 1
            xs = (np.arange(nx) + lo[0] + 0.5) * dx[0] + prob_lo[0]
            ys = (np.arange(ny) + lo[1] + 0.5) * dx[1] + prob_lo[1]
            mask_2d = (np.abs(xs - center_x)[:, None] < width) & (np.abs(ys - 0.5)[None, :] < width)
            mask = np.broadcast_to(mask_2d[:, :, None], (nx, ny, nz)).astype(np.int32).copy()
            tbi.set_tags(mask)

    return _tag


def _tag_nothing(lev, tags, time, ngrow):
    """Tag nothing — forces removal of fine level."""
    pass


def _tag_all(lev, tags, time, ngrow):
    """Tag all cells — forces full refinement."""
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def _vel_func(x, y, z, t):
    return (np.ones_like(x), np.zeros_like(x), np.zeros_like(x))


def _box_counts(phi, mesh):
    return [len(phi.mf[lev].arrays()) for lev in range(mesh.n_levels())]


def _valid_boxes(mf):
    return [(tuple(m.valid_box().small_end()), tuple(m.valid_box().big_end()))
            for m in blockamr.MFIterator(mf)]


def _assert_uniform(phi, mesh, value=1.0):
    """Every valid cell on every level equals ``value``."""
    for lev in range(mesh.n_levels()):
        for mfi in blockamr.MFIterator(phi.mf[lev]):
            arr = phi.mf[lev].copy_to_host(mfi)
            err = float(np.max(np.abs(arr - value)))
            assert err < 1e-12, f"lev {lev}: uniform field drifted by {err:.2e}"


def _do_cpp_step(phi, ff, mesh, t=0.0, dt=0.25):
    """A full DSL step on the explicit C++ path."""
    for lev in range(mesh.n_levels()):
        phi.mf[lev].set_val(1.0)
        phi.fill_patch(lev, t)
        if ff[lev] is not None:
            update_face_fluxes(ff[lev], _vel_func, mesh.geom(lev), t)
    expr = exp.ddt(phi) + Div(ff, phi, scheme=Upwind())
    solve(expr, t=t, dt=dt)


def test_amr_cpp_scratch_invalidates_on_regrid(blockamr_session):
    """The explicit path's scratch MultiFab cache tracks the box array across regrid.

    After a solve step, ``phi._cpp_scratch[lev]`` holds one scratch MultiFab
    per level, keyed on the per-box extents. A regrid that
    changes a level's box array must invalidate that level's scratch (new
    signature → new MultiFab), with the box count tracking ``_box_counts``.
    """
    mesh = _make_amr_mesh(N=64, Nz=4, max_level=1, max_size=8)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    tag_small = _make_tag_func(mesh, width=0.10)
    tag_big = _make_tag_func(mesh, width=0.40)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_small)

    # First step — scratch built per level; box count matches the grid.
    _do_cpp_step(phi, ff, mesh)
    boxes_small = _box_counts(phi, mesh)
    print(f"\ntag_small: {mesh.n_levels()} levels, boxes={boxes_small}")
    assert set(phi._cpp_scratch) == set(range(mesh.n_levels()))
    for lev in range(mesh.n_levels()):
        assert len(phi._cpp_scratch[lev][0]) == boxes_small[lev], (
            f"scratch box count on lev {lev} != grid box count"
        )
    fine_sig = phi._cpp_scratch[1][0]
    fine_mf = phi._cpp_scratch[1][1]

    # Regrid to a much larger fine region → the level-1 box array changes.
    mesh.regrid(0.0, tag=tag_big)
    boxes_big = _box_counts(phi, mesh)
    _do_cpp_step(phi, ff, mesh)
    print(f"tag_big:   {mesh.n_levels()} levels, boxes={boxes_big}")

    for lev in range(mesh.n_levels()):
        assert len(phi._cpp_scratch[lev][0]) == boxes_big[lev]
    assert boxes_big[1] != boxes_small[1], "fixture sanity: fine box count should change"
    assert phi._cpp_scratch[1][0] != fine_sig, "scratch signature not invalidated on regrid"
    assert phi._cpp_scratch[1][1] is not fine_mf, "scratch MultiFab not rebuilt on regrid"


def test_amr_cpp_scratch_rebuilt_when_equal_sized_boxes_move(blockamr_session):
    """A regrid that moves the fine boxes but keeps their sizes rebuilds the scratch.

    The kernels index the scratch with the field's box indices, so a scratch left
    on the old (equal-sized) boxes would be read and written out of bounds.
    """
    mesh = _make_amr_mesh(N=64, Nz=4, max_level=1, max_size=8)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=_make_tag_func(mesh, width=0.10, center_x=0.3))
    _do_cpp_step(phi, ff, mesh, dt=0.001)
    boxes_left = _valid_boxes(phi.mf[1])
    assert _valid_boxes(phi._cpp_scratch[1][1]) == boxes_left

    mesh.regrid(0.0, tag=_make_tag_func(mesh, width=0.10, center_x=0.7))
    boxes_right = _valid_boxes(phi.mf[1])
    sizes = lambda bxs: [tuple(h - l for l, h in zip(*b)) for b in bxs]  # noqa: E731
    assert sizes(boxes_right) == sizes(boxes_left), "fixture sanity: box sizes should match"
    assert boxes_right != boxes_left, "fixture sanity: fine boxes should move"

    _do_cpp_step(phi, ff, mesh, dt=0.001)
    assert _valid_boxes(phi._cpp_scratch[1][1]) == boxes_right, (
        "scratch MultiFab left on the old boxes after regrid"
    )
    _assert_uniform(phi, mesh)


def test_amr_cpp_scratch_reused_on_stable_grid(blockamr_session):
    """Repeated solves on a stable grid, or a same-tag regrid, reuse the scratch."""
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    tag_center = _make_tag_func(mesh, width=0.2)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_center)
    boxes_before = _box_counts(phi, mesh)

    _do_cpp_step(phi, ff, mesh, dt=0.001)
    scratch = {lev: phi._cpp_scratch[lev][1] for lev in range(mesh.n_levels())}
    for _ in range(3):
        _do_cpp_step(phi, ff, mesh, dt=0.001)
    for lev in range(mesh.n_levels()):
        assert phi._cpp_scratch[lev][1] is scratch[lev], f"scratch rebuilt on stable lev {lev}"

    # Regrid with the same tags → same layout → scratch kept.
    mesh.regrid(0.0, tag=tag_center)
    assert _box_counts(phi, mesh) == boxes_before, "Sanity: same tag should give same layout"
    _do_cpp_step(phi, ff, mesh, dt=0.001)
    for lev in range(mesh.n_levels()):
        assert phi._cpp_scratch[lev][1] is scratch[lev], (
            f"scratch rebuilt on lev {lev} after a same-layout regrid"
        )
    _assert_uniform(phi, mesh)


def test_amr_level_remove_and_restore(blockamr_session):
    """Removing and restoring the fine level keeps the solve correct on every level."""
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    tag_center = _make_tag_func(mesh, width=0.2)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_center)
    assert mesh.n_levels() == 2
    boxes_2level = _box_counts(phi, mesh)
    _do_cpp_step(phi, ff, mesh, dt=0.001)
    _assert_uniform(phi, mesh)

    # Remove the fine level.
    mesh.regrid(0.0, tag=_tag_nothing)
    assert mesh.n_levels() == 1
    _do_cpp_step(phi, ff, mesh, dt=0.001)
    _assert_uniform(phi, mesh)

    # Restore it.
    mesh.regrid(0.0, tag=tag_center)
    assert mesh.n_levels() == 2
    assert _box_counts(phi, mesh) == boxes_2level
    _do_cpp_step(phi, ff, mesh, dt=0.001)
    assert len(phi._cpp_scratch[1][0]) == boxes_2level[1]
    _assert_uniform(phi, mesh)
