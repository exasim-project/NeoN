# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Verify recompilation behavior with AMR regrid using CellField + FaceField.

Uses AmrMesh with regrid and the DSL solve (parallel_for under the hood).
Prints instrumentation showing exactly when and why recompiles happen.
"""

import jax
import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.mesh import AmrMesh
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.fillpatch import FillPatchCellConservative
from neon.blockamr.operators.div import Div, update_face_fluxes
from neon.blockamr.dsl import exp, solve
from neon.blockamr.schemes.div_schemes import Upwind


class CompileCounter:
    """Count JAX backend compilations via the monitoring API."""

    def __init__(self):
        self.count = 0
        self._listener = None

    def _on_event(self, event, duration, **kwargs):
        if event == "/jax/core/compile/backend_compile_duration":
            self.count += 1

    def start(self):
        self.count = 0
        self._listener = self._on_event
        jax.monitoring.register_event_duration_secs_listener(self._listener)

    def stop(self):
        if self._listener is not None:
            jax.monitoring.unregister_event_duration_listener(self._listener)
            self._listener = None

    def reset(self):
        self.count = 0


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


def _make_tag_func(mesh, width):
    """Tag cells within `width` of the domain center."""

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
            mask_2d = (np.abs(xs - 0.5)[:, None] < width) & (np.abs(ys - 0.5)[None, :] < width)
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
    return (jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x))


def _box_counts(phi, mesh):
    return [len(phi.mf[lev].arrays()) for lev in range(mesh.n_levels())]


def _do_step(phi, ff, mesh, t=0.0, dt=0.001):
    """Fill boundary + face fluxes + solve — the full DSL step."""
    for lev in range(mesh.n_levels()):
        phi.mf[lev].set_val(1.0)
        phi.fill_patch(lev, t)
        if ff[lev] is not None:
            update_face_fluxes(ff[lev], _vel_func, mesh.geom(lev), t)
    expr = exp.ddt(phi) + Div(ff, phi, scheme=Upwind())
    solve(expr, t=t, dt=dt)
    jax.block_until_ready(None)


def _do_cpp_step(phi, ff, mesh, t=0.0, dt=0.25):
    """A full DSL step on the composable cpp explicit backend."""
    for lev in range(mesh.n_levels()):
        phi.mf[lev].set_val(1.0)
        phi.fill_patch(lev, t)
        if ff[lev] is not None:
            update_face_fluxes(ff[lev], _vel_func, mesh.geom(lev), t)
    expr = exp.ddt(phi) + Div(ff, phi, scheme=Upwind())
    solve(expr, t=t, dt=dt, solution={"backend": "cpp"})


def test_amr_cpp_scratch_invalidates_on_regrid(blockamr_session):
    """The cpp backend's scratch MultiFab cache tracks the box array across regrid.

    After a cpp solve step, ``phi._cpp_scratch[lev]`` holds one scratch MultiFab
    per level, keyed on the ``fab_metadata`` box-size signature. A regrid that
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

    # First cpp step — scratch built per level; box count matches the grid.
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


def test_amr_recompilation_stable_grid(blockamr_session):
    """Repeated solve on a stable AMR grid produces 0 recompiles."""
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    tag_center = _make_tag_func(mesh, width=0.2)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_center)

    boxes = _box_counts(phi, mesh)
    print(f"\n{mesh.n_levels()} levels, boxes={boxes}")

    # Warmup
    _do_step(phi, ff, mesh)

    # Timed: 3 solves on the same grid
    counter = CompileCounter()
    counter.start()
    for _ in range(3):
        _do_step(phi, ff, mesh)
    counter.stop()

    print(f"3 solves on stable grid: {counter.count} recompiles")
    assert counter.count == 0, f"Stable grid should produce 0 recompiles, got {counter.count}"


def test_amr_recompilation_same_tag_regrid(blockamr_session):
    """Regrid with identical tagging produces 0 recompiles."""
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    tag_center = _make_tag_func(mesh, width=0.2)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_center)
    boxes_before = _box_counts(phi, mesh)

    # Warmup
    _do_step(phi, ff, mesh)

    # Regrid with same tags → same box layout
    mesh.regrid(0.0, tag=tag_center)
    boxes_after = _box_counts(phi, mesh)
    print(f"\nBoxes: {boxes_before} → {boxes_after}")

    counter = CompileCounter()
    counter.start()
    _do_step(phi, ff, mesh)
    counter.stop()

    print(f"Solve after same-tag regrid: {counter.count} recompiles")
    assert boxes_after == boxes_before, "Sanity: same tag should give same layout"
    assert counter.count == 0, (
        f"Same layout after regrid should produce 0 recompiles, got {counter.count}"
    )


def test_amr_recompilation_level_change(blockamr_session):
    """Removing and restoring a fine level recompiles predictably.

    The key insight: JAX caches by static shape. Once a configuration
    has been seen (warmup), revisiting it produces 0 recompiles.
    """
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    tag_center = _make_tag_func(mesh, width=0.2)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_center)

    n_levels_2 = mesh.n_levels()
    boxes_2level = _box_counts(phi, mesh)
    print(f"\nInitial: {n_levels_2} levels, boxes={boxes_2level}")

    # Warmup with 2 levels
    _do_step(phi, ff, mesh)

    # Remove fine level (warmup for 1-level config)
    mesh.regrid(0.0, tag=_tag_nothing)
    n_levels_1 = mesh.n_levels()
    boxes_1level = _box_counts(phi, mesh)
    print(f"After remove: {n_levels_1} levels, boxes={boxes_1level}")
    _do_step(phi, ff, mesh)

    # Restore fine level (this config was already compiled in warmup)
    mesh.regrid(0.0, tag=tag_center)
    boxes_restored = _box_counts(phi, mesh)
    print(f"After restore: {mesh.n_levels()} levels, boxes={boxes_restored}")

    counter = CompileCounter()
    counter.start()
    _do_step(phi, ff, mesh)
    counter.stop()

    print(f"Solve after restoring 2 levels: {counter.count} recompiles")

    # Both configs were warmed up → 0 recompiles when revisiting
    if boxes_restored == boxes_2level:
        assert counter.count == 0, (
            f"Restored same layout — expected 0 recompiles, got {counter.count}"
        )


def test_amr_tile_padding_through_regrids(blockamr_session):
    """Box and tile counts pad to power-of-2 tiers and expand correctly.

    Verifies that:
    1. packed_tiles returns correct n_tiles and n_padded for each grid config
    2. n_padded is always a power of 2 ≥ n_tiles
    3. After warmup of both configs, switching between them → 0 recompiles
    """

    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    bf = 4  # blocking factor for tiles

    # --- Config A: tag center (small fine region) ---
    tag_small = _make_tag_func(mesh, width=0.1)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_small)

    configs = []
    for lev in range(mesh.n_levels()):
        mf = phi.mf[lev]
        n_boxes = len(mf.arrays())
        d = mf.packed_tiles(bf)
        n_tiles = int(d["n_tiles"])
        n_padded = int(d["n_padded"])
        is_pow2 = (n_padded & (n_padded - 1)) == 0
        configs.append(
            {
                "lev": lev,
                "n_boxes": n_boxes,
                "n_tiles": n_tiles,
                "n_padded": n_padded,
            }
        )
        print(f"\nConfig A lev {lev}: {n_boxes} boxes, {n_tiles} tiles, padded to {n_padded}")
        assert is_pow2, f"n_padded={n_padded} is not a power of 2"
        assert n_padded >= n_tiles, f"n_padded={n_padded} < n_tiles={n_tiles}"

    # Warmup solve with config A
    _do_step(phi, ff, mesh)

    # --- Config B: tag all (full fine region → more boxes) ---
    mesh.regrid(0.0, tag=_tag_all)
    n_levels_b = mesh.n_levels()

    configs_b = []
    for lev in range(n_levels_b):
        mf = phi.mf[lev]
        n_boxes = len(mf.arrays())
        d = mf.packed_tiles(bf)
        n_tiles = int(d["n_tiles"])
        n_padded = int(d["n_padded"])
        is_pow2 = (n_padded & (n_padded - 1)) == 0
        configs_b.append(
            {
                "lev": lev,
                "n_boxes": n_boxes,
                "n_tiles": n_tiles,
                "n_padded": n_padded,
            }
        )
        print(f"Config B lev {lev}: {n_boxes} boxes, {n_tiles} tiles, padded to {n_padded}")
        assert is_pow2, f"n_padded={n_padded} is not a power of 2"
        assert n_padded >= n_tiles

    # Warmup solve with config B
    _do_step(phi, ff, mesh)

    # --- Switch back to Config A → 0 recompiles (already cached) ---
    mesh.regrid(0.0, tag=tag_small)
    _do_step(phi, ff, mesh)  # extra warmup to ensure face fluxes cached

    counter = CompileCounter()
    counter.start()
    _do_step(phi, ff, mesh)
    counter.reset()

    # Switch to B
    mesh.regrid(0.0, tag=_tag_all)
    _do_step(phi, ff, mesh)
    switch_b = counter.count
    counter.reset()

    # Switch back to A
    mesh.regrid(0.0, tag=tag_small)
    _do_step(phi, ff, mesh)
    switch_a = counter.count
    counter.stop()

    print("\nAfter warmup of both configs:")
    print(f"  Switch A→B: {switch_b} recompiles")
    print(f"  Switch B→A: {switch_a} recompiles")

    # Tile counts should differ between configs (more boxes → more tiles)
    if len(configs_b) > 1 and len(configs) > 1:
        print(f"\nTile growth: lev1 {configs[1]['n_tiles']} → {configs_b[1]['n_tiles']} tiles")

    assert switch_a == 0, (
        f"Switching back to cached config A should be 0 recompiles, got {switch_a}"
    )
    assert switch_b == 0, f"Switching to cached config B should be 0 recompiles, got {switch_b}"


def test_amr_new_tier_recompiles_once(blockamr_session):
    """A new tile-count tier recompiles once, then is cached.

    Uses 3 tag widths that produce 3 different power-of-2 padded sizes.
    Each new tier recompiles on first visit but not on second.
    Revisiting all 3 cached tiers produces 0 recompiles.
    """
    mesh = _make_amr_mesh(N=64, Nz=4, max_level=1, max_size=8)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi", fill_patch=FillPatchCellConservative())
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    widths = [0.10, 0.25, 0.40]
    tags = [_make_tag_func(mesh, w) for w in widths]

    mesh.init_from_scratch(0.0)

    counter = CompileCounter()
    counter.start()

    padded_sizes = []
    for i, (w, tag) in enumerate(zip(widths, tags)):
        mesh.regrid(0.0, tag=tag)

        # Record padded tile count on lev 1
        bf = 4
        if mesh.n_levels() > 1:
            d = phi.mf[1].packed_tiles(bf)
            padded_sizes.append(int(d["n_padded"]))
            print(f"\nWidth={w:.2f} lev1: {int(d['n_tiles'])} tiles, padded={int(d['n_padded'])}")

        # First visit — should recompile (new tier)
        counter.reset()
        _do_step(phi, ff, mesh)
        first = counter.count

        # Second visit — should NOT recompile (cached)
        counter.reset()
        _do_step(phi, ff, mesh)
        second = counter.count

        print(f"  First visit:  {first} recompiles")
        print(f"  Second visit: {second} recompiles")

        assert first > 0, f"Width={w}: new tier should recompile, got 0"
        assert second == 0, f"Width={w}: repeat on same tier should not recompile, got {second}"

    # Verify we actually hit different padded sizes
    unique_padded = set(padded_sizes)
    print(f"\nPadded tiers seen: {sorted(unique_padded)}")
    assert len(unique_padded) >= 2, (
        f"Expected at least 2 different padded tiers, got {unique_padded}"
    )

    # Revisit all 3 cached tiers — all should be 0
    print("\nRevisiting all cached tiers:")
    for w, tag in zip(widths, tags):
        mesh.regrid(0.0, tag=tag)
        counter.reset()
        _do_step(phi, ff, mesh)
        print(f"  Width={w:.2f}: {counter.count} recompiles")
        assert counter.count == 0, (
            f"Width={w}: revisiting cached tier should be 0 recompiles, got {counter.count}"
        )

    counter.stop()
