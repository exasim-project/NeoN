# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Verify that changing the number of boxes triggers JAX recompilation.

When AMR regridding changes the box count on a level, the static
`n_boxes_padded` field of BucketContext changes (if a power-of-2 boundary
is crossed), forcing JAX to retrace every JIT-compiled function.
"""

import jax

import blockamr
from blockamr.mesh import Mesh
from blockamr.field import CellField, FaceField
from blockamr.dsl import exp, solve
from blockamr.operators.div import Div
from blockamr.schemes.div_schemes import Upwind

# jax pinned tier-wide: every count here is a JAX compilation, which only the jax
# backend produces — under the cpp default (B14) the counters would read 0 and the
# zero-recompile assertions would pass vacuously (Q14).
_JAX = {"backend": "jax"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mesh(N, Nz, max_size):
    """Create a periodic single-level mesh with given max_size."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def _setup_fields(mesh):
    """Create a CellField and FaceField with constant values."""
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 1.0
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            arr = ff[0][d].mf.copy_to_host(mfi)
            arr[:] = 1.0
            ff[0][d].mf.copy_from(mfi, arr)

    return phi, ff


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_same_box_count_no_recompile(blockamr_session):
    """Repeated solve with the same box count does not recompile."""
    N, Nz = 16, 4
    mesh = _make_mesh(N, Nz, max_size=16)  # 1 box
    phi, ff = _setup_fields(mesh)

    expr = exp.ddt(phi) + Div(ff, phi, scheme=Upwind())

    # Warmup: first call compiles
    solve(expr, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi.mf[0].contiguous_array())

    # Second call with same mesh → no recompilation
    counter = CompileCounter()
    counter.start()

    # Reset phi to constant so we can re-run
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 1.0
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    solve(expr, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi.mf[0].contiguous_array())
    counter.stop()

    assert counter.count == 0, f"Expected 0 recompilations for same box count, got {counter.count}"


def test_different_box_count_recompiles(blockamr_session):
    """Changing the number of boxes triggers JAX recompilation.

    max_size=32 → 1 box  (n_boxes_padded=1)
    max_size=16 → 4 boxes (n_boxes_padded=4)

    The different n_boxes_padded is a static field in BucketContext,
    so JAX must retrace.
    """
    N, Nz = 32, 4

    # --- Setup A: 1 box ---
    mesh_a = _make_mesh(N, Nz, max_size=32)
    phi_a, ff_a = _setup_fields(mesh_a)
    expr_a = exp.ddt(phi_a) + Div(ff_a, phi_a, scheme=Upwind())

    solve(expr_a, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi_a.mf[0].contiguous_array())

    # --- Setup B: 4 boxes (different n_boxes_padded) ---
    mesh_b = _make_mesh(N, Nz, max_size=16)
    phi_b, ff_b = _setup_fields(mesh_b)
    expr_b = exp.ddt(phi_b) + Div(ff_b, phi_b, scheme=Upwind())

    counter = CompileCounter()
    counter.start()
    solve(expr_b, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi_b.mf[0].contiguous_array())
    counter.stop()

    assert counter.count > 0, (
        "Expected recompilation when box count changes (1 → 4 boxes), but got 0 recompilations"
    )


def test_crossing_power_of_2_boundary_recompiles(blockamr_session):
    """Box counts 3 and 5 both pad differently (4 vs 8), causing recompile.

    Uses a 24x8x4 grid:
      max_size=8  → 3 boxes (24/8 * 8/8 * 4/4 = 3) → padded to 4
      Then a 40x8x4 grid:
      max_size=8  → 5 boxes (40/8 * 8/8 * 4/4 = 5) → padded to 8
    """
    Nz = 4

    # --- 3 boxes → padded to 4 ---
    box_a = blockamr.Box([0, 0, 0], [23, 7, Nz - 1])
    rb_a = blockamr.RealBox([0.0, 0.0, 0.0], [3.0, 1.0, Nz / 8])
    geom_a = blockamr.Geometry(box_a, rb_a, 0, [1, 1, 1])
    ba_a = blockamr.BoxArray(box_a)
    ba_a.max_size(8)
    dm_a = blockamr.DistributionMapping(ba_a)
    mesh_a = Mesh(ba_a, dm_a, geom_a)

    phi_a, ff_a = _setup_fields(mesh_a)
    expr_a = exp.ddt(phi_a) + Div(ff_a, phi_a, scheme=Upwind())
    solve(expr_a, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi_a.mf[0].contiguous_array())

    # --- 5 boxes → padded to 8 (crosses power-of-2 boundary) ---
    box_b = blockamr.Box([0, 0, 0], [39, 7, Nz - 1])
    rb_b = blockamr.RealBox([0.0, 0.0, 0.0], [5.0, 1.0, Nz / 8])
    geom_b = blockamr.Geometry(box_b, rb_b, 0, [1, 1, 1])
    ba_b = blockamr.BoxArray(box_b)
    ba_b.max_size(8)
    dm_b = blockamr.DistributionMapping(ba_b)
    mesh_b = Mesh(ba_b, dm_b, geom_b)

    phi_b, ff_b = _setup_fields(mesh_b)
    expr_b = exp.ddt(phi_b) + Div(ff_b, phi_b, scheme=Upwind())

    counter = CompileCounter()
    counter.start()
    solve(expr_b, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi_b.mf[0].contiguous_array())
    counter.stop()

    assert counter.count > 0, (
        "Expected recompilation when crossing power-of-2 boundary "
        "(3 boxes padded=4 → 5 boxes padded=8), got 0 recompilations"
    )


def test_same_padded_count_no_recompile(blockamr_session):
    """Box counts 3 and 4 both pad to 4, so no recompilation.

    Uses grids that produce 3 vs 4 same-shape boxes, both padding
    to n_boxes_padded=4.
    """
    Nz = 4

    # --- 4 boxes → padded to 4 ---
    box_a = blockamr.Box([0, 0, 0], [31, 7, Nz - 1])
    rb_a = blockamr.RealBox([0.0, 0.0, 0.0], [4.0, 1.0, Nz / 8])
    geom_a = blockamr.Geometry(box_a, rb_a, 0, [1, 1, 1])
    ba_a = blockamr.BoxArray(box_a)
    ba_a.max_size(8)
    dm_a = blockamr.DistributionMapping(ba_a)
    mesh_a = Mesh(ba_a, dm_a, geom_a)

    phi_a, ff_a = _setup_fields(mesh_a)
    expr_a = exp.ddt(phi_a) + Div(ff_a, phi_a, scheme=Upwind())
    solve(expr_a, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi_a.mf[0].contiguous_array())

    # --- 3 boxes → also padded to 4 ---
    box_b = blockamr.Box([0, 0, 0], [23, 7, Nz - 1])
    rb_b = blockamr.RealBox([0.0, 0.0, 0.0], [3.0, 1.0, Nz / 8])
    geom_b = blockamr.Geometry(box_b, rb_b, 0, [1, 1, 1])
    ba_b = blockamr.BoxArray(box_b)
    ba_b.max_size(8)
    dm_b = blockamr.DistributionMapping(ba_b)
    mesh_b = Mesh(ba_b, dm_b, geom_b)

    phi_b, ff_b = _setup_fields(mesh_b)
    expr_b = exp.ddt(phi_b) + Div(ff_b, phi_b, scheme=Upwind())

    counter = CompileCounter()
    counter.start()
    solve(expr_b, t=0.0, dt=0.001, solution=_JAX)
    jax.block_until_ready(phi_b.mf[0].contiguous_array())
    counter.stop()

    assert counter.count == 0, (
        f"Expected 0 recompilations when padded count is the same "
        f"(4 boxes padded=4, 3 boxes padded=4), got {counter.count}"
    )
