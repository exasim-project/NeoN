# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for contiguous array padding with hysteresis."""

import jax
import jax.numpy as jnp

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh
from neon.blockamr.field import CellField, _padded_capacity
from neon.blockamr.fillpatch import FillPatchCellConservative


# ---------------------------------------------------------------------------
# Pure-Python hysteresis tests (no AMReX needed)
# ---------------------------------------------------------------------------


def test_padded_capacity_first_call():
    """First call (current=0) pads by 20%."""
    assert _padded_capacity(1000, 0) == 1200


def test_padded_capacity_within_band():
    """Required in [60%, 100%] of capacity — keep current."""
    assert _padded_capacity(800, 1200) == 1200   # 800 >= 720 (60%)
    assert _padded_capacity(1200, 1200) == 1200  # exact fit
    assert _padded_capacity(720, 1200) == 1200   # boundary (60%)


def test_padded_capacity_grew_past():
    """Required > capacity — repad to ceil(required * 1.2)."""
    assert _padded_capacity(1300, 1200) == 1560


def test_padded_capacity_too_wasteful():
    """Required < 60% of capacity — shrink to ceil(required * 1.2)."""
    assert _padded_capacity(500, 1200) == 600  # 500 < 720


def test_padded_capacity_small_values():
    """Edge cases: very small values."""
    assert _padded_capacity(1, 0) == 2    # ceil(1.2) = 2
    assert _padded_capacity(0, 0) == 0    # nothing to pad
    assert _padded_capacity(10, 0) == 12  # ceil(12.0) = 12


# ---------------------------------------------------------------------------
# Integration tests (require AMReX)
# ---------------------------------------------------------------------------


def _make_mesh(N, Nz=4, max_size=None):
    ms = max_size or N
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(ms)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def test_padded_multifab_construction(blockamr_session):
    """MultiFab with padding: shape includes padding, data is correct."""
    mesh = _make_mesh(8, Nz=4, max_size=8)
    field = CellField(mesh, ncomp=1, ngrow=1, name="test",
                      fill_patch=FillPatchCellConservative())

    mf = field.mf[0]
    required = mf.n_valid_elems()
    cap = field._padded_cap[0]

    # Padded capacity should be ~20% larger
    assert cap > required
    assert cap == _padded_capacity(required, 0)

    # contiguous_array with padding has the padded shape
    arr = mf.contiguous_array(cap)
    assert arr.shape[0] == cap

    # contiguous_array without padding has the valid shape
    arr_valid = mf.contiguous_array()
    assert arr_valid.shape[0] == required

    # Data integrity: set a value and verify
    mf.set_val(7.0)
    arr = mf.contiguous_array(cap)
    assert float(jnp.max(arr[:required])) == 7.0
    # Padding region should be zeros
    if cap > required:
        assert float(jnp.max(jnp.abs(arr[required:]))) == 0.0


def test_padded_copy_from_flat(blockamr_session):
    """copy_from_flat works with both padded and exact-size arrays."""
    mesh = _make_mesh(8, Nz=4, max_size=8)
    field = CellField(mesh, ncomp=1, ngrow=1, name="test",
                      fill_patch=FillPatchCellConservative())

    mf = field.mf[0]
    required = mf.n_valid_elems()
    cap = field._padded_cap[0]

    # Write with exact-size array
    exact = jnp.ones(required, dtype=jnp.float64) * 3.0
    mf.copy_from_flat(exact)
    arr = mf.contiguous_array()
    assert jnp.allclose(arr, 3.0)

    # Write with padded-size array (only valid portion is copied)
    padded = jnp.ones(cap, dtype=jnp.float64) * 5.0
    mf.copy_from_flat(padded)
    arr = mf.contiguous_array()
    assert jnp.allclose(arr, 5.0)


def test_cellfield_contiguous_returns_padded(blockamr_session):
    """CellField.contiguous() returns the padded-size array."""
    mesh = _make_mesh(8, Nz=4, max_size=8)
    field = CellField(mesh, ncomp=1, ngrow=1, name="test",
                      fill_patch=FillPatchCellConservative())

    required = field.mf[0].n_valid_elems()
    arr = field.contiguous(0)

    assert arr.shape[0] == field._padded_cap[0]
    assert arr.shape[0] > required


def test_hysteresis_stable_across_similar_sizes(blockamr_session):
    """Two meshes with different sizes within the band share capacity."""
    # Mesh A: 8x8x4, max_size=8 → 1 box
    mesh_a = _make_mesh(8, Nz=4, max_size=8)
    field = CellField(mesh_a, ncomp=1, ngrow=1, name="test",
                      fill_patch=FillPatchCellConservative())
    cap_a = field._padded_cap[0]
    required_a = field.mf[0].n_valid_elems()

    # Simulate a regrid with a slightly smaller required size
    # that still falls within the 60% band of the capacity
    smaller_required = int(cap_a * 0.7)  # 70% of capacity (> 60% threshold)
    new_cap = _padded_capacity(smaller_required, cap_a)
    assert new_cap == cap_a, (
        f"Should keep same capacity within band: "
        f"required={smaller_required}, capacity={cap_a}, got={new_cap}"
    )


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


def test_no_recompile_same_padded_shape(blockamr_session):
    """A JIT'd function on padded arrays with same shape doesn't recompile."""

    @jax.jit
    def compute(arr):
        return jnp.sum(arr)

    mesh = _make_mesh(8, Nz=4, max_size=8)
    field = CellField(mesh, ncomp=1, ngrow=1, name="test",
                      fill_patch=FillPatchCellConservative())
    field.mf[0].set_val(1.0)

    # Warmup
    arr1 = field.contiguous(0)
    _ = compute(arr1)
    jax.block_until_ready(_)

    # Change values but keep same padded shape
    field.mf[0].set_val(2.0)
    arr2 = field.contiguous(0)
    assert arr2.shape == arr1.shape

    counter = CompileCounter()
    counter.start()
    _ = compute(arr2)
    jax.block_until_ready(_)
    counter.stop()

    assert counter.count == 0, (
        f"Expected 0 recompilations with same padded shape, got {counter.count}"
    )
