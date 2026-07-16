# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""GPU support tests for MultiFab device awareness."""

import jax
import numpy as np
import blockamr


def _make_multifab(ngrow=0):
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return mf


def _has_gpu():
    """Check if AMReX was built with GPU support and a device is available."""
    return jax.default_backend() != "cpu"


def test_multifab_is_host():
    """Default MultiFab reports is_host consistent with build."""
    mf = _make_multifab()
    if _has_gpu():
        assert mf.is_host is False
    else:
        assert mf.is_host is True


def test_multifab_is_device():
    """Default MultiFab reports is_device consistent with build."""
    mf = _make_multifab()
    if _has_gpu():
        assert mf.is_device is True
    else:
        assert mf.is_device is False


def test_multifab_is_managed():
    """Default MultiFab on CPU build should report is_managed=False."""
    mf = _make_multifab()
    assert mf.is_managed is False


def test_multifab_memory_default():
    """MultiFab with memory='default' should behave like no-arg constructor."""
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, 0, memory="default")
    if _has_gpu():
        assert mf.is_host is False
    else:
        assert mf.is_host is True
    assert mf.num_comp() == 1


def test_array_returns_jax():
    """array() should return a JAX array covering the full FAB (incl. ghosts)."""
    mf = _make_multifab(ngrow=1)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        assert isinstance(arr, jax.Array), f"Expected jax.Array, got {type(arr)}"
        assert arr.shape == (34, 34, 34, 1)
        break


def test_grown_array_returns_jax():
    """grown_array() should return a JAX array covering the full FAB."""
    mf = _make_multifab(ngrow=1)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.grown_array(mfi)
        assert isinstance(arr, jax.Array), f"Expected jax.Array, got {type(arr)}"
        assert arr.shape == (34, 34, 34, 1)
        break


def test_array_no_ghosts():
    """array() with ngrow=0 returns just the valid region."""
    mf = _make_multifab(ngrow=0)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        assert isinstance(arr, jax.Array)
        assert arr.shape == (32, 32, 32, 1)
        break


def test_host_array_returns_writable_numpy():
    """copy_to_host() returns a writable numpy copy of the valid region."""
    mf = _make_multifab(ngrow=1)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        assert isinstance(arr, np.ndarray), f"Expected np.ndarray, got {type(arr)}"
        assert arr.shape == (32, 32, 32, 1)
        arr[:, :, :, 0] = 42.0
        mf.copy_from(mfi, arr)
        assert arr[0, 0, 0, 0] == 42.0
        break


def test_grown_array_as_numpy():
    """np.asarray(grown_array()) returns a numpy copy of the full FAB."""
    mf = _make_multifab(ngrow=1)
    # Write via valid region, then fill ghosts
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = 7.0
        mf.copy_from(mfi, arr)
    mf.fill_boundary(geom)
    for mfi in blockamr.MFIterator(mf):
        arr = np.asarray(mf.grown_array(mfi))
        assert isinstance(arr, np.ndarray), f"Expected np.ndarray, got {type(arr)}"
        assert arr.shape == (34, 34, 34, 1)
        assert np.allclose(arr[:, :, :, 0], 7.0)
        break


def test_copy_to_host():
    """copy_to_host() returns an owned numpy copy of the valid region."""
    mf = _make_multifab(ngrow=1)
    # Write some data via copy_to_host + copy_from
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = 3.14
        mf.copy_from(mfi, arr)
    # copy_to_host should return an independent copy
    for mfi in blockamr.MFIterator(mf):
        host = mf.copy_to_host(mfi)
        assert isinstance(host, np.ndarray)
        assert host.shape == (32, 32, 32, 1)
        assert np.allclose(host[:, :, :, 0], 3.14)
        break


def _make_multifab_with_memory(ngrow=0, memory="default"):
    """Create a MultiFab with explicit memory placement."""
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow, memory=memory)
    return mf


def test_copy_from_cpu_to_cpu():
    """copy_from: numpy (CPU) source → host MultiFab."""
    mf = _make_multifab(ngrow=1)
    src = np.full((32, 32, 32), 2.718)
    for mfi in blockamr.MFIterator(mf):
        mf.copy_from(mfi, src)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        assert np.allclose(arr[:, :, :, 0], 2.718)
        break


def test_copy_from_gpu_to_cpu():
    """copy_from: JAX GPU source → host MultiFab."""
    import jax.numpy as jnp

    mf = _make_multifab(ngrow=1)
    src = jnp.full((32, 32, 32), 1.618)
    for mfi in blockamr.MFIterator(mf):
        mf.copy_from(mfi, src)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        assert np.allclose(arr[:, :, :, 0], 1.618)
        break


def test_copy_from_gpu_to_gpu():
    """copy_from: JAX GPU source → device MultiFab."""
    import jax.numpy as jnp

    mf = _make_multifab_with_memory(ngrow=1, memory="device")
    src = jnp.full((32, 32, 32), 3.14)
    for mfi in blockamr.MFIterator(mf):
        mf.copy_from(mfi, src)
    for mfi in blockamr.MFIterator(mf):
        host = mf.copy_to_host(mfi)
        assert np.allclose(host[:, :, :, 0], 3.14)
        break


def test_copy_from_cpu_to_gpu():
    """copy_from: numpy (CPU) source → device MultiFab."""
    mf = _make_multifab_with_memory(ngrow=1, memory="device")
    src = np.full((32, 32, 32), 0.577)
    for mfi in blockamr.MFIterator(mf):
        mf.copy_from(mfi, src)
    for mfi in blockamr.MFIterator(mf):
        host = mf.copy_to_host(mfi)
        assert np.allclose(host[:, :, :, 0], 0.577)
        break


def test_copy_arrays_ncomp3_roundtrip():
    """copy_arrays with ncomp=3 must preserve spatially-varying per-component data.

    Regression test: copy_arrays previously corrupted ncomp>1 data because
    the C-ordered JAX array layout was not converted to AMReX's Fortran order
    for the component dimension.
    """
    import jax.numpy as jnp

    N = 8
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(N)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 3, 0)

    # Create spatially-varying data: comp0 = x-index, comp1 = y-index, comp2 = z-index
    ix = jnp.arange(N, dtype=jnp.float64)
    comp0 = jnp.broadcast_to(ix[:, None, None], (N, N, N))
    comp1 = jnp.broadcast_to(ix[None, :, None], (N, N, N))
    comp2 = jnp.broadcast_to(ix[None, None, :], (N, N, N))
    vals = jnp.stack([comp0, comp1, comp2], axis=-1)  # (N, N, N, 3)

    mf.copy_arrays([vals])

    # Read back and verify each component independently
    arr = mf.arrays()[0]  # (N, N, N, 3)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                got = [float(arr[i, j, k, c]) for c in range(3)]
                expected = [float(i), float(j), float(k)]
                assert got == expected, (
                    f"Cell ({i},{j},{k}): got {got}, expected {expected}"
                )


def test_set_get_executor():
    """Module-level executor config should be settable and queryable."""
    assert blockamr.get_executor() == "cpu"
    blockamr.set_executor("cpu")
    assert blockamr.get_executor() == "cpu"


def test_cellfield_memory_param():
    """CellField should accept a memory parameter."""
    from blockamr.field import CellField
    from blockamr.mesh import Mesh

    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    field = CellField(mesh, ncomp=1, ngrow=1, memory="default")
    assert field.mf[0] is not None
