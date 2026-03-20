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


def test_multifab_is_host():
    """Default MultiFab on CPU build should report is_host=True."""
    mf = _make_multifab()
    assert mf.is_host is True


def test_multifab_is_device():
    """Default MultiFab on CPU build should report is_device=False."""
    mf = _make_multifab()
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
    """host_array() returns a writable numpy view of the valid region."""
    mf = _make_multifab(ngrow=1)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.host_array(mfi)
        assert isinstance(arr, np.ndarray), f"Expected np.ndarray, got {type(arr)}"
        assert arr.shape == (32, 32, 32, 1)
        arr[:, :, :, 0] = 42.0
        assert arr[0, 0, 0, 0] == 42.0
        break


def test_host_grown_array_returns_writable_numpy():
    """host_grown_array() returns a writable numpy view of the full FAB."""
    mf = _make_multifab(ngrow=1)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.host_grown_array(mfi)
        assert isinstance(arr, np.ndarray), f"Expected np.ndarray, got {type(arr)}"
        assert arr.shape == (34, 34, 34, 1)
        arr[:, :, :, 0] = 7.0
        assert arr[0, 0, 0, 0] == 7.0
        break


def test_copy_to_host():
    """copy_to_host() returns an owned numpy copy of the valid region."""
    mf = _make_multifab(ngrow=1)
    # Write some data via host_array
    for mfi in blockamr.MFIterator(mf):
        arr = mf.host_array(mfi)
        arr[:, :, :, 0] = 3.14
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
        arr = mf.host_array(mfi)
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
        arr = mf.host_array(mfi)
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


def test_set_get_executor():
    """Module-level executor config should be settable and queryable."""
    assert blockamr.get_executor() == "cpu"
    blockamr.set_executor("cpu")
    assert blockamr.get_executor() == "cpu"


def test_cellfield_memory_param():
    """CellField should accept a memory parameter."""
    from blockamr.field import CellField

    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box).max_size(32))
    field = CellField(box, dm, geom, ncomp=1, ngrow=1, memory="default")
    assert field.mf.is_host is True
