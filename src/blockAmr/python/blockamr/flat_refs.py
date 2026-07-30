# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Structured (i, j, k) indexing translated to ``plt.load`` on Fortran-order flat
MultiFab buffers, so equinox kernels work unchanged under the tiled Pallas dispatch.

``FlatFaceRef`` carries per-tile offsets (16-int tile metadata, benchmarks only);
``FlatFaceBoxed`` carries per-box offsets indexed by box_id (5-int tiles + box arrays).
"""

import equinox as eqx
import jax.numpy as jnp
from jax.experimental.pallas import triton as plt


class FlatCellRef:
    """Drop-in for CellArray — phi[i, j, k, comp] via plt.load on flat buffer."""

    def __init__(self, ref, offset, sx, sy, sz, sc=0):
        self._ref = ref
        self._offset = offset
        self._sx = sx
        self._sy = sy
        self._sz = sz
        self._sc = sc

    def __getitem__(self, idx):
        if len(idx) == 4:
            i, j, k, comp = idx
            flat_idx = (self._offset + i * self._sx + j * self._sy
                        + k * self._sz + comp * self._sc)
        else:
            i, j, k = idx
            flat_idx = (self._offset + i * self._sx + j * self._sy
                        + k * self._sz)
        return plt.load(self._ref.at[flat_idx])


class _FaceAxisRef:
    """Single-direction face ref — face_axis[i, j, k] via plt.load."""

    def __init__(self, ref, offset, sx, sy, sz):
        self._ref = ref
        self._offset = offset
        self._sx = sx
        self._sy = sy
        self._sz = sz

    def __getitem__(self, idx):
        i, j, k = idx
        flat_idx = self._offset + i * self._sx + j * self._sy + k * self._sz
        return plt.load(self._ref.at[flat_idx])


class FlatFaceRef:
    """Drop-in for FaceArray, wrapping one flat buffer per direction with per-tile
    staggered strides."""

    def __init__(self, fx_ref, fy_ref, fz_ref,
                 fx_off, fx_sx, fx_sy, fx_sz,
                 fy_off, fy_sx, fy_sy, fy_sz,
                 fz_off, fz_sx, fz_sy, fz_sz):
        self._axes = (
            _FaceAxisRef(fx_ref, fx_off, fx_sx, fx_sy, fx_sz),
            _FaceAxisRef(fy_ref, fy_off, fy_sx, fy_sy, fy_sz),
            _FaceAxisRef(fz_ref, fz_off, fz_sx, fz_sy, fz_sz),
        )

    def __getitem__(self, ax):
        return self._axes[int(ax)]


class _FaceAxisBoxed(eqx.Module):
    """Single-direction face ref, per-box offsets indexed by box_id.

    An eqx.Module so buffers/offsets/strides survive tree flatten through Pallas.
    """

    buf: jnp.ndarray           # face contiguous buffer (traced leaf)
    offsets: jnp.ndarray       # (n_boxes_padded,) int32 (traced leaf)
    strides: jnp.ndarray       # (n_boxes_padded, 3) int32 (traced leaf)
    box_id: jnp.ndarray        # int32 scalar (traced leaf)

    def __getitem__(self, idx):
        i, j, k = idx
        off = self.offsets[self.box_id]
        sx = self.strides[self.box_id, 0]
        sy = self.strides[self.box_id, 1]
        sz = self.strides[self.box_id, 2]
        return plt.load(self.buf.at[off + i * sx + j * sy + k * sz])


class FlatFaceBoxed(eqx.Module):
    """Drop-in for FaceArray — face[ax][i,j,k] via per-box offsets + box_id.

    ``box_id`` is a dummy ``jnp.int32(0)`` at build time; ``parallel_for`` replaces it
    with the tile's real box_id inside the Pallas kernel.

    Parameters
    ----------
    fx, fy, fz : jax.Array
        Face contiguous buffers (one per direction).
    fxo, fyo, fzo : jax.Array
        Per-box offsets (n_boxes_padded,) int32.
    fxs, fys, fzs : jax.Array
        Per-box strides (n_boxes_padded, 3) int32.
    box_id : jax.Array
        int32 scalar — dummy at build time, real inside Pallas.
    """

    x: _FaceAxisBoxed
    y: _FaceAxisBoxed
    z: _FaceAxisBoxed

    def __init__(self, fx, fy, fz,
                 fxo, fyo, fzo, fxs, fys, fzs, box_id):
        self.x = _FaceAxisBoxed(buf=fx, offsets=fxo, strides=fxs, box_id=box_id)
        self.y = _FaceAxisBoxed(buf=fy, offsets=fyo, strides=fys, box_id=box_id)
        self.z = _FaceAxisBoxed(buf=fz, offsets=fzo, strides=fzs, box_id=box_id)

    def __getitem__(self, ax):
        return (self.x, self.y, self.z)[int(ax)]

    def with_box_id(self, box_id):
        """Return a copy with box_id replaced (used inside Pallas kernel)."""
        return eqx.tree_at(
            lambda f: (f.x.box_id, f.y.box_id, f.z.box_id),
            self,
            (box_id, box_id, box_id))
