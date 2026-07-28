# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cell-level accessors for flat contiguous buffers.

These classes operate on a 1D flat buffer (from contiguous_array)
and use offset arithmetic to read stencil neighbors without
creating shape-dependent array views.
"""

import equinox as eqx


class StencilAxis(eqx.Module):
    """Reads scalar neighbor values along one axis from a flat buffer.

    Usage: axis[k] returns the value at offset k from center.
    """

    _buf: object
    _base: object
    _stride: object

    def __init__(self, buf, base, stride):
        self._buf = buf
        self._base = base
        self._stride = stride

    def __getitem__(self, k):
        return self._buf[self._base + k * self._stride]


class CellAccessor(eqx.Module):
    """Cell-centred stencil accessor, built from a single cell_idx.

    Converts cell_idx to (i, j, k) in grown-box coordinates, then
    computes the flat-buffer base index. Provides stencil access via
    phi.x[k], phi.y[k], phi.z[k], phi.S(k, ax), and phi.center.
    """

    x: StencilAxis
    y: StencilAxis
    z: StencilAxis
    cell_idx: object

    def __init__(self, cell_buf, box_offset, cell_idx, Nx, Ny, Nz, ng, nc=1, comp=0):
        self.cell_idx = cell_idx

        vNx = Nx - 2 * ng
        vNy = Ny - 2 * ng
        i = ng + cell_idx % vNx
        j = ng + (cell_idx // vNx) % vNy
        k = ng + cell_idx // (vNx * vNy)

        # AMReX planar layout: comp planes are contiguous
        # (i, j, k, comp) → box_offset + comp * Nx * Ny * Nz + i + Nx * j + Nx * Ny * k
        box_plane = Nx * Ny * Nz
        base = box_offset + comp * box_plane + i + Nx * j + Nx * Ny * k
        self.x = StencilAxis(cell_buf, base, 1)
        self.y = StencilAxis(cell_buf, base, Nx)
        self.z = StencilAxis(cell_buf, base, Nx * Ny)

    def S(self, k, ax):
        """Neighbor at offset k along axis ax."""
        return (self.x, self.y, self.z)[ax][k]

    @property
    def center(self):
        return self.x[0]


class StencilAxis3D(eqx.Module):
    """Reads neighbor values along one axis from a 3D array.

    Usage: axis[k] returns the value at offset k from center along this axis.
    """

    _arr: object
    _i: object
    _j: object
    _k: object
    _ax: int = eqx.field(static=True)

    def __getitem__(self, k):
        if self._ax == 0:
            return self._arr[self._i + k, self._j, self._k]
        if self._ax == 1:
            return self._arr[self._i, self._j + k, self._k]
        return self._arr[self._i, self._j, self._k + k]


class CellAccessor3D(eqx.Module):
    """Cell-centred stencil accessor backed by a 3D array.

    Duck-type compatible with CellAccessor — provides the same
    .S(k, ax), .center, .x[k], .y[k], .z[k] interface so that
    existing cell kernels work unchanged.
    """

    x: StencilAxis3D
    y: StencilAxis3D
    z: StencilAxis3D
    cell_idx: object

    def __init__(self, arr_3d, i, j, k, cell_idx=0):
        self.x = StencilAxis3D(arr_3d, i, j, k, _ax=0)
        self.y = StencilAxis3D(arr_3d, i, j, k, _ax=1)
        self.z = StencilAxis3D(arr_3d, i, j, k, _ax=2)
        self.cell_idx = cell_idx

    def S(self, k, ax):
        """Neighbor at offset k along axis ax."""
        return (self.x, self.y, self.z)[ax][k]

    @property
    def center(self):
        return self.x[0]


class FaceAccessor(eqx.Module):
    """Access face fluxes for a cell using .x[k], .y[k], .z[k].

    ff.x[0] = left x-face flux, ff.x[1] = right x-face flux.
    Built from flat face buffers + cell position.

    Parameters
    ----------
    face_bufs : tuple of 3 arrays
        Flat contiguous buffers for (fx, fy, fz).
    face_offsets : tuple of 3 scalars
        Per-direction box offsets into each face buffer.
        Required because face MultiFabs for different directions have
        different grown shapes, so their per-box offsets differ.
    cell_idx : int
        Flat index into valid cells (x-fastest).
    Nx, Ny, Nz : int
        Cell MultiFab grown dimensions.
    ng : int
        Cell MultiFab ghost cell count.
    ng_face : int
        Face MultiFab ghost cell count. May differ from ng (e.g. cell
        ngrow=2 for VanLeer stencil, face ngrow=0 for face fluxes).
        Defaults to ng for backward compatibility.
    """

    x: StencilAxis
    y: StencilAxis
    z: StencilAxis

    def __init__(self, face_bufs, face_offsets, cell_idx, Nx, Ny, Nz, ng,
                 ng_face=None):
        if ng_face is None:
            ng_face = ng

        # Valid cell dimensions (same for cell and face)
        vNx = Nx - 2 * ng
        vNy = Ny - 2 * ng
        vNz = Nz - 2 * ng

        # Valid-local indices from cell_idx
        ix = cell_idx % vNx
        it = (cell_idx // vNx) % vNy
        iz = cell_idx // (vNx * vNy)

        # Face buffer positions: offset by ng_face (not ng)
        i_f = ng_face + ix
        j_f = ng_face + it
        k_f = ng_face + iz

        # x-faces: grown shape (vNx + 1 + 2*ng_face, vNy + 2*ng_face, vNz + 2*ng_face)
        Nx_fx = vNx + 1 + 2 * ng_face
        Ny_fx = vNy + 2 * ng_face
        fx_base = face_offsets[0] + i_f + Nx_fx * j_f + Nx_fx * Ny_fx * k_f
        self.x = StencilAxis(face_bufs[0], fx_base, 1)

        # y-faces: grown shape (vNx + 2*ng_face, vNy + 1 + 2*ng_face, vNz + 2*ng_face)
        Nx_fy = vNx + 2 * ng_face
        Ny_fy = vNy + 1 + 2 * ng_face
        fy_base = face_offsets[1] + i_f + Nx_fy * j_f + Nx_fy * Ny_fy * k_f
        self.y = StencilAxis(face_bufs[1], fy_base, Nx_fy)

        # z-faces: grown shape (vNx + 2*ng_face, vNy + 2*ng_face, vNz + 1 + 2*ng_face)
        Nx_fz = vNx + 2 * ng_face
        Ny_fz = vNy + 2 * ng_face
        fz_base = face_offsets[2] + i_f + Nx_fz * j_f + Nx_fz * Ny_fz * k_f
        self.z = StencilAxis(face_bufs[2], fz_base, Nx_fz * Ny_fz)
