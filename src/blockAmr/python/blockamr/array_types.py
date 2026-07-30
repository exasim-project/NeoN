# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Core array types for 3D functor kernels: ``Axis``, ``CellArray``, ``FaceArray``,
and the reshape helpers from a Fortran-order flat buffer into them.
"""

import enum

import equinox as eqx
import jax
import jax.numpy as jnp


class Axis(enum.IntEnum):
    """Spatial axis with unit direction vector."""

    x = 0
    y = 1
    z = 2

    @property
    def d(self):
        """Unit offset vector: Axis.x.d = (1,0,0), etc."""
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1))[self]


class CellArray(eqx.Module):
    """Ghosted cell-centred array, always 4D (Nx, Ny, Nz, ncomp), as AMReX Array4."""

    data: jnp.ndarray  # (Nx, Ny, Nz, nc)

    def __getitem__(self, idx):
        return self.data[idx]


class FaceArray(eqx.Module):
    """Staggered face arrays: x-faces (Nx+1, Ny, Nz), y-faces (Nx, Ny+1, Nz), etc."""

    x: jnp.ndarray  # x-face fluxes
    y: jnp.ndarray  # y-face fluxes
    z: jnp.ndarray  # z-face fluxes

    def __getitem__(self, ax):
        return getattr(self, Axis(ax).name)


def _reshape_plane(flat_buf, start, Nx, Ny, Nz):
    """Reshape one Fortran-order plane (Nx*Ny*Nz,) to (Nx, Ny, Nz), metadata-only.

    AMReX layout is x-fastest: ``flat[i + Nx*j + Nx*Ny*k]``.
    """
    plane = jax.lax.dynamic_slice(flat_buf, (start,), (Nx * Ny * Nz,))
    return plane.reshape(Nz, Ny, Nx).transpose(2, 1, 0)


def reshape_to_cell_array(flat_buf, offset, Nx, Ny, Nz, ncomp=1):
    """Fortran-order flat buffer → CellArray (Nx, Ny, Nz, ncomp)."""
    plane_size = Nx * Ny * Nz
    planes = [_reshape_plane(flat_buf, offset + c * plane_size, Nx, Ny, Nz)
              for c in range(ncomp)]
    return CellArray(jnp.stack(planes, axis=-1))


def reshape_to_face_array(face_bufs, face_offsets, box_idx,
                          vNx, vNy, vNz, ng_face):
    """Reshape flat face buffers to FaceArray for one box."""
    ngf = ng_face
    Nx_fx = vNx + 1 + 2 * ngf; Ny_fx = vNy + 2 * ngf; Nz_fx = vNz + 2 * ngf
    Nx_fy = vNx + 2 * ngf; Ny_fy = vNy + 1 + 2 * ngf; Nz_fy = vNz + 2 * ngf
    Nx_fz = vNx + 2 * ngf; Ny_fz = vNy + 2 * ngf; Nz_fz = vNz + 1 + 2 * ngf

    fx_3d = _reshape_plane(face_bufs[0], int(face_offsets[0][box_idx]),
                           Nx_fx, Ny_fx, Nz_fx)
    fy_3d = _reshape_plane(face_bufs[1], int(face_offsets[1][box_idx]),
                           Nx_fy, Ny_fy, Nz_fy)
    fz_3d = _reshape_plane(face_bufs[2], int(face_offsets[2][box_idx]),
                           Nx_fz, Ny_fz, Nz_fz)

    return FaceArray(x=fx_3d, y=fy_3d, z=fz_3d)
