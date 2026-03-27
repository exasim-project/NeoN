# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cell-level kernel eqx.Modules for the accessor-based dispatch.

Each kernel is an eqx.Module with
__call__(self, phi: CellAccessor) -> scalar
and for_box(self, bucket, box_idx) -> kernel (for per-box rebinding in vmap).

Nx, Ny, Nz, dh are traced (per-box) fields — they are rebound via for_box()
inside the outer vmap over boxes. This allows boxes with different shapes
to coexist in the same bucket without triggering JAX recompilation.
"""

import equinox as eqx
import jax.numpy as jnp

from .cell_accessor import CellAccessor, FaceAccessor


class CellLaplacianKernel(eqx.Module):
    """Central difference laplacian: coeff * div(gamma * grad(phi)).

    When gamma_buf is None, uses constant gamma (scalar coeff).
    When gamma_buf is provided, builds a CellAccessor for gamma with the
    same box layout as phi to read per-cell gamma values.
    """

    dh: object  # traced: (3,) array per-box cell spacing
    coeff: float = eqx.field(static=True)
    ncomp: int = eqx.field(static=True, default=1)
    has_variable_gamma: bool = eqx.field(static=True, default=False)
    gamma_buf: object = None
    gamma_offsets: object = None
    _gamma_offset: object = 0
    Nx: object = 0  # traced: per-box grown x-dim
    Ny: object = 0  # traced
    Nz: object = 0  # traced
    ng: int = eqx.field(static=True, default=0)

    def for_box(self, bucket, box_idx):
        if not self.has_variable_gamma:
            return eqx.tree_at(
                lambda k: (k.Nx, k.Ny, k.Nz, k.dh),
                self,
                (bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
                 bucket.Nz_arr[box_idx], bucket.dh_arr[box_idx]),
            )
        return eqx.tree_at(
            lambda k: (k._gamma_offset, k.Nx, k.Ny, k.Nz, k.dh),
            self,
            (self.gamma_offsets[box_idx],
             bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
             bucket.Nz_arr[box_idx], bucket.dh_arr[box_idx]),
        )

    def __call__(self, phi):
        if not self.has_variable_gamma:
            return self.coeff * sum(
                (phi.S(1, ax) - 2 * phi.center + phi.S(-1, ax)) / self.dh[ax] ** 2
                for ax in range(3)
            )
        g = CellAccessor(
            self.gamma_buf, self._gamma_offset, phi.cell_idx,
            self.Nx, self.Ny, self.Nz, self.ng,
        )
        total = 0.0
        for ax in range(3):
            g_right = 0.5 * (g.S(0, ax) + g.S(1, ax))
            g_left = 0.5 * (g.S(-1, ax) + g.S(0, ax))
            total += (
                g_right * (phi.S(1, ax) - phi.center)
                - g_left * (phi.center - phi.S(-1, ax))
            ) / self.dh[ax] ** 2
        return self.coeff * total


class CellUpwindDivKernel(eqx.Module):
    """First-order upwind divergence on the flat contiguous buffer."""

    face_bufs: tuple  # traced: (fx_buf, fy_buf, fz_buf)
    face_offsets: tuple  # traced: 3 arrays of (max_boxes,) per direction
    _face_offset: tuple  # traced: 3 scalars — current box offset per direction
    Nx: object  # traced: per-box grown x-dim
    Ny: object  # traced
    Nz: object  # traced
    ng: int = eqx.field(static=True)
    dh: object  # traced: (3,) array per-box cell spacing
    coeff: float = eqx.field(static=True)
    ng_face: int = eqx.field(static=True, default=None)
    ncomp: int = eqx.field(static=True, default=1)

    def for_box(self, bucket, box_idx):
        new_off = tuple(fo[box_idx] for fo in self.face_offsets)
        return eqx.tree_at(
            lambda k: (k._face_offset, k.Nx, k.Ny, k.Nz, k.dh),
            self,
            (new_off, bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
             bucket.Nz_arr[box_idx], bucket.dh_arr[box_idx]),
        )

    def __call__(self, phi):
        ff = FaceAccessor(
            self.face_bufs, self._face_offset, phi.cell_idx,
            self.Nx, self.Ny, self.Nz, self.ng, ng_face=self.ng_face,
        )
        total = 0.0
        for ax in range(3):
            face_ax = (ff.x, ff.y, ff.z)[ax]
            fl = face_ax[0]
            fr = face_ax[1]
            F_l = fl * jnp.where(fl >= 0, phi.S(-1, ax), phi.S(0, ax))
            F_r = fr * jnp.where(fr >= 0, phi.S(0, ax), phi.S(1, ax))
            total = total + (F_r - F_l) / self.dh[ax]
        return self.coeff * total


class CellLinearDivKernel(eqx.Module):
    """Central/linear divergence: F = f * 0.5 * (u_left + u_right)."""

    face_bufs: tuple
    face_offsets: tuple  # traced: 3 arrays of (max_boxes,) per direction
    _face_offset: tuple  # traced: 3 scalars — current box offset per direction
    Nx: object  # traced
    Ny: object  # traced
    Nz: object  # traced
    ng: int = eqx.field(static=True)
    dh: object  # traced: (3,) array
    coeff: float = eqx.field(static=True)
    ng_face: int = eqx.field(static=True, default=None)
    ncomp: int = eqx.field(static=True, default=1)

    def for_box(self, bucket, box_idx):
        new_off = tuple(fo[box_idx] for fo in self.face_offsets)
        return eqx.tree_at(
            lambda k: (k._face_offset, k.Nx, k.Ny, k.Nz, k.dh),
            self,
            (new_off, bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
             bucket.Nz_arr[box_idx], bucket.dh_arr[box_idx]),
        )

    def __call__(self, phi):
        ff = FaceAccessor(
            self.face_bufs, self._face_offset, phi.cell_idx,
            self.Nx, self.Ny, self.Nz, self.ng, ng_face=self.ng_face,
        )
        total = 0.0
        for ax in range(3):
            face_ax = (ff.x, ff.y, ff.z)[ax]
            fl = face_ax[0]
            fr = face_ax[1]
            F_l = fl * 0.5 * (phi.S(-1, ax) + phi.S(0, ax))
            F_r = fr * 0.5 * (phi.S(0, ax) + phi.S(1, ax))
            total = total + (F_r - F_l) / self.dh[ax]
        return self.coeff * total


class CellQUICKDivKernel(eqx.Module):
    """QUICK divergence: 3/8 downstream + 6/8 upwind - 1/8 far-upwind."""

    face_bufs: tuple
    face_offsets: tuple  # traced: 3 arrays of (max_boxes,) per direction
    _face_offset: tuple  # traced: 3 scalars — current box offset per direction
    Nx: object  # traced
    Ny: object  # traced
    Nz: object  # traced
    ng: int = eqx.field(static=True)
    dh: object  # traced: (3,) array
    coeff: float = eqx.field(static=True)
    ng_face: int = eqx.field(static=True, default=None)
    ncomp: int = eqx.field(static=True, default=1)

    def for_box(self, bucket, box_idx):
        new_off = tuple(fo[box_idx] for fo in self.face_offsets)
        return eqx.tree_at(
            lambda k: (k._face_offset, k.Nx, k.Ny, k.Nz, k.dh),
            self,
            (new_off, bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
             bucket.Nz_arr[box_idx], bucket.dh_arr[box_idx]),
        )

    def __call__(self, phi):
        ff = FaceAccessor(
            self.face_bufs, self._face_offset, phi.cell_idx,
            self.Nx, self.Ny, self.Nz, self.ng, ng_face=self.ng_face,
        )
        total = 0.0
        for ax in range(3):
            face_ax = (ff.x, ff.y, ff.z)[ax]
            fl = face_ax[0]
            fr = face_ax[1]

            u_ll = phi.S(-2, ax)
            u_l = phi.S(-1, ax)
            u_r = phi.S(0, ax)
            u_rr = phi.S(1, ax)

            phi_l = jnp.where(
                fl >= 0,
                0.375 * u_r + 0.75 * u_l - 0.125 * u_ll,
                0.375 * u_l + 0.75 * u_r - 0.125 * u_rr,
            )
            F_l = fl * phi_l

            u_l_r = phi.S(0, ax)
            u_r_r = phi.S(1, ax)
            u_rr_r = phi.S(2, ax)

            phi_r = jnp.where(
                fr >= 0,
                0.375 * u_r_r + 0.75 * u_l_r - 0.125 * u_l,
                0.375 * u_l_r + 0.75 * u_r_r - 0.125 * u_rr_r,
            )
            F_r = fr * phi_r

            total = total + (F_r - F_l) / self.dh[ax]

        return self.coeff * total


def _vanleer_limiter(r):
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))


class CellVanLeerDivKernel(eqx.Module):
    """TVD VanLeer divergence on the flat contiguous buffer."""

    face_bufs: tuple  # traced: (fx_buf, fy_buf, fz_buf)
    face_offsets: tuple  # traced: 3 arrays of (max_boxes,) per direction
    _face_offset: tuple  # traced: 3 scalars — current box offset per direction
    Nx: object  # traced
    Ny: object  # traced
    Nz: object  # traced
    ng: int = eqx.field(static=True)
    dh: object  # traced: (3,) array
    coeff: float = eqx.field(static=True)
    ng_face: int = eqx.field(static=True, default=None)
    ncomp: int = eqx.field(static=True, default=1)

    def for_box(self, bucket, box_idx):
        new_off = tuple(fo[box_idx] for fo in self.face_offsets)
        return eqx.tree_at(
            lambda k: (k._face_offset, k.Nx, k.Ny, k.Nz, k.dh),
            self,
            (new_off, bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
             bucket.Nz_arr[box_idx], bucket.dh_arr[box_idx]),
        )

    def __call__(self, phi):
        ff = FaceAccessor(
            self.face_bufs, self._face_offset, phi.cell_idx,
            self.Nx, self.Ny, self.Nz, self.ng, ng_face=self.ng_face,
        )
        total = 0.0
        eps = 1e-30
        for ax in range(3):
            face_ax = (ff.x, ff.y, ff.z)[ax]
            fl = face_ax[0]
            fr = face_ax[1]

            u_ll = phi.S(-2, ax)
            u_l = phi.S(-1, ax)
            u_r = phi.S(0, ax)
            u_rr = phi.S(1, ax)

            delta = u_r - u_l

            r_pos = (u_l - u_ll) / (delta + eps)
            r_neg = (u_rr - u_r) / (delta + eps)
            phi_l = jnp.where(
                fl >= 0,
                u_l + 0.5 * _vanleer_limiter(r_pos) * delta,
                u_r - 0.5 * _vanleer_limiter(r_neg) * delta,
            )
            F_l = fl * phi_l

            u_l_r = phi.S(0, ax)
            u_r_r = phi.S(1, ax)
            u_rr_r = phi.S(2, ax)

            delta_r = u_r_r - u_l_r
            r_pos_r = (u_l_r - u_l) / (delta_r + eps)
            r_neg_r = (u_rr_r - u_r_r) / (delta_r + eps)
            phi_r = jnp.where(
                fr >= 0,
                u_l_r + 0.5 * _vanleer_limiter(r_pos_r) * delta_r,
                u_r_r - 0.5 * _vanleer_limiter(r_neg_r) * delta_r,
            )
            F_r = fr * phi_r

            total = total + (F_r - F_l) / self.dh[ax]

        return self.coeff * total
