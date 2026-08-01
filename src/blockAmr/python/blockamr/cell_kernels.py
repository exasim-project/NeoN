# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cell-level kernel eqx.Modules for the accessor-based dispatch.

Each has ``__call__(self, phi: CellAccessor) -> scalar`` and
``for_box(self, bucket, box_idx) -> kernel``. Nx, Ny, Nz and dh are traced and rebound
per box inside the outer vmap, so boxes of different shapes share a bucket without
forcing a JAX recompilation.
"""

import equinox as eqx
import jax.numpy as jnp

from .cell_accessor import CellAccessor, FaceAccessor


class CellLaplacianKernel(eqx.Module):
    """Central difference laplacian: coeff * div(gamma * grad(phi)).

    ``gamma_buf=None`` means constant gamma folded into ``coeff``; otherwise gamma is
    read per cell through a CellAccessor sharing phi's box layout.
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
    Ny: object
    Nz: object
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
    Nx: object
    Ny: object
    Nz: object
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
    Nx: object
    Ny: object
    Nz: object
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


def _vanleer_correction(d_up, d_down):
    """Harmonic-mean VanLeer: ``_vanleer_limiter(d_up/d_down) * d_down`` in one division.

    Returns 0 when the gradients have opposite signs (the TVD property).
    """
    prod = d_up * d_down
    return jnp.where(prod > 0.0, 2.0 * prod / (d_up + d_down), 0.0)


class CellVanLeerDivKernel(eqx.Module):
    """TVD VanLeer divergence on the flat contiguous buffer."""

    face_bufs: tuple  # traced: (fx_buf, fy_buf, fz_buf)
    face_offsets: tuple  # traced: 3 arrays of (max_boxes,) per direction
    _face_offset: tuple  # traced: 3 scalars — current box offset per direction
    Nx: object
    Ny: object
    Nz: object
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

            s_m2 = phi.S(-2, ax)
            s_m1 = phi.S(-1, ax)
            s_0 = phi.S(0, ax)
            s_p1 = phi.S(1, ax)
            s_p2 = phi.S(2, ax)

            # Left face: upwind from the left when fl >= 0, from the right otherwise.
            d_down_l = s_0 - s_m1
            corr_l_pos = _vanleer_correction(s_m1 - s_m2, d_down_l)
            corr_l_neg = _vanleer_correction(s_p1 - s_0, d_down_l)
            phi_l = jnp.where(fl >= 0,
                              s_m1 + 0.5 * corr_l_pos,
                              s_0 - 0.5 * corr_l_neg)
            F_l = fl * phi_l

            # Right face: the same, shifted by +1.
            d_down_r = s_p1 - s_0
            corr_r_pos = _vanleer_correction(s_0 - s_m1, d_down_r)
            corr_r_neg = _vanleer_correction(s_p2 - s_p1, d_down_r)
            phi_r = jnp.where(fr >= 0,
                              s_0 + 0.5 * corr_r_pos,
                              s_p1 - 0.5 * corr_r_neg)
            F_r = fr * phi_r

            total = total + (F_r - F_l) / self.dh[ax]

        return self.coeff * total
