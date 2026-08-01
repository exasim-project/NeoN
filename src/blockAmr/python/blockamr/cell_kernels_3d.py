# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""3D functor kernels for structured grid stencils.

Each kernel is an eqx.Module with ``__call__(self, box_id, i, j, k, phi) -> scalar``
and a static ``ng`` — the ghost width its stencil needs. ``phi`` is supplied by
``parallel_for``, never stored on the kernel, mirroring AMReX ParallelFor + Array4.
"""

import equinox as eqx
import jax.numpy as jnp

from .array_types import Axis, CellArray, FaceArray


class Laplacian3D(eqx.Module):
    """Central difference laplacian: coeff * sum_ax (phi[+1] - 2*phi + phi[-1]) / dx^2."""

    dh: tuple = eqx.field(static=True)  # (dx, dy, dz)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=1)

    def __call__(self, box_id, i, j, k, phi):
        c = phi[i, j, k, 0]
        total = 0.0
        for ax in Axis:
            d = ax.d
            total += (phi[i+d[0], j+d[1], k+d[2], 0]
                      - 2 * c
                      + phi[i-d[0], j-d[1], k-d[2], 0]) / self.dh[ax]**2
        return self.coeff * total


class VariableGammaLaplacian3D(eqx.Module):
    """div(gamma * grad(phi)) with per-cell variable gamma."""

    gamma: CellArray  # (Nx, Ny, Nz, 1)
    dh: tuple = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=1)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for ax in Axis:
            d = ax.d
            g_r = 0.5 * (self.gamma[i, j, k, 0]
                         + self.gamma[i+d[0], j+d[1], k+d[2], 0])
            g_l = 0.5 * (self.gamma[i-d[0], j-d[1], k-d[2], 0]
                         + self.gamma[i, j, k, 0])
            total += (g_r * (phi[i+d[0], j+d[1], k+d[2], 0]
                             - phi[i, j, k, 0])
                      - g_l * (phi[i, j, k, 0]
                               - phi[i-d[0], j-d[1], k-d[2], 0])
                      ) / self.dh[ax]**2
        return self.coeff * total


class UpwindDiv3D(eqx.Module):
    """First-order upwind divergence."""

    face: FaceArray
    dh: tuple = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=1)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for ax in Axis:
            d = ax.d
            fl = self.face[ax][i, j, k]
            fr = self.face[ax][i+d[0], j+d[1], k+d[2]]
            F_l = fl * jnp.where(fl >= 0,
                                 phi[i-d[0], j-d[1], k-d[2], 0],
                                 phi[i, j, k, 0])
            F_r = fr * jnp.where(fr >= 0,
                                 phi[i, j, k, 0],
                                 phi[i+d[0], j+d[1], k+d[2], 0])
            total += (F_r - F_l) / self.dh[ax]
        return self.coeff * total


class LinearDiv3D(eqx.Module):
    """Central/linear divergence: F = f * 0.5 * (u_left + u_right)."""

    face: FaceArray
    dh: tuple = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=1)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for ax in Axis:
            d = ax.d
            fl = self.face[ax][i, j, k]
            fr = self.face[ax][i+d[0], j+d[1], k+d[2]]
            F_l = fl * 0.5 * (phi[i-d[0], j-d[1], k-d[2], 0]
                              + phi[i, j, k, 0])
            F_r = fr * 0.5 * (phi[i, j, k, 0]
                              + phi[i+d[0], j+d[1], k+d[2], 0])
            total += (F_r - F_l) / self.dh[ax]
        return self.coeff * total


class QUICKDiv3D(eqx.Module):
    """QUICK divergence: 3/8 downstream + 6/8 upwind - 1/8 far-upwind."""

    face: FaceArray
    dh: tuple = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=2)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for ax in Axis:
            d = ax.d
            fl = self.face[ax][i, j, k]
            fr = self.face[ax][i+d[0], j+d[1], k+d[2]]

            u_mm = phi[i-2*d[0], j-2*d[1], k-2*d[2], 0]
            u_m = phi[i-d[0], j-d[1], k-d[2], 0]
            u_0 = phi[i, j, k, 0]
            u_p = phi[i+d[0], j+d[1], k+d[2], 0]
            u_pp = phi[i+2*d[0], j+2*d[1], k+2*d[2], 0]

            phi_l = jnp.where(fl >= 0,
                              0.375 * u_0 + 0.75 * u_m - 0.125 * u_mm,
                              0.375 * u_m + 0.75 * u_0 - 0.125 * u_p)
            F_l = fl * phi_l

            phi_r = jnp.where(fr >= 0,
                              0.375 * u_p + 0.75 * u_0 - 0.125 * u_m,
                              0.375 * u_0 + 0.75 * u_p - 0.125 * u_pp)
            F_r = fr * phi_r

            total += (F_r - F_l) / self.dh[ax]
        return self.coeff * total


def _vanleer_corr(d_up, d_down):
    """Harmonic-mean VanLeer limiter without explicit ratio."""
    prod = d_up * d_down
    return jnp.where(prod > 0.0, 2.0 * prod / (d_up + d_down), 0.0)


class VanLeerDiv3D(eqx.Module):
    """TVD VanLeer divergence with slope limiting."""

    face: FaceArray
    dh: tuple = eqx.field(static=True)
    coeff: float = eqx.field(static=True)
    ng: int = eqx.field(static=True, default=2)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for ax in Axis:
            d = ax.d
            fl = self.face[ax][i, j, k]
            fr = self.face[ax][i+d[0], j+d[1], k+d[2]]

            s = [phi[i+n*d[0], j+n*d[1], k+n*d[2], 0]
                 for n in range(-2, 3)]

            d_down_l = s[2] - s[1]
            corr_l = jnp.where(
                fl >= 0,
                _vanleer_corr(s[1] - s[0], d_down_l),
                _vanleer_corr(s[3] - s[2], d_down_l))
            phi_l = jnp.where(fl >= 0,
                              s[1] + 0.5 * corr_l,
                              s[2] - 0.5 * corr_l)

            d_down_r = s[3] - s[2]
            corr_r = jnp.where(
                fr >= 0,
                _vanleer_corr(s[2] - s[1], d_down_r),
                _vanleer_corr(s[4] - s[3], d_down_r))
            phi_r = jnp.where(fr >= 0,
                              s[2] + 0.5 * corr_r,
                              s[3] - 0.5 * corr_r)

            total += (fr * phi_r - fl * phi_l) / self.dh[ax]
        return self.coeff * total


class FusedEulerKernel(eqx.Module):
    """``phi[i,j,k,0] - dt_over_coeff * sum(spatial_kernels(i,j,k,phi))``.

    ``ng`` is the max over the sub-kernels.
    """

    spatial_kernels: tuple
    dt_over_coeff: float
    ng: int = eqx.field(static=True)
    _n: int = eqx.field(static=True)

    def __init__(self, spatial_kernels, dt_over_coeff):
        self.spatial_kernels = spatial_kernels
        self.dt_over_coeff = jnp.float32(dt_over_coeff)
        self.ng = max(s.ng for s in spatial_kernels)
        self._n = len(spatial_kernels)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for idx in range(self._n):
            total = total + self.spatial_kernels[idx](box_id, i, j, k, phi)
        return phi[i, j, k, 0] - self.dt_over_coeff[()] * total


class CombinedSource(eqx.Module):
    """Sums spatial operator contributions, no time step. ``ng`` is the sub-kernel max."""

    spatial_kernels: tuple
    ng: int = eqx.field(static=True)
    _n: int = eqx.field(static=True)

    def __init__(self, spatial_kernels):
        self.spatial_kernels = spatial_kernels
        self.ng = max(s.ng for s in spatial_kernels)
        self._n = len(spatial_kernels)

    def __call__(self, box_id, i, j, k, phi):
        total = 0.0
        for idx in range(self._n):
            total = total + self.spatial_kernels[idx](box_id, i, j, k, phi)
        return total
