# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Divergence schemes — each `compute()` is a fused JIT kernel returning the source term."""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, ConfigDict, Discriminator

from .stencil import S_wide, face, interior

if TYPE_CHECKING:
    from ..operators.div import BoxFluxData


def _extract_fluxes(flux_x: Array, flux_y: Array, flux_z: Array, ng: int) -> list[Array]:
    """Extract component 0 and trim ghost cells from raw 4D flux arrays.

    *ng* is the MultiFab ngrow — trims to the valid region plus one face.
    Called inside @jax.jit — all slicing is free at trace time.
    """
    raw = [flux_x[:, :, :, 0], flux_y[:, :, :, 0], flux_z[:, :, :, 0]]
    trimmed = []
    for ax in range(3):
        f = raw[ax]
        sl = [slice(None)] * 3
        sl[ax] = slice(ng, -ng) if ng > 0 else slice(None)
        trimmed.append(f[tuple(sl)])
    return trimmed


# ---------------------------------------------------------------------------
# Upwind (1st-order)
# ---------------------------------------------------------------------------
@jax.jit(static_argnums=(5, 6))
def _upwind_compute(
    u_4d: Array, flux_x: Array, flux_y: Array, flux_z: Array, dh: Array, w: int, ng: int
) -> Array:
    """Fused upwind flux balance.  Traced once → single XLA kernel."""
    u = u_4d[:, :, :, 0]
    fluxes = _extract_fluxes(flux_x, flux_y, flux_z, ng)
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)
        F_l: Array = fl * jnp.where(fl >= 0, S_wide(u, -1, ax, ng), S_wide(u, 0, ax, ng))
        F_r: Array = fr * jnp.where(fr >= 0, S_wide(u, 0, ax, ng), S_wide(u, 1, ax, ng))
        total = total + interior((F_r - F_l) / dh[ax], ax, ng)
    return total


class UpwindDivKernel(NamedTuple):
    flux_data: object  # BoxFluxData
    coeff: float

    def __call__(self, phi_4d: Array) -> Array:
        return self.coeff * _upwind_compute(
            phi_4d,
            self.flux_data.flux_x,
            self.flux_data.flux_y,
            self.flux_data.flux_z,
            self.flux_data.dh,
            self.flux_data.stencil_width,
            self.flux_data.ngrow,
        )


class Upwind(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Upwind"] = "Upwind"
    stencil_width: int = 1

    def compute(self, u: Array, fluxes: list[Array], dh: Array, ngrow: int = 0) -> Array:
        u_4d = u[:, :, :, None] if u.ndim == 3 else u
        fx = fluxes[0][:, :, :, None] if fluxes[0].ndim == 3 else fluxes[0]
        fy = fluxes[1][:, :, :, None] if fluxes[1].ndim == 3 else fluxes[1]
        fz = fluxes[2][:, :, :, None] if fluxes[2].ndim == 3 else fluxes[2]
        ng = ngrow if ngrow > 0 else self.stencil_width
        return _upwind_compute(u_4d, fx, fy, fz, dh, self.stencil_width, ng)

    def build_kernel(self, flux_data: BoxFluxData, coeff: float = 1.0) -> UpwindDivKernel:
        return UpwindDivKernel(flux_data=flux_data, coeff=coeff)


# ---------------------------------------------------------------------------
# Linear / central (2nd-order, unbounded)
# ---------------------------------------------------------------------------
@jax.jit(static_argnums=(5, 6))
def _linear_compute(
    u_4d: Array, flux_x: Array, flux_y: Array, flux_z: Array, dh: Array, w: int, ng: int
) -> Array:
    u = u_4d[:, :, :, 0]
    fluxes = _extract_fluxes(flux_x, flux_y, flux_z, ng)
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)
        F_l: Array = fl * 0.5 * (S_wide(u, -1, ax, ng) + S_wide(u, 0, ax, ng))
        F_r: Array = fr * 0.5 * (S_wide(u, 0, ax, ng) + S_wide(u, 1, ax, ng))
        total = total + interior((F_r - F_l) / dh[ax], ax, ng)
    return total


class LinearDivKernel(NamedTuple):
    flux_data: object  # BoxFluxData
    coeff: float

    def __call__(self, phi_4d: Array) -> Array:
        return self.coeff * _linear_compute(
            phi_4d,
            self.flux_data.flux_x,
            self.flux_data.flux_y,
            self.flux_data.flux_z,
            self.flux_data.dh,
            self.flux_data.stencil_width,
            self.flux_data.ngrow,
        )


class Linear(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Linear"] = "Linear"
    stencil_width: int = 1

    def compute(self, u: Array, fluxes: list[Array], dh: Array, ngrow: int = 0) -> Array:
        u_4d = u[:, :, :, None] if u.ndim == 3 else u
        fx = fluxes[0][:, :, :, None] if fluxes[0].ndim == 3 else fluxes[0]
        fy = fluxes[1][:, :, :, None] if fluxes[1].ndim == 3 else fluxes[1]
        fz = fluxes[2][:, :, :, None] if fluxes[2].ndim == 3 else fluxes[2]
        ng = ngrow if ngrow > 0 else self.stencil_width
        return _linear_compute(u_4d, fx, fy, fz, dh, self.stencil_width, ng)

    def build_kernel(self, flux_data: BoxFluxData, coeff: float = 1.0) -> LinearDivKernel:
        return LinearDivKernel(flux_data=flux_data, coeff=coeff)


# ---------------------------------------------------------------------------
# VanLeer (TVD, 2nd-order bounded)
# ---------------------------------------------------------------------------
def _vanleer_limiter(r: Array) -> Array:
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))


@jax.jit(static_argnums=(5, 6))
def _vanleer_compute(
    u_4d: Array, flux_x: Array, flux_y: Array, flux_z: Array, dh: Array, w: int, ng: int
) -> Array:
    """Fused TVD flux balance with vanLeer limiter."""
    u = u_4d[:, :, :, 0]
    fluxes = _extract_fluxes(flux_x, flux_y, flux_z, ng)
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)

        u_ll: Array = S_wide(u, -2, ax, ng)
        u_l: Array = S_wide(u, -1, ax, ng)
        u_r: Array = S_wide(u, 0, ax, ng)
        u_rr: Array = S_wide(u, +1, ax, ng)

        delta_down: Array = u_r - u_l
        eps: float = 1e-30

        # Left face reconstruction
        r_pos_l: Array = (u_l - u_ll) / (delta_down + eps)
        r_neg_l: Array = (u_rr - u_r) / (delta_down + eps)
        phi_face_l: Array = jnp.where(
            fl >= 0,
            u_l + 0.5 * _vanleer_limiter(r_pos_l) * delta_down,
            u_r - 0.5 * _vanleer_limiter(r_neg_l) * delta_down,
        )
        F_l: Array = fl * phi_face_l

        # Right face reconstruction (shift by +1)
        u_ll_r: Array = S_wide(u, -1, ax, ng)
        u_l_r: Array = S_wide(u, 0, ax, ng)
        u_r_r: Array = S_wide(u, +1, ax, ng)
        u_rr_r: Array = S_wide(u, +2, ax, ng)

        delta_down_r: Array = u_r_r - u_l_r
        r_pos_r: Array = (u_l_r - u_ll_r) / (delta_down_r + eps)
        r_neg_r: Array = (u_rr_r - u_r_r) / (delta_down_r + eps)
        phi_face_r: Array = jnp.where(
            fr >= 0,
            u_l_r + 0.5 * _vanleer_limiter(r_pos_r) * delta_down_r,
            u_r_r - 0.5 * _vanleer_limiter(r_neg_r) * delta_down_r,
        )
        F_r: Array = fr * phi_face_r

        total = total + interior((F_r - F_l) / dh[ax], ax, ng)

    return total


class VanLeerDivKernel(NamedTuple):
    flux_data: object  # BoxFluxData
    coeff: float

    def __call__(self, phi_4d: Array) -> Array:
        return self.coeff * _vanleer_compute(
            phi_4d,
            self.flux_data.flux_x,
            self.flux_data.flux_y,
            self.flux_data.flux_z,
            self.flux_data.dh,
            self.flux_data.stencil_width,
            self.flux_data.ngrow,
        )


class VanLeer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["VanLeer"] = "VanLeer"
    stencil_width: int = 2

    def compute(self, u: Array, fluxes: list[Array], dh: Array, ngrow: int = 0) -> Array:
        u_4d = u[:, :, :, None] if u.ndim == 3 else u
        fx = fluxes[0][:, :, :, None] if fluxes[0].ndim == 3 else fluxes[0]
        fy = fluxes[1][:, :, :, None] if fluxes[1].ndim == 3 else fluxes[1]
        fz = fluxes[2][:, :, :, None] if fluxes[2].ndim == 3 else fluxes[2]
        ng = ngrow if ngrow > 0 else self.stencil_width
        return _vanleer_compute(u_4d, fx, fy, fz, dh, self.stencil_width, ng)

    def build_kernel(self, flux_data: BoxFluxData, coeff: float = 1.0) -> VanLeerDivKernel:
        return VanLeerDivKernel(flux_data=flux_data, coeff=coeff)


# ---------------------------------------------------------------------------
# QUICK (3rd-order)
# ---------------------------------------------------------------------------
@jax.jit(static_argnums=(5, 6))
def _quick_compute(
    u_4d: Array, flux_x: Array, flux_y: Array, flux_z: Array, dh: Array, w: int, ng: int
) -> Array:
    """Fused QUICK flux balance (quadratic upstream interpolation)."""
    u = u_4d[:, :, :, 0]
    fluxes = _extract_fluxes(flux_x, flux_y, flux_z, ng)
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)

        u_ll: Array = S_wide(u, -2, ax, ng)
        u_l: Array = S_wide(u, -1, ax, ng)
        u_r: Array = S_wide(u, 0, ax, ng)
        u_rr: Array = S_wide(u, +1, ax, ng)

        # QUICK: 3/8 downstream + 6/8 upwind - 1/8 far-upwind
        phi_face_l: Array = jnp.where(
            fl >= 0,
            0.375 * u_r + 0.75 * u_l - 0.125 * u_ll,
            0.375 * u_l + 0.75 * u_r - 0.125 * u_rr,
        )
        F_l: Array = fl * phi_face_l

        u_ll_r: Array = S_wide(u, -1, ax, ng)
        u_l_r: Array = S_wide(u, 0, ax, ng)
        u_r_r: Array = S_wide(u, +1, ax, ng)
        u_rr_r: Array = S_wide(u, +2, ax, ng)

        phi_face_r: Array = jnp.where(
            fr >= 0,
            0.375 * u_r_r + 0.75 * u_l_r - 0.125 * u_ll_r,
            0.375 * u_l_r + 0.75 * u_r_r - 0.125 * u_rr_r,
        )
        F_r: Array = fr * phi_face_r

        total = total + interior((F_r - F_l) / dh[ax], ax, ng)

    return total


class QUICKDivKernel(NamedTuple):
    flux_data: object  # BoxFluxData
    coeff: float

    def __call__(self, phi_4d: Array) -> Array:
        return self.coeff * _quick_compute(
            phi_4d,
            self.flux_data.flux_x,
            self.flux_data.flux_y,
            self.flux_data.flux_z,
            self.flux_data.dh,
            self.flux_data.stencil_width,
            self.flux_data.ngrow,
        )


class QUICK(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["QUICK"] = "QUICK"
    stencil_width: int = 2

    def compute(self, u: Array, fluxes: list[Array], dh: Array, ngrow: int = 0) -> Array:
        u_4d = u[:, :, :, None] if u.ndim == 3 else u
        fx = fluxes[0][:, :, :, None] if fluxes[0].ndim == 3 else fluxes[0]
        fy = fluxes[1][:, :, :, None] if fluxes[1].ndim == 3 else fluxes[1]
        fz = fluxes[2][:, :, :, None] if fluxes[2].ndim == 3 else fluxes[2]
        ng = ngrow if ngrow > 0 else self.stencil_width
        return _quick_compute(u_4d, fx, fy, fz, dh, self.stencil_width, ng)

    def build_kernel(self, flux_data: BoxFluxData, coeff: float = 1.0) -> QUICKDivKernel:
        return QUICKDivKernel(flux_data=flux_data, coeff=coeff)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------
DivScheme = Annotated[
    Union[Upwind, Linear, VanLeer, QUICK],
    Discriminator("type"),
]
