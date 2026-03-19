# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Divergence schemes — each `compute()` is a fused JIT kernel returning the source term."""
from __future__ import annotations

from typing import Annotated, Literal, NamedTuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, ConfigDict, Discriminator

from blockamr.schemes.stencil import S, S_wide, face, interior


# ---------------------------------------------------------------------------
# Upwind (1st-order)
# ---------------------------------------------------------------------------
@jax.jit
def _upwind_compute(u: Array, fluxes: list[Array], dh: Array) -> Array:
    """Fused upwind flux balance.  Traced once → single XLA kernel."""
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)
        F_l: Array = fl * jnp.where(fl >= 0, S(u, -1, ax), S(u, 0, ax))
        F_r: Array = fr * jnp.where(fr >= 0, S(u, 0, ax), S(u, 1, ax))
        total = total + interior((F_r - F_l) / dh[ax], ax)
    return total


class UpwindDivKernel(NamedTuple):
    fluxes: list
    dh: Array
    coeff: float

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _upwind_compute(phi, self.fluxes, self.dh)


class Upwind(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Upwind"] = "Upwind"
    stencil_width: int = 1

    def compute(self, u: Array, fluxes: list[Array], dh: Array) -> Array:
        return _upwind_compute(u, fluxes, dh)

    def build_kernel(self, fluxes: list[Array], dh: Array, coeff: float = 1.0) -> UpwindDivKernel:
        return UpwindDivKernel(fluxes=fluxes, dh=dh, coeff=coeff)


# ---------------------------------------------------------------------------
# Linear / central (2nd-order, unbounded)
# ---------------------------------------------------------------------------
@jax.jit
def _linear_compute(u: Array, fluxes: list[Array], dh: Array) -> Array:
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)
        F_l: Array = fl * 0.5 * (S(u, -1, ax) + S(u, 0, ax))
        F_r: Array = fr * 0.5 * (S(u, 0, ax) + S(u, 1, ax))
        total = total + interior((F_r - F_l) / dh[ax], ax)
    return total


class LinearDivKernel(NamedTuple):
    fluxes: list
    dh: Array
    coeff: float

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _linear_compute(phi, self.fluxes, self.dh)


class Linear(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Linear"] = "Linear"
    stencil_width: int = 1

    def compute(self, u: Array, fluxes: list[Array], dh: Array) -> Array:
        return _linear_compute(u, fluxes, dh)

    def build_kernel(self, fluxes: list[Array], dh: Array, coeff: float = 1.0) -> LinearDivKernel:
        return LinearDivKernel(fluxes=fluxes, dh=dh, coeff=coeff)


# ---------------------------------------------------------------------------
# VanLeer (TVD, 2nd-order bounded)
# ---------------------------------------------------------------------------
def _vanleer_limiter(r: Array) -> Array:
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))


@jax.jit
def _vanleer_compute(u: Array, fluxes: list[Array], dh: Array) -> Array:
    """Fused TVD flux balance with vanLeer limiter."""
    W: int = 2
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)

        u_ll: Array = S_wide(u, -2, ax, W)
        u_l: Array = S_wide(u, -1, ax, W)
        u_r: Array = S_wide(u, 0, ax, W)
        u_rr: Array = S_wide(u, +1, ax, W)

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
        u_ll_r: Array = S_wide(u, -1, ax, W)
        u_l_r: Array = S_wide(u, 0, ax, W)
        u_r_r: Array = S_wide(u, +1, ax, W)
        u_rr_r: Array = S_wide(u, +2, ax, W)

        delta_down_r: Array = u_r_r - u_l_r
        r_pos_r: Array = (u_l_r - u_ll_r) / (delta_down_r + eps)
        r_neg_r: Array = (u_rr_r - u_r_r) / (delta_down_r + eps)
        phi_face_r: Array = jnp.where(
            fr >= 0,
            u_l_r + 0.5 * _vanleer_limiter(r_pos_r) * delta_down_r,
            u_r_r - 0.5 * _vanleer_limiter(r_neg_r) * delta_down_r,
        )
        F_r: Array = fr * phi_face_r

        total = total + interior((F_r - F_l) / dh[ax], ax, W)

    return total


class VanLeerDivKernel(NamedTuple):
    fluxes: list
    dh: Array
    coeff: float

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _vanleer_compute(phi, self.fluxes, self.dh)


class VanLeer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["VanLeer"] = "VanLeer"
    stencil_width: int = 2

    def compute(self, u: Array, fluxes: list[Array], dh: Array) -> Array:
        return _vanleer_compute(u, fluxes, dh)

    def build_kernel(self, fluxes: list[Array], dh: Array, coeff: float = 1.0) -> VanLeerDivKernel:
        return VanLeerDivKernel(fluxes=fluxes, dh=dh, coeff=coeff)


# ---------------------------------------------------------------------------
# QUICK (3rd-order)
# ---------------------------------------------------------------------------
@jax.jit
def _quick_compute(u: Array, fluxes: list[Array], dh: Array) -> Array:
    """Fused QUICK flux balance (quadratic upstream interpolation)."""
    W: int = 2
    total: Array | float = 0.0
    for ax in range(u.ndim):
        f: Array = fluxes[ax]
        fl: Array = face(f, 0, ax)
        fr: Array = face(f, 1, ax)

        u_ll: Array = S_wide(u, -2, ax, W)
        u_l: Array = S_wide(u, -1, ax, W)
        u_r: Array = S_wide(u, 0, ax, W)
        u_rr: Array = S_wide(u, +1, ax, W)

        # QUICK: 3/8 downstream + 6/8 upwind - 1/8 far-upwind
        phi_face_l: Array = jnp.where(
            fl >= 0,
            0.375 * u_r + 0.75 * u_l - 0.125 * u_ll,
            0.375 * u_l + 0.75 * u_r - 0.125 * u_rr,
        )
        F_l: Array = fl * phi_face_l

        u_ll_r: Array = S_wide(u, -1, ax, W)
        u_l_r: Array = S_wide(u, 0, ax, W)
        u_r_r: Array = S_wide(u, +1, ax, W)
        u_rr_r: Array = S_wide(u, +2, ax, W)

        phi_face_r: Array = jnp.where(
            fr >= 0,
            0.375 * u_r_r + 0.75 * u_l_r - 0.125 * u_ll_r,
            0.375 * u_l_r + 0.75 * u_r_r - 0.125 * u_rr_r,
        )
        F_r: Array = fr * phi_face_r

        total = total + interior((F_r - F_l) / dh[ax], ax, W)

    return total


class QUICKDivKernel(NamedTuple):
    fluxes: list
    dh: Array
    coeff: float

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _quick_compute(phi, self.fluxes, self.dh)


class QUICK(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["QUICK"] = "QUICK"
    stencil_width: int = 2

    def compute(self, u: Array, fluxes: list[Array], dh: Array) -> Array:
        return _quick_compute(u, fluxes, dh)

    def build_kernel(self, fluxes: list[Array], dh: Array, coeff: float = 1.0) -> QUICKDivKernel:
        return QUICKDivKernel(fluxes=fluxes, dh=dh, coeff=coeff)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------
DivScheme = Annotated[
    Union[Upwind, Linear, VanLeer, QUICK],
    Discriminator("type"),
]
