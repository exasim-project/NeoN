# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Laplacian schemes — each `compute()` is a fused JIT kernel."""
from __future__ import annotations

from typing import Annotated, Literal, Union

import jax
from jax import Array
from pydantic import BaseModel, ConfigDict, Discriminator

from .stencil import S_wide, interior


@jax.jit(static_argnums=(3,))
def _central_diff_laplacian_compute(phi: Array, gamma: Array, dh: Array, ng: int) -> Array:
    """Fused central-difference laplacian: sum_ax gamma * d²phi/dx²."""
    # Strip component dimension if present (4D array from GPU MultiFab)
    if phi.ndim == 4:
        phi = phi[:, :, :, 0]
    total: Array | float = 0.0
    for ax in range(3):
        phi_l = S_wide(phi, -1, ax, ng)
        phi_c = S_wide(phi, 0, ax, ng)
        phi_r = S_wide(phi, +1, ax, ng)
        gamma_l = S_wide(gamma, -1, ax, ng)
        gamma_c = S_wide(gamma, 0, ax, ng)
        gamma_r = S_wide(gamma, +1, ax, ng)
        gamma_right = 0.5 * (gamma_c + gamma_r)
        gamma_left = 0.5 * (gamma_l + gamma_c)
        lap = (gamma_right * (phi_r - phi_c) - gamma_left * (phi_c - phi_l)) / dh[ax] ** 2
        total = total + interior(lap, ax, ng)
    return total


class CentralDiffLaplacianKernel:
    """Callable kernel with ngrow as static data for JAX pytree."""

    def __init__(self, gamma, dh, coeff, ngrow):
        self.gamma = gamma
        self.dh = dh
        self.coeff = coeff
        self.ngrow = ngrow

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _central_diff_laplacian_compute(phi, self.gamma, self.dh, self.ngrow)


def _lap_kernel_flatten(k):
    return (k.gamma, k.dh, k.coeff), k.ngrow


def _lap_kernel_unflatten(ngrow, children):
    return CentralDiffLaplacianKernel(children[0], children[1], children[2], ngrow)


jax.tree_util.register_pytree_node(
    CentralDiffLaplacianKernel, _lap_kernel_flatten, _lap_kernel_unflatten
)


class CentralDiffLaplacian(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffLaplacian"] = "CentralDiffLaplacian"
    stencil_width: int = 1

    def compute(self, phi: Array, gamma: Array, dh: Array, ngrow: int = 0) -> Array:
        ng = ngrow if ngrow > 0 else self.stencil_width
        return _central_diff_laplacian_compute(phi, gamma, dh, ng)

    def build_kernel(self, gamma: Array, dh: Array, coeff: float = 1.0, ngrow: int = 0):
        ng = ngrow if ngrow > 0 else self.stencil_width
        return CentralDiffLaplacianKernel(gamma, dh, coeff, ng)


LaplacianScheme = Annotated[
    Union[CentralDiffLaplacian],
    Discriminator("type"),
]
