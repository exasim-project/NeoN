# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Laplacian schemes — each `compute()` is a fused JIT kernel."""
from __future__ import annotations

from typing import Annotated, Literal, NamedTuple, Union

import jax
from jax import Array
from pydantic import BaseModel, ConfigDict, Discriminator

from blockamr.schemes.stencil import S, interior


@jax.jit
def _central_diff_laplacian_compute(phi: Array, gamma: Array, dh: Array) -> Array:
    """Fused central-difference laplacian: sum_ax gamma * d²phi/dx²."""
    total: Array | float = 0.0
    for ax in range(phi.ndim):
        phi_l = S(phi, -1, ax)
        phi_c = S(phi, 0, ax)
        phi_r = S(phi, +1, ax)
        gamma_l = S(gamma, -1, ax)
        gamma_c = S(gamma, 0, ax)
        gamma_r = S(gamma, +1, ax)
        gamma_right = 0.5 * (gamma_c + gamma_r)
        gamma_left = 0.5 * (gamma_l + gamma_c)
        lap = (gamma_right * (phi_r - phi_c) - gamma_left * (phi_c - phi_l)) / dh[ax] ** 2
        total = total + interior(lap, ax)
    return total


class CentralDiffLaplacianKernel(NamedTuple):
    gamma: Array
    dh: Array
    coeff: float

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _central_diff_laplacian_compute(phi, self.gamma, self.dh)


class CentralDiffLaplacian(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffLaplacian"] = "CentralDiffLaplacian"
    stencil_width: int = 1

    def compute(self, phi: Array, gamma: Array, dh: Array) -> Array:
        return _central_diff_laplacian_compute(phi, gamma, dh)

    def build_kernel(self, gamma: Array, dh: Array, coeff: float = 1.0) -> CentralDiffLaplacianKernel:
        return CentralDiffLaplacianKernel(gamma=gamma, dh=dh, coeff=coeff)


LaplacianScheme = Annotated[
    Union[CentralDiffLaplacian],
    Discriminator("type"),
]
