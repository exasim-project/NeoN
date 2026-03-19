# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Gradient schemes — each `compute()` is a fused JIT kernel."""
from __future__ import annotations

from typing import Annotated, Literal, NamedTuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, ConfigDict, Discriminator

from blockamr.schemes.stencil import S, interior


@jax.jit
def _central_diff_grad_compute(phi: Array, dh: Array) -> Array:
    """Fused central-difference gradient: (phi_r - phi_l) / (2*dh) per axis."""
    components: list[Array] = []
    for ax in range(phi.ndim):
        dphi = (S(phi, +1, ax) - S(phi, -1, ax)) / (2.0 * dh[ax])
        dphi = interior(dphi, ax)
        components.append(dphi)
    return jnp.stack(components, axis=-1)


class CentralDiffGradKernel(NamedTuple):
    dh: Array
    coeff: float

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _central_diff_grad_compute(phi, self.dh)


class CentralDiffGrad(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffGrad"] = "CentralDiffGrad"
    stencil_width: int = 1

    def compute(self, phi: Array, dh: Array) -> Array:
        return _central_diff_grad_compute(phi, dh)

    def build_kernel(self, dh: Array, coeff: float = 1.0) -> CentralDiffGradKernel:
        return CentralDiffGradKernel(dh=dh, coeff=coeff)


GradScheme = Annotated[
    Union[CentralDiffGrad],
    Discriminator("type"),
]
