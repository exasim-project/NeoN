# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Gradient schemes — each `compute()` is a fused JIT kernel."""
from __future__ import annotations

from typing import Annotated, Literal, Union

import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, ConfigDict, Discriminator

from ..cpp_kernels import CppGradAcc
from .stencil import S_wide, interior


@jax.jit(static_argnums=(2,))
def _central_diff_grad_compute(phi: Array, dh: Array, ng: int) -> Array:
    """Fused central-difference gradient: (phi_r - phi_l) / (2*dh) per axis."""
    if phi.ndim == 4:
        phi = phi[:, :, :, 0]
    components: list[Array] = []
    for ax in range(3):
        dphi = (S_wide(phi, +1, ax, ng) - S_wide(phi, -1, ax, ng)) / (2.0 * dh[ax])
        dphi = interior(dphi, ax, ng)
        components.append(dphi)
    return jnp.stack(components, axis=-1)


class CentralDiffGradKernel:
    """Callable kernel with ngrow as static data for JAX pytree."""

    def __init__(self, dh, coeff, ngrow):
        self.dh = dh
        self.coeff = coeff
        self.ngrow = ngrow

    def __call__(self, phi: Array) -> Array:
        return self.coeff * _central_diff_grad_compute(phi, self.dh, self.ngrow)


def _grad_kernel_flatten(k):
    return (k.dh, k.coeff), k.ngrow


def _grad_kernel_unflatten(ngrow, children):
    return CentralDiffGradKernel(children[0], children[1], ngrow)


jax.tree_util.register_pytree_node(
    CentralDiffGradKernel, _grad_kernel_flatten, _grad_kernel_unflatten
)


class CentralDiffGrad(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffGrad"] = "CentralDiffGrad"
    stencil_width: int = 1

    def compute(self, phi: Array, dh: Array, ngrow: int = 0) -> Array:
        ng = ngrow if ngrow > 0 else self.stencil_width
        return _central_diff_grad_compute(phi, dh, ng)

    def build_kernel(self, dh: Array, coeff: float = 1.0, ngrow: int = 0):
        ng = ngrow if ngrow > 0 else self.stencil_width
        return CentralDiffGradKernel(dh, coeff, ng)

    def build_cpp_kernel(self):
        return CppGradAcc()


GradScheme = Annotated[
    Union[CentralDiffGrad],
    Discriminator("type"),
]
