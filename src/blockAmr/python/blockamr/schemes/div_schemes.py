# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Divergence schemes."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cpp_kernels import CppDivAcc


class Upwind(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Upwind"] = "Upwind"
    stencil_width: int = 1

    def build_cpp_kernel(self):
        return CppDivAcc("div_upwind_acc")


class Linear(BaseModel):
    """Central / linear divergence: F = f * 0.5 * (u_left + u_right)."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Linear"] = "Linear"
    stencil_width: int = 1

    def build_cpp_kernel(self):
        return CppDivAcc("div_linear_acc")


class VanLeer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["VanLeer"] = "VanLeer"
    stencil_width: int = 2

    def build_cpp_kernel(self):
        return CppDivAcc("div_vanleer_acc")


class QUICK(BaseModel):
    """QUICK: 3/8 downstream + 6/8 upwind - 1/8 far-upwind."""

    model_config = ConfigDict(frozen=True)
    type: Literal["QUICK"] = "QUICK"
    stencil_width: int = 2

    def build_cpp_kernel(self):
        return CppDivAcc("div_quick_acc")


DivScheme = Annotated[
    Union[Upwind, Linear, VanLeer, QUICK],
    Discriminator("type"),
]
