# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Gradient schemes."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cpp_kernels import CppGradAcc


class CentralDiffGrad(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffGrad"] = "CentralDiffGrad"
    stencil_width: int = 1

    def build_cpp_kernel(self):
        return CppGradAcc()


GradScheme = Annotated[
    Union[CentralDiffGrad],
    Discriminator("type"),
]
