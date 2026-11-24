# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Laplacian schemes."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cpp_kernels import CppLaplacianAcc


class CentralDiffLaplacian(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffLaplacian"] = "CentralDiffLaplacian"
    stencil_width: int = 1

    def build_cpp_kernel(self):
        return CppLaplacianAcc()


LaplacianScheme = Annotated[
    Union[CentralDiffLaplacian],
    Discriminator("type"),
]
