# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Laplacian schemes — cell-level kernels on flat contiguous buffers."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cell_kernels import CellLaplacianKernel


class CentralDiffLaplacian(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffLaplacian"] = "CentralDiffLaplacian"
    stencil_width: int = 1

    def build_kernel(self, dh, coeff=1.0, ncomp=1,
                     gamma_buf=None, gamma_offsets=None,
                     Nx=0, Ny=0, Nz=0, ng=0):
        """Return a cell-level laplacian kernel.

        dh: tuple of (dx, dy, dz) as Python floats.
        gamma_buf/gamma_offsets: optional variable gamma data.
        """
        return CellLaplacianKernel(
            dh=dh, coeff=coeff, ncomp=ncomp,
            has_variable_gamma=gamma_buf is not None,
            gamma_buf=gamma_buf, gamma_offsets=gamma_offsets,
            _gamma_offset=0,
            Nx=Nx, Ny=Ny, Nz=Nz, ng=ng,
        )


LaplacianScheme = Annotated[
    Union[CentralDiffLaplacian],
    Discriminator("type"),
]
