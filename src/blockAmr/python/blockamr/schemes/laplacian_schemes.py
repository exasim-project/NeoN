# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Laplacian schemes — cell-level kernels."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cell_kernels import CellLaplacianKernel
from ..cell_kernels_3d import Laplacian3D, VariableGammaLaplacian3D
from ..cpp_kernels import CppLaplacianAcc


class CentralDiffLaplacian(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffLaplacian"] = "CentralDiffLaplacian"
    stencil_width: int = 1

    def build_kernel(self, dh, coeff=1.0, ncomp=1,
                     gamma_buf=None, gamma_offsets=None,
                     Nx=0, Ny=0, Nz=0, ng=0):
        """Return a cell-level laplacian kernel (flat dispatch path)."""
        return CellLaplacianKernel(
            dh=dh, coeff=coeff, ncomp=ncomp,
            has_variable_gamma=gamma_buf is not None,
            gamma_buf=gamma_buf, gamma_offsets=gamma_offsets,
            _gamma_offset=0,
            Nx=Nx, Ny=Ny, Nz=Nz, ng=ng,
        )

    def build_spatial_kernel(self, dh, coeff=1.0, gamma=None):
        """3D functor kernel for parallel_for dispatch.

        ``coeff`` already includes gamma in the constant case; pass ``gamma`` only for
        the per-cell variable one.
        """
        if gamma is not None:
            return VariableGammaLaplacian3D(
                gamma=gamma, dh=dh, coeff=coeff)
        return Laplacian3D(dh=dh, coeff=coeff)

    def build_cpp_kernel(self):
        return CppLaplacianAcc()



LaplacianScheme = Annotated[
    Union[CentralDiffLaplacian],
    Discriminator("type"),
]
