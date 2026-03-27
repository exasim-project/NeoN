# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Divergence schemes — cell-level kernels on flat contiguous buffers."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cell_kernels import (
    CellUpwindDivKernel, CellLinearDivKernel,
    CellVanLeerDivKernel, CellQUICKDivKernel,
)


class Upwind(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Upwind"] = "Upwind"
    stencil_width: int = 1

    def build_kernel(self, face_bufs, face_offsets, Nx, Ny, Nz, ng, dh,
                     coeff=1.0, ncomp=1, ng_face=None):
        return CellUpwindDivKernel(
            face_bufs=face_bufs, face_offsets=face_offsets, _face_offset=(0, 0, 0),
            Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, ng_face=ng_face, dh=dh, coeff=coeff, ncomp=ncomp,
        )


class Linear(BaseModel):
    """Central / linear divergence: F = f * 0.5 * (u_left + u_right)."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Linear"] = "Linear"
    stencil_width: int = 1

    def build_kernel(self, face_bufs, face_offsets, Nx, Ny, Nz, ng, dh,
                     coeff=1.0, ncomp=1, ng_face=None):
        return CellLinearDivKernel(
            face_bufs=face_bufs, face_offsets=face_offsets, _face_offset=(0, 0, 0),
            Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, ng_face=ng_face, dh=dh, coeff=coeff, ncomp=ncomp,
        )


class VanLeer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["VanLeer"] = "VanLeer"
    stencil_width: int = 2

    def build_kernel(self, face_bufs, face_offsets, Nx, Ny, Nz, ng, dh,
                     coeff=1.0, ncomp=1, ng_face=None):
        return CellVanLeerDivKernel(
            face_bufs=face_bufs, face_offsets=face_offsets, _face_offset=(0, 0, 0),
            Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, ng_face=ng_face, dh=dh, coeff=coeff, ncomp=ncomp,
        )


class QUICK(BaseModel):
    """QUICK: 3/8 downstream + 6/8 upwind - 1/8 far-upwind."""

    model_config = ConfigDict(frozen=True)
    type: Literal["QUICK"] = "QUICK"
    stencil_width: int = 2

    def build_kernel(self, face_bufs, face_offsets, Nx, Ny, Nz, ng, dh,
                     coeff=1.0, ncomp=1, ng_face=None):
        return CellQUICKDivKernel(
            face_bufs=face_bufs, face_offsets=face_offsets, _face_offset=(0, 0, 0),
            Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, ng_face=ng_face, dh=dh, coeff=coeff, ncomp=ncomp,
        )


DivScheme = Annotated[
    Union[Upwind, Linear, VanLeer, QUICK],
    Discriminator("type"),
]
