# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Divergence schemes — cell-level kernels."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cell_kernels import (
    CellUpwindDivKernel, CellLinearDivKernel,
    CellVanLeerDivKernel, CellQUICKDivKernel,
)
from ..cell_kernels_3d import (
    UpwindDiv3D, LinearDiv3D, VanLeerDiv3D, QUICKDiv3D,
)
from ..cpp_kernels import CppDivAcc


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

    def build_spatial_kernel(self, face, dh, coeff=1.0):
        return UpwindDiv3D(face=face, dh=dh, coeff=coeff)

    def build_cpp_kernel(self):
        return CppDivAcc("div_upwind_acc")



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

    def build_spatial_kernel(self, face, dh, coeff=1.0):
        return LinearDiv3D(face=face, dh=dh, coeff=coeff)

    def build_cpp_kernel(self):
        return CppDivAcc("div_linear_acc")



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

    def build_spatial_kernel(self, face, dh, coeff=1.0):
        return VanLeerDiv3D(face=face, dh=dh, coeff=coeff)

    def build_cpp_kernel(self):
        return CppDivAcc("div_vanleer_acc")



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

    def build_spatial_kernel(self, face, dh, coeff=1.0):
        return QUICKDiv3D(face=face, dh=dh, coeff=coeff)

    def build_cpp_kernel(self):
        return CppDivAcc("div_quick_acc")



DivScheme = Annotated[
    Union[Upwind, Linear, VanLeer, QUICK],
    Discriminator("type"),
]
