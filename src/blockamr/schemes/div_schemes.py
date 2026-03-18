# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Discriminator


class Upwind(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Upwind"] = "Upwind"
    stencil_width: int = 1

    def face_value(self, phi_left, phi_right, vel_face):
        return jnp.where(vel_face >= 0, phi_left, phi_right)


class Linear(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["Linear"] = "Linear"
    stencil_width: int = 1

    def face_value(self, phi_left, phi_right, vel_face):
        return 0.5 * (phi_left + phi_right)


class VanLeer(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["VanLeer"] = "VanLeer"
    stencil_width: int = 2

    def face_value(self, phi_far_left, phi_left, phi_right, phi_far_right, vel_face):
        def _limiter(r):
            return jnp.where(r > 0, 2.0 * r / (1.0 + r), 0.0)

        d_up_pos = phi_left - phi_far_left
        d_down_pos = phi_right - phi_left
        r_pos = jnp.where(jnp.abs(d_down_pos) > 1e-30, d_up_pos / d_down_pos, 0.0)
        phi_face_pos = phi_left + 0.5 * _limiter(r_pos) * d_down_pos

        d_up_neg = phi_right - phi_far_right
        d_down_neg = phi_left - phi_right
        r_neg = jnp.where(jnp.abs(d_down_neg) > 1e-30, d_up_neg / d_down_neg, 0.0)
        phi_face_neg = phi_right + 0.5 * _limiter(r_neg) * d_down_neg

        return jnp.where(vel_face >= 0, phi_face_pos, phi_face_neg)


class QUICK(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["QUICK"] = "QUICK"
    stencil_width: int = 2

    def face_value(self, phi_far_left, phi_left, phi_right, phi_far_right, vel_face):
        phi_face_pos = (3.0 / 8.0) * phi_right + (6.0 / 8.0) * phi_left - (1.0 / 8.0) * phi_far_left
        phi_face_neg = (
            (3.0 / 8.0) * phi_left + (6.0 / 8.0) * phi_right - (1.0 / 8.0) * phi_far_right
        )
        return jnp.where(vel_face >= 0, phi_face_pos, phi_face_neg)


DivScheme = Annotated[
    Union[Upwind, VanLeer, Linear, QUICK],
    Discriminator("type"),
]
