# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator


class ForwardEuler(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["ForwardEuler"] = "ForwardEuler"


class RungeKutta2(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["RungeKutta2"] = "RungeKutta2"


class RungeKutta4(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["RungeKutta4"] = "RungeKutta4"


DdtScheme = Annotated[
    Union[ForwardEuler, RungeKutta2, RungeKutta4],
    Discriminator("type"),
]
