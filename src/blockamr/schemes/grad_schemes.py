# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator


class CentralDiffGrad(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffGrad"] = "CentralDiffGrad"

    def face_value(self, phi_left, phi_right, dx):
        return (phi_right - phi_left) / (2.0 * dx)


GradScheme = Annotated[
    Union[CentralDiffGrad],
    Discriminator("type"),
]
