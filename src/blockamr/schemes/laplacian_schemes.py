# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator


class CentralDiffLaplacian(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["CentralDiffLaplacian"] = "CentralDiffLaplacian"

    def face_value(self, gamma_left, gamma_right):
        return 0.5 * (gamma_left + gamma_right)


LaplacianScheme = Annotated[
    Union[CentralDiffLaplacian],
    Discriminator("type"),
]
