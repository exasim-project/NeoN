# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Immersed-body geometry — the ``mesh.body`` payload."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Cylinder:
    """Infinite cylinder: circular cross-section perpendicular to ``axis``."""

    centre: Sequence[float]
    radius: float
    axis: int
