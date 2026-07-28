# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Immersed-body geometry — the ``mesh.body`` payload (API doc §6)."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Cylinder:
    """Infinite cylinder: circular cross-section in the plane perpendicular
    to ``axis``, centred at ``centre``."""

    centre: Sequence[float]
    radius: float
    axis: int
