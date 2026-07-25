# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Surface BC models for the immersed boundary — the ``ibm_bc`` payload.

All three are the triple ``(alpha, beta, gamma)`` in the one surface condition
(design §1.3)::

    alpha * phi_w + beta * dphi/dn|_w = gamma

so a single row formula serves them all. ``robin()`` is the whole interface the
row builders use; ``gamma`` may be a scalar or a per-component sequence and is
broadcast to ``(ncomp,)`` by :func:`broadcast_gamma`.
"""

from dataclasses import dataclass

import numpy as np


def broadcast_gamma(value, ncomp):
    """Broadcast a scalar or per-component BC datum to shape ``(ncomp,)``."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr, ncomp)
    if arr.size != ncomp:
        raise ValueError(f"IBM BC datum has {arr.size} components but the field has ncomp={ncomp}")
    return arr


@dataclass
class FixedValue:
    """Dirichlet: ``phi_w = value`` — the triple ``(1, 0, value)``."""

    value: float

    def robin(self):
        return (1.0, 0.0, self.value)


@dataclass
class FixedGradient:
    """Neumann: ``dphi/dn|_w = gradient`` — the triple ``(0, 1, gradient)``."""

    gradient: float

    def robin(self):
        return (0.0, 1.0, self.gradient)


@dataclass
class Mixed:
    """OpenFOAM-style blend of the two, weighted by ``fraction``:
    ``(fraction, 1 - fraction, fraction*value + (1 - fraction)*gradient)``.

    ``fraction=1`` is bitwise :class:`FixedValue` and ``fraction=0`` is bitwise
    :class:`FixedGradient` (the dead term multiplies by an exact zero).
    """

    value: float
    gradient: float
    fraction: float

    def robin(self):
        f = float(self.fraction)
        alpha = f
        beta = 1.0 - f
        gamma = alpha * np.asarray(self.value, dtype=float) + beta * np.asarray(
            self.gradient, dtype=float
        )
        return (alpha, beta, gamma if gamma.ndim else float(gamma))
