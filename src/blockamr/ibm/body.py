# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Immersed-body geometry — the ``mesh.bodies`` payload (API doc §6).

Every body exposes the same two vectorised primitives, which is all the row
builders in :mod:`blockamr.ibm.rows` ever ask of a shape:

``sdf(x, y, z)``
    signed distance, **positive in the fluid, negative inside the solid**.
``normal(x, y, z)``
    unit outward normal ``n̂ = ∇s/|∇s|``, **pointing into the fluid**, returned
    with shape ``(..., 3)``.

Both accept scalars or numpy arrays of any (common) shape.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _centre3(centre, axis):
    """Pad a possibly 2-entry ``centre`` to three components.

    ``Cylinder(centre=(0.5, 0.5), axis=2)`` is the historical spelling: the
    entries are indexed by *global* axis, and the (irrelevant) coordinate along
    ``axis`` may simply be omitted when it is the last one.
    """
    c = [float(v) for v in centre]
    if len(c) == 3:
        return c
    if len(c) == 2 and axis == 2:
        return c + [0.0]
    raise ValueError(f"Cylinder.centre must have 3 entries (or 2 with axis=2); got {centre!r}")


@dataclass
class Cylinder:
    """Infinite cylinder: circular cross-section in the plane perpendicular
    to ``axis``, centred at ``centre``."""

    centre: Sequence[float]
    radius: float
    axis: int

    def _radial(self, x, y, z):
        """(distance from the axis, in-plane offsets, plane axis indices)."""
        axis = int(self.axis)
        centre = _centre3(self.centre, axis)
        plane = [a for a in range(3) if a != axis]
        coords = (
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        d0 = coords[plane[0]] - centre[plane[0]]
        d1 = coords[plane[1]] - centre[plane[1]]
        return np.hypot(d0, d1), d0, d1, plane

    def sdf(self, x, y, z):
        """Signed distance: positive outside the cylinder (fluid)."""
        r, _d0, _d1, _plane = self._radial(x, y, z)
        return r - float(self.radius)

    def normal(self, x, y, z):
        """Outward (into-fluid) unit radial normal, shape ``(..., 3)``.

        On the axis the normal is undefined; the first in-plane direction is
        returned there so downstream arithmetic stays finite.
        """
        r, d0, d1, plane = self._radial(x, y, z)
        safe = np.where(r > 0.0, r, 1.0)
        n = np.zeros(np.shape(safe) + (3,), dtype=float)
        n[..., plane[0]] = np.where(r > 0.0, d0 / safe, 1.0)
        n[..., plane[1]] = np.where(r > 0.0, d1 / safe, 0.0)
        return n


class Plane:
    """Half-space body: solid on the side ``normal`` points away from, so
    ``sdf(x) = (x - point) · n̂`` and the fluid is where that is positive.

    It exists for testability (verification plan §1): a field that is linear
    along the wall normal, ``T = a + b·(x·n̂)``, has a **constant** trace on a
    plane, so a scalar ``FixedValue`` can express its surface BC exactly. That
    turns the sharpest reconstruction test — linear exactness — from an order
    study on a curved body into an exact, tolerance-free equality.

    Not a dataclass, unlike :class:`Cylinder`: the constructor argument
    ``normal`` and the body protocol's ``normal(x, y, z)`` method share a name,
    and a dataclass field would shadow the method.
    """

    def __init__(self, point, normal):
        self.point = tuple(float(v) for v in point)
        n = np.asarray([float(v) for v in normal], dtype=float)
        norm = float(np.linalg.norm(n))
        if norm == 0.0:
            raise ValueError("Plane.normal must be non-zero")
        self._n = n / norm

    def __repr__(self):
        return f"Plane(point={self.point}, normal={tuple(self._n)})"

    def sdf(self, x, y, z):
        """Signed distance to the plane: positive in the fluid half-space."""
        coords = (
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        return sum((coords[d] - self.point[d]) * self._n[d] for d in range(3))

    def normal(self, x, y, z):
        """The (constant) unit normal, broadcast to ``(..., 3)``."""
        shape = np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape
        return np.broadcast_to(self._n, shape + (3,))
