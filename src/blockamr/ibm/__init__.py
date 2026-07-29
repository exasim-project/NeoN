# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""IBM method registry (API doc §6): ``IBM.lookup(name)`` resolves a
per-field ``solution["ibm"]`` name to a strategy class."""

from .bc import FixedGradient, FixedValue, Harmonic, Mixed
from .body import Cylinder, Plane
from .direct_forcing import DirectForcing, DirectForcingData
from .driver import WallEvaluation, wall_ngrow
from .ghost_cell import GhostCell
from .no_ibm import NoIbm
from .samples import WallSamples, wall_gradient, wall_samples

_METHODS = {
    "noIbm": NoIbm,
    "directForcing": DirectForcing,
    "ghostCell": GhostCell,
}
# Names that validate the fvSolution schema shape but are not yet
# implemented — cut-cell support is deferred (task T19, gated on the
# analytic-validation gate).
_NOT_IMPLEMENTED = {"cutCell"}


class IBM:
    """Registry mapping an ``fvSolution.solvers[field].ibm`` name to its
    strategy class."""

    @staticmethod
    def names():
        """Every ``solution["ibm"]`` spelling this build validates, sorted.

        Includes the schema-valid but deferred ones (``cutCell``): the list is
        what an unknown name is offered, and it is the method axis of the
        scheme x method grid (verification §5), so a method that validates but
        refuses has to be in it and has to say so when the grid reaches it.
        """
        return sorted(set(_METHODS) | _NOT_IMPLEMENTED)

    @staticmethod
    def lookup(name):
        if name in _METHODS:
            return _METHODS[name]
        if name in _NOT_IMPLEMENTED:
            raise NotImplementedError(
                f"IBM method '{name}' is not implemented (cut-cell support is "
                "deferred); the name validates the schema, execution refuses."
            )
        raise ValueError(f"Unknown IBM method '{name}'; valid methods: {IBM.names()}")


def evaluation(name, cell_field, spatial_ops):
    """Validate the request and build the per-``evaluate`` IBM driver.

    Returns ``None`` when the IBM path is not entered at all — no
    ``solution["ibm"]`` key, the explicit ``noIbm`` opt-out, or **no ``WALL``
    cell on any level** (a body whose surface this mesh does not cut). All three
    are then a short-circuit *outside* any kernel in the caller, which is what
    makes bitwise equality with the plain operator structural rather than
    maintained (design §6).

    Otherwise the driver is the wall flow of :mod:`blockamr.ibm.driver` — the
    only flow there is: the prolong/restrict schedule that ``ghostCell`` ran on
    until its three ``(operator, method)`` pairs existed is gone with W5.
    """
    if name is None:
        return None
    method = IBM.lookup(name)
    if method.kind == "step":
        raise ValueError(
            f"'{name}' does not support operator-level evaluation (it is a step "
            "method); apply it between steps via mesh.build_ibm([...]) instead."
        )
    if method.requires_bodies:
        _validate_patches(name, cell_field)
    if not method.requires_bodies or _no_wall_cell(method, cell_field, spatial_ops):
        return None
    return WallEvaluation(method, name, cell_field, spatial_ops)


def _no_wall_cell(method, cell_field, spatial_ops):
    """True when no level of this mesh has a ``WALL`` cell for ``method``.

    Asked of the **marker**, through the method's own preprocessed data, whose
    row count *is* the level's wall-cell count (design §2.3). v1 asked the same
    question of the equation's band, at the widest of its terms' widths and in
    the stencil shape they declared; the marker has neither, so the question is
    simply "did the classification find a wall here".

    The ghost width is the equation's (:func:`~blockamr.ibm.driver.wall_ngrow`)
    so that the marker built here is the one the driver goes on to use, and the
    monotonic caches in :class:`~blockamr.ibm.mesh.IbmMesh` are not asked to
    grow a second time.
    """
    mesh = cell_field.mesh
    ngrow = wall_ngrow(spatial_ops)
    return not any(mesh.ibm.wall_data(method, lev, ngrow).nrows for lev in range(mesh.n_levels()))


def _validate_patches(name, cell_field):
    """``ibm_bc`` keys must match ``mesh.bodies`` exactly — the patch-keyed
    contract that makes more than one immersed patch expressible."""
    bodies = cell_field.mesh.bodies
    if not bodies:
        raise ValueError(
            f"IBM method '{name}' was requested but mesh.bodies is empty; set "
            "mesh.bodies = {'<patch>': Cylinder(...)} first."
        )
    ibm_bc = cell_field.ibm_bc
    missing = sorted(set(bodies) - set(ibm_bc))
    if missing:
        raise ValueError(
            f"field '{cell_field.name}' has no ibm_bc entry for patch "
            f"{missing[0]!r} (mesh.bodies: {sorted(bodies)})."
        )
    extra = sorted(set(ibm_bc) - set(bodies))
    if extra:
        raise ValueError(
            f"field '{cell_field.name}' has an ibm_bc entry for patch "
            f"{extra[0]!r}, which is not in mesh.bodies ({sorted(bodies)})."
        )


__all__ = [
    "IBM",
    "Cylinder",
    "DirectForcing",
    "DirectForcingData",
    "FixedGradient",
    "FixedValue",
    "GhostCell",
    "Harmonic",
    "Mixed",
    "NoIbm",
    "Plane",
    "WallEvaluation",
    "WallSamples",
    "wall_ngrow",
    "evaluation",
    "wall_gradient",
    "wall_samples",
]
