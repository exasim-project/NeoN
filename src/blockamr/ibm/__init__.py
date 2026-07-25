# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""IBM method registry (API doc §6): ``IBM.lookup(name)`` resolves a
per-field ``solution["ibm"]`` name to a strategy class."""

from .bc import FixedGradient, FixedValue, Mixed
from .body import Cylinder, Plane
from .context import IbmEvaluation
from .direct_forcing import DirectForcing, DirectForcingData
from .ghost_cell import GhostCell
from .no_ibm import NoIbm

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
    def lookup(name):
        if name in _METHODS:
            return _METHODS[name]
        if name in _NOT_IMPLEMENTED:
            raise NotImplementedError(
                f"IBM method '{name}' is not implemented (cut-cell support is "
                "deferred); the name validates the schema, execution refuses."
            )
        valid = sorted(_METHODS) + sorted(_NOT_IMPLEMENTED)
        raise ValueError(f"Unknown IBM method '{name}'; valid methods: {valid}")


def evaluation(name, cell_field):
    """Validate the request and build the per-``evaluate`` IBM driver.

    ``name is None`` (no ``solution["ibm"]`` key) means the IBM path is not
    entered at all.
    """
    if name is None:
        return IbmEvaluation(None, None, cell_field)
    method = IBM.lookup(name)
    if method.kind == "step":
        raise ValueError(
            f"'{name}' does not support operator-level evaluation (it is a step "
            "method); apply it between steps via mesh.build_ibm([...]) instead."
        )
    if method.requires_bodies:
        _validate_patches(name, cell_field)
    return IbmEvaluation(method, name, cell_field)


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
    "IbmEvaluation",
    "Mixed",
    "NoIbm",
    "Plane",
    "evaluation",
]
