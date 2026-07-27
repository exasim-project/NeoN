# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""IBM method registry (API doc §6): ``IBM.lookup(name)`` resolves a
per-field ``solution["ibm"]`` name to a strategy class."""

from .band_rows import BandRows, band_table
from .bc import FixedGradient, FixedValue, Mixed
from .body import Cylinder, Plane
from .direct_forcing import DirectForcing, DirectForcingData
from .driver import BandEvaluation, equation_band
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
    ``solution["ibm"]`` key, the explicit ``noIbm`` opt-out, or an **empty
    band** (a body whose boundary cells are none of this mesh's). All three are
    then a short-circuit *outside* any kernel in the caller, which is what
    makes bitwise equality with the plain operator structural rather than
    maintained (design §6).

    Otherwise the driver is the band flow of :mod:`blockamr.ibm.driver` — the
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
    if not method.requires_bodies or _band_is_empty(cell_field, spatial_ops):
        return None
    return BandEvaluation(method, name, cell_field, spatial_ops)


def _band_is_empty(cell_field, spatial_ops):
    """True when the equation's band has no cell to correct, on any level.

    Asked of the *equation's* band — the widest of its terms', in the stencil
    shape they declare (:func:`~blockamr.ibm.driver.equation_band`) — because
    that is the one set the rows are built over. Taking the default cross band
    of width 1 instead would short-circuit a corner-reading or a wide scheme
    whose band is not empty.

    The classification is the method-agnostic layer, so this is decided from
    ``mesh.ibm`` alone — before a method's own preprocessing runs and before a
    single row is built.
    """
    mesh = cell_field.mesh
    width, shape = equation_band(spatial_ops)
    return not any(mesh.ibm.band(lev, width, shape).nrows for lev in range(mesh.n_levels()))


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
    "BandEvaluation",
    "BandRows",
    "Cylinder",
    "DirectForcing",
    "DirectForcingData",
    "FixedGradient",
    "FixedValue",
    "GhostCell",
    "Mixed",
    "NoIbm",
    "Plane",
    "band_table",
    "evaluation",
]
