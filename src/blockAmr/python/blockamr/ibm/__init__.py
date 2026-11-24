# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""IBM method registry: ``IBM.lookup(name)`` resolves a per-field ``solution["ibm"]``
name to a strategy class."""

from .body import Cylinder
from .direct_forcing import DirectForcing, DirectForcingData

_METHODS = {"directForcing": DirectForcing}
# Valid fvSolution spellings whose EB (cut-cell) support is deferred on this branch.
_NOT_IMPLEMENTED = {"cutCell", "ghostCell"}


class IBM:
    """Maps an ``fvSolution.solvers[field].ibm`` name to its strategy class."""

    @staticmethod
    def lookup(name):
        if name in _METHODS:
            return _METHODS[name]
        if name in _NOT_IMPLEMENTED:
            raise NotImplementedError(
                f"IBM method '{name}' is not implemented (EB support is deferred "
                "on this branch); the name validates the schema, execution refuses."
            )
        valid = sorted(_METHODS) + sorted(_NOT_IMPLEMENTED)
        raise ValueError(f"Unknown IBM method '{name}'; valid methods: {valid}")


__all__ = ["IBM", "Cylinder", "DirectForcing", "DirectForcingData"]
