# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Scheme-name registry: the names appearing in fvSchemes (e.g. ``div(phi,U) vanLeer;``)
mapped to the engine's scheme classes. Lookups are case-insensitive.
"""

from .ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from .div_schemes import Linear, QUICK, Upwind, VanLeer
from .grad_schemes import CentralDiffGrad
from .laplacian_schemes import CentralDiffLaplacian

SCHEME_REGISTRY = {
    "div": {
        "upwind": Upwind,
        "linear": Linear,
        "vanLeer": VanLeer,
        "quick": QUICK,
    },
    "ddt": {
        "Euler": ForwardEuler,
        "RK2": RungeKutta2,
        "RK4": RungeKutta4,
    },
    "laplacian": {
        "central": CentralDiffLaplacian,
    },
    "grad": {
        "central": CentralDiffGrad,
    },
}


def resolve(operator, name):
    """Return the scheme class for *name* (case-insensitive) of *operator*."""
    table = SCHEME_REGISTRY[operator]
    for spelling, cls in table.items():
        if spelling.lower() == str(name).lower():
            return cls
    valid = ", ".join(table)
    raise ValueError(f"Unknown {operator} scheme '{name}'. Valid options: {valid}")


def lookup_scheme(schemes, keys, operator, default):
    """Resolve a term's scheme from a *schemes* mapping.

    Tries each non-None key in *keys*, then ``"default"``. String values are resolved
    through the registry and INSTANTIATED; scheme objects pass through unchanged.
    Returns *default* when nothing matches or *schemes* is empty.
    """
    if not schemes:
        return default
    for key in [k for k in keys if k is not None] + ["default"]:
        if key in schemes:
            value = schemes[key]
            if isinstance(value, str):
                return resolve(operator, value)()
            return value
    return default
