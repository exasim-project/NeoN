# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from .base import ExplicitBackend
from .cpp_backend import CppBackend
from .jax_backend import JaxBackend

# Explicit-path backend registry: `jax` (Pallas GPU) and `cpp` (composable
# AMReX ParallelFor kernels).
backends = {"jax": JaxBackend(), "cpp": CppBackend()}


def get(name):
    """Return the explicit backend for `name`, listing valid names on error."""
    try:
        return backends[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; valid backends: {sorted(backends)}") from None


__all__ = ["ExplicitBackend", "JaxBackend", "CppBackend", "backends", "get"]
