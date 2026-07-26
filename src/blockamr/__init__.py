# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax

jax.config.update("jax_enable_x64", True)

from ._blockamr import *
from .field import CellField, FaceField, Field, NodalField, PatchData, _FaceFieldLevel
from .fillpatch import FillPatchCellConservative, FillPatchSingleLevel
from .mesh import AmrMesh, Mesh
from . import dsl
from . import schemes
from .runtime import initialized, runtime

_default_executor = "cpu"
_default_backend = "jax"


def set_executor(executor):
    global _default_executor
    _default_executor = executor
    jax.config.update("jax_platform_name", "gpu" if executor == "gpu" else "cpu")


def get_executor():
    return _default_executor


def set_tile_size(bf):
    """Set the Pallas tile size (default 8). Must be a power of 2."""
    from .dsl.solve import set_tile_size as _set
    _set(bf)


def set_backend(backend):
    """Set the default dispatch backend.

    Parameters
    ----------
    backend : str
        "jax"    — jax.vmap over 3D arrays (default)
        "pallas" — Pallas 3D tiled GPU dispatch
        "triton" — Triton kernels with phi(ptr, i, j, k, sx, sy)
    """
    global _default_backend
    if backend not in ("jax", "pallas", "triton"):
        raise ValueError(f"Unknown backend: {backend!r}. "
                         f"Choose from 'jax', 'pallas', 'triton'.")
    _default_backend = backend


def get_backend():
    return _default_backend
