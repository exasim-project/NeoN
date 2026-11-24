# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import gc
from . import initialize, finalize

# AMReX may only be Initialize()'d once per process, so nested ``runtime()`` blocks
# count depth and only the outermost one init/finalizes.
_depth = 0


def initialized():
    """True if an AMReX runtime is currently active in this process."""
    return _depth > 0


class _RuntimeCtx:
    """Context manager for AMReX initialization and finalization."""

    def __enter__(self):
        global _depth
        if _depth == 0:
            initialize()
        _depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _depth
        _depth -= 1
        if _depth == 0:
            gc.collect()
            finalize()
        return False


def runtime(func=None):
    """Run *func* inside an AMReX session, or return a context manager.

    Prefer the callback form: *func*'s locals are destroyed before ``finalize()``, so
    GPU resources are freed while the CUDA context is still alive::

        def run():
            mesh = AmrMesh(geom, info)
            ...

        blockamr.runtime(run)

    In the context-manager form the CALLER must keep AMReX-backed objects from
    outliving the block, e.g. by building them inside a helper function::

        with blockamr.runtime():
            run()
    """
    if func is not None:
        global _depth
        if _depth == 0:
            initialize()
        _depth += 1
        try:
            func()
        finally:
            _depth -= 1
            if _depth == 0:
                gc.collect()
                finalize()
    else:
        return _RuntimeCtx()
