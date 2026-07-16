# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import gc
from . import initialize, finalize

# Re-entrancy guard: AMReX may only be Initialize()'d once per process. Nested
# ``runtime()`` blocks (e.g. a pytest session fixture wrapping a solver ``run()``
# that also opens a runtime) count depth so only the outermost init/finalizes.
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

    Preferred (callback) form — locals in *func* are destroyed before
    ``finalize()``, so GPU resources are freed while the CUDA context is
    still alive::

        def run():
            mesh = AmrMesh(geom, info)
            ...

        blockamr.runtime(run)

    Context-manager form (caller must ensure AMReX-backed objects do not
    outlive the block, e.g. by calling them inside a helper function)::

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
