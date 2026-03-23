# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import gc
from . import initialize, finalize


class _RuntimeCtx:
    """Context manager for AMReX initialization and finalization."""

    def __enter__(self):
        initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):


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
        initialize()
        try:
            func()
        finally:
            gc.collect()
            finalize()
    else:
        return _RuntimeCtx()
