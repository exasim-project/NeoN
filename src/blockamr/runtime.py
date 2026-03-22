# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import contextlib
import gc


@contextlib.contextmanager
def runtime():
    """Context manager for AMReX initialization and finalization.

    Ensures all C++ objects are garbage-collected before the CUDA context
    is destroyed, preventing device-memory-after-finalize crashes.

    Usage::

        with blockamr.runtime():
            mesh = AmrMesh(geom, info)
            ...
    """
    from . import initialize, finalize

    initialize()
    try:
        yield
    finally:
        gc.collect()
        finalize()
