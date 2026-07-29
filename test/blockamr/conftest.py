# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import gc
import os

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import pytest

import blockamr


@pytest.fixture(scope="session", autouse=True)
def blockamr_session():
    """Initialize and finalize blockAMR once for all tests in this directory.

    The ``gc.collect()`` before the context closes is not hygiene — it is the
    difference between exit 0 and ``CUDA error 709: context is destroyed``
    followed by ``MPI_Abort() after MPI_FINALIZE``.

    A ``Mesh`` and its ``mesh.ibm`` (:class:`~blockamr.ibm.mesh.IbmMesh`) hold
    each other, so every mesh that has met an IBM method is a **reference
    cycle**, and the device memory hanging off it — the packed geometry fabs,
    the markers, the method data, the fields — is freed only when the cycle
    collector runs. Unreached, that happens at *interpreter* shutdown, which is
    after this fixture has finalized AMReX and destroyed the CUDA context;
    freeing a device allocation into a destroyed context aborts.

    Collecting here runs those destructors while AMReX is still up. It is the
    session-wide form of the module-scoped ``_release_the_memoised_levels``
    finalizers the pair suites carry, and it is why those files still need
    theirs: a *live* module global is not garbage, so no collection frees it.

    No assertion, no test and no fixture value depends on this call.
    """
    with blockamr.runtime():
        yield
        gc.collect()
