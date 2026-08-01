# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Map a parametrised executor name onto a NeoN executor object.

blockamr's solver entry points take a NeoN executor -- SerialExecutor,
CPUExecutor or GPUExecutor -- rather than a string, so that blockAMR and the
rest of NeoN resolve to the same memoized Ginkgo executor (and the same stream)
via NeoN::la::ginkgo::getGkoExecutor.

The tests keep parametrising over the short names: they read better as pytest
ids than a repr does, and the skip logic has to know which case is the GPU one
so it can skip rather than fail on a machine without a device.

The map holds CLASSES, not instances -- an executor is constructed when a test
asks for one, not at collection time, which would run before the blockamr
runtime fixture has initialised Kokkos.
"""

import neon

_BY_NAME = {
    "reference": neon.SerialExecutor,
    "cpu": neon.CPUExecutor,
    "cuda": neon.GPUExecutor,
}


def gko_executor(name):
    """The NeoN executor for a parametrised name such as "reference" or "cuda"."""
    return _BY_NAME[name]()
