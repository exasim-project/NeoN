# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Process-level lifetime of the AMReX/Kokkos runtime, checked in fresh subprocesses."""

import re
import subprocess
import sys

# Opens the runtime and never exits it, keeping an AMReX object alive at module scope.
_OPEN_RUNTIME = """
import blockamr
blockamr.runtime().__enter__()
ba = blockamr.BoxArray(blockamr.Box([0, 0, 0], [15, 15, 15]))
mf = blockamr.MultiFab(ba, blockamr.DistributionMapping(ba), 1, 0)
"""


def _run(script):
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )


def test_runtime_left_open_exits_cleanly():
    result = _run(_OPEN_RUNTIME)
    assert result.returncode == 0, result.stdout + result.stderr
