# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Process-level lifetime of the AMReX/Kokkos runtime, checked in fresh subprocesses."""

import os
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
    # Without the override, so the built-in arena default is what gets exercised.
    env = {k: v for k, v in os.environ.items() if k != "AMREX_THE_ARENA_INIT_SIZE"}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, env=env
    )


def test_runtime_left_open_exits_cleanly():
    result = _run(_OPEN_RUNTIME)
    assert result.returncode == 0, result.stdout + result.stderr


def test_arena_does_not_reserve_the_device_up_front():
    result = _run(_OPEN_RUNTIME)
    assert result.returncode == 0, result.stdout + result.stderr
    # AMReX reports these at Finalize; the device line is absent on CPU builds.
    total = re.search(r"Total GPU global memory \(MB\).*?\[(\d+)", result.stdout)
    allocated = re.search(r"\[The\s+Arena\] max space \(MB\) allocated.*?\[(\d+)", result.stdout)
    assert allocated is not None, result.stdout
    if total is not None:
        assert int(allocated.group(1)) < int(total.group(1)) // 4
