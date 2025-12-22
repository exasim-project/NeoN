# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
lib_dir = Path.cwd() / "neon"
if lib_dir.exists():
    sys.path.insert(0, str(lib_dir))

import pytest
import neon


def test_import():
    assert neon is not None


def test_serial_executor():
    exec = neon.SerialExecutor()
    assert exec.name() == "SerialExecutor"
    assert repr(exec) == "<SerialExecutor>"


def test_cpu_executor():
    exec = neon.CPUExecutor()
    assert exec.name() == "CPUExecutor"
    assert repr(exec) == "<CPUExecutor>"


def test_gpu_executor():
    try:
        exec = neon.GPUExecutor()
        assert exec.name() == "GPUExecutor"
        assert repr(exec) == "<GPUExecutor>"
    except RuntimeError:
        pytest.skip("GPU not available")


def test_executor_helpers():
    serial = neon.SerialExecutor()
    assert neon.executor_name(serial) == "SerialExecutor"
    assert neon.executor_repr(serial) == "<Executor: SerialExecutor>"

    cpu = neon.CPUExecutor()
    assert neon.executor_name(cpu) == "CPUExecutor"
    assert neon.executor_repr(cpu) == "<Executor: CPUExecutor>"

    try:
        gpu = neon.GPUExecutor()
        assert neon.executor_name(gpu) == "GPUExecutor"
        assert neon.executor_repr(gpu) == "<Executor: GPUExecutor>"
    except RuntimeError:
        pass


def test_executor_equality():
    exec1 = neon.SerialExecutor()
    exec2 = neon.SerialExecutor()
    exec3 = neon.CPUExecutor()
    exec4 = neon.CPUExecutor()

    # same type executors are equal
    assert exec1 == exec2
    assert exec3 == exec4

    # different types are distinguishable
    assert type(exec1) != type(exec3)
