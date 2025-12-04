# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

# Add the neon directory to Python path so we can import the compiled modules
lib_dir = Path.cwd() / "neon"
if lib_dir.exists():
    sys.path.insert(0, str(lib_dir))

import pytest


def test_import():
    """Test that the neon module can be imported"""
    try:
        import neon  # noqa: F401
        assert True
    except ImportError:
        assert False


def test_serial_executor():
    """Test SerialExecutor creation and methods"""
    import neon

    exec = neon.SerialExecutor()
    assert exec.name() == "SerialExecutor"
    assert repr(exec) == "<SerialExecutor>"


def test_cpu_executor():
    """Test CPUExecutor creation and methods"""
    import neon

    exec = neon.CPUExecutor()
    assert exec.name() == "CPUExecutor"
    assert repr(exec) == "<CPUExecutor>"


def test_gpu_executor():
    """Test GPUExecutor creation and methods"""
    import neon

    exec = neon.GPUExecutor()
    assert exec.name() == "GPUExecutor"
    assert repr(exec) == "<GPUExecutor>"


def test_executor_helper_functions():
    """Test executor helper functions that work with the variant"""
    import neon

    # Test with SerialExecutor
    serial_exec = neon.SerialExecutor()
    assert neon.executor_name(serial_exec) == "SerialExecutor"
    assert neon.executor_repr(serial_exec) == "<Executor: SerialExecutor>"

    # Test with CPUExecutor
    cpu_exec = neon.CPUExecutor()
    assert neon.executor_name(cpu_exec) == "CPUExecutor"
    assert neon.executor_repr(cpu_exec) == "<Executor: CPUExecutor>"

    # Test with GPUExecutor
    gpu_exec = neon.GPUExecutor()
    assert neon.executor_name(gpu_exec) == "GPUExecutor"
    assert neon.executor_repr(gpu_exec) == "<Executor: GPUExecutor>"


def test_executor_equality():
    """Test executor equality comparison"""
    import neon

    exec1 = neon.SerialExecutor()
    exec2 = neon.SerialExecutor()
    exec3 = neon.CPUExecutor()

    # Same type executors should be equal
    assert exec1 == exec2

    # Different type executors have different types
    assert type(exec1) != type(exec3)
    assert type(exec2) != type(exec3)
