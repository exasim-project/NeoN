# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

# Add the lib directory to Python path so we can import the compiled modules
lib_dir = Path.cwd() / "lib"
if lib_dir.exists():
    sys.path.insert(0, str(lib_dir))

import pytest


def test_import():
    """Test that the my_ext module can be imported"""
    try:
        import my_ext  # noqa: F401
        assert True
    except ImportError:
        assert False


def test_serial_executor():
    """Test SerialExecutor creation and methods"""
    import my_ext

    exec = my_ext.SerialExecutor()
    assert exec.name() == "SerialExecutor"
    assert repr(exec) == "<SerialExecutor>"


def test_cpu_executor():
    """Test CPUExecutor creation and methods"""
    import my_ext

    exec = my_ext.CPUExecutor()
    assert exec.name() == "CPUExecutor"
    assert repr(exec) == "<CPUExecutor>"


def test_gpu_executor():
    """Test GPUExecutor creation and methods"""
    import my_ext

    exec = my_ext.GPUExecutor()
    assert exec.name() == "GPUExecutor"
    assert repr(exec) == "<GPUExecutor>"


def test_executor_helper_functions():
    """Test executor helper functions that work with the variant"""
    import my_ext

    # Test with SerialExecutor
    serial_exec = my_ext.SerialExecutor()
    assert my_ext.executor_name(serial_exec) == "SerialExecutor"
    assert my_ext.executor_repr(serial_exec) == "<Executor: SerialExecutor>"

    # Test with CPUExecutor
    cpu_exec = my_ext.CPUExecutor()
    assert my_ext.executor_name(cpu_exec) == "CPUExecutor"
    assert my_ext.executor_repr(cpu_exec) == "<Executor: CPUExecutor>"

    # Test with GPUExecutor
    gpu_exec = my_ext.GPUExecutor()
    assert my_ext.executor_name(gpu_exec) == "GPUExecutor"
    assert my_ext.executor_repr(gpu_exec) == "<Executor: GPUExecutor>"


def test_executor_equality():
    """Test executor equality comparison"""
    import my_ext

    exec1 = my_ext.SerialExecutor()
    exec2 = my_ext.SerialExecutor()
    exec3 = my_ext.CPUExecutor()

    # Same type executors should be equal
    assert exec1 == exec2

    # Different type executors have different types
    assert type(exec1) != type(exec3)
    assert type(exec2) != type(exec3)
