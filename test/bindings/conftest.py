# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Test configuration for NeoN Python bindings.

This module automatically locates the neon package from the build directory
and configures pytest fixtures for executor parameterization.
"""

import pytest

# Set up path before importing neon
import neon


@pytest.fixture(scope="session", autouse=True)
def neon_global_session():
    """Initialize NeoN once for all test files in this session."""
    neon.initialize()
    yield  # This is where all tests run
    neon.finalize()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU support")


def get_available_executors():
    """Get list of executors available at build time."""
    executors = []
    if neon.__has_serial__:
        executors.append(("serial", neon.SerialExecutor))
    if neon.__has_cpu__:
        executors.append(("cpu", neon.CPUExecutor))
    if neon.__has_gpu__:
        executors.append(("gpu", neon.GPUExecutor))
    return executors


def pytest_generate_tests(metafunc):
    """Auto-parameterize tests using 'executor' fixture."""
    if "executor" in metafunc.fixturenames:
        available = get_available_executors()
        if not available:
            pytest.skip("No executors available")
        # Create instances directly in parametrization
        executor_instances = [(name, exec_class()) for name, exec_class in available]
        metafunc.parametrize(
            "executor",
            executor_instances,
            ids=[name for name, _ in available]
        )


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if GPU is not available."""
    if hasattr(neon, '__has_gpu__') and not neon.__has_gpu__:
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
