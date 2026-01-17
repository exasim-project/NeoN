# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest
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


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if GPU is not available."""
    if not neon.gpu_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:  # Finds @pytest.mark.gpu
                item.add_marker(skip_gpu)  # Adds the skip automatically
