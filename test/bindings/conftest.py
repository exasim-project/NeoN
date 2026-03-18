# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path
import importlib.util

_neon_so_dir = os.environ.get("NEON_BINDINGS_PATH")
if _neon_so_dir:

    _project_root = Path(__file__).parents[2]
    _init_file = _project_root / "src" / "neon" / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        "neon",
        str(_init_file),
        submodule_search_locations=[_neon_so_dir],
    )
    _neon_module = importlib.util.module_from_spec(spec)
    sys.modules["neon"] = _neon_module
    spec.loader.exec_module(_neon_module)

import pytest

_neon_available = False
try:
    import neon
    neon.initialize  # check that bindings loaded
    _neon_available = True
except (ImportError, AttributeError):
    pass

_blockamr_available = False
try:
    import blockamr
    _blockamr_available = True
except ImportError:
    pass


@pytest.fixture(scope="session", autouse=True)
def neon_global_session():
    """Initialize NeoN and/or blockAMR once for all test files in this session."""
    if _neon_available:
        neon.initialize()
    if _blockamr_available:
        blockamr.initialize()
    yield  # This is where all tests run
    if _blockamr_available:
        blockamr.finalize()
    if _neon_available:
        neon.finalize()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU support")


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if GPU is not available."""
    if not _neon_available or not neon.gpu_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:  # Finds @pytest.mark.gpu
                item.add_marker(skip_gpu)  # Adds the skip automatically
