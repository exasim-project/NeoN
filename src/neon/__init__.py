# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
NeoN - A framework for CFD software

Python bindings for the NeoN CFD framework.
"""

__version__ = "0.1.0"

# Import the C++ extension module
try:
    from ._neon import *  # noqa: F401, F403
except ImportError as e:
    raise ImportError(
        "Failed to import NeoN C++ extension module. "
        "Make sure the package is properly installed. "
        f"Error: {e}"
    ) from e

__all__ = ["__version__"]
