# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
NeoN - A framework for CFD software

Python bindings for the NeoN CFD framework.
"""

__version__ = "0.1.0"

# Import the C++ extension module (optional — not needed for blockamr-only usage)
try:
    from ._neon import *  # noqa: F401, F403
except ImportError:
    pass

__all__ = ["__version__"]
