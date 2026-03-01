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

class MeshWriter:
    """Context manager: write a mesh with multiple VolumeFields in one pass.

    Args:
        mesh:     The UnstructuredMesh to write.
        filepath: Output file path.
        fmt:      ``"vtm"`` (default, XML multiblock) or ``"vtkhdf"`` (HDF5).

    Example::

        with neon.MeshWriter(mesh, "output.vtm") as w:
            w.add_field(pressure)
            w.add_field(velocity)
    """

    def __init__(self, mesh, filepath: str, fmt: str = "vtm"):
        self._mesh = mesh
        self._filepath = filepath
        self._fmt = fmt
        self._fields = FieldSet()

    def add_field(self, field):
        """Add a ScalarVolumeField or VectorVolumeField."""
        self._fields.add_field(field)
        return self

    def write(self):
        """Write the mesh and all registered fields to disk."""
        if self._fmt == "vtm":
            write_vtm(self._mesh, self._fields, self._filepath)
        elif self._fmt == "vtkhdf":
            write_vtk_hdf(self._mesh, self._fields, self._filepath)
        else:
            raise ValueError(f"Unknown format {self._fmt!r}. Use 'vtm' or 'vtkhdf'.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.write()
        return False


__all__ = ["__version__"]
