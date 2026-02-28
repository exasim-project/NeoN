// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/io/vtuMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshReader.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerIO(nb::module_& m)
{
    m.def(
        "write_vtu",
        &NeoN::io::writeVtu,
        "mesh"_a,
        "filepath"_a,
        "Write mesh to VTU (VTK XML unstructured grid) format."
    );

    m.def("write_cgns", &NeoN::io::writeCgns, "mesh"_a, "filepath"_a, "Write mesh to CGNS format.");

    m.def(
        "write_vtk_hdf",
        &NeoN::io::writeVtkHdf,
        "mesh"_a,
        "filepath"_a,
        "Write mesh to VTK HDF5 format."
    );

    m.def("read_cgns", &NeoN::io::readCgns, "filepath"_a, "exec"_a, "Read mesh from CGNS file.");

    m.def(
        "read_vtk_hdf",
        &NeoN::io::readVtkHdf,
        "filepath"_a,
        "exec"_a,
        "Read mesh from VTK HDF5 file."
    );
}

} // namespace NeoN::bindings
