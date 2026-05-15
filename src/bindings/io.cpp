// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/io/vtmMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/fieldWriter.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerIO(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    // --- mesh-only writers ---

    m.def(
        "write_vtm",
        [](const NeoN::UnstructuredMesh& mesh, const std::string& filePath)
        { NeoN::io::writeVtm(mesh, filePath); },
        "mesh"_a,
        "filepath"_a,
        "Write mesh to VTM (multi-block) format with named boundary patches."
    );

    m.def("write_cgns", &NeoN::io::writeCgns, "mesh"_a, "filepath"_a, "Write mesh to CGNS format.");

    m.def(
        "write_vtk_hdf",
        [](const NeoN::UnstructuredMesh& mesh, const std::string& filePath)
        { NeoN::io::writeVtkHdf(mesh, filePath); },
        "mesh"_a,
        "filepath"_a,
        "Write mesh to VTK HDF5 format."
    );

    // --- writers with scalar VolumeField ---

    m.def(
        "write_vtm",
        [](const NeoN::UnstructuredMesh& mesh,
           const fvcc::VolumeField<NeoN::scalar>& field,
           const std::string& filePath) { NeoN::io::writeVtm(mesh, field, filePath); },
        "mesh"_a,
        "field"_a,
        "filepath"_a,
        "Write mesh + scalar VolumeField to VTM format."
    );

    m.def(
        "write_vtk_hdf",
        [](const NeoN::UnstructuredMesh& mesh,
           const fvcc::VolumeField<NeoN::scalar>& field,
           const std::string& filePath) { NeoN::io::writeVtkHdf(mesh, field, filePath); },
        "mesh"_a,
        "field"_a,
        "filepath"_a,
        "Write mesh + scalar VolumeField to VTK HDF5 format."
    );

    // --- writers with Vec3 VolumeField ---

    m.def(
        "write_vtm",
        [](const NeoN::UnstructuredMesh& mesh,
           const fvcc::VolumeField<NeoN::Vec3>& field,
           const std::string& filePath) { NeoN::io::writeVtm(mesh, field, filePath); },
        "mesh"_a,
        "field"_a,
        "filepath"_a,
        "Write mesh + Vec3 VolumeField to VTM format."
    );

    m.def(
        "write_vtk_hdf",
        [](const NeoN::UnstructuredMesh& mesh,
           const fvcc::VolumeField<NeoN::Vec3>& field,
           const std::string& filePath) { NeoN::io::writeVtkHdf(mesh, field, filePath); },
        "mesh"_a,
        "field"_a,
        "filepath"_a,
        "Write mesh + Vec3 VolumeField to VTK HDF5 format."
    );

    // --- FieldSet class ---

    nb::class_<NeoN::io::FieldSet>(
        m, "FieldSet", "Accumulate VolumeFields for a single write pass."
    )
        .def(nb::init<>())
        .def(
            "add_field",
            static_cast<NeoN::io::FieldSet& (NeoN::io::FieldSet::*)(const fvcc::VolumeField<
                                                                    NeoN::scalar>&)>(
                &NeoN::io::FieldSet::add
            ),
            "field"_a,
            nb::rv_policy::reference,
            "Add a scalar VolumeField."
        )
        .def(
            "add_field",
            static_cast<
                NeoN::io::FieldSet& (NeoN::io::FieldSet::*)(const fvcc::VolumeField<NeoN::Vec3>&)>(
                &NeoN::io::FieldSet::add
            ),
            "field"_a,
            nb::rv_policy::reference,
            "Add a Vec3 VolumeField."
        );

    // --- writers with FieldSet ---

    m.def(
        "write_vtm",
        [](const NeoN::UnstructuredMesh& mesh,
           const NeoN::io::FieldSet& fs,
           const std::string& filePath) { NeoN::io::writeVtm(mesh, fs, filePath); },
        "mesh"_a,
        "fields"_a,
        "filepath"_a,
        "Write mesh + FieldSet to VTM format."
    );

    m.def(
        "write_vtk_hdf",
        [](const NeoN::UnstructuredMesh& mesh,
           const NeoN::io::FieldSet& fs,
           const std::string& filePath) { NeoN::io::writeVtkHdf(mesh, fs, filePath); },
        "mesh"_a,
        "fields"_a,
        "filepath"_a,
        "Write mesh + FieldSet to VTK HDF5 format."
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
