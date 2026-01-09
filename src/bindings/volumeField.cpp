// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerVolumeField(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    nb::class_<fvcc::VolumeBoundary<NeoN::scalar>>(
        m, "ScalarVolumeBoundary", "Volume boundary for scalar fields"
    )
        .def("patch_id", &fvcc::VolumeBoundary<NeoN::scalar>::patchID)
        .def("patch_size", &fvcc::VolumeBoundary<NeoN::scalar>::patchSize);

    nb::class_<fvcc::VolumeBoundary<NeoN::Vec3>>(
        m, "VectorVolumeBoundary", "Volume boundary for Vec3 fields"
    )
        .def("patch_id", &fvcc::VolumeBoundary<NeoN::Vec3>::patchID)
        .def("patch_size", &fvcc::VolumeBoundary<NeoN::Vec3>::patchSize);

    nb::class_<fvcc::VolumeField<NeoN::scalar>>(
        m, "ScalarVolumeField", "Volume field for scalar values"
    )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::scalar>& field,
               const NeoN::SerialExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::scalar>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a scalar VolumeField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::scalar>& field,
               const NeoN::CPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::scalar>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a scalar VolumeField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::scalar>& field,
               const NeoN::GPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::scalar>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a scalar VolumeField with calculated boundary conditions"
        )
        .def(
            nb::init<
                const NeoN::Executor&,
                std::string,
                const NeoN::UnstructuredMesh&,
                const std::vector<fvcc::VolumeBoundary<NeoN::scalar>>&>(),
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "boundary_conditions"_a,
            "Create a scalar VolumeField with custom boundary conditions"
        )
        .def(
            "internal_vector",
            static_cast<NeoN::Vector<NeoN::scalar>& (fvcc::VolumeField<NeoN::scalar>::*)()>(
                &fvcc::VolumeField<NeoN::scalar>::internalVector
            ),
            nb::rv_policy::reference_internal,
            "Get the internal vector"
        )
        .def(
            "mesh",
            &fvcc::VolumeField<NeoN::scalar>::mesh,
            nb::rv_policy::reference_internal,
            "Get the mesh"
        )
        .def(
            "exec",
            &fvcc::VolumeField<NeoN::scalar>::exec,
            nb::rv_policy::reference_internal,
            "Get the executor"
        )
        .def("size", &fvcc::VolumeField<NeoN::scalar>::size, "Get the field size")
        .def(
            "correct_boundary_conditions",
            &fvcc::VolumeField<NeoN::scalar>::correctBoundaryConditions,
            "Apply boundary conditions"
        )
        .def("has_database", &fvcc::VolumeField<NeoN::scalar>::hasDatabase)
        .def_rw("name", &fvcc::VolumeField<NeoN::scalar>::name);

    nb::class_<fvcc::VolumeField<NeoN::Vec3>>(
        m, "VectorVolumeField", "Volume field for Vec3 values"
    )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::Vec3>& field,
               const NeoN::SerialExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh);
                new (&field
                ) fvcc::VolumeField<NeoN::Vec3>(NeoN::Executor {exec}, std::move(name), mesh, bcs);
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a Vec3 VolumeField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::Vec3>& field,
               const NeoN::CPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh);
                new (&field
                ) fvcc::VolumeField<NeoN::Vec3>(NeoN::Executor {exec}, std::move(name), mesh, bcs);
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a Vec3 VolumeField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::Vec3>& field,
               const NeoN::GPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh);
                new (&field
                ) fvcc::VolumeField<NeoN::Vec3>(NeoN::Executor {exec}, std::move(name), mesh, bcs);
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a Vec3 VolumeField with calculated boundary conditions"
        )
        .def(
            nb::init<
                const NeoN::Executor&,
                std::string,
                const NeoN::UnstructuredMesh&,
                const std::vector<fvcc::VolumeBoundary<NeoN::Vec3>>&>(),
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "boundary_conditions"_a
        )
        .def(
            "internal_vector",
            static_cast<NeoN::Vector<NeoN::Vec3>& (fvcc::VolumeField<NeoN::Vec3>::*)()>(
                &fvcc::VolumeField<NeoN::Vec3>::internalVector
            ),
            nb::rv_policy::reference_internal
        )
        .def("mesh", &fvcc::VolumeField<NeoN::Vec3>::mesh, nb::rv_policy::reference_internal)
        .def("exec", &fvcc::VolumeField<NeoN::Vec3>::exec, nb::rv_policy::reference_internal)
        .def("size", &fvcc::VolumeField<NeoN::Vec3>::size)
        .def(
            "correct_boundary_conditions", &fvcc::VolumeField<NeoN::Vec3>::correctBoundaryConditions
        )
        .def("has_database", &fvcc::VolumeField<NeoN::Vec3>::hasDatabase)
        .def_rw("name", &fvcc::VolumeField<NeoN::Vec3>::name);

    m.def(
        "create_calculated_volume_bcs_scalar",
        [](const NeoN::UnstructuredMesh& mesh)
        { return fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh); },
        "mesh"_a,
        "Create calculated scalar volume boundary conditions"
    );

    m.def(
        "create_calculated_volume_bcs_vec3",
        [](const NeoN::UnstructuredMesh& mesh)
        { return fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh); },
        "mesh"_a,
        "Create calculated Vec3 volume boundary conditions"
    );
}

} // namespace NeoN::bindings
