// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerSurfaceField(nb::module_& m)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    nb::class_<fvcc::SurfaceBoundary<NeoN::scalar>>(
        m, "ScalarSurfaceBoundary", "Surface boundary for scalar fields"
    )
        .def("patch_id", &fvcc::SurfaceBoundary<NeoN::scalar>::patchID)
        .def("patch_size", &fvcc::SurfaceBoundary<NeoN::scalar>::patchSize);

    nb::class_<fvcc::SurfaceBoundary<NeoN::Vec3>>(
        m, "VectorSurfaceBoundary", "Surface boundary for Vec3 fields"
    )
        .def("patch_id", &fvcc::SurfaceBoundary<NeoN::Vec3>::patchID)
        .def("patch_size", &fvcc::SurfaceBoundary<NeoN::Vec3>::patchSize);

    nb::class_<fvcc::SurfaceField<NeoN::scalar>>(
        m, "ScalarSurfaceField", "Surface field for scalar values"
    )
        .def(
            "__init__",
            [](fvcc::SurfaceField<NeoN::scalar>& field,
               const NeoN::SerialExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
                new (&field) fvcc::SurfaceField<NeoN::scalar>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a scalar SurfaceField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceField<NeoN::scalar>& field,
               const NeoN::CPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
                new (&field) fvcc::SurfaceField<NeoN::scalar>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a scalar SurfaceField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceField<NeoN::scalar>& field,
               const NeoN::GPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
                new (&field) fvcc::SurfaceField<NeoN::scalar>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a scalar SurfaceField with calculated boundary conditions"
        )
        .def(
            nb::init<
                const NeoN::Executor&,
                std::string,
                const NeoN::UnstructuredMesh&,
                const std::vector<fvcc::SurfaceBoundary<NeoN::scalar>>&>(),
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "boundary_conditions"_a,
            "Create a scalar SurfaceField with custom boundary conditions"
        )
        .def(
            "internal_vector",
            static_cast<NeoN::Vector<NeoN::scalar>& (fvcc::SurfaceField<NeoN::scalar>::*)()>(
                &fvcc::SurfaceField<NeoN::scalar>::internalVector
            ),
            nb::rv_policy::reference_internal,
            "Get the internal vector"
        )
        .def(
            "mesh",
            &fvcc::SurfaceField<NeoN::scalar>::mesh,
            nb::rv_policy::reference_internal,
            "Get the mesh"
        )
        .def(
            "exec",
            &fvcc::SurfaceField<NeoN::scalar>::exec,
            nb::rv_policy::reference_internal,
            "Get the executor"
        )
        .def(
            "boundary_data_value",
            [](fvcc::SurfaceField<NeoN::scalar>& field) -> NeoN::Vector<NeoN::scalar>&
            { return field.boundaryData().value(); },
            nb::rv_policy::reference_internal,
            "Get the boundary data value vector"
        )
        .def("size", &fvcc::SurfaceField<NeoN::scalar>::size, "Get the field size")
        .def(
            "correct_boundary_conditions",
            &fvcc::SurfaceField<NeoN::scalar>::correctBoundaryConditions,
            "Apply boundary conditions"
        )
        .def_rw("name", &fvcc::SurfaceField<NeoN::scalar>::name)
        .def(
            "__add__",
            [](const fvcc::SurfaceField<NeoN::scalar>& a, const fvcc::SurfaceField<NeoN::scalar>& b)
            { return a + b; },
            "other"_a,
            "Element-wise addition of two scalar surface fields"
        )
        .def(
            "__mul__",
            [](const fvcc::SurfaceField<NeoN::scalar>& a, const fvcc::SurfaceField<NeoN::scalar>& b)
            { return a * b; },
            "other"_a,
            "Element-wise multiplication of two scalar surface fields"
        );

    nb::class_<fvcc::SurfaceField<NeoN::Vec3>>(
        m, "VectorSurfaceField", "Surface field for Vec3 values"
    )
        .def(
            "__init__",
            [](fvcc::SurfaceField<NeoN::Vec3>& field,
               const NeoN::SerialExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::Vec3>>(mesh);
                new (&field
                ) fvcc::SurfaceField<NeoN::Vec3>(NeoN::Executor {exec}, std::move(name), mesh, bcs);
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a Vec3 SurfaceField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceField<NeoN::Vec3>& field,
               const NeoN::CPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::Vec3>>(mesh);
                new (&field
                ) fvcc::SurfaceField<NeoN::Vec3>(NeoN::Executor {exec}, std::move(name), mesh, bcs);
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a Vec3 SurfaceField with calculated boundary conditions"
        )
        .def(
            "__init__",
            [](fvcc::SurfaceField<NeoN::Vec3>& field,
               const NeoN::GPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::Vec3>>(mesh);
                new (&field
                ) fvcc::SurfaceField<NeoN::Vec3>(NeoN::Executor {exec}, std::move(name), mesh, bcs);
            },
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "Create a Vec3 SurfaceField with calculated boundary conditions"
        )
        .def(
            nb::init<
                const NeoN::Executor&,
                std::string,
                const NeoN::UnstructuredMesh&,
                const std::vector<fvcc::SurfaceBoundary<NeoN::Vec3>>&>(),
            "exec"_a,
            "name"_a,
            "mesh"_a,
            "boundary_conditions"_a
        )
        .def(
            "internal_vector",
            static_cast<NeoN::Vector<NeoN::Vec3>& (fvcc::SurfaceField<NeoN::Vec3>::*)()>(
                &fvcc::SurfaceField<NeoN::Vec3>::internalVector
            ),
            nb::rv_policy::reference_internal
        )
        .def("mesh", &fvcc::SurfaceField<NeoN::Vec3>::mesh, nb::rv_policy::reference_internal)
        .def("exec", &fvcc::SurfaceField<NeoN::Vec3>::exec, nb::rv_policy::reference_internal)
        .def("size", &fvcc::SurfaceField<NeoN::Vec3>::size)
        .def(
            "correct_boundary_conditions",
            &fvcc::SurfaceField<NeoN::Vec3>::correctBoundaryConditions
        )
        .def_rw("name", &fvcc::SurfaceField<NeoN::Vec3>::name);

    m.def(
        "create_calculated_surface_bcs_scalar",
        [](const NeoN::UnstructuredMesh& mesh)
        { return fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh); },
        "mesh"_a,
        "Create calculated scalar surface boundary conditions"
    );

    m.def(
        "create_calculated_surface_bcs_vec3",
        [](const NeoN::UnstructuredMesh& mesh)
        { return fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::Vec3>>(mesh); },
        "mesh"_a,
        "Create calculated Vec3 surface boundary conditions"
    );

    // old-time and rotate for scalar SurfaceField (field must be registered in VectorCollection)
    m.def(
        "old_time",
        [](fvcc::SurfaceField<NeoN::scalar>& field) -> fvcc::SurfaceField<NeoN::scalar>&
        { return fvcc::oldTime(field); },
        "field"_a,
        nb::rv_policy::reference,
        "Get or create the old-time scalar surface field"
    );

    m.def(
        "rotate_old_times",
        [](fvcc::SurfaceField<NeoN::scalar>& field) { fvcc::rotateOldTimes(field); },
        "field"_a,
        "Rotate old-time scalar surface field (φ^n → φ^{n-1}) — field must be registered in "
        "VectorCollection"
    );
}

} // namespace NeoN::bindings
