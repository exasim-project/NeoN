// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
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
        .def_rw("name", &fvcc::VolumeField<NeoN::scalar>::name)
        .def(
            "assign",
            [](fvcc::VolumeField<NeoN::scalar>& self, const fvcc::VolumeField<NeoN::scalar>& other)
            { self.internalVector() = other.internalVector(); },
            "other"_a,
            "Deep-copy internal vector from another scalar field (mirrors C++ = operator)"
        )
        .def(
            "__add__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, const fvcc::VolumeField<NeoN::scalar>& b)
            { return a + b; },
            "other"_a,
            "Element-wise addition of two scalar volume fields"
        )
        .def(
            "__sub__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, const fvcc::VolumeField<NeoN::scalar>& b)
            { return a - b; },
            "other"_a,
            "Element-wise subtraction of two scalar volume fields"
        )
        .def(
            "__mul__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, const fvcc::VolumeField<NeoN::scalar>& b)
            { return a * b; },
            "other"_a,
            "Element-wise multiplication of two scalar volume fields"
        )
        .def(
            "__truediv__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, const fvcc::VolumeField<NeoN::scalar>& b)
            { return a / b; },
            "other"_a,
            "Element-wise division of two scalar volume fields"
        )
        // Scalar operations
        .def(
            "__add__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, NeoN::scalar s)
            {
                fvcc::VolumeField<NeoN::scalar> result(a);
                result += s;
                return result;
            },
            "scalar"_a,
            "Add a scalar to all elements"
        )
        .def(
            "__radd__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, NeoN::scalar s)
            {
                fvcc::VolumeField<NeoN::scalar> result(a);
                result += s;
                return result;
            },
            "scalar"_a,
            "Add a scalar to all elements (reflected)"
        )
        .def(
            "__sub__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, NeoN::scalar s)
            {
                fvcc::VolumeField<NeoN::scalar> result(a);
                result -= s;
                return result;
            },
            "scalar"_a,
            "Subtract a scalar from all elements"
        )
        .def(
            "__rsub__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, NeoN::scalar s)
            {
                // s - a: negate a, then add s
                fvcc::VolumeField<NeoN::scalar> result(a);
                result.internalVector() *= NeoN::scalar(-1.0);
                result.boundaryData().value() *= NeoN::scalar(-1.0);
                result += s;
                return result;
            },
            "scalar"_a,
            "Subtract field from a scalar (reflected)"
        )
        .def(
            "__mul__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, NeoN::scalar s)
            {
                fvcc::VolumeField<NeoN::scalar> result(a);
                result.internalVector() *= s;
                result.boundaryData().value() *= s;
                return result;
            },
            "scalar"_a,
            "Multiply all elements by a scalar"
        )
        .def(
            "__rmul__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, NeoN::scalar s)
            {
                fvcc::VolumeField<NeoN::scalar> result(a);
                result.internalVector() *= s;
                result.boundaryData().value() *= s;
                return result;
            },
            "scalar"_a,
            "Multiply all elements by a scalar (reflected)"
        )
        .def(
            "__truediv__",
            [](const fvcc::VolumeField<NeoN::scalar>& a, NeoN::scalar s)
            {
                fvcc::VolumeField<NeoN::scalar> result(a);
                NeoN::scalar inv = NeoN::scalar(1.0) / s;
                result.internalVector() *= inv;
                result.boundaryData().value() *= inv;
                return result;
            },
            "scalar"_a,
            "Divide all elements by a scalar"
        )
        .def(
            "__neg__",
            [](const fvcc::VolumeField<NeoN::scalar>& a)
            {
                fvcc::VolumeField<NeoN::scalar> result(a);
                result.internalVector() *= NeoN::scalar(-1.0);
                result.boundaryData().value() *= NeoN::scalar(-1.0);
                return result;
            },
            "Negate all elements"
        );

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
        .def_rw("name", &fvcc::VolumeField<NeoN::Vec3>::name)
        .def(
            "assign",
            [](fvcc::VolumeField<NeoN::Vec3>& self, const fvcc::VolumeField<NeoN::Vec3>& other)
            { self.internalVector() = other.internalVector(); },
            "other"_a,
            "Deep-copy internal vector from another Vec3 field (mirrors C++ = operator)"
        );

    // --- Tensor VolumeField ---
    nb::class_<fvcc::VolumeBoundary<NeoN::Tensor>>(
        m, "TensorVolumeBoundary", "Volume boundary for Tensor fields"
    )
        .def("patch_id", &fvcc::VolumeBoundary<NeoN::Tensor>::patchID)
        .def("patch_size", &fvcc::VolumeBoundary<NeoN::Tensor>::patchSize);

    nb::class_<fvcc::VolumeField<NeoN::Tensor>>(
        m, "TensorVolumeField", "Volume field for Tensor values"
    )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::Tensor>& field,
               const NeoN::SerialExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::Tensor>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::Tensor>& field,
               const NeoN::CPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::Tensor>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::Tensor>& field,
               const NeoN::GPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::Tensor>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a
        )
        .def(
            "internal_vector",
            static_cast<NeoN::Vector<NeoN::Tensor>& (fvcc::VolumeField<NeoN::Tensor>::*)()>(
                &fvcc::VolumeField<NeoN::Tensor>::internalVector
            ),
            nb::rv_policy::reference_internal
        )
        .def("mesh", &fvcc::VolumeField<NeoN::Tensor>::mesh, nb::rv_policy::reference_internal)
        .def("exec", &fvcc::VolumeField<NeoN::Tensor>::exec, nb::rv_policy::reference_internal)
        .def("size", &fvcc::VolumeField<NeoN::Tensor>::size)
        .def(
            "correct_boundary_conditions",
            &fvcc::VolumeField<NeoN::Tensor>::correctBoundaryConditions
        )
        .def_rw("name", &fvcc::VolumeField<NeoN::Tensor>::name);

    // --- SymmTensor VolumeField ---
    nb::class_<fvcc::VolumeBoundary<NeoN::SymmTensor>>(
        m, "SymmTensorVolumeBoundary", "Volume boundary for SymmTensor fields"
    )
        .def("patch_id", &fvcc::VolumeBoundary<NeoN::SymmTensor>::patchID)
        .def("patch_size", &fvcc::VolumeBoundary<NeoN::SymmTensor>::patchSize);

    nb::class_<fvcc::VolumeField<NeoN::SymmTensor>>(
        m, "SymmTensorVolumeField", "Volume field for SymmTensor values"
    )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::SymmTensor>& field,
               const NeoN::SerialExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::SymmTensor>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::SymmTensor>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::SymmTensor>& field,
               const NeoN::CPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::SymmTensor>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::SymmTensor>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a
        )
        .def(
            "__init__",
            [](fvcc::VolumeField<NeoN::SymmTensor>& field,
               const NeoN::GPUExecutor& exec,
               std::string name,
               const NeoN::UnstructuredMesh& mesh)
            {
                auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::SymmTensor>>(mesh);
                new (&field) fvcc::VolumeField<NeoN::SymmTensor>(
                    NeoN::Executor {exec}, std::move(name), mesh, bcs
                );
            },
            "exec"_a,
            "name"_a,
            "mesh"_a
        )
        .def(
            "internal_vector",
            static_cast<NeoN::Vector<NeoN::SymmTensor>& (fvcc::VolumeField<NeoN::SymmTensor>::*)()>(
                &fvcc::VolumeField<NeoN::SymmTensor>::internalVector
            ),
            nb::rv_policy::reference_internal
        )
        .def("mesh", &fvcc::VolumeField<NeoN::SymmTensor>::mesh, nb::rv_policy::reference_internal)
        .def("exec", &fvcc::VolumeField<NeoN::SymmTensor>::exec, nb::rv_policy::reference_internal)
        .def("size", &fvcc::VolumeField<NeoN::SymmTensor>::size)
        .def(
            "correct_boundary_conditions",
            &fvcc::VolumeField<NeoN::SymmTensor>::correctBoundaryConditions
        )
        .def_rw("name", &fvcc::VolumeField<NeoN::SymmTensor>::name);

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

    // oldTime free functions (require field to be registered in a VectorCollection)
    m.def(
        "old_time",
        [](fvcc::VolumeField<NeoN::scalar>& field) -> fvcc::VolumeField<NeoN::scalar>&
        { return fvcc::oldTime(field); },
        "field"_a,
        nb::rv_policy::reference,
        "Get or create the old-time scalar volume field"
    );

    m.def(
        "old_time",
        [](fvcc::VolumeField<NeoN::Vec3>& field) -> fvcc::VolumeField<NeoN::Vec3>&
        { return fvcc::oldTime(field); },
        "field"_a,
        nb::rv_policy::reference,
        "Get or create the old-time vector volume field"
    );

    m.def(
        "rotate_old_times",
        [](fvcc::VolumeField<NeoN::scalar>& field) { fvcc::rotateOldTimes(field); },
        "field"_a,
        "Rotate old-time scalar volume field (φ^n → φ^{n-1}) — field must be registered in "
        "VectorCollection"
    );

    m.def(
        "rotate_old_times",
        [](fvcc::VolumeField<NeoN::Vec3>& field) { fvcc::rotateOldTimes(field); },
        "field"_a,
        "Rotate old-time Vec3 volume field (φ^n → φ^{n-1}) — field must be registered in "
        "VectorCollection"
    );
}

} // namespace NeoN::bindings
