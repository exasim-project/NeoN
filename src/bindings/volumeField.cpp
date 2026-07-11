// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

namespace
{
using ScalarVol = NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>;

// Allocate a scalar VolumeField sized like `proto` (same executor/mesh, calculated
// boundaries), pre-seeded with `proto`'s internal values — the mutable target an
// elementwise arithmetic kernel then overwrites. These operators return fresh
// fields so a closure reads like the maths (``Cmu * k * k / epsilon``); the result
// carries placeholder (calculated) boundaries — assign it into a properly
// boundary-conditioned field and ``correct_boundary_conditions`` for solver use.
ScalarVol seededLike(const ScalarVol& proto)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(proto.mesh());
    ScalarVol result(proto.exec(), "tmp", proto.mesh(), bcs);
    result.internalVector() = proto.internalVector();
    return result;
}

// The elementwise kernels live in free functions (not inside the binding lambdas):
// NEON_LAMBDA expands to an extended __device__ lambda under CUDA, which nvcc
// forbids from being defined inside another lambda.
ScalarVol mulFieldField(const ScalarVol& a, const ScalarVol& b)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    auto bv = b.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] * bv[i]; },
        "fieldMul"
    );
    return r;
}

ScalarVol mulFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] * s; },
        "fieldMulScalar"
    );
    return r;
}

ScalarVol divFieldField(const ScalarVol& a, const ScalarVol& b)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    auto bv = b.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] / bv[i]; },
        "fieldDiv"
    );
    return r;
}

ScalarVol divFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    const NeoN::scalar inv = NeoN::scalar(1.0) / s;
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] * inv; },
        "fieldDivScalar"
    );
    return r;
}

ScalarVol addFieldField(const ScalarVol& a, const ScalarVol& b)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    auto bv = b.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] + bv[i]; },
        "fieldAdd"
    );
    return r;
}

ScalarVol addFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] + s; },
        "fieldAddScalar"
    );
    return r;
}

ScalarVol subFieldField(const ScalarVol& a, const ScalarVol& b)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    auto bv = b.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] - bv[i]; },
        "fieldSub"
    );
    return r;
}

ScalarVol maxFieldScalar(const ScalarVol& a, NeoN::scalar low)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] > low ? rv[i] : low; },
        "fieldMaxScalar"
    );
    return r;
}

// field - scalar (the existing __sub__ only covered field - field).
ScalarVol subFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] - s; },
        "fieldSubScalar"
    );
    return r;
}

// scalar - field (for __rsub__, e.g. Spalart-Allmaras fv2 = 1 - chi/(...)).
ScalarVol rsubScalarField(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = s - rv[i]; },
        "fieldRsubScalar"
    );
    return r;
}

// scalar / field (for __rtruediv__, e.g. the fw ratio in Spalart-Allmaras).
ScalarVol rdivScalarField(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = s / rv[i]; },
        "fieldRdivScalar"
    );
    return r;
}

// -field (for __neg__).
ScalarVol negField(const ScalarVol& a)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = -rv[i]; },
        "fieldNeg"
    );
    return r;
}

// field ** exponent — elementwise power (the only way to express a fractional
// power such as the Spalart-Allmaras fw term (...)^(1/6) as a field op).
ScalarVol powFieldScalar(const ScalarVol& a, NeoN::scalar e)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = Kokkos::pow(rv[i], e); },
        "fieldPowScalar"
    );
    return r;
}

// Elementwise max(field, field) — e.g. Spalart-Allmaras Stilda = max(Omega + ..., Cs*Omega).
ScalarVol maxFieldField(const ScalarVol& a, const ScalarVol& b)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    auto bv = b.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] > bv[i] ? rv[i] : bv[i]; },
        "fieldMaxField"
    );
    return r;
}

// Elementwise min(field, scalar) — e.g. the Spalart-Allmaras r clamp min(..., 10).
ScalarVol minFieldScalar(const ScalarVol& a, NeoN::scalar high)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] < high ? rv[i] : high; },
        "fieldMinScalar"
    );
    return r;
}

// Elementwise min(field, field) — e.g. the kOmegaSST production limiter min(G, ...).
ScalarVol minFieldField(const ScalarVol& a, const ScalarVol& b)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    auto bv = b.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] < bv[i] ? rv[i] : bv[i]; },
        "fieldMinField"
    );
    return r;
}

// Elementwise tanh — the kOmegaSST blending functions F1 = tanh(arg^4), F2 = tanh(arg^2).
ScalarVol tanhField(const ScalarVol& a)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = Kokkos::tanh(rv[i]); },
        "fieldTanh"
    );
    return r;
}

// Elementwise sqrt — e.g. sqrt(k) in the kOmegaSST blending arguments (also field**0.5).
ScalarVol sqrtField(const ScalarVol& a)
{
    ScalarVol r = seededLike(a);
    auto rv = r.internalVector().view();
    NeoN::parallelFor(
        r.exec(),
        {0, r.internalVector().size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = Kokkos::sqrt(rv[i]); },
        "fieldSqrt"
    );
    return r;
}
} // namespace

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
            static_cast<void (fvcc::VolumeField<NeoN::scalar>::*)()>(
                &fvcc::VolumeField<NeoN::scalar>::correctBoundaryConditions
            ),
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
        // Elementwise arithmetic — each returns a fresh scalar field so a
        // turbulence closure can be written as readable field maths. The field
        // operands must share the same mesh; the scalar overloads broadcast.
        .def("__mul__", &mulFieldField, nb::is_operator())
        .def("__mul__", &mulFieldScalar, nb::is_operator())
        .def("__rmul__", &mulFieldScalar, nb::is_operator())
        .def("__truediv__", &divFieldField, nb::is_operator())
        .def("__truediv__", &divFieldScalar, nb::is_operator())
        .def("__add__", &addFieldField, nb::is_operator())
        .def("__add__", &addFieldScalar, nb::is_operator())
        .def("__radd__", &addFieldScalar, nb::is_operator())
        .def("__sub__", &subFieldField, nb::is_operator())
        .def("__sub__", &subFieldScalar, nb::is_operator())
        .def("__rsub__", &rsubScalarField, nb::is_operator())
        .def("__rtruediv__", &rdivScalarField, nb::is_operator())
        .def("__pow__", &powFieldScalar, nb::is_operator())
        .def("__neg__", &negField, nb::is_operator());

    // Elementwise lower bound: max(field, low) — used to keep k / epsilon positive.
    m.def(
        "field_max",
        &maxFieldScalar,
        "field"_a,
        "low"_a,
        "Elementwise max(field, low) — lower-bound a scalar field (e.g. bound k, epsilon)"
    );
    // Elementwise max(field, field) — e.g. Spalart-Allmaras Stilda.
    m.def(
        "field_max",
        &maxFieldField,
        "a"_a,
        "b"_a,
        "Elementwise max(a, b) of two scalar fields"
    );
    // Elementwise min(field, high) — e.g. the Spalart-Allmaras r clamp.
    m.def(
        "field_min",
        &minFieldScalar,
        "field"_a,
        "high"_a,
        "Elementwise min(field, high) — upper-bound a scalar field"
    );
    // Elementwise min(field, field) — e.g. the kOmegaSST production limiter.
    m.def(
        "field_min",
        &minFieldField,
        "a"_a,
        "b"_a,
        "Elementwise min(a, b) of two scalar fields"
    );
    // Elementwise tanh / sqrt — kOmegaSST blending functions and sqrt(k) arguments.
    m.def("tanh", &tanhField, "field"_a, "Elementwise tanh of a scalar field");
    m.def("sqrt", &sqrtField, "field"_a, "Elementwise sqrt of a scalar field");

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
            "correct_boundary_conditions",
            static_cast<void (fvcc::VolumeField<NeoN::Vec3>::*)()>(
                &fvcc::VolumeField<NeoN::Vec3>::correctBoundaryConditions
            ),
            "Apply boundary conditions"
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
