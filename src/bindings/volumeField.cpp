// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

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
// boundaries), pre-seeded with `proto`'s internal AND boundary values — the mutable
// target an elementwise arithmetic kernel then overwrites. These operators return
// fresh fields so a closure reads like the maths (``Cmu * k * k / epsilon``).
ScalarVol seededLike(const ScalarVol& proto)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(proto.mesh());
    ScalarVol result(proto.exec(), "tmp", proto.mesh(), bcs);
    result.internalVector() = proto.internalVector();
    result.boundaryData().value() = proto.boundaryData().value();
    return result;
}

// Guard every field-field operation. The kernels iterate over the LEFT operand's size and index
// the right one directly, so mismatched operands would read out of bounds on the device instead
// of raising in Python. Mesh identity is the real precondition (element i must mean the same cell
// in both fields); executor and vector sizes are checked too so a mismatch reports the specific
// cause. std::invalid_argument surfaces as a Python ValueError through nanobind.
void requireCompatible(const ScalarVol& a, const ScalarVol& b, const char* op)
{
    if (&a.mesh() != &b.mesh())
    {
        throw std::invalid_argument(std::string(op) + ": operands must live on the same mesh");
    }
    if (a.exec() != b.exec())
    {
        throw std::invalid_argument(std::string(op) + ": operands must live on the same executor");
    }
    if (a.internalVector().size() != b.internalVector().size()
        || a.boundaryData().value().size() != b.boundaryData().value().size())
    {
        throw std::invalid_argument(
            std::string(op) + ": operands must have matching internal and boundary sizes"
        );
    }
}

// The elementwise kernels live in free functions on raw vectors (not inside the
// binding lambdas): NEON_LAMBDA expands to an extended __device__ lambda under
// CUDA, which nvcc forbids from being defined inside another lambda. Each field
// operator applies its kernel to the internal AND the boundary vector — NeoN
// volume-field maths mirrors OpenFOAM GeometricField arithmetic, which evaluates
// boundary values too (a closure's nut/nuEff boundary values feed the momentum
// wall fluxes; internal-only maths leaves them stale at their on-disk values).
using ScalarVec = NeoN::Vector<NeoN::scalar>;

void mulVecVec(const NeoN::Executor& exec, ScalarVec& r, const ScalarVec& b, std::string label)
{
    auto [rv, bv] = NeoN::views(r, b);
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] * bv[i]; },
        std::move(label)
    );
}

void divVecVec(const NeoN::Executor& exec, ScalarVec& r, const ScalarVec& b, std::string label)
{
    auto [rv, bv] = NeoN::views(r, b);
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] / bv[i]; },
        std::move(label)
    );
}

void addVecVec(const NeoN::Executor& exec, ScalarVec& r, const ScalarVec& b, std::string label)
{
    auto [rv, bv] = NeoN::views(r, b);
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] + bv[i]; },
        std::move(label)
    );
}

void subVecVec(const NeoN::Executor& exec, ScalarVec& r, const ScalarVec& b, std::string label)
{
    auto [rv, bv] = NeoN::views(r, b);
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] - bv[i]; },
        std::move(label)
    );
}

void maxVecVec(const NeoN::Executor& exec, ScalarVec& r, const ScalarVec& b, std::string label)
{
    auto [rv, bv] = NeoN::views(r, b);
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] > bv[i] ? rv[i] : bv[i]; },
        std::move(label)
    );
}

void minVecVec(const NeoN::Executor& exec, ScalarVec& r, const ScalarVec& b, std::string label)
{
    auto [rv, bv] = NeoN::views(r, b);
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] < bv[i] ? rv[i] : bv[i]; },
        std::move(label)
    );
}

void mulVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar s, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] * s; },
        std::move(label)
    );
}

void addVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar s, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] + s; },
        std::move(label)
    );
}

void subVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar s, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] - s; },
        std::move(label)
    );
}

void rsubVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar s, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = s - rv[i]; },
        std::move(label)
    );
}

void rdivVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar s, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = s / rv[i]; },
        std::move(label)
    );
}

void maxVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar low, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] > low ? rv[i] : low; },
        std::move(label)
    );
}

void minVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar high, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = rv[i] < high ? rv[i] : high; },
        std::move(label)
    );
}

void powVecScalar(const NeoN::Executor& exec, ScalarVec& r, NeoN::scalar e, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = Kokkos::pow(rv[i], e); },
        std::move(label)
    );
}

void negVec(const NeoN::Executor& exec, ScalarVec& r, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = -rv[i]; },
        std::move(label)
    );
}

void tanhVec(const NeoN::Executor& exec, ScalarVec& r, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = Kokkos::tanh(rv[i]); },
        std::move(label)
    );
}

void sqrtVec(const NeoN::Executor& exec, ScalarVec& r, std::string label)
{
    auto rv = r.view();
    NeoN::parallelFor(
        exec,
        {0, r.size()},
        NEON_LAMBDA(const NeoN::localIdx i) { rv[i] = Kokkos::sqrt(rv[i]); },
        std::move(label)
    );
}

ScalarVol mulFieldField(const ScalarVol& a, const ScalarVol& b)
{
    requireCompatible(a, b, "field * field");
    ScalarVol r = seededLike(a);
    mulVecVec(r.exec(), r.internalVector(), b.internalVector(), "fieldMul");
    mulVecVec(r.exec(), r.boundaryData().value(), b.boundaryData().value(), "fieldMul::b");
    return r;
}

ScalarVol mulFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    mulVecScalar(r.exec(), r.internalVector(), s, "fieldMulScalar");
    mulVecScalar(r.exec(), r.boundaryData().value(), s, "fieldMulScalar::b");
    return r;
}

ScalarVol divFieldField(const ScalarVol& a, const ScalarVol& b)
{
    requireCompatible(a, b, "field / field");
    ScalarVol r = seededLike(a);
    divVecVec(r.exec(), r.internalVector(), b.internalVector(), "fieldDiv");
    divVecVec(r.exec(), r.boundaryData().value(), b.boundaryData().value(), "fieldDiv::b");
    return r;
}

ScalarVol divFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    const NeoN::scalar inv = NeoN::scalar(1.0) / s;
    mulVecScalar(r.exec(), r.internalVector(), inv, "fieldDivScalar");
    mulVecScalar(r.exec(), r.boundaryData().value(), inv, "fieldDivScalar::b");
    return r;
}

ScalarVol addFieldField(const ScalarVol& a, const ScalarVol& b)
{
    requireCompatible(a, b, "field + field");
    ScalarVol r = seededLike(a);
    addVecVec(r.exec(), r.internalVector(), b.internalVector(), "fieldAdd");
    addVecVec(r.exec(), r.boundaryData().value(), b.boundaryData().value(), "fieldAdd::b");
    return r;
}

ScalarVol addFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    addVecScalar(r.exec(), r.internalVector(), s, "fieldAddScalar");
    addVecScalar(r.exec(), r.boundaryData().value(), s, "fieldAddScalar::b");
    return r;
}

ScalarVol subFieldField(const ScalarVol& a, const ScalarVol& b)
{
    requireCompatible(a, b, "field - field");
    ScalarVol r = seededLike(a);
    subVecVec(r.exec(), r.internalVector(), b.internalVector(), "fieldSub");
    subVecVec(r.exec(), r.boundaryData().value(), b.boundaryData().value(), "fieldSub::b");
    return r;
}

ScalarVol maxFieldScalar(const ScalarVol& a, NeoN::scalar low)
{
    ScalarVol r = seededLike(a);
    maxVecScalar(r.exec(), r.internalVector(), low, "fieldMaxScalar");
    maxVecScalar(r.exec(), r.boundaryData().value(), low, "fieldMaxScalar::b");
    return r;
}

// field - scalar (the existing __sub__ only covered field - field).
ScalarVol subFieldScalar(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    subVecScalar(r.exec(), r.internalVector(), s, "fieldSubScalar");
    subVecScalar(r.exec(), r.boundaryData().value(), s, "fieldSubScalar::b");
    return r;
}

// scalar - field (for __rsub__, e.g. Spalart-Allmaras fv2 = 1 - chi/(...)).
ScalarVol rsubScalarField(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    rsubVecScalar(r.exec(), r.internalVector(), s, "fieldRsubScalar");
    rsubVecScalar(r.exec(), r.boundaryData().value(), s, "fieldRsubScalar::b");
    return r;
}

// scalar / field (for __rtruediv__, e.g. the fw ratio in Spalart-Allmaras).
ScalarVol rdivScalarField(const ScalarVol& a, NeoN::scalar s)
{
    ScalarVol r = seededLike(a);
    rdivVecScalar(r.exec(), r.internalVector(), s, "fieldRdivScalar");
    rdivVecScalar(r.exec(), r.boundaryData().value(), s, "fieldRdivScalar::b");
    return r;
}

// -field (for __neg__).
ScalarVol negField(const ScalarVol& a)
{
    ScalarVol r = seededLike(a);
    negVec(r.exec(), r.internalVector(), "fieldNeg");
    negVec(r.exec(), r.boundaryData().value(), "fieldNeg::b");
    return r;
}

// field ** exponent — elementwise power (the only way to express a fractional
// power such as the Spalart-Allmaras fw term (...)^(1/6) as a field op).
ScalarVol powFieldScalar(const ScalarVol& a, NeoN::scalar e)
{
    ScalarVol r = seededLike(a);
    powVecScalar(r.exec(), r.internalVector(), e, "fieldPowScalar");
    powVecScalar(r.exec(), r.boundaryData().value(), e, "fieldPowScalar::b");
    return r;
}

// Elementwise max(field, field) — e.g. Spalart-Allmaras Stilda = max(Omega + ..., Cs*Omega).
ScalarVol maxFieldField(const ScalarVol& a, const ScalarVol& b)
{
    requireCompatible(a, b, "field_max(a, b)");
    ScalarVol r = seededLike(a);
    maxVecVec(r.exec(), r.internalVector(), b.internalVector(), "fieldMaxField");
    maxVecVec(r.exec(), r.boundaryData().value(), b.boundaryData().value(), "fieldMaxField::b");
    return r;
}

// Elementwise min(field, scalar) — e.g. the Spalart-Allmaras r clamp min(..., 10).
ScalarVol minFieldScalar(const ScalarVol& a, NeoN::scalar high)
{
    ScalarVol r = seededLike(a);
    minVecScalar(r.exec(), r.internalVector(), high, "fieldMinScalar");
    minVecScalar(r.exec(), r.boundaryData().value(), high, "fieldMinScalar::b");
    return r;
}

// Elementwise min(field, field) — e.g. the kOmegaSST production limiter min(G, ...).
ScalarVol minFieldField(const ScalarVol& a, const ScalarVol& b)
{
    requireCompatible(a, b, "field_min(a, b)");
    ScalarVol r = seededLike(a);
    minVecVec(r.exec(), r.internalVector(), b.internalVector(), "fieldMinField");
    minVecVec(r.exec(), r.boundaryData().value(), b.boundaryData().value(), "fieldMinField::b");
    return r;
}

// Elementwise tanh — the kOmegaSST blending functions F1 = tanh(arg^4), F2 = tanh(arg^2).
ScalarVol tanhField(const ScalarVol& a)
{
    ScalarVol r = seededLike(a);
    tanhVec(r.exec(), r.internalVector(), "fieldTanh");
    tanhVec(r.exec(), r.boundaryData().value(), "fieldTanh::b");
    return r;
}

// Elementwise sqrt — e.g. sqrt(k) in the kOmegaSST blending arguments (also field**0.5).
ScalarVol sqrtField(const ScalarVol& a)
{
    ScalarVol r = seededLike(a);
    sqrtVec(r.exec(), r.internalVector(), "fieldSqrt");
    sqrtVec(r.exec(), r.boundaryData().value(), "fieldSqrt::b");
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
            "boundary_data_value",
            [](fvcc::VolumeField<NeoN::scalar>& field) -> NeoN::Vector<NeoN::scalar>&
            { return field.boundaryData().value(); },
            nb::rv_policy::reference_internal,
            "Get the boundary data value vector"
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
            {
                requireCompatible(self, other, "assign");
                self.internalVector() = other.internalVector();
                self.boundaryData().value() = other.boundaryData().value();
            },
            "other"_a,
            "Deep-copy internal + boundary values from another scalar field (mirrors C++ =)"
        )
        // Elementwise arithmetic — each returns a fresh scalar field so a
        // turbulence closure can be written as readable field maths. Field-field
        // operands must share mesh, executor and sizes (enforced by
        // requireCompatible, raising ValueError); the scalar overloads broadcast.
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
    m.def("field_max", &maxFieldField, "a"_a, "b"_a, "Elementwise max(a, b) of two scalar fields");
    // Elementwise min(field, high) — e.g. the Spalart-Allmaras r clamp.
    m.def(
        "field_min",
        &minFieldScalar,
        "field"_a,
        "high"_a,
        "Elementwise min(field, high) — upper-bound a scalar field"
    );
    // Elementwise min(field, field) — e.g. the kOmegaSST production limiter.
    m.def("field_min", &minFieldField, "a"_a, "b"_a, "Elementwise min(a, b) of two scalar fields");
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
            {
                self.internalVector() = other.internalVector();
                self.boundaryData().value() = other.boundaryData().value();
            },
            "other"_a,
            "Deep-copy internal + boundary values from another Vec3 field (mirrors C++ =)"
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
