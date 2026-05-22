// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <catch2/catch_approx.hpp>

#include <random>

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

using Vector = NeoN::Vector<NeoN::scalar>;
using Coeff = NeoN::dsl::Coeff;
using Operator = NeoN::dsl::Operator;
using Executor = NeoN::Executor;
using localIdx = NeoN::localIdx;
using VolumeField = fvcc::VolumeField<NeoN::scalar>;
using OperatorMixin = NeoN::dsl::OperatorMixin<VolumeField>;
using BoundaryData = NeoN::BoundaryData<NeoN::scalar>;

/* helper struct to create a vector in the database
 */
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    NeoN::scalar value = 0;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
        NeoN::Field<NeoN::scalar> domainVector(
            mesh.exec(),
            NeoN::Vector<NeoN::scalar>(mesh.exec(), mesh.nCells(), 1.0),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<NeoN::scalar> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");
        NeoN::fill(vf.internalVector(), value);
        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateVectorDoc
        );
    }
};

template<typename ValueType>
struct CreateVolumeVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;

    // initial value for the internal field
    ValueType value {}; // e.g. 0.0 for scalars, Vec3{0,0,0} for vectors

    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db) const
    {
        using VF = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
        using VB = NeoN::finiteVolume::cellCentred::VolumeBoundary<ValueType>;

        std::vector<VB> bcs; // empty is fine for the test

        // Domain storage (Field<T>) with proper sizes/offsets
        NeoN::Field<ValueType> domainField(
            mesh.exec(),
            NeoN::Vector<ValueType>(mesh.exec(), mesh.nCells(), value),
            mesh.boundaryMesh().offset()
        );

        VF vf(mesh.exec(), name, mesh, domainField, bcs, db, /*dbKey*/ "", /*collection*/ "");
        NeoN::fill(vf.internalVector(), value);

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            NeoN::finiteVolume::cellCentred::validateVectorDoc
        );
    }
};

template<typename ValueType>
struct CreateSurfaceVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    const std::vector<NeoN::finiteVolume::cellCentred::SurfaceBoundary<ValueType>>* bcs = nullptr;

    ValueType value {}; // initial face value

    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db) const
    {
        using SF = NeoN::finiteVolume::cellCentred::SurfaceField<ValueType>;

        // Face storage: internalVector holds only internal faces
        NeoN::Field<ValueType> domainField(
            mesh.exec(), mesh.nInternalFaces(), mesh.boundaryMesh().offset()
        );
        NeoN::fill(domainField.internalVector(), value);
        NeoN::fill(domainField.boundaryData().refValue(), value);
        NeoN::fill(domainField.boundaryData().value(), value);

        // Safe default if caller didn’t pass BCs
        std::vector<NeoN::finiteVolume::cellCentred::SurfaceBoundary<ValueType>> local_bcs;
        const auto& use_bcs = (bcs) ? *bcs : local_bcs;

        SF sf(mesh.exec(), name, mesh, domainField, use_bcs, db, /*dbKey*/ "", /*collection*/ "");

        return NeoN::Document(
            {{"name", sf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", sf}},
            NeoN::finiteVolume::cellCentred::validateVectorDoc
        );
    }
};

/* A dummy implementation of a SpatialOperator
 * following the SpatialOperator interface */
template<typename ValueType>
class Dummy : public NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    Dummy(fvcc::VolumeField<ValueType>& field)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit
        )
    {}

    Dummy(fvcc::VolumeField<ValueType>& field, Operator::Type type)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, type
        )
    {}

    void read(const NeoN::Input&) {}

    void explicitOperation(NeoN::Vector<ValueType>& source) const
    {
        auto sourceView = source.view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            NEON_LAMBDA(const localIdx i) { sourceView[i] += coeff[i] * fieldView[i]; }
        );
    }

    void implicitOperation(la::LinearSystem<ValueType, la::CSRMatrix<ValueType, localIdx>>& ls
    ) const
    {
        auto values = ls.matrix().values().view();
        auto rhs = ls.rhs().view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();

        // update diag
        NeoN::parallelFor(
            this->exec(),
            {0, values.size()},
            NEON_LAMBDA(const localIdx i) { values[i] += coeff[i] * fieldView[i]; }
        );

        // update rhs
        NeoN::parallelFor(
            this->exec(),
            ls.rhs().range(),
            NEON_LAMBDA(const localIdx i) { rhs[i] += coeff[i] * fieldView[i]; }
        );
    }

    std::string getName() const { return "Dummy"; }
};

/* A dummy implementation of a SpatialOperator
 * following the SpatialOperator interface */
template<typename ValueType>
class TemporalDummy : public NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    TemporalDummy(fvcc::VolumeField<ValueType>& field)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit
        )
    {}

    TemporalDummy(fvcc::VolumeField<ValueType>& field, Operator::Type type)
        : NeoN::dsl::OperatorMixin<fvcc::VolumeField<ValueType>>(
            field.exec(), dsl::Coeff(1.0), field, type
        )
    {}

    void read(const NeoN::Input&) {}

    void explicitOperation(NeoN::Vector<ValueType>& source, NeoN::scalar, NeoN::scalar)
    {
        auto sourceView = source.view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            NEON_LAMBDA(const localIdx i) { sourceView[i] += coeff[i] * fieldView[i]; }
        );
    }

    void implicitOperation(
        la::LinearSystem<ValueType, la::CSRMatrix<ValueType, localIdx>>& ls,
        NeoN::scalar,
        NeoN::scalar
    )
    {
        auto values = ls.matrix().values().view();
        auto rhs = ls.rhs().view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();

        // update diag
        NeoN::parallelFor(
            this->exec(),
            {0, values.size()},
            NEON_LAMBDA(const localIdx i) { values[i] += coeff[i] * fieldView[i]; }
        );

        // update rhs
        NeoN::parallelFor(
            this->exec(),
            ls.rhs().range(),
            NEON_LAMBDA(const localIdx i) { rhs[i] += coeff[i] * fieldView[i]; }
        );
    }

    std::string getName() const { return "TemporalDummy"; }
};

template<typename ValueType>
ValueType getVector(const NeoN::Vector<ValueType>& source)
{
    auto sourceVector = source.copyToHost();
    return sourceVector.view()[0];
}

template<typename ValueType>
ValueType getDiag(const la::LinearSystem<ValueType, la::CSRMatrix<ValueType, localIdx>>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.matrix().values().view()[0];
}

template<typename ValueType>
ValueType getRhs(const la::LinearSystem<ValueType, la::CSRMatrix<ValueType, localIdx>>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.rhs().view()[0];
}


template<typename VectorValueType>
void randomizeVector(NeoN::Vector<VectorValueType>& a)
{
    // std::random_device rd;  // Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> dis(1.0, 2.0);

    auto b = a.copyToHost();
    auto bV = b.view();

    for (int i = 0; i < b.size(); i++)
    {
        bV[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    auto c = b.copyToExecutor(a.exec());
    a = c;
}

template<typename VectorValueType>
void randomizeVector(fvcc::VolumeField<VectorValueType>& a)
{
    randomizeVector(a.internalVector());
}
