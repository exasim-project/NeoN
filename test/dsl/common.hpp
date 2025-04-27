// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

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

    void explicitOperation(NeoN::Vector<ValueType>& source)
    {
        auto sourceView = source.view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const localIdx i) { sourceView[i] += coeff[i] * fieldView[i]; }
        );
    }

    void implicitOperation(la::LinearSystem<ValueType, NeoN::localIdx>& ls)
    {
        auto values = ls.matrix().values().view();
        auto rhs = ls.rhs().view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();

        // update diag
        NeoN::parallelFor(
            this->exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const localIdx i) { values[i] += coeff[i] * fieldView[i]; }
        );

        // update rhs
        NeoN::parallelFor(
            this->exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const localIdx i) { rhs[i] += coeff[i] * fieldView[i]; }
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

    void explicitOperation(NeoN::Vector<ValueType>& source, NeoN::scalar, NeoN::scalar)
    {
        auto sourceView = source.view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const localIdx i) { sourceView[i] += coeff[i] * fieldView[i]; }
        );
    }

    void
    implicitOperation(la::LinearSystem<ValueType, NeoN::localIdx>& ls, NeoN::scalar, NeoN::scalar)
    {
        auto values = ls.matrix().values().view();
        auto rhs = ls.rhs().view();
        auto fieldView = this->field_.internalVector().view();
        auto coeff = this->getCoefficient();

        // update diag
        NeoN::parallelFor(
            this->exec(),
            {0, values.size()},
            KOKKOS_LAMBDA(const localIdx i) { values[i] += coeff[i] * fieldView[i]; }
        );

        // update rhs
        NeoN::parallelFor(
            this->exec(),
            ls.rhs().range(),
            KOKKOS_LAMBDA(const localIdx i) { rhs[i] += coeff[i] * fieldView[i]; }
        );
    }

    la::LinearSystem<ValueType, NeoN::localIdx> createEmptyLinearSystem() const
    {
        NeoN::Vector<ValueType> values(this->exec(), 1, NeoN::zero<ValueType>());
        NeoN::Vector<NeoN::localIdx> colIdx(this->exec(), 1, 0.0);
        NeoN::Vector<NeoN::localIdx> rowOffs(this->exec(), {0, 1});
        NeoN::la::CSRMatrix<ValueType, NeoN::localIdx> csrMatrix(values, colIdx, rowOffs);

        NeoN::Vector<ValueType> rhs(this->exec(), 1, NeoN::zero<ValueType>());
        NeoN::la::LinearSystem<ValueType, NeoN::localIdx> linearSystem(csrMatrix, rhs);
        return linearSystem;
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
ValueType getDiag(const la::LinearSystem<ValueType, NeoN::localIdx>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.matrix().values().view()[0];
}

template<typename ValueType>
ValueType getRhs(const la::LinearSystem<ValueType, NeoN::localIdx>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.rhs().view()[0];
}
