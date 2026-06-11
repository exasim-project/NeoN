// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <catch2/matchers/catch_matchers_all.hpp>
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


template<typename ValueType>
ValueType getVector(const NeoN::Vector<ValueType>& source)
{
    auto sourceVector = source.copyToHost();
    return sourceVector.view()[0];
}

template<typename ValueType>
ValueType getDiag(const la::LinearSystem<ValueType>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.matrix().values().view()[0];
}

template<typename ValueType>
ValueType getRhs(const la::LinearSystem<ValueType>& ls)
{
    auto hostLs = ls.copyToHost();
    return hostLs.rhs().view()[0];
}
