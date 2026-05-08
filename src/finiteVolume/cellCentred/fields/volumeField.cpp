// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <tuple>

#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/macros.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string nameIn,
    const UnstructuredMesh& mesh,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(
        exec, nameIn, mesh, Field<ValueType>(exec, mesh.nCells(), mesh.boundaryMesh().offset())
    ),
      FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string nameIn,
    const UnstructuredMesh& mesh,
    const Vector<ValueType>& internalVector,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(
        exec, nameIn, mesh, Field<ValueType>(exec, internalVector, mesh.boundaryMesh().offset())
    ),
      FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string nameIn,
    const UnstructuredMesh& mesh,
    const Vector<ValueType>& internalVector,
    const BoundaryData<ValueType>& boundaryVectors,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(exec, nameIn, mesh, internalVector, boundaryVectors),
      FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
{}


template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string fieldName,
    const UnstructuredMesh& mesh,
    const Field<ValueType>& domainVector,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions,
    Database& db,
    std::string dbKey,
    std::string collectionName
)
    : DomainMixin<ValueType>(exec, fieldName, mesh, domainVector),
      FieldDatabaseMixin(db, std::move(dbKey), std::move(collectionName)),
      boundaryConditions_(boundaryConditions)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(const VolumeField& other)
    : DomainMixin<ValueType>(other), FieldDatabaseMixin(other),
      boundaryConditions_(other.boundaryConditions_)
{}

template<typename ValueType>
VolumeField<ValueType>& VolumeField<ValueType>::operator+=(const ValueType rhs)
{
    add(this->internalVector(), rhs);
    correctBoundaryConditions();
    return *this;
}

template<typename ValueType>
VolumeField<ValueType>& VolumeField<ValueType>::operator-=(const ValueType rhs)
{
    sub(this->internalVector(), rhs);
    correctBoundaryConditions();
    return *this;
}

template<typename ValueType>
void VolumeField<ValueType>::correctBoundaryConditions()
{
    NeoN::mpi::Environment mpiEnviron;

    // Identify processor patches and collect their (start, end) ranges paired with
    // the neighbour rank for sorting. communicateBoundaryData walks ranks in
    // ascending order and consumes the j-th procPatchOffset entry for the j-th
    // rank with non-zero send count, so the offsets MUST be in ascending
    // neighbour-rank order. Mesh-order is not the same as ascending rank-order
    // in general (decomposition-dependent), so we sort here.
    const auto& bm = this->mesh().boundaryMesh();
    const auto& nbrRanks = bm.neighbourRank();
    const auto totalPatches = bm.nBoundaries();
    const auto procPatchCount = bm.nProcBoundaryPatches();
    const auto firstProcPatch = totalPatches - procPatchCount;

    std::vector<std::tuple<localIdx, localIdx, localIdx>> procTriples;
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_);
        if (procPatchCount > 0 && boundaryCondition.patchID() >= firstProcPatch)
        {
            const auto procIdx = boundaryCondition.patchID() - firstProcPatch;
            auto [start, end] = boundaryCondition.range();
            procTriples.emplace_back(nbrRanks[procIdx], start, end);
        }
    }

    if (!procTriples.empty())
    {
        std::sort(
            procTriples.begin(),
            procTriples.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); }
        );
        std::vector<std::pair<localIdx, localIdx>> procPatchOffset;
        procPatchOffset.reserve(procTriples.size());
        for (const auto& t : procTriples)
        {
            procPatchOffset.emplace_back(std::get<1>(t), std::get<2>(t));
        }

        // FIXME dont recompute communication pattern
        // exchange processor boundary data
        auto commPattern = computeCommunicationPattern(this->mesh());
        communicateBoundaryData(commPattern, procPatchOffset, this->field_.boundaryData().value());
    }
}

template<typename ValueType>
void VolumeField<ValueType>::correctBoundaryConditions(const BoundaryContext& ctx)
{
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_, ctx);
    }
}

#define NN_DECLARE_FIELD(TYPENAME) template class VolumeField<TYPENAME>

NN_FOR_ALL_VALUE_TYPES(NN_DECLARE_FIELD);

}
