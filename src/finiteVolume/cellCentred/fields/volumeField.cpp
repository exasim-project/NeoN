// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <type_traits>

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

    // Collect proc-patch (start, end) ranges in MESH-BOUNDARY ORDER, paired
    // with their target neighbour ranks. communicateBoundaryData uses
    // targetRanks to compute per-rank Alltoallv displacements, so mesh-order
    // is preserved end-to-end (which is what setProcBoundarySparsityPattern
    // and the matrix layout expect).
    const auto& bm = this->mesh().boundaryMesh();
    const auto& nbrRanks = bm.neighbourRank();
    const auto totalPatches = bm.nBoundaries();
    const auto procPatchCount = bm.nProcBoundaryPatches();
    const auto firstProcPatch = totalPatches - procPatchCount;

    const bool trace = (std::getenv("NF_PROC_BC_TRACE") != nullptr);
    if (trace)
    {
        std::fprintf(
            stderr,
            "[NF_PROC_BC_TRACE][rank %d][VolumeField::correctBC field='%s'] nBC=%zu "
            "totalPatches=%lld procPatchCount=%lld firstProcPatch=%lld nbrRanks.size=%zu "
            "iVec.size=%lld value.size=%lld\n",
            mpiEnviron.rank(),
            this->name.c_str(),
            boundaryConditions_.size(),
            (long long)totalPatches,
            (long long)procPatchCount,
            (long long)firstProcPatch,
            nbrRanks.size(),
            (long long)this->field_.internalVector().size(),
            (long long)this->field_.boundaryData().value().size()
        );

        // After loop: print first few proc-tail values of the field BEFORE
        // and AFTER communicateBoundaryData. Done by capturing here pre-loop.
    }

    std::vector<std::pair<localIdx, localIdx>> procPatchOffset;
    std::vector<int> targetRanks;
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_);
        if (trace)
        {
            auto [bs, be] = boundaryCondition.range();
            std::fprintf(
                stderr,
                "[NF_PROC_BC_TRACE][rank %d][VolumeField::correctBC] BC patchID=%lld "
                "range=[%lld,%lld) isProc=%d\n",
                mpiEnviron.rank(),
                (long long)boundaryCondition.patchID(),
                (long long)bs,
                (long long)be,
                (int)(procPatchCount > 0 && boundaryCondition.patchID() >= firstProcPatch)
            );
        }
        if (procPatchCount > 0 && boundaryCondition.patchID() >= firstProcPatch)
        {
            const auto procIdx = boundaryCondition.patchID() - firstProcPatch;
            auto [start, end] = boundaryCondition.range();
            procPatchOffset.emplace_back(start, end);
            targetRanks.push_back(static_cast<int>(nbrRanks[procIdx]));
        }
    }

    if (trace)
    {
        std::fprintf(
            stderr,
            "[NF_PROC_BC_TRACE][rank %d][VolumeField::correctBC] procPatchOffset.size=%zu "
            "communicateBoundaryData will %s\n",
            mpiEnviron.rank(),
            procPatchOffset.size(),
            procPatchOffset.empty() ? "BE SKIPPED" : "RUN"
        );
    }

    if (!procPatchOffset.empty())
    {
        // FIXME dont recompute communication pattern
        // exchange processor boundary data
        auto commPattern = computeCommunicationPattern(this->mesh());
        if (trace)
        {
            std::fprintf(
                stderr,
                "[NF_PROC_BC_TRACE][rank %d][VolumeField::correctBC] commPattern: "
                "sendCounts.size=%zu recvIdx.size=%zu boundaryMapVector.size=%zu\n",
                mpiEnviron.rank(),
                commPattern.sendCounts.size(),
                commPattern.recvIdx.size(),
                commPattern.boundaryMapVector.size()
            );
        }
        communicateBoundaryData(
            commPattern, procPatchOffset, targetRanks, this->field_.boundaryData().value()
        );
        if (trace)
        {
            auto bvH = this->field_.boundaryData().value().copyToHost();
            const auto bv = bvH.view();
            const auto sz = bvH.size();
            // Print first 6 entries at the proc-tail (start of first proc patch).
            const auto procStart = procPatchOffset[0].first;
            if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
            {
                std::fprintf(
                    stderr,
                    "[NF_PROC_BC_TRACE][rank %d][VolumeField::correctBC field='%s'] "
                    "POST-MPI boundaryValue[%lld..%lld) first6=[%.6e %.6e %.6e %.6e %.6e %.6e]\n",
                    mpiEnviron.rank(),
                    this->name.c_str(),
                    (long long)procStart,
                    (long long)sz,
                    (procStart + 0 < sz) ? (double)bv[procStart + 0] : 0.0,
                    (procStart + 1 < sz) ? (double)bv[procStart + 1] : 0.0,
                    (procStart + 2 < sz) ? (double)bv[procStart + 2] : 0.0,
                    (procStart + 3 < sz) ? (double)bv[procStart + 3] : 0.0,
                    (procStart + 4 < sz) ? (double)bv[procStart + 4] : 0.0,
                    (procStart + 5 < sz) ? (double)bv[procStart + 5] : 0.0
                );
            }
        }
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
