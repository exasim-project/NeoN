// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

/**
 * @brief Processor boundary condition for partitioned meshes.
 *
 * Reads ghost cell values from the internal vector (at ghost indices beyond nCells)
 * and sets valueFraction=1.0 so that operator assembly treats proc-boundary faces
 * like internal faces with the ghost value moved to the RHS.
 */
template<typename ValueType>
class ProcBoundary :
    public VolumeBoundaryFactory<ValueType>::template Register<ProcBoundary<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<ProcBoundary<ValueType>>;

public:

    ProcBoundary(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        const auto& ghostMap =
            *mesh_.stencilDB().template get<std::shared_ptr<std::vector<localIdx>>>(
                "partition::procBoundaryGhostMap"
            );
        auto procBndStart = *mesh_.stencilDB().template get<std::shared_ptr<localIdx>>(
            "partition::procBoundaryStartOffset"
        );

        auto iVector = domainVector.internalVector().view();
        auto [value, refValue, valueFraction, refGrad] = views(
            domainVector.boundaryData().value(),
            domainVector.boundaryData().refValue(),
            domainVector.boundaryData().valueFraction(),
            domainVector.boundaryData().refGrad()
        );

        auto [rangeStart, rangeEnd] = this->range();

        // Copy ghost map to a device-accessible array for this patch's portion
        localIdx nPatchFaces = rangeEnd - rangeStart;
        localIdx mapOffset = rangeStart - procBndStart;

        // Build a device-accessible copy of the ghost indices for this patch
        std::vector<localIdx> patchGhostIdxs(static_cast<std::size_t>(nPatchFaces));
        for (localIdx i = 0; i < nPatchFaces; ++i)
        {
            patchGhostIdxs[static_cast<std::size_t>(i)] =
                ghostMap[static_cast<std::size_t>(mapOffset + i)];
        }
        Vector<localIdx> ghostIdxs(domainVector.exec(), patchGhostIdxs);
        auto ghostIdxsV = ghostIdxs.view();

        NeoN::parallelFor(
            domainVector.exec(),
            {rangeStart, rangeEnd},
            NEON_LAMBDA(const localIdx i) {
                localIdx ghostIdx = ghostIdxsV[i - rangeStart];
                ValueType ghostVal = iVector[ghostIdx];
                value[i] = ghostVal;
                valueFraction[i] = 1.0;
                refValue[i] = ghostVal;
                refGrad[i] = zero<ValueType>();
            },
            "procBoundaryCorrection"
        );
    }

    static std::string name() { return "procBoundary"; }

    static std::string doc() { return "Processor boundary condition for ghost cell coupling"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<ProcBoundary>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

}
