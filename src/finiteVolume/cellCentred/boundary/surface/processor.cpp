// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/processor.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary::detail
{

// One-time (set): zero the unused mixed-BC coefficients (refGradient / valueFraction / refValue)
// on the processor patch. They are constant after construction, so the field runs this once via
// Processor::set() rather than on every correction; the actual halo exchange is posted each
// iteration by Processor::update().
//
// Note what is deliberately NOT done here: the surface BC does NOT seed value_ from the owner
// cell, unlike the volume BC. The reason is an index-space mismatch, not a timing choice:
//   * The volume BC does value[i] = internalVector[faceCells[i]] — internalVector is CELL data,
//     faceCells[i] is an owner-CELL id, so this copies the owner cell's value into the ghost.
//   * On a SurfaceField, internalVector() is FACE data (size nInternalFaces) indexed by face id.
//     Indexing it with an owner-CELL id (faceCells[i]) would read an unrelated internal face's
//     flux and ship garbage to the neighbour, corrupting the exchanged proc-face flux on a rank
//     with >= 2 processor patches.
// The face value to send already lives in the proc-patch tail of value_ (written there by the
// operator that produced the surface field, or by constructFrom() at construction), so there is
// nothing to reconstruct — Processor::update() just exchanges it. value_ is not touched here, so
// this kernel never drains a pending proc-patch exchange.
template<typename ValueType>
void setProcBoundaryCoefficients(
    Field<ValueType>& domainVector, std::pair<localIdx, localIdx> range
)
{
    auto [refGradient, valueFraction, refValue] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refGradient[i] = zero<ValueType>();
            valueFraction[i] = 0.0;
            refValue[i] = zero<ValueType>();
        },
        "setProcBoundaryCoefficients"
    );
}

template void setProcBoundaryCoefficients<scalar>(Field<scalar>&, std::pair<localIdx, localIdx>);
template void setProcBoundaryCoefficients<Vec3>(Field<Vec3>&, std::pair<localIdx, localIdx>);

} // namespace NeoN::finiteVolume::cellCentred::surfaceBoundary::detail

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{
template class Processor<scalar>;
template class Processor<Vec3>;
}
