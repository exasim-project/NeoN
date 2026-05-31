// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/processor.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary::detail
{

template<typename ValueType>
void setProcBoundaryValue(
    Field<ValueType>& domainVector,
    [[maybe_unused]] const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    // Unlike the volume processor BC, a SurfaceField's proc-tail already holds the LOCAL
    // face value to be sent to the neighbour: it is written by the operator that produced
    // the field (flux() / updateFaceVelocity() / interpolation) or by constructFrom() at
    // construction. There is no owner-cell -> face reconstruction to perform here.
    //
    // The volume BC seeds value[i] = internalVector[faceCells[i]] (owner-cell value ->
    // ghost). Copying that verbatim into the surface BC is WRONG: a surface field's
    // internalVector() is FACE data (size nInternalFaces) while faceCells[i] is an
    // owner-CELL id, so it reads an unrelated internal face's flux and ships garbage to
    // the neighbour. On a rank with >= 2 processor patches this corrupts the exchanged
    // proc-face flux (e.g. the t=0 phi = flux(U) exchange), inflating the reported Courant
    // number and perturbing the distributed solution. So leave value_ untouched and only
    // normalise the unused mixed-BC fields. value_ is NOT accessed (no valueNoWait/value),
    // so no pending proc-patch exchange is drained.
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
        "setProcBoundaryValue"
    );
}

template void setProcBoundaryValue<
    scalar>(Field<scalar>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
template void
setProcBoundaryValue<Vec3>(Field<Vec3>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);

} // namespace NeoN::finiteVolume::cellCentred::surfaceBoundary::detail

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{
template class Processor<scalar>;
template class Processor<Vec3>;
}
