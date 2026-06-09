// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/processor.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail
{

template<typename ValueType>
void setProcBoundaryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();

    // NoWaitAccess: seeding the owner value must NOT drain a previously-posted proc-patch
    // exchange. Draining mid-loop (one patch at a time) breaks the second proc patch's halo on
    // ranks that own more than one (see BoundaryData::valueNoWait docs). All patches post first;
    // the exchange completes together on the next value() read.
    auto [refGradient, value, valueFraction, refValue, faceCells] = views(
        domainVector.boundaryData().refGrad(),
        NoWaitAccess::value(domainVector.boundaryData()),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refGradient[i] = zero<ValueType>();
            value[i] = iVector[faceCells[i]];
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
template void setProcBoundaryValue<
    Tensor>(Field<Tensor>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{
template class Processor<scalar>;
template class Processor<Vec3>;
// Tensor processor BC: the gradient of a Vec3 field is a Tensor cell field, and the corrected /
// limitedCorrected face-normal gradient halo-exchanges that gradient via correctBoundaryConditions.
template class Processor<Tensor>;
}
