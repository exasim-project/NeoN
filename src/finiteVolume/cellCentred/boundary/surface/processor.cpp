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
    const auto internalVector = domainVector.internalVector().view();
    // valueNoWait(): see volume/processor.cpp — seeding must not drain a pending proc-patch
    // exchange, or the second proc patch on a multi-patch rank loses its halo.
    auto [refGradient, valueFraction, value, refValue, faceCells] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().valueNoWait(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refGradient[i] = zero<ValueType>();
            valueFraction[i] = 0.0;
            value[i] = internalVector[faceCells[i]];
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
