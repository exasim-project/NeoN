// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
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
    //
    // The passkey call is bound to a named reference instead of being nested inline in the
    // views(...) pack: that nesting is the one construct here the surface processor BC lacks, and
    // it crashes include-what-you-use (clang-19) during analysis. Hoisting it keeps the structured
    // binding to plain boundaryData() accessors (as in the surface BC) and avoids the crash.
    auto& valueNoWait = NoWaitAccess::value(domainVector.boundaryData());
    auto [refGradient, value, valueFraction, refValue, faceCells] = views(
        domainVector.boundaryData().refGrad(),
        valueNoWait,
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

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{
template class Processor<scalar>;
template class Processor<Vec3>;
}
