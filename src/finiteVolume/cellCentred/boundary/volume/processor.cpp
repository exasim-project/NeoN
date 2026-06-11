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

// One-time (set): zero the unused mixed-BC coefficients (refGradient / valueFraction / refValue).
// They are constant for the lifetime of the field, so this is split out of the per-iteration
// update() and runs only once. value_ is intentionally NOT touched here, so this kernel never
// drains a pending proc-patch exchange; the owner value is seeded each iteration by
// updateProcBoundaryOwnerValue().
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

// Per iteration (update): copy the owner-cell value into the proc-patch ghost so it can be shipped
// to the neighbour rank.
template<typename ValueType>
void updateProcBoundaryOwnerValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();

    // NoWaitAccess: seeding the owner value must NOT drain a proc-patch exchange that another
    // patch on this rank has already posted. boundaryData().value() runs waitAll() on access (see
    // BoundaryData::valueNoWait docs), which would complete — and consume — an in-flight recv
    // before its data is used, dropping the second proc patch's halo on a rank that owns more than
    // one. NoWaitAccess::value() writes the ghost without that drain; the field's
    // correctBoundaryConditions() drains all patches together (waitAll) once every patch has
    // posted.
    auto [value, faceCells] =
        views(NoWaitAccess::value(domainVector.boundaryData()), mesh.boundaryMesh().faceOwners());

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) { value[i] = iVector[faceCells[i]]; },
        "updateProcBoundaryOwnerValue"
    );
}

template void setProcBoundaryCoefficients<scalar>(Field<scalar>&, std::pair<localIdx, localIdx>);
template void setProcBoundaryCoefficients<Vec3>(Field<Vec3>&, std::pair<localIdx, localIdx>);
template void setProcBoundaryCoefficients<Tensor>(Field<Tensor>&, std::pair<localIdx, localIdx>);

template void updateProcBoundaryOwnerValue<
    scalar>(Field<scalar>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
template void updateProcBoundaryOwnerValue<
    Vec3>(Field<Vec3>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
template void updateProcBoundaryOwnerValue<
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
