// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
// Recompute the boundary value from the EXISTING per-face refGrad (does NOT overwrite
// refGrad with a stored uniform, unlike setGradientValue). This is the fixedFluxPressure
// semantics: an external caller (constrainPressure) sets refGrad each corrector so the
// projection cancels the wall face flux; correctBoundaryCondition only re-derives the
// boundary value from whatever refGrad currently holds.
template<typename ValueType>
void applyFixedFluxPressure(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();

    auto [refGradient, value, valueFraction, refValue, boundaryFaceOwners, deltaCoeffs] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            // reads (does not write) refGradient[i], set externally by constrainPressure
            value[i] = iVector[boundaryFaceOwners[i]] + refGradient[i] * (1 / deltaCoeffs[i]);
            valueFraction[i] = 0.0;          // only use refGrad
            refValue[i] = zero<ValueType>(); // not used
        },
        "applyFixedFluxPressure"
    );
}
}

/* @brief Fixed-flux pressure wall boundary condition.
 *
 * Mirrors OpenFOAM's fixedFluxPressureFvPatchScalarField: the per-face gradient (refGrad)
 * is set EXTERNALLY (by NeoFOAM::constrainPressure) so that the pressure projection cancels
 * the prescribed boundary face flux; correctBoundaryCondition only recomputes the boundary
 * value from the current refGrad, and never clobbers it with a stored uniform. refGrad is
 * zero-initialised by the boundary-data allocation, so the BC behaves as zeroGradient until
 * the first constrainPressure call.
 */
template<typename ValueType>
class FixedFluxPressure :
    public VolumeBoundaryFactory<ValueType>::template Register<FixedFluxPressure<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedFluxPressure<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    FixedFluxPressure(const UnstructuredMesh& mesh, const Dictionary&, localIdx patchID)
        : Base(mesh, {}, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::applyFixedFluxPressure(domainVector, mesh_, this->range());
    }

    static std::string name() { return "fixedFluxPressure"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Fixed-flux pressure wall BC (externally updatable per-face refGrad).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<FixedFluxPressure>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

}
