// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Kernel behind freestreamVelocity, OpenFOAM's freestreamVelocityFvPatchVectorField (a mixed BC
// with refValue = freestreamValue, refGrad = 0):
//
//   valueFraction = 0.5 - 0.5 (Up . nf) / |Up|
//
// where Up is the patch's OWN current value (the previous evaluation, as OpenFOAM reads *this in
// updateCoeffs()). The blend is smooth between the two limits:
//   outflow (Up . nf > 0): valueFraction -> 0, zero-gradient
//   inflow  (Up . nf < 0): valueFraction -> 1, fixed to freestreamValue
// The evaluated face value is valueFraction*freestreamValue + (1 - valueFraction)*owner cell.
//
// A zero patch value would make the ratio 0/0. OpenFOAM starts the patch at freestreamValue, so
// the direction test falls back to freestreamValue there.
inline void setFreestreamVelocityValue(
    Field<Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    Vec3 freestreamValue
)
{
    const auto internalV = domainVector.internalVector().view();
    auto [refGradV, valueV, valueFractionV, refValueV, boundaryFaceOwnersV, faceAreaVectorV] =
        views(
            domainVector.boundaryData().refGrad(),
            domainVector.boundaryData().value(),
            domainVector.boundaryData().valueFraction(),
            domainVector.boundaryData().refValue(),
            mesh.boundaryMesh().faceOwners(),
            mesh.boundaryMesh().faceNormals()
        );
    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = boundaryFaceOwnersV[i];
            const Vec3 internal = internalV[owner];

            Vec3 up = valueV[i];
            NeoN::scalar magUp = NeoN::mag(up);
            if (!(magUp > NeoN::scalar(1e-300)))
            {
                up = freestreamValue;
                magUp = NeoN::mag(up);
            }

            // boundaryMesh().faceNormals() is the face area vector Sf; nf = Sf/|Sf|
            const Vec3 sf = faceAreaVectorV[i];
            const NeoN::scalar magSf = Kokkos::max(NeoN::mag(sf), NeoN::scalar(1e-300));
            const NeoN::scalar upDotNf = (up[0] * sf[0] + up[1] * sf[1] + up[2] * sf[2]) / magSf;

            const NeoN::scalar valueFraction =
                NeoN::scalar(0.5)
                - NeoN::scalar(0.5) * upDotNf / Kokkos::max(magUp, NeoN::scalar(1e-300));

            valueFractionV[i] = valueFraction;
            refValueV[i] = freestreamValue;
            refGradV[i] = NeoN::zero<Vec3>();
            valueV[i] =
                freestreamValue * valueFraction + internal * (NeoN::scalar(1) - valueFraction);
        },
        "setFreestreamVelocityValue"
    );
}

}

// freestreamVelocity: far-field velocity that blends the freestream value with zero-gradient
// according to the direction of the patch velocity relative to the face normal. It mirrors
// OpenFOAM's freestreamVelocity (a mixedFvPatchVectorField). Unlike inletOutlet it does not need
// a flux: the direction comes from the patch's own previous value, so the plain and the
// context-aware overload behave the same.
//
// Dictionary: freestreamValue (Vec3, required).
class FreestreamVelocity :
    public VolumeBoundaryFactory<NeoN::Vec3>::template Register<FreestreamVelocity>
{
    using Base = VolumeBoundaryFactory<NeoN::Vec3>::template Register<FreestreamVelocity>;

public:

    using Base::correctBoundaryCondition;

    // assignable = false, as OpenFOAM's mixed patch fields (constrainHbyA then sets HbyA to the
    // patch value). fixesValue = false and
    // mixed = true: the per-face valueFraction drives the operators via the mixed-BC kernels, and
    // OpenFOAM's mixed patches report fixesValue() = true (see BoundaryAttributes::mixed).
    FreestreamVelocity(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = false, .mixed = true}),
          mesh_(mesh), freestreamValue_(dict.get<NeoN::Vec3>("freestreamValue"))
    {}

    virtual void correctBoundaryCondition(Field<NeoN::Vec3>& domainVector) final
    {
        detail::setFreestreamVelocityValue(domainVector, mesh_, this->range(), freestreamValue_);
    }

    virtual void
    correctBoundaryCondition(Field<NeoN::Vec3>& domainVector, const BoundaryContext&) final
    {
        correctBoundaryCondition(domainVector);
    }

    static std::string name() { return "freestreamVelocity"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Far-field velocity (OpenFOAM's freestreamVelocity): valueFraction = 0.5 - 0.5 "
               "(U.nf)/|U| from the patch's own previous value, so inflow takes freestreamValue "
               "and outflow is zero-gradient, blending smoothly in between.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::Vec3>> clone() const final
    {
        return std::make_unique<FreestreamVelocity>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    NeoN::Vec3 freestreamValue_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
