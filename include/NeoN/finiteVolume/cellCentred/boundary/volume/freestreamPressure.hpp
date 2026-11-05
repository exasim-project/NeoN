// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Kernel behind freestreamPressure, OpenFOAM's freestreamPressureFvPatchScalarField (a mixed BC
// with refValue = freestreamValue, refGrad = 0), the complement of freestreamVelocity:
//
//   valueFraction = 0.5 + 0.5 (Up . nf) / |Up|
//
// with Up the patch value of the velocity field. Outflow (Up . nf > 0) fixes the pressure to
// freestreamValue (valueFraction -> 1), inflow (Up . nf < 0) is zero-gradient (valueFraction -> 0),
// and the blend is smooth in between. Without a velocity (hasU == false) every face is
// zero-gradient, as the other flow-direction BCs degrade without their context.
inline void setFreestreamPressureValue(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NeoN::scalar freestreamValue,
    View<const Vec3> boundaryU,
    bool hasU
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
            const NeoN::scalar internal = internalV[owner];

            NeoN::scalar valueFraction = NeoN::scalar(0);
            if (hasU)
            {
                const Vec3 up = boundaryU[i];
                const Vec3 sf = faceAreaVectorV[i];
                const NeoN::scalar magSf = Kokkos::max(NeoN::mag(sf), NeoN::scalar(1e-300));
                const NeoN::scalar upDotNf =
                    (up[0] * sf[0] + up[1] * sf[1] + up[2] * sf[2]) / magSf;
                valueFraction = NeoN::scalar(0.5)
                              + NeoN::scalar(0.5) * upDotNf
                                    / Kokkos::max(NeoN::mag(up), NeoN::scalar(1e-300));
            }

            valueFractionV[i] = valueFraction;
            refValueV[i] = freestreamValue;
            refGradV[i] = NeoN::zero<NeoN::scalar>();
            valueV[i] =
                freestreamValue * valueFraction + internal * (NeoN::scalar(1) - valueFraction);
        },
        "setFreestreamPressureValue"
    );
}

}

// freestreamPressure: far-field pressure that blends the freestream value with zero-gradient
// according to the direction of the velocity at the patch, mirroring OpenFOAM's
// freestreamPressure (a mixedFvPatchScalarField without the supersonic option).
//
// The velocity is looked up by name (dict key "U", default "U") from the BoundaryContext of the
// context-aware correctBoundaryCondition overload; the plain overload has none and degenerates to
// zeroGradient.
//
// Dictionary: freestreamValue (scalar, required), U (optional field name).
class FreestreamPressure :
    public VolumeBoundaryFactory<NeoN::scalar>::template Register<FreestreamPressure>
{
    using Base = VolumeBoundaryFactory<NeoN::scalar>::template Register<FreestreamPressure>;

public:

    using Base::correctBoundaryCondition;

    // assignable = false, as OpenFOAM's mixed patch fields (constrainHbyA then sets HbyA to the
    // patch value). fixesValue = false and
    // mixed = true, as for freestreamVelocity.
    FreestreamPressure(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = false, .mixed = true}),
          mesh_(mesh), freestreamValue_(dict.get<NeoN::scalar>("freestreamValue")),
          UName_(dict.contains("U") ? dict.get<std::string>("U") : std::string("U"))
    {}

    // No context => no velocity: behave as zero-gradient.
    virtual void correctBoundaryCondition(Field<NeoN::scalar>& domainVector) final
    {
        detail::setFreestreamPressureValue(
            domainVector, mesh_, this->range(), freestreamValue_, View<const Vec3> {}, false
        );
    }

    virtual void
    correctBoundaryCondition(Field<NeoN::scalar>& domainVector, const BoundaryContext& ctx) final
    {
        if (ctx.hasVector(UName_))
        {
            const auto& U = ctx.vectorFieldPtr(UName_);
            detail::setFreestreamPressureValue(
                domainVector,
                mesh_,
                this->range(),
                freestreamValue_,
                U.boundaryData().value().view(),
                true
            );
        }
        else
        {
            correctBoundaryCondition(domainVector);
        }
    }

    static std::string name() { return "freestreamPressure"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Far-field pressure (OpenFOAM's freestreamPressure): valueFraction = 0.5 + 0.5 "
               "(U.nf)/|U| from the patch velocity, so outflow takes freestreamValue and inflow "
               "is zero-gradient, blending smoothly in between.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::scalar>> clone() const final
    {
        return std::make_unique<FreestreamPressure>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    NeoN::scalar freestreamValue_;
    std::string UName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
