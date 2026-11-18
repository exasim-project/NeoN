// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Shared kernel behind PressureInletOutletVelocity, the velocity counterpart to a
// pressure-driven outlet (OpenFOAM's pressureInletOutletVelocityFvPatchVectorField, a
// directionMixedFvPatchVectorField). Literal OpenFOAM blends with a TENSOR-valued fraction
//   valueFraction() = neg(phi) * (I - n (x) n)
// i.e. the tangential components are forced to a fixed tangentialVelocity on inflow and
// zero-gradient on outflow, while the normal component is *always* left to zero-gradient
// extrapolation (the pressure equation, not this BC, sets the normal velocity). NeoN's
// boundaryData() valueFraction is a single scalar shared by all 3 components of a Vec3 field
// (see e.g. gaussGreenDiv.cpp/gaussGreenLaplacian.cpp, which multiply a whole Vec3 refValue by
// one scalar fraction), so that tensor cannot be represented directly.
//
// It does not need to be: the tensor blend is reproduced exactly by making refValue itself carry
// the normal-direction extrapolation, recomputed every call from the CURRENT owner-cell value
// (the same "recompute from live field data" pattern already used by e.g. totalPressure):
//   n              = faceNormal / faceArea                       (unit outward normal)
//   refValue       = n*(n & ownerValue) + (tangentialVelocity - n*(n & tangentialVelocity))
//   valueFraction  = neg(phi)                                     (plain scalar, as inletOutlet)
//   refGrad        = 0
//   value = valueFraction*refValue + (1 - valueFraction)*ownerValue
// On inflow (valueFraction = 1): value = n*(n & ownerValue) + tangentialVelocity_projected,
// identical to literal's tensor-blend inflow value. On outflow (valueFraction = 0): value =
// ownerValue, identical to literal's outflow value (full zero-gradient on all 3 components). This
// holds for any U, not just a flow aligned with a coordinate axis -- no approximation, and no
// change to NeoN's scalar valueFraction machinery.
//
// When no flux is available (hasFlux == false) every face is treated as outflow, degenerating to
// a pure zeroGradient BC -- the same fallback inletOutlet uses.
inline void setPressureInletOutletVelocityValue(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NeoN::Vec3 tangentialVelocity,
    View<const NeoN::scalar> boundaryFlux,
    bool hasFlux
)
{
    const auto internalV = domainVector.internalVector().view();

    auto [faceNormalsV, faceAreasV, boundaryFaceOwnersV] = views(
        mesh.boundaryMesh().faceNormals(),
        mesh.boundaryMesh().faceAreas(),
        mesh.boundaryMesh().faceOwners()
    );

    auto [refGradV, valueV, valueFractionV, refValueV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = boundaryFaceOwnersV[i];
            const NeoN::Vec3 ownerValue = internalV[owner];
            const NeoN::Vec3 n = faceNormalsV[i] / faceAreasV[i];

            const NeoN::scalar valueFraction =
                (hasFlux && boundaryFlux[i] < 0.0) ? NeoN::scalar(1) : NeoN::scalar(0);

            const NeoN::Vec3 tangentialVelocityProjected =
                tangentialVelocity - n * (n & tangentialVelocity);
            const NeoN::Vec3 refValue = n * (n & ownerValue) + tangentialVelocityProjected;

            refGradV[i] = NeoN::zero<NeoN::Vec3>();
            refValueV[i] = refValue;
            valueFractionV[i] = valueFraction;
            valueV[i] = valueFraction * refValue + (NeoN::scalar(1) - valueFraction) * ownerValue;
        },
        "setPressureInletOutletVelocityValue"
    );
}

} // namespace detail

// PressureInletOutletVelocity: velocity outlet counterpart to a pressure-driven boundary
// (mirrors OpenFOAM's pressureInletOutletVelocity). See detail::setPressureInletOutletVelocityValue
// for the exact formula and its exact equivalence to OpenFOAM's tensor-valued directionMixed
// blend.
//
// Needs, via the context-aware correctBoundaryCondition(domainVector, ctx) overload, a
// surface-scalar flux, key `phiName_` (dict key "phi", default "phi"). Without it, degrades to
// plain zero-gradient (the same fallback the plain, context-less overload always uses).
class PressureInletOutletVelocity :
    public VolumeBoundaryFactory<NeoN::Vec3>::template Register<PressureInletOutletVelocity>
{
    using Base = VolumeBoundaryFactory<NeoN::Vec3>::template Register<PressureInletOutletVelocity>;

public:

    using Base::correctBoundaryCondition;

    // assignable = true: recomputed every correctBoundaryConditions() call. fixesValue = false:
    // this is a mixed BC (per-face valueFraction), not a global Dirichlet patch.
    PressureInletOutletVelocity(
        const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID
    )
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh),
          tangentialVelocity_(
              dict.contains("tangentialVelocity") ? dict.get<NeoN::Vec3>("tangentialVelocity")
                                                  : NeoN::Vec3 {}
          ),
          phiName_(dict.contains("phi") ? dict.get<std::string>("phi") : std::string("phi"))
    {}

    // No context => no flux available: behave as zero-gradient (all faces treated as outflow).
    virtual void correctBoundaryCondition(Field<NeoN::Vec3>& domainVector) final
    {
        detail::setPressureInletOutletVelocityValue(
            domainVector,
            mesh_,
            this->range(),
            tangentialVelocity_,
            View<const NeoN::scalar> {},
            false
        );
    }

    // Context-aware: read the face flux phi and switch inflow/outflow per face.
    virtual void
    correctBoundaryCondition(Field<NeoN::Vec3>& domainVector, const BoundaryContext& ctx) final
    {
        if (ctx.hasSurfaceScalar(phiName_))
        {
            const auto& phi = ctx.surfaceScalarField(phiName_);
            detail::setPressureInletOutletVelocityValue(
                domainVector,
                mesh_,
                this->range(),
                tangentialVelocity_,
                phi.boundaryData().value().view(),
                true
            );
        }
        else
        {
            correctBoundaryCondition(domainVector);
        }
    }

    static std::string name() { return "pressureInletOutletVelocity"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Velocity outlet paired with a pressure-driven boundary (OpenFOAM's "
               "pressureInletOutletVelocity): on inflow faces (phi < 0) the tangential velocity "
               "is set to tangentialVelocity (default zero) while the normal component is left "
               "to zero-gradient; on outflow faces all 3 components are zero-gradient. Switched "
               "per face by the sign of the face flux phi.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::Vec3>> clone() const final
    {
        return std::make_unique<PressureInletOutletVelocity>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    NeoN::Vec3 tangentialVelocity_;
    std::string phiName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
