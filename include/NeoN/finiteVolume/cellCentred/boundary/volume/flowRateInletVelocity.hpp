// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <string>

#include <Kokkos_Core.hpp>

#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/detail/patchSum.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Shared kernel behind FlowRateInletVelocity, mirroring OpenFOAM's
// flowRateInletVelocityFvPatchVectorField::updateValues(rho). With n the unit outward normal of a
// face, magSf its area and rho the density of the face (1 for a volumetric flow rate):
//
// uniform profile (default):
//   avgU  = -flowRate / sum(rho * magSf)                      (sum over the whole patch, all ranks)
//   value = avgU * n
//
// extrapolateProfile:
//   Up  = ownerValue;  nUp = n & Up;  Up -= nUp * n           (keep the tangential part)
//   nUp = min(nUp, 0)                                          (remove reverse flow)
//   estimated = -sum(rho * magSf * nUp)
//   if (estimated > 0.5 * flowRate)  nUp *= |flowRate| / |estimated|
//   else                             nUp -= (flowRate - estimated) / sum(rho * magSf)
//   value = Up + nUp * n
//
// The patch is a Dirichlet patch (valueFraction = 1, refGrad = 0). When the density the flow rate
// needs is not available (see FlowRateInletVelocity) the patch degrades to zero-gradient, the same
// fallback every context-aware NeoN BC uses.
inline void setFlowRateInletVelocity(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NeoN::scalar flowRate,
    View<const NeoN::scalar> boundaryRho,
    NeoN::scalar constantRho,
    bool useBoundaryRho,
    bool extrapolateProfile
)
{
    const auto internalV = domainVector.internalVector().view();
    auto [faceNormalsV, faceAreasV, boundaryFaceOwnersV] = views(
        mesh.boundaryMesh().faceNormals(),
        mesh.boundaryMesh().faceAreas(),
        mesh.boundaryMesh().faceOwners()
    );

    // density of face i: (useBoundaryRho ? boundaryRho[i] : constantRho). Written out in every
    // kernel instead of a shared lambda: device lambdas must not be nested.
    const NeoN::scalar rhoAreaSum = patchSum(
        mesh,
        range,
        NEON_LAMBDA(const localIdx i, NeoN::scalar& sum) {
            sum += (useBoundaryRho ? boundaryRho[i] : constantRho) * faceAreasV[i];
        }
    );

    auto [refGradV, valueV, valueFractionV, refValueV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );

    if (!extrapolateProfile)
    {
        const NeoN::scalar avgU = -flowRate / rhoAreaSum;
        NeoN::parallelFor(
            domainVector.exec(),
            range,
            NEON_LAMBDA(const localIdx i) {
                const NeoN::Vec3 n = faceNormalsV[i] / faceAreasV[i];
                const NeoN::Vec3 newValue = avgU * n;
                refGradV[i] = NeoN::zero<NeoN::Vec3>();
                refValueV[i] = newValue;
                valueFractionV[i] = NeoN::scalar(1);
                valueV[i] = newValue;
            },
            "setFlowRateInletVelocity"
        );
        return;
    }

    // flow rate carried by the inflow part of the patch normal velocity of the adjacent cells
    const NeoN::scalar estimatedFlowRate = -patchSum(
        mesh,
        range,
        NEON_LAMBDA(const localIdx i, NeoN::scalar& sum) {
            const NeoN::Vec3 n = faceNormalsV[i] / faceAreasV[i];
            const NeoN::scalar nUp = n & internalV[boundaryFaceOwnersV[i]];
            const NeoN::scalar inflowNUp = nUp < NeoN::scalar(0) ? nUp : NeoN::scalar(0);
            sum += (useBoundaryRho ? boundaryRho[i] : constantRho) * faceAreasV[i] * inflowNUp;
        }
    );

    const bool scale = estimatedFlowRate > 0.5 * flowRate;
    const NeoN::scalar factor =
        scale ? Kokkos::abs(flowRate) / Kokkos::abs(estimatedFlowRate) : NeoN::scalar(1);
    const NeoN::scalar shift =
        scale ? NeoN::scalar(0) : (flowRate - estimatedFlowRate) / rhoAreaSum;

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const NeoN::Vec3 n = faceNormalsV[i] / faceAreasV[i];
            const NeoN::Vec3 ownerValue = internalV[boundaryFaceOwnersV[i]];
            const NeoN::scalar ownerNUp = n & ownerValue;
            const NeoN::Vec3 tangential = ownerValue - n * ownerNUp;
            const NeoN::scalar inflowNUp = ownerNUp < NeoN::scalar(0) ? ownerNUp : NeoN::scalar(0);
            const NeoN::scalar nUp = inflowNUp * factor - shift;
            const NeoN::Vec3 newValue = tangential + nUp * n;
            refGradV[i] = NeoN::zero<NeoN::Vec3>();
            refValueV[i] = newValue;
            valueFractionV[i] = NeoN::scalar(1);
            valueV[i] = newValue;
        },
        "setFlowRateInletVelocityExtrapolated"
    );
}

// Zero-gradient fallback, used when the density a mass flow rate needs is not available.
inline void setFlowRateInletVelocityZeroGradient(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto internalV = domainVector.internalVector().view();
    const auto boundaryFaceOwnersV = mesh.boundaryMesh().faceOwners().view();
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
            const NeoN::Vec3 ownerValue = internalV[boundaryFaceOwnersV[i]];
            refGradV[i] = NeoN::zero<NeoN::Vec3>();
            refValueV[i] = ownerValue;
            valueFractionV[i] = NeoN::scalar(0);
            valueV[i] = ownerValue;
        },
        "setFlowRateInletVelocityZeroGradient"
    );
}

} // namespace detail

// FlowRateInletVelocity: inlet velocity that carries a prescribed flow rate through the patch,
// mirroring OpenFOAM's flowRateInletVelocity. See detail::setFlowRateInletVelocity for the exact
// formulas.
//
// Dictionary entries (flow rates are plain scalars; OpenFOAM's Function1 form
// `massFlowRate constant X` has to be flattened by the case reader):
//   volumetricFlowRate  volumetric flow rate in m^3/s (takes precedence), or
//   massFlowRate        mass flow rate in kg/s, which needs the density, see below
//   rho                 name of the density field (default "rho"; "none" means density 1)
//   rhoInlet            constant density, used when the density field is not available
//   extrapolateProfile  keep the tangential velocity and the shape of the inflow of the adjacent
//                       cells instead of a uniform profile (default false)
//
// For a mass flow rate the density is read from the boundary values of the field `rho` in the
// BoundaryContext. Without such a field the constant `rhoInlet` is used. If neither exists the
// context-aware overload throws, as OpenFOAM stops with an error; the plain overload (no context
// at all, e.g. while the field is created) degrades to zero-gradient.
class FlowRateInletVelocity :
    public VolumeBoundaryFactory<NeoN::Vec3>::template Register<FlowRateInletVelocity>
{
    using Base = VolumeBoundaryFactory<NeoN::Vec3>::template Register<FlowRateInletVelocity>;

public:

    using Base::correctBoundaryCondition;

    // A Dirichlet patch, like OpenFOAM's (it derives from fixedValue): assignable = false, so a
    // pressure-velocity coupling that constrains HbyA on non-assignable velocity patches
    // (constrainHbyA) feeds the prescribed inlet velocity into the flux, and fixesValue = true.
    // The value is still recomputed from the flow rate on every call.
    FlowRateInletVelocity(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true}), mesh_(mesh),
          flowRate_(0), volumetric_(dict.contains("volumetricFlowRate")),
          rhoName_(dict.contains("rho") ? dict.get<std::string>("rho") : std::string("rho")),
          rhoInlet_(
              dict.contains("rhoInlet") ? dict.get<NeoN::scalar>("rhoInlet")
                                        : -std::numeric_limits<NeoN::scalar>::max()
          ),
          extrapolateProfile_(
              dict.contains("extrapolateProfile") ? dict.get<bool>("extrapolateProfile") : false
          )
    {
        if (volumetric_)
        {
            flowRate_ = dict.get<NeoN::scalar>("volumetricFlowRate");
        }
        else if (dict.contains("massFlowRate"))
        {
            flowRate_ = dict.get<NeoN::scalar>("massFlowRate");
        }
        else
        {
            NF_THROW(
                "flowRateInletVelocity boundary condition on patch " + std::to_string(patchID)
                + " requires either 'volumetricFlowRate' or 'massFlowRate'"
            );
        }
    }

    // No context => no density field: fine for a volumetric flow rate, rho "none" or a constant
    // rhoInlet, otherwise zero-gradient.
    virtual void correctBoundaryCondition(Field<NeoN::Vec3>& domainVector) final
    {
        if (volumetric_ || rhoName_ == "none")
        {
            applyConstantDensity(domainVector, NeoN::scalar(1));
        }
        else if (rhoInlet_ >= NeoN::scalar(0))
        {
            applyConstantDensity(domainVector, rhoInlet_);
        }
        else
        {
            detail::setFlowRateInletVelocityZeroGradient(domainVector, mesh_, this->range());
        }
    }

    // Context-aware, in OpenFOAM's order: density 1 for a volumetric flow rate or rho "none";
    // else the boundary values of the field `rho`, if the context has it; else the constant
    // rhoInlet; else an error.
    virtual void
    correctBoundaryCondition(Field<NeoN::Vec3>& domainVector, const BoundaryContext& ctx) final
    {
        if (volumetric_ || rhoName_ == "none")
        {
            applyConstantDensity(domainVector, NeoN::scalar(1));
        }
        else if (ctx.hasScalar(rhoName_))
        {
            detail::setFlowRateInletVelocity(
                domainVector,
                mesh_,
                this->range(),
                flowRate_,
                ctx.scalarFieldPtr(rhoName_).boundaryData().value().view(),
                NeoN::scalar(1),
                true,
                extrapolateProfile_
            );
        }
        else if (rhoInlet_ >= NeoN::scalar(0))
        {
            applyConstantDensity(domainVector, rhoInlet_);
        }
        else
        {
            NF_THROW(
                "flowRateInletVelocity boundary condition on patch "
                + std::to_string(this->patchID()) + ": a mass flow rate needs the density field '"
                + rhoName_ + "' in the boundary context or a constant 'rhoInlet'"
            );
        }
    }

    static std::string name() { return "flowRateInletVelocity"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Inlet velocity carrying a prescribed volumetric or mass flow rate through the "
               "patch "
               "(OpenFOAM's flowRateInletVelocity): a uniform profile along the inward normal, or "
               "with extrapolateProfile the adjacent cells' tangential velocity and inflow shape, "
               "rescaled to the flow rate.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::Vec3>> clone() const final
    {
        return std::make_unique<FlowRateInletVelocity>(*this);
    }

private:

    void applyConstantDensity(Field<NeoN::Vec3>& domainVector, NeoN::scalar rho)
    {
        detail::setFlowRateInletVelocity(
            domainVector,
            mesh_,
            this->range(),
            flowRate_,
            View<const NeoN::scalar> {},
            rho,
            false,
            extrapolateProfile_
        );
    }

    const UnstructuredMesh& mesh_;
    NeoN::scalar flowRate_;
    bool volumetric_;
    std::string rhoName_;
    NeoN::scalar rhoInlet_;
    bool extrapolateProfile_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
