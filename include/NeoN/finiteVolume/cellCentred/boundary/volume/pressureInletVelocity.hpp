// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Mirrors OpenFOAM's pressureInletVelocityFvPatchVectorField::updateCoeffs(): the patch velocity
// is the face flux turned into a velocity along the outward normal n,
//   volumetric flux:  U_b = n * phi_b / |S|
//   mass flux:        U_b = n * phi_b / (rho_b * |S|)
// so an inflow (phi < 0) gives a velocity pointing into the domain. A Dirichlet patch
// (valueFraction = 1, refGrad = 0). Without a flux (the context-less path) the patch is at rest.
inline void setPressureInletVelocity(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    View<const NeoN::scalar> boundaryPhi,
    View<const NeoN::scalar> boundaryRho,
    bool hasFlux,
    bool massFlux
)
{
    auto [faceNormalsV, faceAreasV] =
        views(mesh.boundaryMesh().faceNormals(), mesh.boundaryMesh().faceAreas());

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
            NeoN::Vec3 newValue = NeoN::zero<NeoN::Vec3>();
            if (hasFlux)
            {
                const NeoN::Vec3 n = faceNormalsV[i] / faceAreasV[i];
                const NeoN::scalar rho = massFlux ? boundaryRho[i] : NeoN::scalar(1);
                newValue = n * (boundaryPhi[i] / (rho * faceAreasV[i]));
            }
            refGradV[i] = NeoN::zero<NeoN::Vec3>();
            refValueV[i] = newValue;
            valueFractionV[i] = NeoN::scalar(1);
            valueV[i] = newValue;
        },
        "setPressureInletVelocity"
    );
}

} // namespace detail

// PressureInletVelocity: the patch velocity follows the flux through the patch (OpenFOAM's
// pressureInletVelocity), for a boundary where the pressure, not the velocity, drives the flow.
//
// Needs, via the context-aware correctBoundaryCondition(domainVector, ctx) overload, the face
// flux under `phiName_` (dict key "phi", default "phi"). OpenFOAM decides between a volumetric
// and a mass flux by the dimensions of phi, which NeoN fields do not carry: if the context holds a
// density field under `rhoName_` (dict key "rho", default "rho") phi is a mass flux and the
// boundary density is used, otherwise phi is volumetric. Without the flux the patch is at rest.
//
// A Dirichlet patch that is recomputed from the flux, like OpenFOAM's: assignable = true, so
// constrainHbyA leaves HbyA alone on it, and fixesValue = true.
class PressureInletVelocity :
    public VolumeBoundaryFactory<NeoN::Vec3>::template Register<PressureInletVelocity>
{
    using Base = VolumeBoundaryFactory<NeoN::Vec3>::template Register<PressureInletVelocity>;

public:

    using Base::correctBoundaryCondition;

    PressureInletVelocity(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = true}), mesh_(mesh),
          phiName_(dict.contains("phi") ? dict.get<std::string>("phi") : std::string("phi")),
          rhoName_(dict.contains("rho") ? dict.get<std::string>("rho") : std::string("rho"))
    {}

    virtual void correctBoundaryCondition(Field<NeoN::Vec3>& domainVector) final
    {
        detail::setPressureInletVelocity(
            domainVector,
            mesh_,
            this->range(),
            View<const NeoN::scalar> {},
            View<const NeoN::scalar> {},
            false,
            false
        );
    }

    virtual void
    correctBoundaryCondition(Field<NeoN::Vec3>& domainVector, const BoundaryContext& ctx) final
    {
        if (!ctx.hasSurfaceScalar(phiName_))
        {
            correctBoundaryCondition(domainVector);
            return;
        }
        const bool massFlux = ctx.hasScalar(rhoName_);
        detail::setPressureInletVelocity(
            domainVector,
            mesh_,
            this->range(),
            ctx.surfaceScalarField(phiName_).boundaryData().value().view(),
            massFlux ? ctx.scalarFieldPtr(rhoName_).boundaryData().value().view()
                     : View<const NeoN::scalar> {},
            true,
            massFlux
        );
    }

    static std::string name() { return "pressureInletVelocity"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Patch velocity along the outward normal that carries the face flux through the "
               "patch (OpenFOAM's pressureInletVelocity).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::Vec3>> clone() const final
    {
        return std::make_unique<PressureInletVelocity>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    std::string phiName_;
    std::string rhoName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
