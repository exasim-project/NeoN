// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Shared kernel behind WaveTransmissive, a non-reflecting compressible outflow BC. Mirrors
// OpenFOAM's waveTransmissiveFvPatchField (itself a specialisation of advectiveFvPatchField,
// Euler ddt only -- this solver's own scope is BDF1-only anyway, see e.g.
// neorhoPimpleFoam.cpp's own DdtScheme::None assert).
//
// Per boundary face, with the outward wave speed
//   w = max(0, Un + c),  Un = (phi/rho)/magSf (normal velocity),  c = sqrt(gamma/psi)
// (psi = 1/(R*T), so gamma/psi = gamma*R*T = c^2 -- the local speed of sound) and
//   alpha = w*dt*deltaCoeffs
// the patch is a mixed BC (refGrad = 0 throughout):
//   lInf > 0 (relax toward fieldInf over a finite length lInf, the usual non-reflecting case):
//     k = w*dt/lInf
//     refValue       = (p_oldTime + k*fieldInf) / (1 + k)
//     valueFraction  = (1 + k) / (1 + alpha + k)
//   lInf <= 0 (pure wave transmission, no far-field relaxation):
//     refValue       = p_oldTime
//     valueFraction  = 1 / (1 + alpha)
// value = valueFraction*refValue + (1 - valueFraction)*ownerCellValue
//
// Only ever evaluated for scalar fields in practice (OpenFOAM templates it generically, but
// every real case uses it on p) -- kept as a bare function over scalar views rather than
// templated on ValueType to avoid an unused, unverified vector/tensor path.
inline void setWaveTransmissiveValue(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NeoN::scalar gamma,
    NeoN::scalar fieldInf,
    NeoN::scalar lInf,
    NeoN::scalar dt,
    View<const NeoN::scalar> boundaryPhi,
    View<const NeoN::scalar> boundaryRho,
    View<const NeoN::scalar> boundaryPsi,
    View<const NeoN::scalar> boundaryPOld,
    bool haveContext
)
{
    const auto internalV = domainVector.internalVector().view();
    const auto [faceAreasV, deltaCoeffsV, boundaryFaceOwnersV] = views(
        mesh.boundaryMesh().faceAreas(),
        mesh.boundaryMesh().deltaCoeffs(),
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
            const NeoN::scalar ownerValue = internalV[owner];

            refGradV[i] = NeoN::scalar(0);

            if (!haveContext)
            {
                // Degenerate fallback with no phi/rho/psi/old-time/dt available: behave as a
                // plain zero-gradient (owner-cell) extrapolation.
                valueFractionV[i] = NeoN::scalar(0);
                refValueV[i] = ownerValue;
                valueV[i] = ownerValue;
                return;
            }

            const NeoN::scalar Un = (boundaryPhi[i] / boundaryRho[i]) / faceAreasV[i];
            const NeoN::scalar c = Kokkos::sqrt(gamma / boundaryPsi[i]);
            const NeoN::scalar w = Kokkos::max(Un + c, NeoN::scalar(0));
            const NeoN::scalar alpha = w * dt * deltaCoeffsV[i];

            NeoN::scalar refValue;
            NeoN::scalar valueFraction;
            if (lInf > NeoN::scalar(0))
            {
                const NeoN::scalar k = w * dt / lInf;
                refValue = (boundaryPOld[i] + k * fieldInf) / (NeoN::scalar(1) + k);
                valueFraction = (NeoN::scalar(1) + k) / (NeoN::scalar(1) + alpha + k);
            }
            else
            {
                refValue = boundaryPOld[i];
                valueFraction = NeoN::scalar(1) / (NeoN::scalar(1) + alpha);
            }

            refValueV[i] = refValue;
            valueFractionV[i] = valueFraction;
            valueV[i] = valueFraction * refValue + (NeoN::scalar(1) - valueFraction) * ownerValue;
        },
        "setWaveTransmissiveValue"
    );
}

} // namespace detail

// WaveTransmissive: a non-reflecting compressible outflow BC, mirroring OpenFOAM's
// waveTransmissive (built on advectiveFvPatchField). See detail::setWaveTransmissiveValue for
// the exact formula and OpenFOAM source cross-reference.
//
// Needs, via the context-aware correctBoundaryCondition(domainVector, ctx) overload:
//   - a surface-scalar flux, key `phiName_` (dict key "phi", default "phi")
//   - a volume-scalar density, key `rhoName_` (dict key "rho", default "rho")
//   - a volume-scalar compressibility psi = 1/(R*T), key `psiName_` (dict key "psi", default
//     "psi")
//   - the field this BC is itself attached to, inserted under `fieldName_` (dict key "field",
//     default "p") -- needed for its oldTime() boundary value. NeoN's BC strategy objects are
//     not told which field they belong to, so the caller must additionally insert that same
//     field into the context under this name for this BC to find its own history.
//   - the current timestep, key "deltaT" (BoundaryContext::insertValue), via
//     BoundaryContext::hasValue/value rather than a field slot.
// Any one missing degrades to a plain zero-gradient extrapolation (the same degeneration the
// plain, context-less correctBoundaryCondition(domainVector) overload always uses), since a
// partial wave-speed calculation would silently give the wrong far-field behaviour rather than
// an honest fallback.
class WaveTransmissive :
    public VolumeBoundaryFactory<NeoN::scalar>::template Register<WaveTransmissive>
{
    using Base = VolumeBoundaryFactory<NeoN::scalar>::template Register<WaveTransmissive>;

public:

    using Base::correctBoundaryCondition;

    // assignable = true: recomputed every correctBoundaryConditions() call. fixesValue =
    // false: this is a mixed BC (per-face valueFraction), not a global Dirichlet patch.
    WaveTransmissive(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh),
          gamma_(dict.get<NeoN::scalar>("gamma")), // required: no physically-sane default
          fieldInf_(
              dict.contains("fieldInf") ? dict.get<NeoN::scalar>("fieldInf") : NeoN::scalar(0)
          ),
          lInf_(dict.contains("lInf") ? dict.get<NeoN::scalar>("lInf") : NeoN::scalar(-1)),
          phiName_(dict.contains("phi") ? dict.get<std::string>("phi") : std::string("phi")),
          rhoName_(dict.contains("rho") ? dict.get<std::string>("rho") : std::string("rho")),
          psiName_(dict.contains("psi") ? dict.get<std::string>("psi") : std::string("psi")),
          fieldName_(dict.contains("field") ? dict.get<std::string>("field") : std::string("p"))
    {}

    // No context => degenerate to zero-gradient (see detail kernel's haveContext == false path).
    virtual void correctBoundaryCondition(Field<NeoN::scalar>& domainVector) final
    {
        detail::setWaveTransmissiveValue(
            domainVector,
            mesh_,
            this->range(),
            gamma_,
            fieldInf_,
            lInf_,
            NeoN::scalar(0),
            View<const NeoN::scalar> {},
            View<const NeoN::scalar> {},
            View<const NeoN::scalar> {},
            View<const NeoN::scalar> {},
            false
        );
    }

    // Context-aware: the real non-reflecting wave-speed calculation.
    virtual void
    correctBoundaryCondition(Field<NeoN::scalar>& domainVector, const BoundaryContext& ctx) final
    {
        const bool haveContext = ctx.hasSurfaceScalar(phiName_) && ctx.hasScalar(rhoName_)
                              && ctx.hasScalar(psiName_) && ctx.hasScalar(fieldName_)
                              && ctx.hasValue("deltaT");

        if (!haveContext)
        {
            correctBoundaryCondition(domainVector);
            return;
        }

        const auto& phi = ctx.surfaceScalarField(phiName_);
        const auto& rho = ctx.scalarFieldPtr(rhoName_);
        const auto& psi = ctx.scalarFieldPtr(psiName_);
        const auto& pOld = oldTime(ctx.scalarFieldPtr(fieldName_));
        const NeoN::scalar dt = ctx.value("deltaT");

        detail::setWaveTransmissiveValue(
            domainVector,
            mesh_,
            this->range(),
            gamma_,
            fieldInf_,
            lInf_,
            dt,
            phi.boundaryData().value().view(),
            rho.boundaryData().value().view(),
            psi.boundaryData().value().view(),
            pOld.boundaryData().value().view(),
            true
        );
    }

    static std::string name() { return "waveTransmissive"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Non-reflecting compressible outflow (OpenFOAM's waveTransmissive): relaxes "
               "toward fieldInf over length lInf (or holds its own old-time value if lInf is "
               "unset) at a rate set by the local wave speed Un + sqrt(gamma/psi). Needs phi, "
               "rho, psi, the field's own old-time value, and the current deltaT in the "
               "BoundaryContext; without them it degrades to zero-gradient.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<NeoN::scalar>> clone() const final
    {
        return std::make_unique<WaveTransmissive>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    NeoN::scalar gamma_;
    NeoN::scalar fieldInf_;
    NeoN::scalar lInf_;
    std::string phiName_;
    std::string rhoName_;
    std::string psiName_;
    std::string fieldName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
