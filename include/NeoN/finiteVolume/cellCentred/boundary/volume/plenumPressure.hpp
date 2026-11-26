// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <Kokkos_Core.hpp>

#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/detail/patchSum.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Face values of OpenFOAM's plenumPressureFvPatchScalarField::updateCoeffs(), given the plenum
// state of this update. pPlenum is the plenum pressure rho_pl*R*T_pl; per face
//   U_e = |U_b / inletAreaRatio|^2                      (squared exit velocity)
//   r   = 1 - (gamma - 1) U_e / (2 gamma R T_pl)        (exit to plenum temperature ratio)
//   a   = (1 - r) / (r^2 Cd^2)                          (quadratic coefficient, b = 1, c = -1)
//   s   = 2 / (1 + sqrt(1 + 4 a))                       (isentropic temperature ratio)
//   t   = s^(gamma / (gamma - 1))                       (exit to plenum pressure ratio)
//   p_new = phi >= 0 ? max(p, pPlenum) : t * pPlenum    (no outflow limit: inflow to the domain
//                                                        has phi < 0)
//   value = p_old + fraction * (p_new - p_old)
// with fraction = 1 for timeScale <= dt, else dt / timeScale. A Dirichlet patch.
inline void setPlenumPressureValue(
    Field<NeoN::scalar>& domainVector,
    std::pair<localIdx, localIdx> range,
    View<const NeoN::scalar> boundaryPhi,
    View<const NeoN::Vec3> boundaryU,
    View<const NeoN::scalar> boundaryPOld,
    NeoN::scalar gamma,
    NeoN::scalar gasConstant,
    NeoN::scalar plenumTemperature,
    NeoN::scalar plenumPressure,
    NeoN::scalar inletAreaRatio,
    NeoN::scalar dischargeCoefficient,
    NeoN::scalar fraction
)
{
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
            const NeoN::Vec3 uExit = boundaryU[i] / inletAreaRatio;
            const NeoN::scalar uE = uExit & uExit;
            const NeoN::scalar r =
                NeoN::scalar(1)
                - (gamma - NeoN::scalar(1)) * uE
                      / (NeoN::scalar(2) * gamma * gasConstant * plenumTemperature);
            const NeoN::scalar a =
                (NeoN::scalar(1) - r) / (r * r * dischargeCoefficient * dischargeCoefficient);
            const NeoN::scalar s =
                NeoN::scalar(2)
                / (NeoN::scalar(1) + Kokkos::sqrt(NeoN::scalar(1) + NeoN::scalar(4) * a));
            const NeoN::scalar t = Kokkos::pow(s, gamma / (gamma - NeoN::scalar(1)));
            const NeoN::scalar pNew = boundaryPhi[i] >= NeoN::scalar(0)
                                        ? Kokkos::max(valueV[i], plenumPressure)
                                        : t * plenumPressure;
            const NeoN::scalar newValue = boundaryPOld[i] + fraction * (pNew - boundaryPOld[i]);
            refGradV[i] = NeoN::scalar(0);
            refValueV[i] = newValue;
            valueFractionV[i] = NeoN::scalar(1);
            valueV[i] = newValue;
        },
        "setPlenumPressureValue"
    );
}

} // namespace detail

// PlenumPressure: the pressure of a plenum that feeds the domain through the patch, modelled as a
// lumped volume (OpenFOAM's plenumPressure). The plenum receives the supply mass flow and loses
// what flows into the domain; its density and temperature follow from mass and energy balance,
//   rho_pl = rho_old + dt/V (m_supply - m)
//   T_pl   = T_old + dt / (rho_pl cv V) (m_supply (cp T_supply - cv T_pl) - m R T_pl)
// with m = -sum(phi) over the patch (the mass flow into the domain), T_pl on the right the value of
// the previous update, as OpenFOAM, and the patch pressure follows from
// the plenum pressure and the channel exit flow (see detail::setPlenumPressureValue).
//
// The patch is stateful. Like OpenFOAM it keeps the plenum state at the start of the time index
// (rho_old, T_old); every update integrates from that state, so repeated updates within a time
// step (outer correctors) recompute the step from the latest flux instead of integrating twice,
// and the first update of a new time index adopts the last state as the new old state.
//
// Needs, via the context-aware correctBoundaryCondition(domainVector, ctx) overload:
//   - the mass flux, a surface scalar under `phiName_` (dict key "phi", default "phi")
//   - the velocity, a volume vector under `UName_` (dict key "U", default "U")
//   - the field this BC is attached to under `fieldName_` (dict key "field", default "p"), for
//     its old-time boundary value, as WaveTransmissive does
//   - the values "deltaT" and "timeIndex" (BoundaryContext::insertValue)
// A missing one is an error, since the plenum state would silently be wrong. The evaluation that
// follows the solve of the field (context value evaluateOnly > 0) keeps the values, as OpenFOAM,
// which updates the patch once when the matrix is built. Without a context the patch holds the
// "value" of the dictionary (the adjacent cell values if there is none) and the state is not
// touched.
//
// A volumetric flux, which OpenFOAM combines with a given `rho`, is not supported: a `rho` entry
// is rejected.
//
// A Dirichlet patch like OpenFOAM's (it derives from fixedValue): assignable = false and
// fixesValue = true.
class PlenumPressure : public VolumeBoundaryFactory<NeoN::scalar>::template Register<PlenumPressure>
{
    using Base = VolumeBoundaryFactory<NeoN::scalar>::template Register<PlenumPressure>;

public:

    using Base::correctBoundaryCondition;

    PlenumPressure(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true}), mesh_(mesh),
          gamma_(required(dict, "gamma", patchID)), gasConstant_(required(dict, "R", patchID)),
          supplyMassFlowRate_(required(dict, "supplyMassFlowRate", patchID)),
          supplyTotalTemperature_(required(dict, "supplyTotalTemperature", patchID)),
          plenumVolume_(required(dict, "plenumVolume", patchID)),
          plenumDensity_(required(dict, "plenumDensity", patchID)),
          plenumDensityOld_(plenumDensity_),
          plenumTemperature_(required(dict, "plenumTemperature", patchID)),
          plenumTemperatureOld_(plenumTemperature_),
          inletAreaRatio_(required(dict, "inletAreaRatio", patchID)),
          dischargeCoefficient_(required(dict, "inletDischargeCoefficient", patchID)),
          timeScale_(dict.contains("timeScale") ? dict.get<NeoN::scalar>("timeScale") : 0),
          hasInitialValue_(dict.contains("value")),
          initialValue_(hasInitialValue_ ? dict.get<NeoN::scalar>("value") : 0), timeIndex_(-1),
          phiName_(word(dict, "phi", "phi")), UName_(word(dict, "U", "U")),
          fieldName_(word(dict, "field", "p"))
    {
        if (dict.contains("rho"))
        {
            NF_THROW(
                "plenumPressure boundary condition on patch " + std::to_string(patchID)
                + ": a volumetric flux (entry 'rho') is not supported, phi has to be a mass flux"
            );
        }
        if (plenumVolume_ <= NeoN::scalar(0) || gamma_ <= NeoN::scalar(1)
            || gasConstant_ <= NeoN::scalar(0))
        {
            NF_THROW(
                "plenumPressure boundary condition on patch " + std::to_string(patchID)
                + ": plenumVolume and R must be positive and gamma greater than 1"
            );
        }
    }

    // No context: no flux to integrate. Hold the initial value (or the adjacent cells).
    virtual void correctBoundaryCondition(Field<NeoN::scalar>& domainVector) final
    {
        const auto internalV = domainVector.internalVector().view();
        const auto ownersV = mesh_.boundaryMesh().faceOwners().view();
        auto [refGradV, valueV, valueFractionV, refValueV] = views(
            domainVector.boundaryData().refGrad(),
            domainVector.boundaryData().value(),
            domainVector.boundaryData().valueFraction(),
            domainVector.boundaryData().refValue()
        );
        const bool useInitial = hasInitialValue_;
        const NeoN::scalar initial = initialValue_;
        NeoN::parallelFor(
            domainVector.exec(),
            this->range(),
            NEON_LAMBDA(const localIdx i) {
                const NeoN::scalar newValue = useInitial ? initial : internalV[ownersV[i]];
                refGradV[i] = NeoN::scalar(0);
                refValueV[i] = newValue;
                valueFractionV[i] = NeoN::scalar(1);
                valueV[i] = newValue;
            },
            "plenumPressureHold"
        );
    }

    virtual void
    correctBoundaryCondition(Field<NeoN::scalar>& domainVector, const BoundaryContext& ctx) final
    {
        if (ctx.hasValue(evaluateOnlyKey) && ctx.value(evaluateOnlyKey) > NeoN::scalar(0)) return;

        if (!(ctx.hasSurfaceScalar(phiName_) && ctx.hasVector(UName_) && ctx.hasScalar(fieldName_)
              && ctx.hasValue("deltaT") && ctx.hasValue("timeIndex")))
        {
            NF_THROW(
                "plenumPressure boundary condition on patch " + std::to_string(this->patchID())
                + " needs the flux '" + phiName_ + "', the velocity '" + UName_ + "', the field '"
                + fieldName_ + "' and the values deltaT and timeIndex in the boundary context"
            );
        }

        const auto& phi = ctx.surfaceScalarField(phiName_);
        const auto& U = ctx.vectorFieldPtr(UName_);
        const auto& pOld = oldTime(ctx.scalarFieldPtr(fieldName_));
        const NeoN::scalar dt = ctx.value("deltaT");
        const NeoN::scalar timeIndex = ctx.value("timeIndex");

        if (timeIndex != timeIndex_)
        {
            timeIndex_ = timeIndex;
            plenumDensityOld_ = plenumDensity_;
            plenumTemperatureOld_ = plenumTemperature_;
        }

        const auto phiV = phi.boundaryData().value().view();
        const NeoN::scalar massFlowRate = -detail::patchSum(
            mesh_,
            this->range(),
            NEON_LAMBDA(const localIdx i, NeoN::scalar& sum) { sum += phiV[i]; }
        );

        const NeoN::scalar cv = gasConstant_ / (gamma_ - NeoN::scalar(1));
        const NeoN::scalar cp = gasConstant_ * gamma_ / (gamma_ - NeoN::scalar(1));
        plenumDensity_ =
            plenumDensityOld_ + (dt / plenumVolume_) * (supplyMassFlowRate_ - massFlowRate);
        plenumTemperature_ =
            plenumTemperatureOld_
            + (dt / (plenumDensity_ * cv * plenumVolume_))
                  * (supplyMassFlowRate_ * (cp * supplyTotalTemperature_ - cv * plenumTemperature_)
                     - massFlowRate * gasConstant_ * plenumTemperature_);
        const NeoN::scalar plenumPressure = plenumDensity_ * gasConstant_ * plenumTemperature_;

        const NeoN::scalar oneByFraction = timeScale_ / dt;
        const NeoN::scalar fraction =
            oneByFraction < NeoN::scalar(1) ? NeoN::scalar(1) : NeoN::scalar(1) / oneByFraction;

        detail::setPlenumPressureValue(
            domainVector,
            this->range(),
            phiV,
            U.boundaryData().value().view(),
            pOld.boundaryData().value().view(),
            gamma_,
            gasConstant_,
            plenumTemperature_,
            plenumPressure,
            inletAreaRatio_,
            dischargeCoefficient_,
            fraction
        );
    }

    // the plenum state, for output and tests
    NeoN::scalar plenumDensity() const { return plenumDensity_; }

    NeoN::scalar plenumTemperature() const { return plenumTemperature_; }

    NeoN::scalar plenumPressure() const
    {
        return plenumDensity_ * gasConstant_ * plenumTemperature_;
    }

    static std::string name() { return "plenumPressure"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Pressure of a lumped plenum that feeds the domain through the patch, from the "
               "mass and energy balance of the plenum and the isentropic exit flow (OpenFOAM's "
               "plenumPressure).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::scalar>> clone() const final
    {
        return std::make_unique<PlenumPressure>(*this);
    }

private:

    static NeoN::scalar required(const Dictionary& dict, const std::string& key, localIdx patchID)
    {
        if (!dict.contains(key))
        {
            NF_THROW(
                "plenumPressure boundary condition on patch " + std::to_string(patchID)
                + " requires a '" + key + "' entry"
            );
        }
        return dict.get<NeoN::scalar>(key);
    }

    static std::string word(const Dictionary& dict, const std::string& key, const char* fallback)
    {
        return dict.contains(key) ? dict.get<std::string>(key) : std::string(fallback);
    }

    const UnstructuredMesh& mesh_;
    NeoN::scalar gamma_;
    NeoN::scalar gasConstant_;
    NeoN::scalar supplyMassFlowRate_;
    NeoN::scalar supplyTotalTemperature_;
    NeoN::scalar plenumVolume_;
    NeoN::scalar plenumDensity_;
    NeoN::scalar plenumDensityOld_;
    NeoN::scalar plenumTemperature_;
    NeoN::scalar plenumTemperatureOld_;
    NeoN::scalar inletAreaRatio_;
    NeoN::scalar dischargeCoefficient_;
    NeoN::scalar timeScale_;
    bool hasInitialValue_;
    NeoN::scalar initialValue_;
    NeoN::scalar timeIndex_;
    std::string phiName_;
    std::string UName_;
    std::string fieldName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
