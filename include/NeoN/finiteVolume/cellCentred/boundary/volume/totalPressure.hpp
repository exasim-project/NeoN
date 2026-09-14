// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
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

// Shared kernel behind totalPressure, for the incompressible form
//
//   p = p0 - 0.5 * (1 - pos0(phi)) * magSqr(U)
//
// i.e. on outflow (phi >= 0) the static pressure is the specified total pressure, and on
// inflow (phi < 0) the dynamic head 0.5*|U|^2 is subtracted so that the *total* pressure of
// the entering stream is p0. The patch is Dirichlet either way: valueFraction = 1 with
// refGrad = 0, only the value differs per face.
//
// Without a flux (hasFlux == false, the plain correctBoundaryCondition path with no
// BoundaryContext) every face is treated as outflow, degenerating to a fixedValue at p0.
// Without a velocity the dynamic head is zero, which is the same degeneration.
template<typename ValueType>
void setTotalPressureValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    ValueType p0,
    View<const NeoN::scalar> boundaryFlux,
    bool hasFlux,
    View<const Vec3> boundaryVelocity,
    bool hasVelocity
)
{
    auto [refGradV, valueV, valueFractionV, refValueV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );
    (void)mesh;

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            // pos0(phi): 1 on outflow (phi >= 0), 0 on inflow.
            const bool inflow = hasFlux && boundaryFlux[i] < NeoN::scalar(0);

            ValueType p = p0;
            if constexpr (std::is_same<ValueType, NeoN::scalar>::value)
            {
                if (inflow && hasVelocity)
                {
                    const Vec3 u = boundaryVelocity[i];
                    const NeoN::scalar magSqrU = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                    p = p0 - NeoN::scalar(0.5) * magSqrU;
                }
            }

            valueFractionV[i] = NeoN::scalar(1);
            refValueV[i] = p;
            refGradV[i] = NeoN::zero<ValueType>();
            valueV[i] = p;
        },
        "setTotalPressureValue"
    );
}

} // namespace detail

// totalPressure fixes the *total* pressure on a patch: the static pressure it imposes is the
// specified p0 on outflow, and p0 minus the dynamic head on inflow, so a stream entering the
// domain carries total pressure p0. It mirrors OpenFOAM's incompressible totalPressure
// (a fixedValue whose value is p0 - 0.5*(1 - pos0(phi))*magSqr(U)).
//
// The flux and velocity are looked up by name from the BoundaryContext passed to the
// context-aware correctBoundaryCondition overload (keys "phi" and "U", both overridable via
// the dictionary). The plain overload has neither and degenerates to a fixedValue at p0, so a
// solver that wants the dynamic head must call correctBoundaryConditions(ctx) with phi and U
// inserted.
//
// Not modelled: the compressible forms (psi/rho variants) and a time-varying p0.
template<typename ValueType>
class TotalPressure :
    public VolumeBoundaryFactory<ValueType>::template Register<TotalPressure<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<TotalPressure<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    // assignable = true: the value is recomputed on every correctBoundaryCondition, so a
    // downstream overwrite is reversible. fixesValue = true: the patch is Dirichlet on every
    // face, which is what the pressure equation needs to pin its datum here.
    TotalPressure(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = true}), mesh_(mesh),
          p0_(dict.get<ValueType>("p0")), // required: an omitted datum is a case-setup error
          phiName_(dict.contains("phi") ? dict.get<std::string>("phi") : std::string("phi")),
          uName_(dict.contains("U") ? dict.get<std::string>("U") : std::string("U"))
    {}

    // No context => no flux or velocity: a plain fixedValue at p0.
    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setTotalPressureValue(
            domainVector,
            mesh_,
            this->range(),
            p0_,
            View<const NeoN::scalar> {},
            false,
            View<const Vec3> {},
            false
        );
    }

    // Context-aware: subtract the dynamic head on inflow faces.
    virtual void
    correctBoundaryCondition(Field<ValueType>& domainVector, const BoundaryContext& ctx) final
    {
        const bool hasFlux = ctx.hasSurfaceScalar(phiName_);
        const bool hasVelocity = ctx.hasVector(uName_);

        detail::setTotalPressureValue(
            domainVector,
            mesh_,
            this->range(),
            p0_,
            hasFlux ? ctx.surfaceScalarField(phiName_).boundaryData().value().view()
                    : View<const NeoN::scalar> {},
            hasFlux,
            hasVelocity ? ctx.vectorFieldPtr(uName_).boundaryData().value().view()
                        : View<const Vec3> {},
            hasVelocity
        );
    }

    static std::string name() { return "totalPressure"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Total-pressure patch: static pressure p0 on outflow, p0 minus the dynamic "
               "head 0.5*magSqr(U) on inflow, so the entering stream carries total pressure "
               "p0. Needs phi and U in the BoundaryContext; without them it is a fixedValue "
               "at p0.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<TotalPressure>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    ValueType p0_;
    std::string phiName_;
    std::string uName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
