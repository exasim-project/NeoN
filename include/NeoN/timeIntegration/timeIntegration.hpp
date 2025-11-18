// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/dsl/expression.hpp"

namespace NeoN::timeIntegration
{

/* @class Factory class to create time integration method by a given name
 * using NeoNs runTimeFactory mechanism
 */
template<typename SolutionType>
class TimeIntegratorBase :
    public RuntimeSelectionFactory<
        TimeIntegratorBase<SolutionType>,
        Parameters<const Dictionary&, const Dictionary&>>
{

public:

    using ValueType = typename SolutionType::VectorValueType;
    using Expression = NeoN::dsl::Expression<ValueType>;
    using VolVectorField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
    using SurfScalarField = NeoN::finiteVolume::cellCentred::SurfaceField<scalar>;

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegratorBase(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : schemeDict_(schemeDict), solutionDict_(solutionDict)
    {}

    virtual ~TimeIntegratorBase() {}

    virtual void solve(
        Expression& eqn, SolutionType& sol, scalar t, scalar dt
    ) = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegratorBase> clone() const = 0;

    virtual bool explicitIntegration() const { return true; }

    virtual SurfScalarField
    ddtPhiCorr(const VolVectorField& U, const SurfScalarField& phi, scalar dt) const
    {
        // construct a surface field with the same mesh & default BCs
        auto surfaceBCs = NeoN::finiteVolume::cellCentred::createCalculatedBCs<
            NeoN::finiteVolume::cellCentred::SurfaceBoundary<scalar>>(phi.mesh());
        SurfScalarField out(phi.exec(), std::string("ddtPhiCorr"), phi.mesh(), surfaceBCs);

        // fill internal & boundary with zero
        NeoN::fill(out.internalVector(), scalar(0));
        NeoN::fill(out.boundaryData().value(), scalar(0));
        NeoN::fill(out.boundaryData().refValue(), scalar(0));
        NF_INFO("Warning! ddtPhiCorr not implemented for this ddtScheme."
                "Falling back to phiHbyA = flux(HbyA).");
        return out;
    }

protected:

    const Dictionary& schemeDict_;
    const Dictionary& solutionDict_;
};

/**
 * @class Factory class to create time integration method by a given name
 * using NeoNs runTimeFactory mechanism
 *
 * @tparam SolutionVectorType Type of the solution field eg, volumeVector or just a plain Vector
 */
template<typename SolutionVectorType>
class TimeIntegration
{

public:

    using ValueType = typename SolutionVectorType::VectorValueType;
    using Expression = NeoN::dsl::Expression<ValueType>;
    using VolVectorField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
    using SurfScalarField = NeoN::finiteVolume::cellCentred::SurfaceField<scalar>;

    TimeIntegration(const TimeIntegration& timeIntegrator)
        : timeIntegratorStrategy_(timeIntegrator.timeIntegratorStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrator)
        : timeIntegratorStrategy_(std::move(timeIntegrator.timeIntegratorStrategy_)) {};

    TimeIntegration(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : timeIntegratorStrategy_(TimeIntegratorBase<SolutionVectorType>::create(
            schemeDict.get<std::string>("type"), schemeDict, solutionDict
        )) {};

    void solve(Expression& eqn, SolutionVectorType& sol, scalar t, scalar dt)
    {
        timeIntegratorStrategy_->solve(eqn, sol, t, dt);
    }

    bool explicitIntegration() const { return timeIntegratorStrategy_->explicitIntegration(); }

    SurfScalarField ddtPhiCorr(const VolVectorField& U, const SurfScalarField& phi, scalar dt) const
    {
        return timeIntegratorStrategy_->ddtPhiCorr(U, phi, dt);
    }

private:

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> timeIntegratorStrategy_;
};


} // namespace NeoN::timeIntegration
