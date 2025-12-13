// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"
#include "NeoN/timeIntegration/ddtPhiCorrBoundary.hpp"

namespace NeoN::timeIntegration
{

template<typename SolutionVectorType>
class ForwardEuler :
    public TimeIntegratorBase<SolutionVectorType>::template Register<
        ForwardEuler<SolutionVectorType>>
{

public:

    using ValueType = typename SolutionVectorType::VectorValueType;
    using Base =
        TimeIntegratorBase<SolutionVectorType>::template Register<ForwardEuler<SolutionVectorType>>;
    using VolVectorField = typename Base::VolVectorField;
    using SurfScalarField = typename Base::SurfScalarField;

    ForwardEuler(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : Base(schemeDict, solutionDict)
    {}

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(
        dsl::Expression<ValueType>& eqn,
        SolutionVectorType& solutionVector,
        [[maybe_unused]] scalar t,
        scalar dt
    ) override
    {
        auto source = eqn.explicitOperation(solutionVector.size());
        SolutionVectorType& oldSolutionVector =
            NeoN::finiteVolume::cellCentred::oldTime(solutionVector);

        solutionVector.internalVector() = oldSolutionVector.internalVector() - source * dt;
        solutionVector.correctBoundaryConditions();

        fence(eqn.exec());
    };

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
    SurfScalarField
    ddtPhiCorr(const VolVectorField& U, const SurfScalarField& phi, scalar dt) const override
    {
        const auto& mesh = U.mesh();
        const auto exec = phi.exec();
        const scalar rDeltaT = 1.0 / dt;

        const VolVectorField& U0 = NeoN::finiteVolume::cellCentred::oldTime(U);
        const SurfScalarField& phi0 = NeoN::finiteVolume::cellCentred::oldTime(phi);

        NeoN::finiteVolume::cellCentred::SurfaceInterpolation<Vec3> interp(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );
        auto Uf0 = interp.interpolate(U0); // SurfaceField<Vec3>

        auto surfaceBCs = createPhiCorrBCsFromU(mesh, U); // Custom BC that implements OF logic
        SurfScalarField phiCorr(exec, std::string("ddtPhiCorr"), mesh, surfaceBCs);

        auto [outV, phi0V, UfV, SfV] = views(
            phiCorr.internalVector(), phi0.internalVector(), Uf0.internalVector(), mesh.faceAreas()
        );
        const size_t N = outV.size();

        NeoN::parallelFor(
            exec,
            {size_t(0), N},
            KOKKOS_LAMBDA(const localIdx i) {
                const auto d = (SfV[i] & UfV[i]); // dot product via operator&
                const auto tphiCorr = (phi0V[i] - d);
                const auto ratio = mag(tphiCorr)
                                 / (mag(phi0V[i]) + scalar(1e-30)
                                 ); // TODO currently hardcoded OpenFOAM double prescision SMALL
                const auto coeff = scalar(1.0) - Kokkos::min(ratio, scalar(1));
                outV[i] = coeff * rDeltaT * tphiCorr;
            }
        );

        phiCorr.correctBoundaryConditions();

        return phiCorr;
    }
};


} // namespace NeoN
