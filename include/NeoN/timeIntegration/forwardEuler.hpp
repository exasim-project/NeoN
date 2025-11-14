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
    using VolVector = typename Base::VolVector;
    using SurfScalar = typename Base::SurfScalar;

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
    SurfScalar ddtPhiCorr(const VolVector& U, const SurfScalar& phi, scalar dt) const override
    {
        const auto& mesh = U.mesh();
        const auto exec = phi.exec();
        const scalar rDeltaT = 1.0 / dt;

        const VolVector& U0 = NeoN::finiteVolume::cellCentred::oldTime(U);
        const SurfScalar& phi0 = NeoN::finiteVolume::cellCentred::oldTime(phi);

        NeoN::finiteVolume::cellCentred::SurfaceInterpolation<Vec3> interp(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );
        auto Uf0 = interp.interpolate(U0); // SurfaceField<Vec3>

        // auto surfaceBCs =
        //     NeoN::finiteVolume::cellCentred::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh
        //     );
        auto surfaceBCs = createPhiCorrBCsFromU(mesh, U); // Custom BC that implements OF logic
        SurfScalar phiCorr(exec, std::string("ddtPhiCorr"), mesh, surfaceBCs);

        auto out = phiCorr.internalVector().view();
        auto ph0 = phi0.internalVector().view();
        auto Uf = Uf0.internalVector().view();
        auto Sf = mesh.faceAreas().view(); // boundaryMesh().sf().view();
        const size_t N = out.size();

        NF_INFO(
            "N: " + std::to_string(N) + ", ph0: " + std::to_string(ph0.size())
            + ", Uf: " + std::to_string(Uf.size()) + ", Sf: " + std::to_string(Sf.size())
        );

        NeoN::parallelFor(
            exec,
            {size_t(0), N},
            KOKKOS_LAMBDA(const localIdx i) {
                const auto d = (Sf[i] & Uf[i]); // <— dot product via operator&
                const auto tphiCorr = (ph0[i] - d);
                const auto ratio = mag(tphiCorr) / (mag(ph0[i]) + scalar(1e-30));
                const auto coeff = scalar(1.0) - Kokkos::min(ratio, scalar(1));
                out[i] = coeff * rDeltaT * tphiCorr;
            }
        );

        phiCorr.correctBoundaryConditions();

        return phiCorr;
    }
};


} // namespace NeoN
