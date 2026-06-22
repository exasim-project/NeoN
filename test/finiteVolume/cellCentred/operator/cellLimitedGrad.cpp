// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#include <algorithm>
#include <any>
#include <cmath>


namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

// The cellLimited scheme wraps a base gradient (Gauss linear) and clips the
// reconstructed gradient with the minmod slope limiter. These tests run on a 1D
// uniform mesh where the gradient has a single non-zero (x) component, so all
// assertions compare component [0]. Only interior cells [1, nCells-1) are
// checked — they are unaffected by the boundary-face contributions.
TEST_CASE("cellLimited gradient")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);

    // Build a cellLimited grad operator from the scheme tokens
    //   cellLimited Gauss linear <coeff>
    // via the runtime factory, exactly as a gradSchemes entry would.
    auto makeLimited = [&](const fvcc::VolumeField<scalar>& phi, std::any coeff)
    {
        Input input = TokenList(
            {std::string("cellLimited"), std::string("Gauss"), std::string("linear"), coeff}
        );
        auto op = fvcc::GradOperatorFactory<Vec3>::create(exec, mesh, input);
        return op->grad(phi, dsl::Coeff {});
    };

    auto linearField = [&]()
    {
        fvcc::VolumeField<scalar> phi(exec, "phi", mesh, bcs);
        parallelFor(
            phi.internalVector(), NEON_LAMBDA(const localIdx i) { return scalar(2) * scalar(i); }
        );
        fill(phi.boundaryData().value(), scalar(0));
        phi.correctBoundaryConditions();
        return phi;
    };

    auto stepField = [&]()
    {
        fvcc::VolumeField<scalar> phi(exec, "phi", mesh, bcs);
        const localIdx half = nCells / 2;
        parallelFor(
            phi.internalVector(),
            NEON_LAMBDA(const localIdx i) { return i < half ? scalar(0) : scalar(1); }
        );
        fill(phi.boundaryData().value(), scalar(0));
        phi.correctBoundaryConditions();
        return phi;
    };

    SECTION("is registered in the grad operator factory on " + execName)
    {
        auto entries = fvcc::GradOperatorFactory<Vec3>::entries();
        REQUIRE(
            std::find(entries.begin(), entries.end(), std::string("cellLimited")) != entries.end()
        );
    }

    SECTION("linear field is not clipped — equals base Gauss gradient on " + execName)
    {
        auto phi = linearField();

        fvcc::GaussGreenGrad gauss(exec, mesh);
        auto gBase = gauss.grad(phi);
        auto gLim = makeLimited(phi, scalar(1.0));

        auto hBase = gBase.internalVector().copyToHost();
        auto hLim = gLim.internalVector().copyToHost();
        auto hBaseV = hBase.view();
        auto hLimV = hLim.view();

        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            REQUIRE(std::abs(hBaseV[i][0]) > 1e-6); // gradient is genuinely non-trivial
            REQUIRE(hLimV[i][0] == Catch::Approx(hBaseV[i][0]).margin(1e-12));
        }
    }

    SECTION("step field is clipped — limited magnitude below base on " + execName)
    {
        auto phi = stepField();

        fvcc::GaussGreenGrad gauss(exec, mesh);
        auto gBase = gauss.grad(phi);
        auto gLim = makeLimited(phi, scalar(1.0));

        auto hBase = gBase.internalVector().copyToHost();
        auto hLim = gLim.internalVector().copyToHost();
        auto hBaseV = hBase.view();
        auto hLimV = hLim.view();

        bool anyClipped = false;
        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            const scalar magBase = std::abs(hBaseV[i][0]);
            const scalar magLim = std::abs(hLimV[i][0]);
            // the limiter lies in [0,1], so the limited gradient never grows
            REQUIRE(magLim <= magBase + 1e-12);
            if (magLim < magBase - 1e-9)
            {
                anyClipped = true;
            }
        }
        REQUIRE(anyClipped); // the jump must be limited somewhere
    }

    SECTION("k=0 disables limiting — equals base gradient even at a jump on " + execName)
    {
        auto phi = stepField();

        fvcc::GaussGreenGrad gauss(exec, mesh);
        auto gBase = gauss.grad(phi);
        auto gLim = makeLimited(phi, scalar(0.0));

        auto hBase = gBase.internalVector().copyToHost();
        auto hLim = gLim.internalVector().copyToHost();
        auto hBaseV = hBase.view();
        auto hLimV = hLim.view();

        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            REQUIRE(hLimV[i][0] == Catch::Approx(hBaseV[i][0]).margin(1e-12));
        }
    }

    SECTION("integer coefficient token (label) parses on " + execName)
    {
        // gradSchemes writes the coefficient as an integer, e.g.
        //   grad(p)  cellLimited Gauss linear 1;
        // which is tokenized as a label rather than a scalar — the scheme must
        // accept it. k=1 (full limiting) leaves a linear field unclipped.
        auto phi = linearField();

        fvcc::GaussGreenGrad gauss(exec, mesh);
        auto gBase = gauss.grad(phi);
        auto gLim = makeLimited(phi, label(1));

        auto hBase = gBase.internalVector().copyToHost();
        auto hLim = gLim.internalVector().copyToHost();
        auto hBaseV = hBase.view();
        auto hLimV = hLim.view();

        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            REQUIRE(hLimV[i][0] == Catch::Approx(hBaseV[i][0]).margin(1e-12));
        }
    }
}

// Tensor path: grad(U) where U is a vector field. On the 1D mesh the only non-zero
// spatial derivative is d/dx, so component (i,0) of the tensor = dU_i/dx; the limiter
// is per U-component (three independent minmod limiters). Interior cells only.
TEST_CASE("cellLimited tensor gradient")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto bcs = fvcc::createCalculatedProcBCs<fvcc::VolumeBoundary<Vec3>>(mesh);

    auto makeLimitedTensor = [&](const fvcc::VolumeField<Vec3>& u, std::any coeff)
    {
        Input input = TokenList(
            {std::string("cellLimited"), std::string("Gauss"), std::string("linear"), coeff}
        );
        auto op = fvcc::GradOperatorFactory<Vec3>::create(exec, mesh, input);
        auto gradU = fvcc::VolumeField<Tensor>(
            exec, "gradU", mesh, fvcc::createCalculatedProcBCs<fvcc::VolumeBoundary<Tensor>>(mesh)
        );
        fill(gradU.internalVector(), zero<Tensor>());
        op->gradTensor(u, gradU, dsl::Coeff {});
        return gradU;
    };

    // U with each component a distinct linear ramp: U = (2 i, -3 i, i).
    auto linearU = [&]()
    {
        fvcc::VolumeField<Vec3> u(exec, "U", mesh, bcs);
        parallelFor(
            u.internalVector(),
            NEON_LAMBDA(const localIdx i) {
                return Vec3(scalar(2) * scalar(i), scalar(-3) * scalar(i), scalar(i));
            }
        );
        fill(u.boundaryData().value(), zero<Vec3>());
        u.correctBoundaryConditions();
        return u;
    };

    // U with a per-component step at the mid-plane.
    auto stepU = [&]()
    {
        fvcc::VolumeField<Vec3> u(exec, "U", mesh, bcs);
        const localIdx half = nCells / 2;
        parallelFor(
            u.internalVector(),
            NEON_LAMBDA(const localIdx i) {
                return i < half ? zero<Vec3>() : Vec3(scalar(1), scalar(1), scalar(1));
            }
        );
        fill(u.boundaryData().value(), zero<Vec3>());
        u.correctBoundaryConditions();
        return u;
    };

    SECTION("linear U is not clipped — equals base Gauss tensor gradient on " + execName)
    {
        auto u = linearU();

        fvcc::GaussGreenGrad gauss(exec, mesh);
        auto gBase = gauss.gradTensor(u);
        auto gLim = makeLimitedTensor(u, scalar(1.0));

        auto hBase = gBase.internalVector().copyToHost();
        auto hLim = gLim.internalVector().copyToHost();
        auto hBaseV = hBase.view();
        auto hLimV = hLim.view();

        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            for (size_t cmpt = 0; cmpt < 3; ++cmpt)
            {
                // gradient of U_cmpt in x is genuinely non-trivial and unclipped
                REQUIRE(std::abs(hBaseV[i](cmpt, 0)) > 1e-6);
                REQUIRE(hLimV[i](cmpt, 0) == Catch::Approx(hBaseV[i](cmpt, 0)).margin(1e-12));
            }
        }
    }

    SECTION("step U is clipped per component — limited magnitude below base on " + execName)
    {
        auto u = stepU();

        fvcc::GaussGreenGrad gauss(exec, mesh);
        auto gBase = gauss.gradTensor(u);
        auto gLim = makeLimitedTensor(u, scalar(1.0));

        auto hBase = gBase.internalVector().copyToHost();
        auto hLim = gLim.internalVector().copyToHost();
        auto hBaseV = hBase.view();
        auto hLimV = hLim.view();

        bool anyClipped = false;
        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            for (size_t cmpt = 0; cmpt < 3; ++cmpt)
            {
                const scalar magBase = std::abs(hBaseV[i](cmpt, 0));
                const scalar magLim = std::abs(hLimV[i](cmpt, 0));
                REQUIRE(magLim <= magBase + 1e-12);
                if (magLim < magBase - 1e-9)
                {
                    anyClipped = true;
                }
            }
        }
        REQUIRE(anyClipped);
    }

    SECTION("k=0 disables tensor limiting — equals base gradient at a jump on " + execName)
    {
        auto u = stepU();

        fvcc::GaussGreenGrad gauss(exec, mesh);
        auto gBase = gauss.gradTensor(u);
        auto gLim = makeLimitedTensor(u, scalar(0.0));

        auto hBase = gBase.internalVector().copyToHost();
        auto hLim = gLim.internalVector().copyToHost();
        auto hBaseV = hBase.view();
        auto hLimV = hLim.view();

        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            for (size_t cmpt = 0; cmpt < 3; ++cmpt)
            {
                REQUIRE(hLimV[i](cmpt, 0) == Catch::Approx(hBaseV[i](cmpt, 0)).margin(1e-12));
            }
        }
    }
}

} // namespace NeoN
