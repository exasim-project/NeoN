// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Steady 1D heat conduction with a constant heat source
//
//     -d/dx (gamma dT/dx) = S,  x in [0, 1],  T(0) = 0,  T(1) = 1
//
// exact solution: T(x) = x + S/(2 gamma) x (1 - x)
// The problem is solved on refined meshes to check that the L2 error converges with second order.

#include <cmath>
#include <iostream>
#include <vector>

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

using NeoN::scalar;
using NeoN::localIdx;

constexpr scalar diffusivity = 1.0; // gamma
constexpr scalar heatSource = 10.0; // S

scalar exactSolution(scalar x) { return x + heatSource / (2.0 * diffusivity) * x * (1.0 - x); }

/* @brief solves the problem on a mesh with nCells and returns the L2 error */
scalar solveSteadyDiffusion(const NeoN::Executor& exec, localIdx nCells)
{
    // patch 0 is at x = 0, patch 1 at x = 1
    auto mesh = NeoN::create1DUniformMesh(exec, nCells);

    // the mesh frees its cell centres after the first solve, so copy them here
    auto centres = mesh.cellCenters().copyToHost();

    // fixed temperature at both ends
    std::vector<fvcc::VolumeBoundary<scalar>> bcs;
    bcs.emplace_back(
        mesh, NeoN::Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", 0.0}}), 0
    );
    bcs.emplace_back(
        mesh, NeoN::Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", 1.0}}), 1
    );

    fvcc::VolumeField<scalar> T(exec, "T", mesh, bcs);
    NeoN::fill(T.internalVector(), 0.0);
    T.correctBoundaryConditions();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gammaF(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gammaF.internalVector(), diffusivity);
    NeoN::fill(gammaF.boundaryData().value(), diffusivity);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> source(exec, "S", mesh, volumeBCs);
    NeoN::fill(source.internalVector(), heatSource);

    NeoN::Dictionary fvSchemes {
        {{"laplacianSchemes",
          NeoN::Dictionary {
              {{"laplacian(gamma,T)",
                NeoN::TokenList(
                    {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                )}}
          }},
         {"timeIntegration", NeoN::Dictionary {{{"type", std::string("steadyState")}}}}}
    };

    NeoN::Dictionary fvSolution {
        {{"solver", std::string("Ginkgo")},
         {"type", std::string("solver::Cg")},
         {"criteria", NeoN::Dictionary {{{"iteration", 1000}, {"relative_residual_norm", 1e-12}}}}}
    };

    // imp:: terms are assembled into the matrix, exp:: terms into the rhs
    dsl::Expression<scalar> eqn = dsl::imp::laplacian(gammaF, T) + dsl::exp::source(source);

    dsl::solve(eqn, T, 0.0, 1.0, fvSchemes, fvSolution);

    // L2 error
    auto Th = T.internalVector().copyToHost();
    auto vol = mesh.cellVolumes().copyToHost();
    scalar err = 0.0;
    for (localIdx i = 0; i < nCells; ++i)
    {
        const scalar diff = Th.view()[i] - exactSolution(centres.view()[i][0]);
        err += diff * diff * vol.view()[i];
    }
    return std::sqrt(err);
}

int main(int argc, char* argv[])
{
    NeoN::initialize(argc, argv);
    int result = 0;
    {
        // runs on the GPU if NeoN was built with GPU support
        NeoN::Executor exec = NeoN::createDefaultExecutor();
        std::cout << "Running on " << NeoN::executorName(exec) << "\n\n";

        std::cout << "  nCells    L2 error    order\n";
        scalar prevErr = 0.0;
        scalar order = 0.0;
        for (localIdx nCells : {10, 20, 40, 80})
        {
            const scalar err = solveSteadyDiffusion(exec, nCells);
            std::cout << "  " << nCells << "\t    " << err;
            if (prevErr > 0.0)
            {
                order = std::log2(prevErr / err);
                std::cout << "    " << order;
            }
            std::cout << "\n";
            prevErr = err;
        }

        if (order < 1.8)
        {
            std::cerr << "\nUnexpected convergence order " << order << "\n";
            result = 1;
        }
    }
    NeoN::finalize();
    return result;
}
