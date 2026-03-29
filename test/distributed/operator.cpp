// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

/** @brief helper function to set the processor boundaries of a distributed field */
template<typename BoundaryType>
auto setProcessorBoundaryHelper(std::vector<BoundaryType> bcs, size_t rank)
{
    return bcs;
}

auto generateInput = [](std::string scheme, std::string post)
{
    auto constructDiv = [](auto post) { return "div(phi" + post + ",U" + post + ")"; };
    auto constructGamma = [](auto post) { return "laplacian(gamma" + post + ",U" + post + ")"; };

    return NeoN::Dictionary {
        {
            "laplacianSchemes",
            NeoN::Dictionary {
                {constructGamma(post),
                 NeoN::TokenList(
                     {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                 )}
            },
        },
        {"divSchemes",
         NeoN::Dictionary {{constructDiv(post), NeoN::TokenList({std::string("Gauss"), scheme})}}}
    };
};

TEST_CASE("Distributed")
{
    // start with non distributed setup
    float epsilon = 1e-32;


    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto input = generateInput("upwind", "");
    auto inputPart = generateInput("upwind", "Part");

    auto nCells = 12;
    auto meshGlobal = create1DUniformMesh(exec, nCells);
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    randomizeVector(U);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh, surfaceBCs);

    fill(phi.internalVector(), 1.0);
    randomizeVector(phi.internalVector());
    fill(gamma.internalVector(), 2.0);

    // assembly
    auto expr = NeoN::dsl::Expression<NeoN::scalar>(
        // NeoN::dsl::imp::div(phi, U)
        // -
        NeoN::dsl::imp::laplacian(gamma, U)
    );
    expr.read(input);
    auto [sp, ls] = expr.assemble(mesh, 1.0, 1.0);

    NeoN::mpi::Environment mpiEnviron;
    auto [meshPart, commPattern] =
        create1DUniformMeshPart(exec, meshGlobal.nCells() / mpiEnviron.sizeRank(), mpiEnviron);

    // partition fields and data
    auto volBCsII = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(meshPart);
    auto volBCsPart = setProcessorBoundaryHelper(volBCsII, mpiEnviron.rank());
    auto uPart = partitionVolField(U, meshPart, volBCsPart, mpiEnviron);
    auto surfaceBCsII = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(meshPart);
    auto surfaceBCsPart = setProcessorBoundaryHelper(surfaceBCsII, mpiEnviron.rank());
    auto phiPart = partitionSurfaceField(phi, meshPart, surfaceBCsPart, mpiEnviron, false);
    auto gammaPart = partitionSurfaceField(gamma, meshPart, surfaceBCsPart, mpiEnviron, false);

    auto exprDist = NeoN::dsl::Expression<NeoN::scalar>(
        // NeoN::dsl::imp::div(phiPart, uPart)
        // -
        NeoN::dsl::imp::laplacian(gammaPart, uPart)
    );

    exprDist.read(inputPart);

    auto [spDst, lsDst] = exprDist.assembleDistributed(meshPart, 1.0, 1.0, commPattern);

    localIdx firstElement = 0;
    localIdx lastElement = 0;

    if (mpiEnviron.rank() == 0)
    {
        lastElement = 10;
    }
    if (mpiEnviron.rank() == 1)
    {
        firstElement = 12;
        lastElement = 22;
    }
    if (mpiEnviron.rank() == 2)
    {
        firstElement = 24;
        lastElement = 34;
    }

    SECTION_IF(mpiEnviron.rank() == 0, "Correct mtx on rank 0")
    {
        compare(
            take(ls.matrix().values(), firstElement, lastElement),
            lsDst.matrix().values(),
            ApproxScalar(1e-15)
        );
    }

    SECTION_IF(mpiEnviron.rank() == 1, "Correct mtx on rank 1")
    {
        compare(
            take(ls.matrix().values(), firstElement, lastElement),
            lsDst.matrix().values(),
            ApproxScalar(1e-15)
        );
    }

    SECTION_IF(mpiEnviron.rank() == 2, "Correct mtx on rank 2")
    {
        compare(
            take(ls.matrix().values(), firstElement, lastElement),
            lsDst.matrix().values(),
            ApproxScalar(1e-15)
        );
    }

    Dictionary solverDict {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Cg"},
         {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
    };

    // Create solver
    auto solver = NeoN::la::Solver(exec, solverDict);
    auto x = Vector<scalar>(exec, 12);
    fill(x, 0.0);

    auto xPart = Vector<scalar>(exec, 4);
    fill(xPart, 0.0);

    auto solverStats = solver.solve(ls, x);
    auto solverStatsDist = solver.solveDist(lsDst, xPart, commPattern);

    auto [numIterDist, initResNormDist, finalResNormDist, solveTimeDist] =
        solverStatsDist.entries[0];
    auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];

    // REQUIRE(numIterDist != 0);
    REQUIRE(numIterDist == numIter);
    REQUIRE(initResNormDist == initResNorm);
}

}
