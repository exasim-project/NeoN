// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
auto partitionMeshHelper(auto mesh, size_t rank) { return mesh; }

/** @brief helper function to set the processor boundaries of a distributed field */
template<typename BoundaryType>
auto setProcessorBoundaryHelper(std::vector<BoundaryType> bcs, size_t rank)
{
    return bcs;
}

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
auto partitionField(auto field, auto bcs, size_t rank) { return field; }

TEST_CASE("Distributed")
{
    // start with non distributed setup
    float epsilon = 1e-32;

    auto input = NeoN::Dictionary {
        {
            "laplacianSchemes",
            NeoN::Dictionary {
                {"laplacian(gamma,U)",
                 NeoN::TokenList(
                     {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                 )}
            },
        },
        {"divSchemes",
         NeoN::Dictionary {
             {"div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}
         }}
    };

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 12;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    // randomizeVector(U);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    // auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh,
    // surfaceBCs);

    fill(phi.internalVector(), 1.0);
    // fill(gamma.internalVector(), 2.0);

    // partition fields and data
    NeoN::mpi::Environment mpiEnviron;

    auto meshPart = partitionMeshHelper(mesh, mpiEnviron.rank());
    auto volBCsPart = setProcessorBoundaryHelper(volBCs, mpiEnviron.rank());
    auto uPart = partitionField(U, volBCsPart, mpiEnviron.rank());
    auto surfaceBCsPart = setProcessorBoundaryHelper(surfaceBCs, mpiEnviron.rank());
    auto phiPart = partitionField(phi, surfaceBCsPart, mpiEnviron.rank());


    // assembly
    auto expr = NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::div(phi, U)
    ); // - NeoN::dsl::imp::laplacian(gamma, U);
    expr.read(input);
    auto [sp, ls] = expr.assemble(mesh, 1.0, 1.0);

    SECTION("Can assemble distributed " + execName)
    {
        auto exprDist = NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::div(phi, U)
        ); // - NeoN::dsl::imp::laplacian(gamma, U);
        exprDist.read(input);

        mpi::Environment env;

        std::vector<localIdx> commIdx {};
        std::vector<int> sendCounts {};
        std::vector<int> commRanks {};

        if (env.rank() == 0)
        {
            // communicate the interior value which is
            commIdx = std::vector<localIdx> {1};
            sendCounts = std::vector<int> {1, 0, 0, 1};
        }
        if (env.rank() == 1)
        {
            // communicate the interior value which is
            commIdx = std::vector<localIdx> {0, 1};
            sendCounts = std::vector<int> {1, 1, 0, 2};
        }
        if (env.rank() == 2)
        {
            // communicate the interior value which is
            commIdx = std::vector<localIdx> {0};
            sendCounts = std::vector<int> {0, 1, 0, 1};
        }


        auto commPattern = CommunicationPattern(commIdx, sendCounts, env);

        auto [sp, ls] = expr.assembleDistributed(mesh, 1.0, 1.0, commPattern);

        // auto rhs = ls.rhs();
        // SECTION("Has correct RHS") { compare(rhs, rhsOpt, ApproxScalar(epsilon)); }

        // auto matDiag = ls.matrix().diag();
        // auto matOptDiag = lsOpt.matrix().diag();
        // SECTION("Has correct diagonal") { compare(matDiag, matOptDiag, ApproxScalar(epsilon)); }

        // auto matUpper = upper(ls.matrix());
        // auto matOptUpper = upper(lsOpt.matrix());
        // SECTION("Has correct upper") { compare(matUpper, matOptUpper, ApproxScalar(epsilon)); }
    }
}

}
