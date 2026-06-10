// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "common.hpp"

namespace dsl = NeoN::dsl;
using Catch::randomizeVector;


namespace NeoN
{

TEST_CASE("Optimizer")
{
    float epsilon = 1e-32;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;
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
    fill(gamma.internalVector(), 2.0);

    SECTION("Can optimize div + laplacian " + execName)
    {
        auto expr = NeoN::dsl::imp::laplacian(gamma, U) - NeoN::dsl::imp::div(phi, U);
        auto exprOpt = dsl::optimize(expr);

        REQUIRE(expr.size() == 2);
        REQUIRE(exprOpt.size() == 1);
    }

    SECTION("Can optimize div + laplacian face based" + execName)
    {
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

        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(gamma, U);
        auto exprOpt = dsl::optimize(expr);

        REQUIRE(expr.size() == 2);
        REQUIRE(exprOpt.size() == 1);

        expr.read(input);
        exprOpt.read(input);

        auto ls = expr.assemble(mesh, 1.0, 1.0);
        auto lsOpt = exprOpt.assemble(mesh, 1.0, 1.0);

        auto rhs = ls.rhs();
        auto rhsOpt = lsOpt.rhs();
        SECTION("Has correct RHS") { compare(rhs, rhsOpt, ApproxScalar(epsilon)); }

        auto matDiag = ls.matrix().diag();
        auto matOptDiag = lsOpt.matrix().diag();
        SECTION("Has correct diagonal") { compare(matDiag, matOptDiag, ApproxScalar(epsilon)); }

        auto matUpper = upper(ls.matrix());
        auto matOptUpper = upper(lsOpt.matrix());
        SECTION("Has correct upper") { compare(matUpper, matOptUpper, ApproxScalar(epsilon)); }
    }

    SECTION("Can optimize div + laplacian and assemble cell based " + execName)
    {
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

        auto expr = NeoN::dsl::imp::laplacian(gamma, U) - NeoN::dsl::imp::div(phi, U);
        auto exprOpt = dsl::optimize(expr);

        REQUIRE(expr.size() == 2);
        REQUIRE(exprOpt.size() == 1);

        expr.read(input);
        exprOpt.read(input);

        auto ls = expr.assemble(mesh, 1.0, 1.0);
        auto lsOpt = exprOpt.assemble(mesh, 1.0, 1.0);

        auto rhs = ls.rhs();
        auto rhsOpt = lsOpt.rhs();
        SECTION("Has correct RHS") { compare(rhs, rhsOpt, ApproxScalar(epsilon)); }

        auto matDiag = ls.matrix().diag();
        auto matOptDiag = lsOpt.matrix().diag();
        SECTION("Has correct diagonal") { compare(matDiag, matOptDiag, ApproxScalar(epsilon)); }

        auto matUpper = upper(ls.matrix());
        auto matOptUpper = upper(lsOpt.matrix());
        SECTION("Has correct upper") { compare(matUpper, matOptUpper, ApproxScalar(epsilon)); }
    }


    // SECTION("Can optimize cellBased ddt + div + laplacian " + execName)
    // {
    //     auto input = NeoN::Dictionary {
    //         {
    //             "laplacianSchemes",
    //             NeoN::Dictionary {
    //                 {"laplacian(gamma,U)",
    //                  NeoN::TokenList(
    //                      {std::string("Gauss"), std::string("linear"),
    //                      std::string("uncorrected")}
    //                  )}
    //             },
    //         },
    //         {"divSchemes",
    //          NeoN::Dictionary {
    //              {"div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}
    //          }}
    //     };

    //     auto expr = NeoN::dsl::imp::ddt(U) + NeoN::dsl::imp::laplacian(gamma, U)
    //               - NeoN::dsl::imp::div(phi, U);

    //     auto mi = NeoN::la::createSparsityPatternFaceToMatrixAddress<NeoN::localIdx>(mesh);
    //     auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
    //     cellIterator->setComputeCellBasedData(mesh, mi);

    //     auto optimizer = std::vector<
    //         std::shared_ptr<NeoN::dsl::Optimizer<NeoN::dsl::Expression<NeoN::scalar>>>> {
    //         std::make_shared<NeoN::dsl::DdtDivLapOptimizer<NeoN::dsl::Expression<NeoN::scalar>>>()
    //     };

    //     auto lsOpt = NeoN::la::createEmptyLinearSystem<scalar>(mesh, cellIterator);
    //     auto exprOpt = dsl::optimize(expr, optimizer);

    //     REQUIRE(expr.size() == 3);
    //     REQUIRE(exprOpt.size() == 1);

    //     //         expr.read(input);
    //     exprOpt.read(input);

    //     //        auto [sp, ls] = expr.assemble(mesh, 1.0, 1.0);
    //     exprOpt.assemble(1.0, 1.0, lsOpt);
    // }
}

}
