// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

/**@brief helper function to produce assembled linear system
** Assemble a diagonally-dominant linear system from a 1D laplacian.
** The laplacian implicit operation produces a valid CSR matrix with non-zero
** diagonals, making it suitable for both the diagonal solver and iterative solvers.
*/
template<typename ValueType>
la::LinearSystem<ValueType>
assembleLS(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh)
{
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<ValueType>>(mesh);
    fvcc::VolumeField<ValueType> phi(exec, "phi", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<ValueType>());

    NeoN::Input input =
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});

    auto ls = la::createEmptyLinearSystem<ValueType>(mesh);
    auto op = fvcc::LaplacianOperator<ValueType>(Operator::Type::Implicit, gamma, phi, input);
    op.implicitOperation(ls);

    return ls;
}

TEMPLATE_TEST_CASE(
    "Solver::1D", "[bench]", NeoN::scalar
) //"Template" for consistency with other benchmarks.
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);
    auto ls = assembleLS<TestType>(exec, mesh);
    NeoN::Vector<TestType> x(exec, mesh.nCells(), NeoN::zero<TestType>());

    const std::string sectionName = std::to_string(size);

    DYNAMIC_SECTION(sectionName + " - DiagonalSolver")
    {
        NeoN::Dictionary solverDict {{{"solver", std::string {"diagonal"}}}};
        auto solver = NeoN::la::Solver(exec, solverDict);
        BENCHMARK(std::string(execName)) { solver.solve(ls, x); };
    }

#if NF_WITH_GINKGO
    DYNAMIC_SECTION(sectionName + " - GinkgoCG")
    {
        NeoN::Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", NeoN::Dictionary {{{"iteration", 100}, {"relative_residual_norm", 1e-6}}}}
            }
        };
        auto solver = NeoN::la::Solver(exec, solverDict);
        BENCHMARK(std::string(execName)) { solver.solve(ls, x); };
    }
#endif
}
