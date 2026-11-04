// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

/**@brief Benchmark the source term operator for explicit and implicit evaluation.
 *
 * Constructs a scalar coefficient field coeff and a volume field phi, then
 * benchmarks explicit evaluation into a source vector and implicit assembly
 * into a linear system.
 *
 * @tparam TestType Field value type (e.g. NeoN::scalar, NeoN::Vec3)
 * @param execName    Name of the executor, used as benchmark label
 * @param exec        Executor on which all fields and operations run
 * @param mesh        Unstructured mesh over which the operator is applied
 * @param sectionName Catch2 section label, typically the mesh size string (e.g. "256x256")
 */
template<typename TestType>
void runSourceTermBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    // Source coeff
    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> coeff(exec, "coeff", mesh, coeffBCs);
    NeoN::fill(coeff.internalVector(), 2.0);
    NeoN::fill(coeff.boundaryData().value(), 0.0);
    coeff.correctBoundaryConditions();

    // Create a vector field to hold the variable
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());
    NeoN::fill(phi.boundaryData().value(), NeoN::zero<TestType>());
    phi.correctBoundaryConditions();

    DYNAMIC_SECTION(sectionName + " - Explicit")
    {
        auto op = fvcc::SourceTerm<TestType>(Operator::Type::Explicit, coeff, phi);

        NeoN::Vector<TestType> source(exec, phi.size(), NeoN::zero<TestType>());

        BENCHMARK(std::string(execName)) { op.explicitOperation(source); };
    }

    DYNAMIC_SECTION(sectionName + " - Implicit")
    {
        // Build sparsity pattern and allocate linear system once - output goes to ls
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);

        auto op = fvcc::SourceTerm<TestType>(Operator::Type::Implicit, coeff, phi);

        BENCHMARK(std::string(execName)) { op.implicitOperation(ls); };
    }
}

TEMPLATE_TEST_CASE("SourceTerm::2D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(256, 512, 1024);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);

    const std::string sectionName =
        std::to_string(nCellsPerDim) + "x" + std::to_string(nCellsPerDim);

    runSourceTermBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}

TEMPLATE_TEST_CASE("SourceTerm::3D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(32, 64, 128);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh =
        NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);

    const std::string sectionName = std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim);

    runSourceTermBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}
