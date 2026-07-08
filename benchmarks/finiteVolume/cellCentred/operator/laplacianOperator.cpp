// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

/**@brief Benchmark the Laplacian operator for explicit and implicit evaluation.
 *
 * Constructs a diffusion coefficient surface field gamma and a volume field phi,
 * then benchmarks explicit evaluation into an output field and implicit assembly
 * into a linear system using Gauss-linear-uncorrected discretisation.
 *
 * @tparam TestType Field value type (e.g. NeoN::scalar, NeoN::Vec3)
 * @param execName    Name of the executor, used as benchmark label
 * @param exec        Executor on which all fields and operations run
 * @param mesh        Unstructured mesh over which the operator is applied
 * @param sectionName Catch2 section label, typically the mesh size string (e.g. "256x256")
 */
template<typename TestType>
void runLaplacianBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    // Create a SurfaceField<scalar> named gamma; initialise with 1.0
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);

    // Create a scalar field phi and initialise with 1.0
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());

    NeoN::Input input =
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});

    DYNAMIC_SECTION(sectionName + " - Explicit")
    {
        // Create a scalar field to hold the laplacian value - output field
        fvcc::VolumeField<TestType> lapPhi(exec, "lapPhi", mesh, volumeBCs);
        NeoN::fill(lapPhi.internalVector(), NeoN::zero<TestType>());

        auto op = fvcc::LaplacianOperator<TestType>(Operator::Type::Explicit, gamma, phi, input);

        BENCHMARK(std::string(execName)) { op.explicitOperation(lapPhi.internalVector()); };
    }

    DYNAMIC_SECTION(sectionName + " - Implicit")
    {
        // Build sparsity pattern and allocate linear system once - output goes to ls
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);

        auto op = fvcc::LaplacianOperator<TestType>(Operator::Type::Implicit, gamma, phi, input);

        BENCHMARK(std::string(execName)) { op.implicitOperation(ls); };
    }
}

TEMPLATE_TEST_CASE("LaplacianOperator::2D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(256, 512, 1024);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);

    const std::string sectionName =
        std::to_string(nCellsPerDim) + "x" + std::to_string(nCellsPerDim);

    runLaplacianBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}

TEMPLATE_TEST_CASE("LaplacianOperator::3D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(32, 64, 128);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh =
        NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);

    const std::string sectionName = std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim);

    runLaplacianBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}
