// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

/**@brief Benchmark the gradient operator for explicit evaluation.
 *
 * Constructs a TestType volume field phi and benchmarks the explicit Gauss-linear
 * gradient operation into a Vec3 output field. Only explicit is implemented.
 *
 * @tparam TestType   Field value type (e.g. NeoN::scalar)
 * @param execName    Name of the executor, used as benchmark label
 * @param exec        Executor on which all fields and operations run
 * @param mesh        Unstructured mesh over which the operator is applied
 * @param sectionName Catch2 section label, typically the mesh size string (e.g. "256x256")
 */
template<typename TestType>
void runGradBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    // Create a TestType (scalar) field phi and initialise with 1.0
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());

    // Create a vector field to hold the gradient value
    auto gradVecBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh);
    fvcc::VolumeField<NeoN::Vec3> gradPhi(exec, "gradPhi", mesh, gradVecBCs);
    NeoN::fill(gradPhi.internalVector(), NeoN::zero<NeoN::Vec3>());

    NeoN::Input input = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

    DYNAMIC_SECTION(sectionName + " - Explicit")
    {
        auto op = fvcc::GradOperator<NeoN::Vec3>(Operator::Type::Explicit, phi, input);

        // Note: only explicit is implemented. Implicit is not available since
        // GaussGreenGrad::grad(phi, coeff, LinearSystem<Vec3>&) calls NF_ERROR_EXIT.
        BENCHMARK(std::string(execName)) { op.explicitOperation(gradPhi.internalVector()); };
    }
}

TEMPLATE_TEST_CASE(
    "GradOperator::2D", "[bench]", NeoN::scalar
) //"Template" for consistency with other benchmarks.
{
    auto nCellsPerDim = GENERATE(256, 512, 1024);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);

    const std::string sectionName =
        std::to_string(nCellsPerDim) + "x" + std::to_string(nCellsPerDim);

    runGradBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}

TEMPLATE_TEST_CASE(
    "GradOperator::3D", "[bench]", NeoN::scalar
) //"Template" for consistency with other benchmarks.
{
    auto nCellsPerDim = GENERATE(32, 64, 128);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh =
        NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);

    const std::string sectionName = std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim);

    runGradBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}
