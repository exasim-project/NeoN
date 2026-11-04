// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

/**@brief Benchmark the divergence operator for explicit and implicit evaluation.
 *
 * Constructs a face flux field and a volume field phi, then benchmarks explicit
 * evaluation into a output volume field and implicit assembly into a linear system.
 *
 * @tparam TestType Field value type (e.g. NeoN::scalar, NeoN::Vec3)
 * @param execName    Name of the executor, used as benchmark label
 * @param exec        Executor on which all fields and operations run
 * @param mesh        Unstructured mesh over which the operator is applied
 * @param sectionName Catch2 section label, typically the mesh size string (e.g. "256x256")
 */
template<typename TestType>
void runDivBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    // Boundary fields
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    // FaceFlux
    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);

    fvcc::VolumeField<TestType> phi(exec, "vf", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());

    NeoN::Input input = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

    DYNAMIC_SECTION(sectionName + " - Explicit")
    {
        // Create a scalar field to hold the div value - output field
        fvcc::VolumeField<TestType> divPhi(exec, "vf", mesh, volumeBCs);
        NeoN::fill(divPhi.internalVector(), NeoN::zero<TestType>());

        auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);

        BENCHMARK(std::string(execName)) { op.explicitOperation(divPhi.internalVector()); };
    }

    DYNAMIC_SECTION(sectionName + " - Implicit")
    {
        // Build sparsity pattern and allocate linear system once - output goes to ls
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);

        auto op = fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, input);

        BENCHMARK(std::string(execName)) { op.implicitOperation(ls); };
    }
}

TEMPLATE_TEST_CASE("DivOperator::2D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(256, 512, 1024);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);

    const std::string sectionName =
        std::to_string(nCellsPerDim) + "x" + std::to_string(nCellsPerDim);

    runDivBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}

TEMPLATE_TEST_CASE("DivOperator::3D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(32, 64, 128);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh =
        NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);

    const std::string sectionName = std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim);

    runDivBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}
