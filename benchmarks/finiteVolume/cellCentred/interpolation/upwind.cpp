// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

/**@brief Benchmark upwind surface interpolation from a volume field to a surface field.
 *
 * Constructs a volume field and a face flux field, both initialised to one, and
 * benchmarks the upwind interpolation onto the mesh faces using the flux direction.
 *
 * @tparam TestType Field value type (e.g. NeoN::scalar, NeoN::Vec3)
 * @param execName    Name of the executor, used as benchmark label
 * @param exec        Executor on which all fields and operations run
 * @param mesh        Unstructured mesh over which the interpolation is applied
 * @param sectionName Catch2 section label, typically the mesh size string (e.g. "256x256")
 */
template<typename TestType>
void runUpwindBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    NeoN::Input input = NeoN::TokenList({std::string("upwind")});

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);
    auto upwind = fvcc::SurfaceInterpolation<TestType>(exec, mesh, input);

    auto in = fvcc::VolumeField<TestType>(exec, "in", mesh, {});
    auto flux = fvcc::SurfaceField<NeoN::scalar>(exec, "flux", mesh, {});
    auto out = fvcc::SurfaceField<TestType>(exec, "out", mesh, surfaceBCs);

    NeoN::fill(flux.internalVector(), NeoN::one<NeoN::scalar>());
    NeoN::fill(in.internalVector(), NeoN::one<TestType>());

    DYNAMIC_SECTION(sectionName + " - Interpolate")
    {
        BENCHMARK(std::string(execName)) { upwind.interpolate(flux, in, out); };
    }
}

TEMPLATE_TEST_CASE("Upwind::2D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(256, 512, 1024);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);

    const std::string sectionName =
        std::to_string(nCellsPerDim) + "x" + std::to_string(nCellsPerDim);

    runUpwindBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}

TEMPLATE_TEST_CASE("Upwind::3D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(32, 64, 128);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh =
        NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);

    const std::string sectionName = std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim);

    runUpwindBenchmark<TestType>(std::string(execName), exec, mesh, sectionName);
}
