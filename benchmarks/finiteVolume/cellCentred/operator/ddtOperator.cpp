// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "benchmarks/finiteVolume/cellCentred/common.hpp"

using Operator = NeoN::dsl::Operator;

/**@brief Benchmark the ddt operator for explicit and implicit time integration.
 *
 * Constructs a VolumeField phi with two old time levels and benchmarks explicit
 * evaluation as well as implicit assembly using BDF1 and BDF2 time schemes.
 *
 * @tparam TestType Field value type (e.g. NeoN::scalar, NeoN::Vec3)
 * @param execName       Name of the executor, used as benchmark label
 * @param exec           Executor on which all fields and operations run
 * @param mesh           Unstructured mesh over which the operator is applied
 * @param collectionName Name of the VectorCollection used to register phi with time levels
 * @param sectionName    Catch2 section label, typically the mesh size string (e.g. "256x256")
 */
template<typename TestType>
void runDdtBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& collectionName,
    const std::string& sectionName
)
{
    NeoN::Database db;

    fvcc::VectorCollection& fieldCollection = fvcc::VectorCollection::instance(db, collectionName);

    fvcc::VolumeField<TestType>& phi = fieldCollection.registerVector<fvcc::VolumeField<TestType>>(
        CreateVector<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());
    NeoN::fill(phi.boundaryData().value(), NeoN::zero<TestType>());
    phi.correctBoundaryConditions();

    // Ensure old-time storage exists and is initialized
    NeoN::fill(oldTime(phi).internalVector(), 0.5 * NeoN::one<TestType>());           // phi^n
    NeoN::fill(oldTime(oldTime(phi)).internalVector(), 0.25 * NeoN::one<TestType>()); // phi^{n-1}

    const NeoN::scalar t = 1.0;
    const NeoN::scalar dt = 0.5;

    DYNAMIC_SECTION(sectionName + " - Explicit")
    {
        // Create a scalar field to hold the ddt value - output field
        NeoN::Vector<TestType> source(exec, phi.size(), NeoN::zero<TestType>());

        auto op = fvcc::DdtOperator(Operator::Type::Explicit, phi);

        BENCHMARK(std::string(execName)) { op.explicitOperation(source, t, dt); };
    }

    DYNAMIC_SECTION(sectionName + " - Implicit")
    {
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        // Build sparsity pattern and allocate linear system once - output goes to ls
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);

        auto op = fvcc::DdtOperator(Operator::Type::Implicit, phi);
        op.read(fvSchemes);

        BENCHMARK(std::string(execName)) { op.implicitOperation(ls, t, dt); };
    }

    DYNAMIC_SECTION(sectionName + " - Implicit_BDF2")
    {
        // Select implicit ddt scheme (BDF2)
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        // Build sparsity pattern and allocate linear system once - output goes to ls
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);

        auto op = fvcc::DdtOperator(Operator::Type::Implicit, phi);
        op.read(fvSchemes);

        // Ensure oldTimeLevel >= 2 to confirm true BDF2
        NF_ASSERT(
            oldTimeLevel(phi) >= 2,
            std::format("Expected oldTimeLevel(phi) >= 2 but got {}", oldTimeLevel(phi))
        );

        BENCHMARK(std::string(execName)) { op.implicitOperation(ls, t, dt); };
    }
}

TEMPLATE_TEST_CASE("DdtOperator::2D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(256, 512, 1024);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);

    const std::string sectionName =
        std::to_string(nCellsPerDim) + "x" + std::to_string(nCellsPerDim);

    runDdtBenchmark<TestType>(
        std::string(execName), exec, mesh, "benchVectorCollection2D", sectionName
    );
}

TEMPLATE_TEST_CASE("DdtOperator::3D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(32, 64, 128);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh =
        NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);

    const std::string sectionName = std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim);

    runDdtBenchmark<TestType>(
        std::string(execName), exec, mesh, "benchVectorCollection3D", sectionName
    );
}
