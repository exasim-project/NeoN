// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "benchmarks/finiteVolume/cellCentred/common.hpp"

#include "NeoN/dsl/explicit.hpp"
#include "NeoN/dsl/implicit.hpp"
#include "NeoN/dsl/expression.hpp"

/**@brief Benchmark the combined transport operator for explicit and implicit time integration.
 *
 * Assembles the full transport equation ddt(phi) + div(faceFlux, phi) + laplacian(gamma, phi)
 * + source(coeff, phi) and benchmarks explicit evaluation as well as implicit assembly using
 * BDF1 and BDF2 time schemes.
 *
 * @tparam TestType Field value type (e.g. NeoN::scalar, NeoN::Vec3)
 * @param execName  Name of the executor, used as benchmark label
 * @param exec      Executor on which all fields and operations run
 * @param mesh      Unstructured mesh over which the operators are applied
 * @param collectionName Name of the VectorCollection used to register phi with time levels
 * @param sectionName    Catch2 section label, typically the mesh size string (e.g. "256x256")
 */
template<typename TestType>
void runTransportBenchmark(
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

    NeoN::fill(oldTime(phi).internalVector(), NeoN::one<TestType>());          // phi^n
    NeoN::fill(oldTime(oldTime(phi)).internalVector(), NeoN::one<TestType>()); // phi^{n-1}

    const NeoN::scalar t = 1.0;
    const NeoN::scalar dt = 0.5;

    // Boundary fields
    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);

    // FaceFlux
    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);

    // Diffusion coefficient
    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);

    // Source coeff
    fvcc::VolumeField<NeoN::scalar> coeff(exec, "coeff", mesh, coeffBCs);
    NeoN::fill(coeff.internalVector(), 2.0);
    NeoN::fill(coeff.boundaryData().value(), 0.0);
    coeff.correctBoundaryConditions();

    auto makeFvSchemes = [&]()
    {
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary divSchemes;
        divSchemes.insert(
            "div(faceFlux,phi)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})
        );
        fvSchemes.insert("divSchemes", divSchemes);

        NeoN::Dictionary lapSchemes;
        lapSchemes.insert(
            "laplacian(gamma,phi)",
            NeoN::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            )
        );
        fvSchemes.insert("laplacianSchemes", lapSchemes);
        return fvSchemes;
    };

    DYNAMIC_SECTION(sectionName + " - Explicit")
    {
        NeoN::Vector<TestType> rhs(exec, phi.size(), NeoN::zero<TestType>());

        // Build Explicit Expression
        auto expr = NeoN::dsl::exp::ddt(phi) + NeoN::dsl::exp::div(faceFlux, phi)
                  + NeoN::dsl::exp::laplacian(gamma, phi) + NeoN::dsl::exp::source(coeff, phi);

        expr.read(makeFvSchemes());

        BENCHMARK(std::string(execName))
        {
            // rhs += ddt(phi)
            expr.explicitOperation(rhs, t, dt);

            // rhs += div(faceFlux, phi) + laplacian(gamma, phi) + source(coeff, phi)
            expr.explicitOperation(rhs);
        };
    }

    DYNAMIC_SECTION(sectionName + " - Implicit")
    {
        auto fvSchemes = makeFvSchemes();
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        // Build sparsity pattern and allocate linear system once - output goes to ls
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);

        // Build Implicit Expression - First order
        auto expr = NeoN::dsl::imp::ddt(phi) + NeoN::dsl::imp::div(faceFlux, phi)
                  + NeoN::dsl::imp::laplacian(gamma, phi) + NeoN::dsl::imp::source(coeff, phi);

        expr.read(fvSchemes);

        BENCHMARK(std::string(execName)) { expr.assemble(t, dt, ls); };
    }

    DYNAMIC_SECTION(sectionName + " - Implicit_BDF2")
    {
        // Select implicit ddt scheme (BDF2)
        auto fvSchemes = makeFvSchemes();
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        // Build sparsity pattern and allocate linear system once - output goes to ls
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);

        // Ensure oldTimeLevel >= 2 to confirm true BDF2
        NF_ASSERT(
            oldTimeLevel(phi) >= 2,
            std::format("Expected oldTimeLevel(phi) >= 2 but got {}", oldTimeLevel(phi))
        );

        // Build Implicit Expression - Second order
        auto expr = NeoN::dsl::imp::ddt(phi) + NeoN::dsl::imp::div(faceFlux, phi)
                  + NeoN::dsl::imp::laplacian(gamma, phi) + NeoN::dsl::imp::source(coeff, phi);

        expr.read(fvSchemes);

        BENCHMARK(std::string(execName)) { expr.assemble(t, dt, ls); };
    }
}

TEMPLATE_TEST_CASE("TransportOperator::2D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(256, 512, 1024);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);

    const std::string sectionName =
        std::to_string(nCellsPerDim) + "x" + std::to_string(nCellsPerDim);

    runTransportBenchmark<TestType>(
        std::string(execName), exec, mesh, "benchVectorCollection2D", sectionName
    );
}

TEMPLATE_TEST_CASE("TransportOperator::3D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto nCellsPerDim = GENERATE(32, 64, 128);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh =
        NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);

    const std::string sectionName = std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim) + "x"
                                  + std::to_string(nCellsPerDim);

    runTransportBenchmark<TestType>(
        std::string(execName), exec, mesh, "benchVectorCollection3D", sectionName
    );
}
