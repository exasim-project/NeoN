// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include "NeoN/NeoN.hpp"
#include "mpiGlobals.hpp"
#include "mpiReporter.hpp"
#include "mpiSerialization.hpp"

std::pair<std::string, NeoN::Executor> createExecutorFromEnv();
std::pair<NeoN::localIdx, NeoN::localIdx> createWeakScalingSizeFromEnv();
std::pair<NeoN::localIdx, NeoN::localIdx> createStrongScalingSizeFromEnv();

#ifdef NF_WITH_MPI_SUPPORT

/**@brief Create boundary conditions for a distributed mesh part.
 *
 * Regular (physical) patches receive a "calculated" boundary condition, while
 * processor patches - which BoundaryMesh stores after the regular patches -
 * receive a "processor" boundary condition so that halo values are exchanged
 * with the neighbouring rank during correctBoundaryConditions.
 * TODO remove when PR528 is merged.
 */
template<typename BoundaryType>
std::vector<BoundaryType> createDistributedBCs(const NeoN::UnstructuredMesh& mesh)
{
    const auto nBoundaries = mesh.nBoundaries();
    const auto nProcPatches = mesh.boundaryMesh().nProcBoundaryPatches();
    const auto nRegular = nBoundaries - nProcPatches;

    std::vector<BoundaryType> bcs;
    bcs.reserve(static_cast<std::size_t>(nBoundaries));
    for (NeoN::localIdx patchID = 0; patchID < nBoundaries; ++patchID)
    {
        const std::string type = (patchID >= nRegular) ? "processor" : "calculated";
        NeoN::Dictionary patchDict({{"type", type}});
        bcs.emplace_back(mesh, patchDict, patchID);
    }
    return bcs;
}

/**@brief Benchmark Div and Laplacian operators on a distributed mesh.
 *
 * @tparam TestType Field value type (e.g. NeoN::scalar, NeoN::Vec3)
 * @param execName    Executor label used as benchmark name
 * @param exec        Executor for all fields and operations
 * @param mesh        Distributed mesh part owned by this rank
 * @param sectionName Catch2 section label prefix
 */
template<typename TestType>
void runDistributedSingleOperatorBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    auto surfaceBCs = createDistributedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);
    gamma.correctBoundaryConditions();

    auto volumeBCs = createDistributedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());

    auto divSchemes = [&]()
    {
        NeoN::Dictionary schemes;
        schemes.insert(
            "div(gamma,phi)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})
        );
        NeoN::Dictionary fv;
        fv.insert("divSchemes", schemes);
        return fv;
    };

    auto lapSchemes = [&]()
    {
        NeoN::Dictionary schemes;
        schemes.insert(
            "laplacian(gamma,phi)",
            NeoN::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            )
        );
        NeoN::Dictionary fv;
        fv.insert("laplacianSchemes", schemes);
        return fv;
    };

    DYNAMIC_SECTION(sectionName + " - Div")
    {
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);
        NeoN::dsl::Expression<TestType> expr = NeoN::dsl::imp::div(gamma, phi);
        expr.read(divSchemes());
        MPI_Barrier(MPI_COMM_WORLD);
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    phi.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - Lap")
    {
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);
        NeoN::dsl::Expression<TestType> expr = NeoN::dsl::imp::laplacian(gamma, phi);
        expr.read(lapSchemes());
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    phi.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }
}

void runDistributedPoissonBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
);

void runDistributedMomentumBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
);

#endif // NF_WITH_MPI_SUPPORT
