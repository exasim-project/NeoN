// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER

#include <fstream>
#include <iostream>
#include <thread>

#include "common.hpp"

CATCH_REGISTER_REPORTER("mpi", MpiReporter);

namespace dsl = NeoN::dsl;
using Operator = NeoN::dsl::Operator;

std::pair<std::string, NeoN::Executor> createExecutorFromEnv()
{
    const char* env = std::getenv("EXECUTOR");
    const std::string name = (env != nullptr) ? env : "Serial";
    if (name == "CPU") return {"CPU", NeoN::CPUExecutor {}};
    if (name == "GPU") return {"GPU", NeoN::GPUExecutor {}};
    return {"Serial", NeoN::SerialExecutor {}};
}

std::pair<NeoN::localIdx, NeoN::localIdx> createWeakScalingSizeFromEnv()
{
    const char* minEnv = std::getenv("WEAK_SCALING_MIN_SIZE");
    const char* maxEnv = std::getenv("WEAK_SCALING_MAX_SIZE");
    const NeoN::localIdx minSize = minEnv ? static_cast<NeoN::localIdx>(std::stoi(minEnv)) : 64;
    const NeoN::localIdx maxSize = maxEnv ? static_cast<NeoN::localIdx>(std::stoi(maxEnv)) : 512;
    return {minSize, maxSize};
}

std::pair<NeoN::localIdx, NeoN::localIdx> createStrongScalingSizeFromEnv()
{
    const char* minEnv = std::getenv("STRONG_SCALING_MIN_SIZE");
    const char* maxEnv = std::getenv("STRONG_SCALING_MAX_SIZE");
    const NeoN::localIdx minSize = minEnv ? static_cast<NeoN::localIdx>(std::stoi(minEnv)) : 128;
    const NeoN::localIdx maxSize = maxEnv ? static_cast<NeoN::localIdx>(std::stoi(maxEnv)) : 1024;
    return {minSize, maxSize};
}

#ifdef NF_WITH_MPI_SUPPORT

void runDistributedPoissonBenchmark(
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

    auto volumeBCs = createDistributedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> p(exec, "p", mesh, volumeBCs);
    NeoN::fill(p.internalVector(), 1.0);
    p.correctBoundaryConditions();

    auto fvSchemes = [&]()
    {
        NeoN::Dictionary schemes;
        schemes.insert(
            "laplacian(gamma,p)",
            NeoN::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            )
        );
        NeoN::Dictionary fv;
        fv.insert("laplacianSchemes", schemes);
        return fv;
    };

    DYNAMIC_SECTION(sectionName + " - Poisson assemble - face based")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        fvcc::SurfaceField<NeoN::scalar> phiHbyA(exec, "gamma", mesh, surfaceBCs);
        NeoN::fill(phiHbyA.internalVector(), 1.0);
        NeoN::dsl::Expression<NeoN::scalar> expr =
            NeoN::dsl::imp::laplacian(gamma, p) - NeoN::dsl::exp::div(phiHbyA);
        expr.read(fvSchemes());
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    p.correctBoundaryConditions();
                    phiHbyA.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    ls.reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - Poisson assemble - cell based")
    {
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh, cellIterator);

        fvcc::SurfaceField<NeoN::scalar> phiHbyA(exec, "gamma", mesh, surfaceBCs);
        NeoN::fill(phiHbyA.internalVector(), 1.0);
        NeoN::dsl::Expression<NeoN::scalar> expr =
            NeoN::dsl::imp::laplacian(gamma, p) - NeoN::dsl::exp::div(phiHbyA);
        expr.read(fvSchemes());
        cellIterator->setComputeCellBasedData(
            mesh, ls.matrix().sparsity(), ls.faceToMatrixAddress()
        );
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    p.correctBoundaryConditions();
                    phiHbyA.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    ls.reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - Poisson assemble + Gko no solve")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        fvcc::SurfaceField<NeoN::scalar> phiHbyA(exec, "gamma", mesh, surfaceBCs);
        NeoN::fill(phiHbyA.internalVector(), 1.0);
        NeoN::dsl::Expression<NeoN::scalar> expr =
            NeoN::dsl::imp::laplacian(gamma, p) - NeoN::dsl::exp::div(phiHbyA);
        expr.read(fvSchemes());

        NeoN::Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Gmres"},
             {"criteria", NeoN::Dictionary {{{"iteration", 0}, {"relative_residual_norm", 1.0}}}}}
        };

        auto solver = NeoN::la::Solver(exec, solverDict);
        auto x = NeoN::Vector<NeoN::scalar>(exec, mesh.nCells());
        NeoN::fill(x, 0.0);

        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    p.correctBoundaryConditions();
                    phiHbyA.correctBoundaryConditions();

                    auto solver = NeoN::la::Solver(exec, solverDict);
                    expr.assemble(0.0, 1.0, ls);
                    auto solverStats = solver.solve(ls, x);

                    fence(exec);
                    ls.reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }
}

void runDistributedMomentumBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    auto surfaceBCs = createDistributedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> phi(exec, "phi", mesh, surfaceBCs);
    NeoN::fill(phi.internalVector(), 1.0);
    phi.correctBoundaryConditions();

    fvcc::SurfaceField<NeoN::scalar> nu(exec, "nu", mesh, surfaceBCs);
    NeoN::fill(nu.internalVector(), 1.0);
    nu.correctBoundaryConditions();

    auto volumeBCs = createDistributedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh);
    fvcc::VolumeField<NeoN::Vec3> U(exec, "U", mesh, volumeBCs);
    NeoN::fill(U.internalVector(), NeoN::one<NeoN::Vec3>());
    U.correctBoundaryConditions();

    auto fvSchemes = [&]()
    {
        NeoN::Dictionary divSchemes;
        divSchemes.insert(
            "div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})
        );
        NeoN::Dictionary lapSchemes;
        lapSchemes.insert(
            "laplacian(nu,U)",
            NeoN::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            )
        );
        NeoN::Dictionary gradSchemes;
        gradSchemes.insert(
            "grad(p)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})
        );
        NeoN::Dictionary fv;
        fv.insert("divSchemes", divSchemes);
        fv.insert("laplacianSchemes", lapSchemes);
        fv.insert("gradSchemes", gradSchemes);
        return fv;
    };

    auto scalarVolumeBCs = createDistributedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> p(exec, "p", mesh, scalarVolumeBCs);
    NeoN::fill(p.internalVector(), 1.0);
    p.correctBoundaryConditions();

    DYNAMIC_SECTION(sectionName + " - momentum assemble - face based - vec3")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::Vec3>(mesh);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U)
                  + NeoN::dsl::exp::grad(p);
        expr.read(fvSchemes());
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - momentum assemble - face based - scalar")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::Vec3>(mesh);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U)
                  + NeoN::dsl::exp::grad(p);
        expr.read(fvSchemes());
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    expr.assemble<NeoN::scalar>(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - momentum assemble - cell based - Vec3")
    {
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::Vec3>(mesh, cellIterator);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U)
                  + NeoN::dsl::exp::grad(p);
        expr.read(fvSchemes());
        cellIterator->setComputeCellBasedData(
            mesh, ls.matrix().sparsity(), ls.faceToMatrixAddress()
        );
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - momentum assemble - cell based - scalar")
    {
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::Vec3>(mesh, cellIterator);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U)
                  + NeoN::dsl::exp::grad(p);
        expr.read(fvSchemes());
        cellIterator->setComputeCellBasedData(
            mesh, ls.matrix().sparsity(), ls.faceToMatrixAddress()
        );
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    expr.assemble<NeoN::scalar>(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    auto fvSchemesOpt = [&]()
    {
        NeoN::Dictionary divSchemes;
        divSchemes.insert(
            "div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})
        );
        NeoN::Dictionary lapSchemes;
        lapSchemes.insert(
            "laplacian(nu,U)",
            NeoN::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            )
        );
        NeoN::Dictionary fv;
        fv.insert("divSchemes", divSchemes);
        fv.insert("laplacianSchemes", lapSchemes);
        return fv;
    };

    DYNAMIC_SECTION(sectionName + " - momentum assemble - face based - optimized")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::Vec3>(mesh);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U);
        auto exprOpt = dsl::optimize(expr);
        exprOpt.read(fvSchemesOpt());
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    exprOpt.assemble<NeoN::scalar>(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - momentum assemble - cell based - optimized")
    {
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::Vec3>(mesh, cellIterator);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U);
        auto exprOpt = dsl::optimize(expr);
        exprOpt.read(fvSchemesOpt());
        cellIterator->setComputeCellBasedData(
            mesh, ls.matrix().sparsity(), ls.faceToMatrixAddress()
        );
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    exprOpt.assemble<NeoN::scalar>(0.0, 1.0, ls);
                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }


    DYNAMIC_SECTION(sectionName + " - momentum assemble + Gko no solve - scalar")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::Vec3>(mesh);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U)
                  + NeoN::dsl::exp::grad(p);
        expr.read(fvSchemes());

        NeoN::Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", NeoN::Dictionary {{{"iteration", 0}, {"relative_residual_norm", 1.0}}}}}
        };

        auto x = NeoN::Vector<NeoN::Vec3>(exec, mesh.nCells());
        NeoN::fill(x, NeoN::zero<NeoN::Vec3>());

        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();

                    auto solver = NeoN::la::Solver(exec, solverDict);
                    expr.assemble<NeoN::scalar>(0.0, 1.0, ls);
                    auto solverStats = solver.solve(ls, x);

                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - momentum assemble + Gko no solve - Vec3")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::Vec3>(mesh);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(nu, U)
                  + NeoN::dsl::exp::grad(p);
        expr.read(fvSchemes());

        NeoN::Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", NeoN::Dictionary {{{"iteration", 0}, {"relative_residual_norm", 1.0}}}}}
        };

        auto x = NeoN::Vector<NeoN::Vec3>(exec, mesh.nCells());
        NeoN::fill(x, NeoN::zero<NeoN::Vec3>());

        auto solver = NeoN::la::Solver(exec, solverDict);
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();

                    expr.assemble<NeoN::Vec3>(0.0, 1.0, ls);
                    auto solverStats = solver.solve(ls, x);

                    fence(exec);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }
}

void runDistributedFusedDivLapBenchmark(
    const std::string& execName,
    const NeoN::Executor& exec,
    NeoN::UnstructuredMesh& mesh,
    const std::string& sectionName
)
{
    auto surfaceBCs = createDistributedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> phi(exec, "phi", mesh, surfaceBCs);
    NeoN::fill(phi.internalVector(), 1.0);
    phi.correctBoundaryConditions();

    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);
    gamma.correctBoundaryConditions();

    auto volumeBCs = createDistributedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> U(exec, "U", mesh, volumeBCs);
    NeoN::fill(U.internalVector(), 1.0);
    U.correctBoundaryConditions();

    auto fvSchemes = [&]()
    {
        NeoN::Dictionary divSchemes;
        divSchemes.insert(
            "div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})
        );
        NeoN::Dictionary lapSchemes;
        lapSchemes.insert(
            "laplacian(gamma,U)",
            NeoN::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            )
        );
        NeoN::Dictionary fv;
        fv.insert("divSchemes", divSchemes);
        fv.insert("laplacianSchemes", lapSchemes);
        return fv;
    };

    DYNAMIC_SECTION(sectionName + " - DivLap assemble - non fused cell-based")
    {
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto ls = la::createEmptyLinearSystem<NeoN::scalar>(mesh, cellIterator);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(gamma, U);
        expr.read(fvSchemes());
        cellIterator->setComputeCellBasedData(
            mesh, ls.matrix().sparsity(), ls.faceToMatrixAddress()
        );
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    ls.reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - DivLap assemble - non fused face-based")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(gamma, U);
        expr.read(fvSchemes());
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    expr.assemble(0.0, 1.0, ls);
                    fence(exec);
                    ls.reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }


    DYNAMIC_SECTION(sectionName + " - DivLap assemble - fused face-based")
    {
        auto ls = la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(gamma, U);
        auto exprOpt = dsl::optimize(expr);
        exprOpt.read(fvSchemes());
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    exprOpt.assemble(0.0, 1.0, ls);
                    fence(exec);
                    ls.reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }

    DYNAMIC_SECTION(sectionName + " - DivLap assemble - fused cell-based")
    {
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh, cellIterator);
        auto expr = NeoN::dsl::imp::div(phi, U) - NeoN::dsl::imp::laplacian(gamma, U);
        auto exprOpt = dsl::optimize(expr);
        exprOpt.read(fvSchemes());
        cellIterator->setComputeCellBasedData(
            mesh, ls.matrix().sparsity(), ls.faceToMatrixAddress()
        );
        BENCHMARK_ADVANCED(std::string(execName))(Catch::Benchmark::Chronometer meter)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            meter.measure(
                [&]
                {
                    U.correctBoundaryConditions();
                    exprOpt.assemble(0.0, 1.0, ls);
                    fence(exec);
                    ls.reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            );
        };
    }
}


#endif // NF_WITH_MPI_SUPPORT

int main(int argc, char* argv[])
{
    int result = 0;
    {
        NeoN::mpi::Init mpiInit(argc, argv);

        MPI_Comm_rank(COMM, &RANK);
        MPI_Comm_size(COMM, &COMM_SIZE);
        IS_ROOT = RANK == ROOT;

        bool threadShutdown = false;
        std::thread serializeIOThread {serializeIO, &threadShutdown};

        NeoN::initialize(argc, argv);

        std::cout << std::flush;
        std::cerr << std::flush;
        MPI_Barrier(COMM);

        Catch::Session session;
        const int returnCode = session.applyCommandLine(argc, argv);
        if (returnCode != 0)
        {
            result = returnCode;
        }
        else
        {
            std::ofstream devnull;
            std::streambuf* savedCout = nullptr;
            if (!IS_ROOT)
            {
                devnull.open("/dev/null");
                savedCout = std::cout.rdbuf(devnull.rdbuf());
            }

            result = session.run();
            MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, COMM);

            if (savedCout) std::cout.rdbuf(savedCout);
        }

        MPI_Barrier(COMM);
        threadShutdown = true;
        serializeIOThread.join();

        NeoN::finalize();
    }

    std::_Exit(result); // Note: Fix for bench_allocator failure
}
