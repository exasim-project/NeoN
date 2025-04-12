// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"
#include <catch2/catch_approx.hpp>


#define KOKKOS_ENABLE_SERIAL

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

#if NF_WITH_PETSC

using NeoN::Executor;
using NeoN::Dictionary;
using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Field;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;
using NeoN::la::Solver;

TEST_CASE("MatrixAssembly - Petsc")
{


    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);


    SECTION("Solve linear system " + execName)
    {

        NeoN::Database db;

        Field<NeoN::scalar> values(exec, {10.0, 4.0, 7.0, 2.0, 10.0, 8.0, 3.0, 6.0, 10.0});
        // TODO work on support for unsingned types
        Field<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
        Field<localIdx> rowPtrs(exec, {0, 3, 6, 9});
        CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

        Field<NeoN::scalar> rhs(exec, {1.0, 2.0, 3.0});
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Field<NeoN::scalar> x(exec, {0.0, 0.0, 0.0});

        NeoN::Dictionary solverDict {{
            {"solver", std::string {"Petsc"}},
        }};

        // Create solver
        auto solver = NeoN::la::Solver(exec, solverDict);

        // Solve system
        solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();
        auto hostXS = hostX.view();
        REQUIRE((hostXS[0]) == Catch::Approx(3. / 205.).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(8. / 205.).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(53. / 205.).margin(1e-8));


        SECTION("Solve linear system second time " + execName)
        {
            // NeoN::Database db;
            NeoN::Field<NeoN::scalar> values(
                exec, {10.0, 2.0, 3.0, 5.0, 20.0, 2.0, 4.0, 4.0, 30.0}
            );

            Field<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
            Field<localIdx> rowPtrs(exec, {0, 3, 6, 9});
            CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

            Field<NeoN::scalar> rhs(exec, {1.0, 2.0, 3.0});
            LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
            Field<NeoN::scalar> x(exec, {0.0, 0.0, 0.0});

            NeoN::Dictionary solverDict {{
                {"solver", std::string {"Petsc"}},
            }};

            // Create solver
            auto solver = NeoN::la::Solver(exec, solverDict);

            // Solve system
            solver.solve(linearSystem, x);

            auto hostX = x.copyToHost();
            auto hostXS = hostX.view();
            REQUIRE((hostXS[0]) == Catch::Approx(8. / 341.).margin(1e-8));
            REQUIRE((hostXS[1]) == Catch::Approx(27. / 341.).margin(1e-8));
            REQUIRE((hostXS[2]) == Catch::Approx(63. / 682.).margin(1e-8));
        }
    }
}

#endif
