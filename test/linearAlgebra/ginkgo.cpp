// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


#if NF_WITH_GINKGO

using NeoN::Executor;
using NeoN::Dictionary;
using NeoN::scalar;
using NeoN::Vec3;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::LinearSystem;
using NeoN::la::CsrSparsityPattern;
using NeoN::la::CooSparsityPattern;
using NeoN::la::CSRMatrix;
using NeoN::la::COOMatrix;
using NeoN::la::ELLMatrix;
using NeoN::la::EllSparsityPattern;
using NeoN::la::EllSparsityView;
using NeoN::la::Solver;
using NeoN::la::Dimensions;

TEST_CASE("Dictionary Parsing - Ginkgo")
{
    SECTION("String")
    {
        NeoN::Dictionary dict {{{"key", std::string("value")}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Const Char *")
    {
        NeoN::Dictionary dict {{{"key", "value"}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Int")
    {
        NeoN::Dictionary dict {{{"key", 10}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {10}}});
        CHECK(node == expected);
    }
    SECTION("Double")
    {
        NeoN::Dictionary dict {{{"key", 1.0}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
        CHECK(node == expected);
    }
    SECTION("Float")
    {
        NeoN::Dictionary dict {{{"key", 1.0f}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
        CHECK(node == expected);
    }
    SECTION("Dict")
    {
        NeoN::Dictionary dict;
        dict.insert("key", NeoN::Dictionary {{"key", "value"}});

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"key", gko::config::pnode({{"key", gko::config::pnode {"value"}}})}}
        );
        CHECK(node == expected);
    }
    SECTION("Throws")
    {
        NeoN::Dictionary dict({{"key", std::pair<int*, std::vector<double>> {}}});

        REQUIRE_THROWS_AS(NeoN::la::ginkgo::parse(dict), NeoN::NeoNException);
    }
}

TEST_CASE("gkoVecView - Ginkgo")
{
    NeoN::Executor exec = NeoN::SerialExecutor {};
    auto gkoExec = NeoN::la::ginkgo::getGkoExecutor(exec);

    SECTION("scalar mutable: 1-column non-owning Dense")
    {
        localIdx n = 4;
        Vector<scalar> v(exec, {1.0, 2.0, 3.0, 4.0});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {1});
        CHECK(dense->get_stride() == gko::size_type {1});
        CHECK(dense->get_values() == v.data());
    }

    SECTION("scalar const: 1-column non-owning Dense")
    {
        localIdx n = 4;
        const Vector<scalar> v(exec, {1.0, 2.0, 3.0, 4.0});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {1});
        CHECK(dense->get_stride() == gko::size_type {1});
        CHECK(dense->get_const_values() == v.data());
    }

    SECTION("Vec3 mutable: 3-column non-owning Dense")
    {
        localIdx n = 3;
        Vector<Vec3> v(exec, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {3});
        CHECK(dense->get_stride() == gko::size_type {3});
        CHECK(dense->get_values() == reinterpret_cast<scalar*>(v.data()));
    }

    SECTION("Vec3 const: 3-column non-owning Dense")
    {
        localIdx n = 3;
        const Vector<Vec3> v(exec, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {3});
        CHECK(dense->get_stride() == gko::size_type {3});
        CHECK(dense->get_const_values() == reinterpret_cast<const scalar*>(v.data()));
    }
}

TEST_CASE("MatrixConversion - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto values = Vector<scalar>(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    auto rowIdx = Vector<localIdx>(exec, {0, 0, 1, 1, 1, 2, 2, 2, 3, 3});
    auto colIdx = Vector<localIdx>(exec, {0, 1, 0, 1, 2, 1, 2, 3, 2, 3});
    auto rowPtr = Vector<localIdx>(exec, {0, 2, 5, 8, 10});

    SECTION("CSRMatrix " + execName)
    {
        auto csrMatrix = CSRMatrix<scalar, localIdx>(values, colIdx, rowPtr, {4, 4});
        auto gkoCsrMtx = NeoN::la::ginkgo::createGkoMtx(csrMatrix);
    }

    SECTION("COOMatrix " + execName)
    {
        auto cooMatrix = COOMatrix<scalar, localIdx>(values, colIdx, rowIdx, {4, 4});
        auto gkoCooMtx = NeoN::la::ginkgo::createGkoMtx(cooMatrix);
    }

    // Same logical 4x4 matrix as the CSR/COO sections above (row0:{0:1,1:2},
    // row1:{0:3,1:4,2:5}, row2:{1:6,2:7,3:8}, row3:{2:9,3:10}), stored ELL-style: widest
    // rows (1 and 2) need 3 slots, stride = nRows = 4, column-major with padding trailing.
    SECTION("ELLMatrix " + execName)
    {
        const auto INV = EllSparsityView<localIdx>::invalidIndex();
        auto ellValues =
            Vector<scalar>(exec, {1.0, 3.0, 6.0, 9.0, 2.0, 4.0, 7.0, 10.0, 0.0, 5.0, 8.0, 0.0});
        auto ellColIdx = Vector<localIdx>(exec, {0, 0, 1, 2, 1, 1, 2, 3, INV, 2, 3, INV});
        auto ellSp = std::make_shared<const EllSparsityPattern<localIdx>>(
            std::move(ellColIdx), Dimensions {4, 4}, 3, 4, 10
        );
        auto ellMatrix = ELLMatrix<scalar, localIdx>(ellValues, ellSp);

        auto gkoEllMtx = NeoN::la::ginkgo::createGkoMtx(ellMatrix);
        auto gkoEll =
            std::dynamic_pointer_cast<const gko::matrix::Ell<scalar, localIdx>>(gkoEllMtx);
        REQUIRE(gkoEll != nullptr);

        // Zero-copy: Ginkgo's arrays must alias NeoN's own memory, not a copy of it.
        REQUIRE(gkoEll->get_const_values() == ellMatrix.values().view().data());
        REQUIRE(gkoEll->get_const_col_idxs() == ellMatrix.sparsity()->colIdxs().view().data());
        REQUIRE(
            gkoEll->get_num_stored_elements_per_row()
            == static_cast<gko::size_type>(ellMatrix.sparsity()->numStoredElementsPerRow())
        );
        REQUIRE(
            gkoEll->get_stride() == static_cast<gko::size_type>(ellMatrix.sparsity()->stride())
        );

        // Correctness: apply() against the same logical matrix as the CSR section above
        // must give the same result.
        auto csrMatrix = CSRMatrix<scalar, localIdx>(values, colIdx, rowPtr, {4, 4});
        auto gkoCsrMtx = NeoN::la::ginkgo::createGkoMtx(csrMatrix);

        auto gkoExec = NeoN::la::ginkgo::getGkoExecutor(exec);
        Vector<scalar> xVec(exec, {1.0, 1.0, 1.0, 1.0});
        Vector<scalar> yCsr(exec, 4, 0.0);
        Vector<scalar> yEll(exec, 4, 0.0);

        auto xDense = NeoN::la::ginkgo::gkoVecView<scalar>(gkoExec, xVec.view().data(), 4);
        auto yCsrDense = NeoN::la::ginkgo::gkoVecView<scalar>(gkoExec, yCsr.view().data(), 4);
        auto yEllDense = NeoN::la::ginkgo::gkoVecView<scalar>(gkoExec, yEll.view().data(), 4);

        gkoCsrMtx->apply(xDense, yCsrDense);
        gkoEllMtx->apply(xDense, yEllDense);

        REQUIRE_THAT(yEll, Equals(yCsr, Approx {1e-10}));
    }
}

// End-to-end: a native NeoN ELL system, wrapped zero-copy by Ginkgo, solves to the same
// solution and residual as the equivalent CSR system. Constructs GinkgoSolver directly and
// calls solveImpl<SystemMatrixType> on it -- SolverFactory/Solver stay CSR-only until the
// DSL's own format selection is generalized, same scoping as every other ELL proof so far.
TEST_CASE("Solve - Ginkgo ELL vs CSR")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    const localIdx nFaces = nCells - 1;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto csrLs =
        NeoN::la::createEmptyLinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>(mesh);
    auto ellLs =
        NeoN::la::createEmptyLinearSystem<scalar, scalar, ELLMatrix<scalar, localIdx>>(mesh);

    // Diagonally dominant 1D stencil: each face adds 2.0 to each endpoint's diagonal and a
    // single -1.0 off-diagonal, so diag always dominates the row's off-diagonal sum.
    auto assemble = [&](auto& ls)
    {
        auto ma = ls.matrix().faceToMatrixView();
        auto matrixV = ls.matrix().values().view();
        parallelFor(
            exec,
            {0, nFaces},
            NEON_LAMBDA(const localIdx facei) {
                const localIdx own = facei;
                const localIdx nei = facei + 1;
                matrixV[ma.upperIdx(own, facei)] = -1.0;
                matrixV[ma.lowerIdx(nei, facei)] = -1.0;
                Kokkos::atomic_add(&matrixV[ma.diagIdx(own)], 2.0);
                Kokkos::atomic_add(&matrixV[ma.diagIdx(nei)], 2.0);
            },
            "assembleDiagDominant"
        );
        fill(ls.rhs(), 1.0);
    };
    assemble(csrLs);
    assemble(ellLs);

    Dictionary solverDict {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Cg"},
         {"criteria", Dictionary {{{"iteration", 200}, {"relative_residual_norm", 1e-12}}}}}
    };
    NeoN::la::ginkgo::GinkgoSolver ginkgoSolver(exec, solverDict);

    Vector<scalar> xCsr(exec, nCells, 0.0);
    Vector<scalar> xEll(exec, nCells, 0.0);
    auto csrStats = ginkgoSolver.solveImpl<CSRMatrix<scalar, localIdx>>(csrLs, xCsr);
    auto ellStats = ginkgoSolver.solveImpl<ELLMatrix<scalar, localIdx>>(ellLs, xEll);

    // Both must report an actual, non-trivial solve -- catches e.g. a solver that returns
    // immediately without iterating but still leaves some (wrong) value in x.
    REQUIRE(csrStats.entries.size() == 1);
    REQUIRE(ellStats.entries.size() == 1);
    REQUIRE(csrStats.entries[0].numIter > 0);
    REQUIRE(ellStats.entries[0].numIter > 0);

    REQUIRE_THAT(xEll, Equals(xCsr, Approx {1e-8}));

    // True residual ||b - Ax|| for each -- computeResidual() itself is CSR-only (reads
    // sparsity.rowOffs directly, which EllSparsityView doesn't have), so this reuses the
    // already-proven generic createGkoMtx()/apply() path instead of extending that helper.
    auto gkoExec = NeoN::la::ginkgo::getGkoExecutor(exec);
    auto residualNorm = [&](auto& ls, auto& x)
    {
        auto gkoMtx = NeoN::la::ginkgo::createGkoMtx(ls.matrix());
        Vector<scalar> ax(exec, nCells, 0.0);
        auto xDense = NeoN::la::ginkgo::gkoVecView<scalar>(gkoExec, x.view().data(), nCells);
        auto axDense = NeoN::la::ginkgo::gkoVecView<scalar>(gkoExec, ax.view().data(), nCells);
        gkoMtx->apply(xDense, axDense);

        auto rhsHost = ls.rhs().copyToHost();
        auto rhsHostV = rhsHost.view();
        auto axHost = ax.copyToHost();
        auto axHostV = axHost.view();
        scalar residNormSq = 0.0;
        for (localIdx i = 0; i < nCells; ++i)
        {
            const scalar r = rhsHostV[i] - axHostV[i];
            residNormSq += r * r;
        }
        return std::sqrt(residNormSq);
    };

    REQUIRE(residualNorm(csrLs, xCsr) < 1e-6);
    REQUIRE(residualNorm(ellLs, xEll) < 1e-6);
}

TEST_CASE("MatrixAssembly - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    Vector<localIdx> colIdx(exec, {0, 1, 0, 1, 2, 1, 2});
    Vector<localIdx> rowOffs(exec, {0, 2, 5, 7});
    Vector<localIdx> bColIdx(exec, {});
    Vector<localIdx> bRowOffs(exec, {});

    const auto nRows = static_cast<localIdx>(rowOffs.size()) - 1;
    auto sparsity = std::make_shared<CsrSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(rowOffs), Dimensions {nRows, nRows}
    );
    auto bSparsity = std::make_shared<CooSparsityPattern<localIdx>>(
        std::move(bColIdx), std::move(bRowOffs), Dimensions {0, 0}
    );

    SECTION("Solve linear system scalar " + execName)
    {
        Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
        Vector<scalar> rhs(exec, {1.0, 2.0, 3.0});

        Vector<scalar> bValues(exec, {});
        COOMatrix<scalar, localIdx> bCooMatrix(bValues, bSparsity);
        Vector<scalar> bRhs(exec, {});

        auto linearSystem = LinearSystem<scalar>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);

        Vector<scalar> x(exec, {0.0, 0.0, 0.0});

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
        };

        // Create solver
        auto solver = NeoN::la::Solver(exec, solverDict);

        // Solve system
        auto solverStats = solver.solve(linearSystem, x);
        auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];

        auto hostX = x.copyToHost();
        auto hostXS = hostX.view();
        REQUIRE((hostXS[0]) == Catch::Approx(1.24489796).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(2.44897959).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(3.24489796).margin(1e-8));
        REQUIRE(numIter == 3);
        REQUIRE(initResNorm == Catch::Approx(3.741657386).margin(1e-8));
        REQUIRE(finalResNorm < 1.0e-04);
    }

    SECTION("Solve linear system vector " + execName)
    {
        Vector<Vec3> values(
            exec,
            {{1.0, 1.0, 1.0},
             {-0.1, -0.1, -0.1},
             {-0.1, -0.1, -0.1},
             {1.0, 1.0, 1.0},
             {-0.1, -0.1, -0.1},
             {-0.1, -0.1, -0.1},
             {1.0, 1.0, 1.0}}
        );

        CSRMatrix<Vec3, localIdx> csrMatrix(values, sparsity);
        Vector<Vec3> bValues(exec, {});
        COOMatrix<Vec3, localIdx> bCooMatrix(bValues, bSparsity);
        Vector<Vec3> bRhs(exec, {});

        Vector<Vec3> rhs(exec, {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}});
        Vector<Vec3> x(exec, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});

        auto linearSystem = LinearSystem<Vec3>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);

        SECTION("Segregated" + execName)
        {

            Dictionary solverDict {
                {{"solver", std::string {"Ginkgo"}},
                 {"type", "solver::Cg"},
                 {"coupled", false},
                 {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
            };

            // Create solver
            auto solver = NeoN::la::Solver(exec, solverDict);

            // Solve system
            auto solverStats = solver.solve(linearSystem, x);
            for (auto entry : solverStats.entries)
            {
                auto [numIter, initResNorm, finalResNorm, solveTime] = entry;
                auto hostX = x.copyToHost();
                auto hostXS = hostX.view();
                REQUIRE((hostXS[0][0]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][0]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][0]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][1]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][1]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][1]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][2]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][2]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][2]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE(numIter == 3);
                REQUIRE(initResNorm == Catch::Approx(3.741657386).margin(1e-8));
                REQUIRE(finalResNorm < 1.0e-04);
            }
        }
        SECTION("Coupled" + execName)
        {

            Dictionary solverDict {
                {{"solver", std::string {"Ginkgo"}},
                 {"type", "solver::Cg"},
                 {"coupled", true},
                 {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
            };

            // Create solver
            auto solver = NeoN::la::Solver(exec, solverDict);

            // Solve system
            auto solverStats = solver.solve(linearSystem, x);
            for (auto entry : solverStats.entries)
            {
                auto [numIter, initResNorm, finalResNorm, solveTime] = entry;
                auto hostX = x.copyToHost();
                auto hostXS = hostX.view();
                REQUIRE((hostXS[0][0]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][0]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][0]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][1]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][1]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][1]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][2]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][2]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][2]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE(numIter == 3);
                REQUIRE(initResNorm == Catch::Approx(6.4807406984).margin(1e-8));
                REQUIRE(finalResNorm < 1.0e-04);
            }
        }

        SECTION("Solve linear system wo boundary scalar with multiple rhs " + execName)
        {
            Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
            CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
            Vector<Vec3> rhs(exec, {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}});

            Vector<scalar> bValues(exec, {});
            COOMatrix<scalar, localIdx> bCsrMatrix(bValues, bSparsity);
            Vector<Vec3> bRhs(exec, {});

            auto linearSystem = LinearSystem<
                scalar,
                NeoN::Vec3,
                NeoN::la::CSRMatrix<scalar, NeoN::localIdx>,
                NeoN::la::COOMatrix<scalar, NeoN::localIdx>>(csrMatrix, rhs, bCsrMatrix, bRhs);

            Vector<Vec3> x(exec, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});

            Dictionary solverDict {
                {{"solver", std::string {"Ginkgo"}},
                 {"type", "solver::Cg"},
                 {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
            };

            // Create solver
            auto solver = NeoN::la::Solver(exec, solverDict);

            // Solve system
            auto solverStats = solver.solve(linearSystem, x);
            auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];

            auto hostX = x.copyToHost();
            auto hostXS = hostX.view();
            for (int c = 0; c < 3; ++c)
            {
                REQUIRE((hostXS[0][c]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][c]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][c]) == Catch::Approx(3.24489796).margin(1e-8));
            }
            REQUIRE(numIter == 3);
            REQUIRE(initResNorm == Catch::Approx(3.741657386).margin(1e-8));
            REQUIRE(finalResNorm < 1.0e-04);
        }
    }
}

// Exercises the implicit transform-BC solver path: a scalar matrix with a Vec3 RHS plus a
// per-component diagonal correction (diagCmpt). The three components are solved segregated, with
// each column's correction temporarily subtracted from the shared diagonal and then restored.
// Using a purely diagonal matrix gives an analytic answer x_c = b_c / (D - diagCmpt_c).
TEST_CASE("Implicit transform diagonal correction solve - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 4;
    auto mesh = NeoN::create1DUniformMesh(exec, nCells);
    auto ls = NeoN::la::createEmptyLinearSystem<scalar, Vec3>(mesh);

    // diagonal D·I (off-diagonals left at zero)
    const scalar D = 10.0;
    {
        auto values = ls.matrix().values().view();
        const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().rowOffs().view());
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx c) { values[ma.diagIdx(c)] = D; },
            "setDiag"
        );
    }

    // per-component diagonal correction (subtracted by the solver) and a uniform RHS
    const Vec3 dc(1.0, 2.0, 3.0);
    NeoN::fill(ls.ensureDiagCmpt(), dc);
    const scalar bVal = 6.0;
    NeoN::fill(ls.rhs(), Vec3(bVal, bVal, bVal));

    // analytic per-component solution: x_c = b_c / (D - diagCmpt_c)
    auto requireAnalytic = [&](const Vector<Vec3>& sol)
    {
        auto host = sol.copyToHost();
        auto v = host.view();
        for (localIdx i = 0; i < nCells; ++i)
        {
            REQUIRE(v[i][0] == Catch::Approx(bVal / (D - dc[0])).margin(1e-8));
            REQUIRE(v[i][1] == Catch::Approx(bVal / (D - dc[1])).margin(1e-8));
            REQUIRE(v[i][2] == Catch::Approx(bVal / (D - dc[2])).margin(1e-8));
        }
    };

    SECTION("standard stopping criterion " + execName)
    {
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 20}, {"relative_residual_norm", 1e-10}}}}}
        };
        auto solver = NeoN::la::Solver(exec, solverDict);

        Vector<Vec3> x(exec, nCells, Vec3(0.0, 0.0, 0.0));
        auto stats = solver.solve(ls, x);
        REQUIRE(stats.entries.size() == 3); // three segregated component solves
        requireAnalytic(x);

        // Re-solving must give the same answer: only holds if the shared diagonal was restored
        // after each column (otherwise it would be doubly corrected, D - 2·dc).
        Vector<Vec3> x2(exec, nCells, Vec3(0.0, 0.0, 0.0));
        solver.solve(ls, x2);
        requireAnalytic(x2);
    }

    SECTION("l1ScaledResidual stopping criterion " + execName)
    {
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"l1ScaledResidual", true},
             {"criteria", Dictionary {{{"iteration", 20}, {"absolute_residual_norm", 1e-10}}}}}
        };
        auto solver = NeoN::la::Solver(exec, solverDict);

        Vector<Vec3> x(exec, nCells, Vec3(0.0, 0.0, 0.0));
        auto stats = solver.solve(ls, x);
        REQUIRE(stats.entries.size() == 3);
        requireAnalytic(x);
    }
}
#endif
