// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "common.hpp"

namespace NeoN
{

#if NF_WITH_GINKGO
// Full vertical slice: dsl::solve() itself (the production PDE entry point, not
// Expression::assemble() + Solver::solve() called separately as every other ELL test in this
// series does) assembles and solves a complete implicit equation -- div, laplacian, and a source
// term together -- through both CSR and ELL, on a mesh with real internal faces and boundary
// contributions. ddt is deliberately left out: DdtOperator's oldTime/Database machinery has a
// known pre-existing crash when combined with a multi-cell mesh (unrelated to ELL, found and
// left out of scope earlier in this series; ddt's own ELL support is already proven separately,
// on a single-cell mesh, in ddtOperator.cpp), so including it here would risk resurfacing that
// bug rather than testing anything new.
TEST_CASE("dsl::solve assembles and solves a complete PDE via ELL, matches CSR")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 0.1);
    fill(faceFlux.boundaryData().value(), 0.1);
    fill(gamma.internalVector(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> sourceCoeff(exec, "sourceCoeff", mesh, volumeBCs);
    // Positive Sp keeps the system strictly diagonally dominant (regularizes what would
    // otherwise be a near-singular pure-diffusion problem under calculated/extrapolated BCs).
    fill(sourceCoeff.internalVector(), 0.5);
    fill(sourceCoeff.boundaryData().value(), 0.0);

    fvcc::VolumeField<scalar> phiCsr(exec, "phi", mesh, volumeBCs);
    fvcc::VolumeField<scalar> phiEll(exec, "phi", mesh, volumeBCs);
    fill(phiCsr.internalVector(), 1.0);
    fill(phiEll.internalVector(), 1.0);
    phiCsr.correctBoundaryConditions();
    phiEll.correctBoundaryConditions();

    Dictionary divSchemes;
    divSchemes.insert(
        "div(faceFlux,phi)", TokenList({std::string("Gauss"), std::string("upwind")})
    );
    Dictionary lapSchemes;
    lapSchemes.insert(
        "laplacian(gamma,phi)",
        TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    // No temporal operators in this expression, so the choice here is irrelevant to the solve
    // itself -- just needs to be a registered scheme so TimeIntegration<VolumeField> constructs.
    // "steadyState" isn't registered in this test binary (only where SteadyState<VolumeField> is
    // explicitly instantiated, in test/timeIntegration/steadyState.cpp's own TU).
    Dictionary timeIntegrationDict;
    timeIntegrationDict.insert("type", std::string("backwardEuler"));
    Dictionary fvSchemes;
    fvSchemes.insert("divSchemes", divSchemes);
    fvSchemes.insert("laplacianSchemes", lapSchemes);
    fvSchemes.insert("timeIntegration", timeIntegrationDict);

    Dictionary fvSolution {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Cg"},
         {"criteria", Dictionary {{{"iteration", 500}, {"relative_residual_norm", 1e-10}}}}}
    };

    // div - laplacian + source: two implicit spatial operators plus a diagonal one, exactly the
    // combination GaussGreenDiv/GaussGreenLaplacian/SourceTerm's own ELL work targeted.
    auto exprCsr = dsl::imp::div(faceFlux, phiCsr) - dsl::imp::laplacian(gamma, phiCsr)
                 + dsl::imp::source(sourceCoeff, phiCsr);
    auto exprEll = dsl::imp::div(faceFlux, phiEll) - dsl::imp::laplacian(gamma, phiEll)
                 + dsl::imp::source(sourceCoeff, phiEll);

    using VolumeFieldScalar = fvcc::VolumeField<scalar>;
    auto csrStats = dsl::solve<VolumeFieldScalar, localIdx, CSRMatrix>(
        exprCsr, phiCsr, 0.0, 1.0, fvSchemes, fvSolution
    );
    auto ellStats = dsl::solve<VolumeFieldScalar, localIdx, ELLMatrix>(
        exprEll, phiEll, 0.0, 1.0, fvSchemes, fvSolution
    );

    REQUIRE(csrStats.has_value());
    REQUIRE(ellStats.has_value());
    REQUIRE_FALSE(csrStats->entries.empty());
    REQUIRE_FALSE(ellStats->entries.empty());
    REQUIRE(csrStats->entries.front().numIter > 0);
    REQUIRE(ellStats->entries.front().numIter > 0);
    REQUIRE(csrStats->entries.front().finalResNorm <= csrStats->entries.front().initResNorm);
    REQUIRE(ellStats->entries.front().finalResNorm <= ellStats->entries.front().initResNorm);

    REQUIRE_THAT(phiCsr.internalVector(), Equals(phiEll.internalVector(), Approx {1e-6}));
}

// SetReference's ELL route (PostAssemblyBase::applyELL), exercised through dsl::solve() itself.
// A pure-diffusion equation under calculated (extrapolated, zero-gradient-like) boundaries has no
// Dirichlet contribution anywhere, so the matrix is singular (the classic constant null space) --
// SetReference::applyELL is what makes this solvable at all, not just a nicety.
TEST_CASE("dsl::solve with SetReference solves a singular Laplacian via ELL, matches CSR")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phiCsr(exec, "phi", mesh, volumeBCs);
    fvcc::VolumeField<scalar> phiEll(exec, "phi", mesh, volumeBCs);
    // Rows sum to zero (pure Laplacian, no source, no Dirichlet contribution from calculated
    // BCs), so once SetReference pins the diagonal, the exact solution is always the uniform
    // field equal to refValue (1.0 below) -- starting from that same uniform value would give CG
    // zero initial residual and 0 iterations, proving nothing. Start from 0 instead.
    fill(phiCsr.internalVector(), 0.0);
    fill(phiEll.internalVector(), 0.0);
    phiCsr.correctBoundaryConditions();
    phiEll.correctBoundaryConditions();

    Dictionary lapSchemes;
    lapSchemes.insert(
        "laplacian(gamma,phi)",
        TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    Dictionary timeIntegrationDict;
    timeIntegrationDict.insert("type", std::string("backwardEuler"));
    Dictionary fvSchemes;
    fvSchemes.insert("laplacianSchemes", lapSchemes);
    fvSchemes.insert("timeIntegration", timeIntegrationDict);

    Dictionary fvSolution {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Cg"},
         {"criteria", Dictionary {{{"iteration", 500}, {"relative_residual_norm", 1e-10}}}}}
    };

    dsl::SetReference<scalar, localIdx> setRef(0, 1.0);

    dsl::Expression<scalar> exprCsr(dsl::imp::laplacian(gamma, phiCsr));
    dsl::Expression<scalar> exprEll(dsl::imp::laplacian(gamma, phiEll));

    using VolumeFieldScalar = fvcc::VolumeField<scalar>;
    auto csrStats = dsl::solve<VolumeFieldScalar, localIdx, CSRMatrix>(
        exprCsr, phiCsr, 0.0, 1.0, fvSchemes, fvSolution, {&setRef}
    );
    auto ellStats = dsl::solve<VolumeFieldScalar, localIdx, ELLMatrix>(
        exprEll, phiEll, 0.0, 1.0, fvSchemes, fvSolution, {&setRef}
    );

    REQUIRE(csrStats.has_value());
    REQUIRE(ellStats.has_value());
    REQUIRE_FALSE(csrStats->entries.empty());
    REQUIRE_FALSE(ellStats->entries.empty());
    REQUIRE(csrStats->entries.front().numIter > 0);
    REQUIRE(ellStats->entries.front().numIter > 0);
    REQUIRE(csrStats->entries.front().finalResNorm <= csrStats->entries.front().initResNorm);
    REQUIRE(ellStats->entries.front().finalResNorm <= ellStats->entries.front().initResNorm);

    REQUIRE_THAT(phiCsr.internalVector(), Equals(phiEll.internalVector(), Approx {1e-6}));

    // CSR and ELL could agree on the same wrong answer -- check against the known exact solution
    // too (see the comment above: rows sum to zero and rhs is zero everywhere except the
    // SetReference-pinned cell, so the unique solution is the uniform field equal to refValue).
    Vector<scalar> expected(exec, mesh.nCells(), 1.0);
    REQUIRE_THAT(phiCsr.internalVector(), Equals(expected, Approx {1e-6}));
    REQUIRE_THAT(phiEll.internalVector(), Equals(expected, Approx {1e-6}));
}

// Segregated vector-solve form (scalar matrix, Vec3 rhs) of the two dsl::solve() tests above --
// the momentum-equation shape (div + laplacian, scalar-matrix assembly). Exercises, together, all
// of this pass's segregated-ELL solve() plumbing: dsl::solve()'s AssemblyType derivation (was
// hardcoded to VectorType::ElementType, i.e. Vec3, which mismatches an ELL matrix's own scalar
// value type), GinkgoSolver::solveSegregatedImpl<ELLMatrix>, and SetReference's new
// applyScalarMatrixELL override -- SetReference<Vec3> is exactly the PostAssemblyBase gap noted
// for the segregated-ELL case, and pinning is what makes this singular Laplacian solvable at all.
TEST_CASE(
    "dsl::solve assembles, pins, and solves a singular Laplacian via ELL, matches CSR, segregated"
)
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> phiCsr(exec, "phi", mesh, volumeBCs);
    fvcc::VolumeField<Vec3> phiEll(exec, "phi", mesh, volumeBCs);
    // Same rationale as the scalar singular-Laplacian test above: start away from the expected
    // uniform solution so CG actually does work (a zero initial residual proves nothing).
    fill(phiCsr.internalVector(), zero<Vec3>());
    fill(phiEll.internalVector(), zero<Vec3>());
    phiCsr.correctBoundaryConditions();
    phiEll.correctBoundaryConditions();

    Dictionary lapSchemes;
    lapSchemes.insert(
        "laplacian(gamma,phi)",
        TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    Dictionary timeIntegrationDict;
    timeIntegrationDict.insert("type", std::string("backwardEuler"));
    Dictionary fvSchemes;
    fvSchemes.insert("laplacianSchemes", lapSchemes);
    fvSchemes.insert("timeIntegration", timeIntegrationDict);

    Dictionary fvSolution {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Cg"},
         {"criteria", Dictionary {{{"iteration", 500}, {"relative_residual_norm", 1e-10}}}}}
    };

    const Vec3 refValue {1.0, 2.0, 3.0};
    dsl::SetReference<Vec3, localIdx> setRef(0, refValue);

    dsl::Expression<Vec3> exprCsr(dsl::imp::laplacian(gamma, phiCsr));
    dsl::Expression<Vec3> exprEll(dsl::imp::laplacian(gamma, phiEll));

    using VolumeFieldVec3 = fvcc::VolumeField<Vec3>;
    auto csrStats = dsl::solve<VolumeFieldVec3, localIdx, CSRMatrix>(
        exprCsr, phiCsr, 0.0, 1.0, fvSchemes, fvSolution, {&setRef}
    );
    auto ellStats = dsl::solve<VolumeFieldVec3, localIdx, ELLMatrix>(
        exprEll, phiEll, 0.0, 1.0, fvSchemes, fvSolution, {&setRef}
    );

    REQUIRE(csrStats.has_value());
    REQUIRE(ellStats.has_value());
    REQUIRE_FALSE(csrStats->entries.empty());
    REQUIRE_FALSE(ellStats->entries.empty());
    REQUIRE(csrStats->entries.front().numIter > 0);
    REQUIRE(ellStats->entries.front().numIter > 0);
    REQUIRE(csrStats->entries.front().finalResNorm <= csrStats->entries.front().initResNorm);
    REQUIRE(ellStats->entries.front().finalResNorm <= ellStats->entries.front().initResNorm);

    REQUIRE_THAT(phiCsr.internalVector(), Equals(phiEll.internalVector(), Approx {1e-6}));

    // Same closed-form check as the scalar test: rows sum to zero and rhs is zero everywhere
    // except the pinned cell, so the unique solution is the uniform field equal to refValue.
    Vector<Vec3> expected(exec, mesh.nCells(), refValue);
    REQUIRE_THAT(phiCsr.internalVector(), Equals(expected, Approx {1e-6}));
    REQUIRE_THAT(phiEll.internalVector(), Equals(expected, Approx {1e-6}));
}

// dsl::solve() with two cells pinned to two different prescribed values -- the two pins already
// make the system well-posed on their own (no need for the singular-Neumann setup above), so this
// uses a randomized phi and just checks the constrained cells land exactly on their prescribed
// values in the final solution, on both formats.
TEST_CASE("dsl::solve with FixedValueConstraints solves via ELL, matches CSR")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phiCsr(exec, "phi", mesh, volumeBCs);
    fvcc::VolumeField<scalar> phiEll(exec, "phi", mesh, volumeBCs);
    fill(phiCsr.internalVector(), 0.0);
    fill(phiEll.internalVector(), 0.0);
    phiCsr.correctBoundaryConditions();
    phiEll.correctBoundaryConditions();

    Dictionary lapSchemes;
    lapSchemes.insert(
        "laplacian(gamma,phi)",
        TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    Dictionary timeIntegrationDict;
    timeIntegrationDict.insert("type", std::string("backwardEuler"));
    Dictionary fvSchemes;
    fvSchemes.insert("laplacianSchemes", lapSchemes);
    fvSchemes.insert("timeIntegration", timeIntegrationDict);

    Dictionary fvSolution {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Cg"},
         {"criteria", Dictionary {{{"iteration", 500}, {"relative_residual_norm", 1e-10}}}}}
    };

    Vector<scalar> mask(exec, nCells, 0.0);
    Vector<scalar> values(exec, nCells, 0.0);
    auto maskV = mask.view();
    auto valuesV = values.view();
    parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx) {
            maskV[0] = 1.0;
            valuesV[0] = 3.0;
        }
    );
    parallelFor(
        exec,
        {nCells - 1, nCells},
        NEON_LAMBDA(const localIdx i) {
            maskV[i] = 1.0;
            valuesV[i] = -2.0;
        }
    );

    dsl::FixedValueConstraints<scalar> constraint(mask.view(), values.view(), nCells);

    dsl::Expression<scalar> exprCsr(dsl::imp::laplacian(gamma, phiCsr));
    dsl::Expression<scalar> exprEll(dsl::imp::laplacian(gamma, phiEll));

    using VolumeFieldScalar = fvcc::VolumeField<scalar>;
    auto csrStats = dsl::solve<VolumeFieldScalar, localIdx, CSRMatrix>(
        exprCsr, phiCsr, 0.0, 1.0, fvSchemes, fvSolution, {&constraint}
    );
    auto ellStats = dsl::solve<VolumeFieldScalar, localIdx, ELLMatrix>(
        exprEll, phiEll, 0.0, 1.0, fvSchemes, fvSolution, {&constraint}
    );

    REQUIRE(csrStats.has_value());
    REQUIRE(ellStats.has_value());
    REQUIRE_FALSE(csrStats->entries.empty());
    REQUIRE_FALSE(ellStats->entries.empty());
    REQUIRE(csrStats->entries.front().numIter > 0);
    REQUIRE(ellStats->entries.front().numIter > 0);

    REQUIRE_THAT(phiCsr.internalVector(), Equals(phiEll.internalVector(), Approx {1e-6}));

    auto phiCsrHost = phiCsr.internalVector().copyToHost();
    auto phiEllHost = phiEll.internalVector().copyToHost();
    REQUIRE(phiCsrHost.view()[0] == Catch::Approx(3.0).margin(1e-6));
    REQUIRE(phiCsrHost.view()[nCells - 1] == Catch::Approx(-2.0).margin(1e-6));
    REQUIRE(phiEllHost.view()[0] == Catch::Approx(3.0).margin(1e-6));
    REQUIRE(phiEllHost.view()[nCells - 1] == Catch::Approx(-2.0).margin(1e-6));
}
#endif

// Direct CSR-vs-ELL comparison of FixedValueConstraints itself (no solver, so this runs
// regardless of NF_WITH_GINKGO): two cells pinned to two different values on a real assembled
// Laplacian, checking every logical matrix entry, rhs, and ELL padding, plus the two properties
// FixedValueConstraints's own CSR-only unit test (test/dsl/constraints.cpp) already proves for
// CSR -- pinned-row off-diagonals zeroed and pinned rhs == diagonal * prescribed value -- also
// hold on ELL.
TEST_CASE("FixedValueConstraints matches for CSR and ELL")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto gammaV = gamma.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            gammaV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
        }
    );
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Input faceNormalGradientInput = TokenList({std::string("uncorrected")});
    fvcc::FaceNormalGradient<scalar> faceNormalGradient(exec, mesh, faceNormalGradientInput);

    auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, scalar, CSRMatrix>(mesh);
    auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, scalar, ELLMatrix>(mesh);
    fvcc::computeLaplacianIntImpl(csrLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);
    fvcc::computeLaplacianIntImpl(ellLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);

    Vector<scalar> mask(exec, nCells, 0.0);
    Vector<scalar> values(exec, nCells, 0.0);
    auto maskV = mask.view();
    auto valuesV = values.view();
    parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx) {
            maskV[0] = 1.0;
            valuesV[0] = 3.0;
        }
    );
    parallelFor(
        exec,
        {nCells - 1, nCells},
        NEON_LAMBDA(const localIdx i) {
            maskV[i] = 1.0;
            valuesV[i] = -2.0;
        }
    );

    // Snapshot pre-constraint entries at the two pinned columns, across every row, to explicitly
    // verify column-cut absorption below -- not just that CSR and ELL happen to agree afterward,
    // which the full logical-entry comparison further down already proves indirectly.
    auto preHost = csrLs.copyToExecutor(SerialExecutor());
    auto preSparsity = preHost.matrix().sparsity()->view();
    auto preMatView = preHost.matrix().view();
    auto preRhsV = preHost.rhs().view();
    std::vector<scalar> preColEntry0(nCells, 0.0);
    std::vector<scalar> preColEntryLast(nCells, 0.0);
    std::vector<scalar> expectedRhs(nCells, 0.0);
    for (localIdx row = 0; row < nCells; ++row)
    {
        expectedRhs[row] = preRhsV[row];
        if (preSparsity.findEntry(row, 0) != decltype(preSparsity)::invalidIndex())
        {
            preColEntry0[row] = preMatView.entry(row, 0);
        }
        if (preSparsity.findEntry(row, nCells - 1) != decltype(preSparsity)::invalidIndex())
        {
            preColEntryLast[row] = preMatView.entry(row, nCells - 1);
        }
    }
    for (localIdx row = 0; row < nCells; ++row)
    {
        if (row == 0 || row == nCells - 1) continue; // pinned rows verified separately below
        expectedRhs[row] -= preColEntry0[row] * scalar(3.0);
        expectedRhs[row] -= preColEntryLast[row] * scalar(-2.0);
    }

    dsl::FixedValueConstraints<scalar> constraint(mask.view(), values.view(), nCells);
    constraint(csrLs);
    constraint.applyELL(ellLs);

    REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag(), Approx {1e-10}));
    REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs(), Approx {1e-10}));

    auto csrLsHost = csrLs.copyToExecutor(SerialExecutor());
    auto ellLsHost = ellLs.copyToExecutor(SerialExecutor());
    auto csrSparsityView = csrLsHost.matrix().sparsity()->view();
    auto csrMatView = csrLsHost.matrix().view();
    auto ellMatView = ellLsHost.matrix().view();

    std::vector<scalar> csrEntries;
    std::vector<scalar> ellEntries;
    for (localIdx row = 0; row < nCells; ++row)
    {
        for (localIdx col = 0; col < nCells; ++col)
        {
            if (csrSparsityView.findEntry(row, col) != decltype(csrSparsityView)::invalidIndex())
            {
                csrEntries.push_back(csrMatView.entry(row, col));
                ellEntries.push_back(ellMatView.entry(row, col));
            }
        }
    }
    REQUIRE(csrEntries.size() == ellEntries.size());
    REQUIRE_THAT(Vector<scalar>(SerialExecutor(), ellEntries), Equals(csrEntries, Approx {1e-10}));

    // Every ELL slot whose column index is the padding sentinel must stay untouched.
    auto colIdxHostV = ellLsHost.matrix().sparsity()->colIdxs().view();
    auto ellValuesHostV = ellLsHost.matrix().values().view();
    for (localIdx i = 0; i < colIdxHostV.size(); ++i)
    {
        if (colIdxHostV[i] == decltype(ellLsHost.matrix().sparsity()->view())::invalidIndex())
        {
            REQUIRE(ellValuesHostV[i] == zero<scalar>());
        }
    }

    // Pinned rows: off-diagonals zero, rhs == diagonal * prescribed value (same properties
    // constraints.cpp's hand-built CSR test proves, checked here on ELL too).
    auto ellMatValuesV = ellLsHost.matrix().values().view();
    auto ellRhsV = ellLsHost.rhs().view();
    auto ellSparsity = ellLsHost.matrix().sparsity()->view();
    for (localIdx pinnedRow : {localIdx {0}, nCells - 1})
    {
        const auto diagIdx = ellSparsity.findEntry(pinnedRow, pinnedRow);
        const auto diagVal = ellMatValuesV[diagIdx];
        const auto expectedVal = (pinnedRow == 0) ? scalar(3.0) : scalar(-2.0);
        REQUIRE(ellRhsV[pinnedRow] == Catch::Approx(diagVal * expectedVal));
        for (localIdx col = 0; col < nCells; ++col)
        {
            if (col == pinnedRow) continue;
            const auto idx = ellSparsity.findEntry(pinnedRow, col);
            if (idx != decltype(ellSparsity)::invalidIndex())
            {
                REQUIRE(ellMatValuesV[idx] == Catch::Approx(0.0).margin(1e-12));
            }
        }
    }

    // Column-cut: every other row's entries in the two pinned columns must be zeroed, and its
    // rhs must have absorbed exactly that dropped coupling (computed from the pre-constraint
    // snapshot above), on ELL specifically -- not just inferred from CSR/ELL agreeing.
    for (localIdx row = 0; row < nCells; ++row)
    {
        if (row == 0 || row == nCells - 1) continue;
        REQUIRE(ellRhsV[row] == Catch::Approx(expectedRhs[row]).margin(1e-8));
        for (localIdx pinnedCol : {localIdx {0}, nCells - 1})
        {
            const auto idx = ellSparsity.findEntry(row, pinnedCol);
            if (idx != decltype(ellSparsity)::invalidIndex())
            {
                REQUIRE(ellMatValuesV[idx] == Catch::Approx(0.0).margin(1e-12));
            }
        }
    }
}

// A minimal PostAssemblyBase subclass that deliberately does not override applyELL(), standing in
// for "some functor whose author hasn't added ELL support yet". FixedValueConstraints no longer
// serves this purpose now that it implements applyELL() itself.
struct UnimplementedEllFunctor : public dsl::PostAssemblyBase<scalar, localIdx>
{
    void operator()(NeoN::la::LinearSystem<scalar, scalar, NeoN::la::CSRMatrix<scalar, localIdx>>&)
        const override
    {}
};

// No solver involved (pure assembly), so this runs regardless of NF_WITH_GINKGO.
// PostAssemblyBase's default applyELL() throws instead of silently doing nothing, so a functor
// that hasn't implemented ELL support must fail loudly rather than quietly not applying itself.
TEST_CASE("Unimplemented ELL post-assembly functor throws rather than being silently dropped")
{
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 2, 2);

    UnimplementedEllFunctor unimplemented;

    dsl::Expression<scalar> expr(exec);
    REQUIRE_THROWS(expr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0, {&unimplemented}));
}

} // namespace NeoN
