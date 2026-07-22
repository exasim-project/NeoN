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

    REQUIRE_FALSE(csrStats.entries.empty());
    REQUIRE_FALSE(ellStats.entries.empty());
    REQUIRE(csrStats.entries.front().numIter > 0);
    REQUIRE(ellStats.entries.front().numIter > 0);
    REQUIRE(csrStats.entries.front().finalResNorm <= csrStats.entries.front().initResNorm);
    REQUIRE(ellStats.entries.front().finalResNorm <= ellStats.entries.front().initResNorm);

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

    REQUIRE_FALSE(csrStats.entries.empty());
    REQUIRE_FALSE(ellStats.entries.empty());
    REQUIRE(csrStats.entries.front().numIter > 0);
    REQUIRE(ellStats.entries.front().numIter > 0);
    REQUIRE(csrStats.entries.front().finalResNorm <= csrStats.entries.front().initResNorm);
    REQUIRE(ellStats.entries.front().finalResNorm <= ellStats.entries.front().initResNorm);

    REQUIRE_THAT(phiCsr.internalVector(), Equals(phiEll.internalVector(), Approx {1e-6}));

    // CSR and ELL could agree on the same wrong answer -- check against the known exact solution
    // too (see the comment above: rows sum to zero and rhs is zero everywhere except the
    // SetReference-pinned cell, so the unique solution is the uniform field equal to refValue).
    Vector<scalar> expected(exec, mesh.nCells(), 1.0);
    REQUIRE_THAT(phiCsr.internalVector(), Equals(expected, Approx {1e-6}));
    REQUIRE_THAT(phiEll.internalVector(), Equals(expected, Approx {1e-6}));
}
#endif

// No solver involved (pure assembly), so this runs regardless of NF_WITH_GINKGO.
// FixedValueConstraints has no applyELL() override; PostAssemblyBase's default now throws instead
// of silently doing nothing, so an unsupported functor passed to ELL assembly must fail loudly
// rather than quietly not applying its constraint.
TEST_CASE("Unimplemented ELL post-assembly functor throws rather than being silently dropped")
{
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 2, 2);
    auto nCells = mesh.nCells();

    Vector<scalar> mask(exec, nCells, 0.0);
    Vector<scalar> pinVals(exec, nCells, 0.0);
    dsl::FixedValueConstraints<scalar> unimplemented(mask.view(), pinVals.view(), nCells);

    dsl::Expression<scalar> expr(exec);
    REQUIRE_THROWS(expr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0, {&unimplemented}));
}

} // namespace NeoN
