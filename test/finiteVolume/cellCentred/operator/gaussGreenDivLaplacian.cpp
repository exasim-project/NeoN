// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

auto MAKE_FV_SCHEMES = []()
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
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    fvSchemes.insert("laplacianSchemes", lapSchemes);
    return fvSchemes;
};

TEMPLATE_TEST_CASE("Div + Laplacian Operator ", "[template]", NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());
    NeoN::fill(phi.boundaryData().value(), NeoN::zero<TestType>());
    phi.correctBoundaryConditions();

    auto fvSchemes = MAKE_FV_SCHEMES();

    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);

    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);

    auto expr = NeoN::dsl::imp::div(faceFlux, phi) + NeoN::dsl::imp::laplacian(gamma, phi);

    expr.read(fvSchemes);

    auto t = NeoN::scalar(1.0);
    auto dt = NeoN::scalar(1.0);

    SECTION("Can assemble to scalar matrix with rhs<Vec3>")
    {
        auto ls = expr.template assemble<NeoN::scalar>(mesh, t, dt);

        // Matrix coefficients are scalar; rhs values are Vec3.
        static_assert(std::is_same_v<
                      std::decay_t<decltype(ls.matrix().values())>,
                      NeoN::Vector<NeoN::scalar>>);
        static_assert(std::is_same_v<std::decay_t<decltype(ls.rhs())>, NeoN::Vector<NeoN::Vec3>>);

        // Assembly must have produced non-zero matrix entries.
        auto matHost = ls.matrix().values().copyToHost();
        auto matView = matHost.view();
        NeoN::scalar matAbsSum = 0.0;
        for (localIdx i = 0; i < matView.size(); ++i)
        {
            matAbsSum += std::abs(matView[i]);
        }
        REQUIRE(matAbsSum > 0.0);
    }

    SECTION("Can assemble to Vec3 matrix with rhs<Vec3>")
    {
        auto ls = expr.template assemble<NeoN::Vec3>(mesh, t, dt);

        // Matrix coefficients and rhs values are both Vec3.
        static_assert(std::is_same_v<
                      std::decay_t<decltype(ls.matrix().values())>,
                      NeoN::Vector<NeoN::Vec3>>);
        static_assert(std::is_same_v<std::decay_t<decltype(ls.rhs())>, NeoN::Vector<NeoN::Vec3>>);

        // Assembly must have produced non-zero matrix entries.
        auto matHost = ls.matrix().values().copyToHost();
        auto matView = matHost.view();
        NeoN::scalar matMagSum = 0.0;
        for (localIdx i = 0; i < matView.size(); ++i)
        {
            matMagSum += mag(matView[i]);
        }
        REQUIRE(matMagSum > 0.0);
    }

#if NF_WITH_GINKGO
    SECTION("Can solve with multiple RHS")
    {
        auto ls = expr.template assemble<NeoN::scalar>(mesh, t, dt);
        fill(ls.rhs(), 2.0 * one<Vec3>());

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Gmres"},
             {"criteria", Dictionary {{{"iteration", 200}, {"relative_residual_norm", 1e-7}}}}}
        };

        auto solver = NeoN::la::Solver(exec, solverDict);
        Vector<Vec3> x(exec, mesh.nCells(), zero<Vec3>());
        auto solverStats = solver.solve(ls, x);

        REQUIRE(solverStats.entries.size() == 3); // one entry per Vec3 component
        const auto& stats = solverStats.entries[0];

        // The solver must have iterated at least once.
        REQUIRE(stats.numIter > 0);

        // Residual should not have grown.
        REQUIRE(stats.finalResNorm <= stats.initResNorm);

        // x should be non-zero after solving.
        auto xHost = x.copyToHost();
        auto xView = xHost.view();
        NeoN::scalar xMagSum = 0.0;
        for (localIdx i = 0; i < xView.size(); ++i)
        {
            xMagSum += mag(xView[i]);
        }
        REQUIRE(xMagSum > 0.0);
    }
#endif // NF_WITH_GINKGO
}

TEMPLATE_TEST_CASE(
    "Face-based and cell-based GaussGreenDivLaplacian give same results", "[template]", NeoN::scalar
)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);
    NeoN::fill(faceFlux.boundaryData().value(), 1.0);

    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 2.0);
    NeoN::fill(gamma.boundaryData().value(), 2.0);

    NeoN::Dictionary fvSchemes;
    fvSchemes.insert(
        "divSchemes",
        NeoN::Dictionary {
            {"div(faceFlux,phi)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}
        }
    );
    fvSchemes.insert(
        "laplacianSchemes",
        NeoN::Dictionary {
            {"laplacian(gamma,phi)",
             NeoN::TokenList(
                 {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
             )}
        }
    );

    // Build fused operator by extracting configs from the individual operators
    dsl::SpatialOperator<TestType> divOp = dsl::imp::div(faceFlux, phi);
    dsl::SpatialOperator<TestType> lapOp = dsl::imp::laplacian(gamma, phi);
    divOp.read(fvSchemes);
    lapOp.read(fvSchemes);

    dsl::SpatialOperator<TestType> fusedOp =
        fvcc::GaussGreenDivLaplacian<TestType>(exec, divOp.getConfig(), lapOp.getConfig());
    fusedOp.read(fvSchemes);

    auto lsFaceBased = NeoN::la::createEmptyLinearSystem<TestType>(mesh);
    auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
    auto lsCellBased = NeoN::la::createEmptyLinearSystem<TestType>(mesh, cellIterator);

    fusedOp.implicitOperation(lsFaceBased);
    fusedOp.implicitOperation(lsCellBased);

    REQUIRE_THAT(
        lsFaceBased.matrix().values(), Equals(lsCellBased.matrix().values(), Approx {1e-12})
    );
    REQUIRE_THAT(lsFaceBased.rhs(), Equals(lsCellBased.rhs(), Approx {1e-12}));
}

// Full vertical slice, going through dsl::optimize() -- the point being that a realistic
// div(flux,phi) + laplacian(gamma,phi) expression loses nothing when the default optimizer
// pipeline fuses it into GaussGreenDivLaplacian, the same way it doesn't for either operator
// unfused (gaussGreenDiv.cpp, laplacianOperator.cpp). Fused-vs-unfused CSR equivalence is already
// covered by test/dsl/optimizer.cpp; this test is specifically about the fused operator's own
// ELL support.
TEST_CASE("Expression assembles fused GaussGreenDivLaplacian into ELL, matches CSR")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto fluxV = faceFlux.internalVector().view();
    auto gammaV = gamma.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // signed, alternating flux -- exercises both upwind directions, not just one
            fluxV[facei] = (facei % 2 == 0 ? 1.0 : -1.0) * (1.0 + 0.1 * static_cast<scalar>(facei));
            gammaV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
        }
    );
    fill(faceFlux.boundaryData().value(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Dictionary divSchemes;
    divSchemes.insert(
        "div(faceFlux,phi)", TokenList({std::string("Gauss"), std::string("upwind")})
    );
    Dictionary lapSchemes;
    lapSchemes.insert(
        "laplacian(gamma,phi)",
        TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    Dictionary fvSchemes;
    fvSchemes.insert("divSchemes", divSchemes);
    fvSchemes.insert("laplacianSchemes", lapSchemes);

    // div + laplacian, both implicit -- DivLapOptimizer fuses this into a single
    // GaussGreenDivLaplacian operator (verified: expr.size() == 2, optExpr.size() == 1, matching
    // test/dsl/optimizer.cpp's own check).
    auto expr = dsl::imp::div(faceFlux, phi) + dsl::imp::laplacian(gamma, phi);
    auto optExpr = dsl::optimize(expr);
    REQUIRE(expr.size() == 2);
    REQUIRE(optExpr.size() == 1);
    optExpr.read(fvSchemes);

    auto csrLs = optExpr.assemble<scalar, CSRMatrix>(mesh, 0.0, 0.0);
    auto ellLs = optExpr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0);

    // Direct fused-vs-unfused ELL comparison, so this test doesn't rely on combining "fused CSR
    // == fused ELL" (below) with test/dsl/optimizer.cpp's separate "unfused CSR == fused CSR" to
    // imply the ELL result is also correct.
    expr.read(fvSchemes);
    auto unfusedEllLs = expr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0);
    REQUIRE_THAT(unfusedEllLs.matrix().diag(), Equals(ellLs.matrix().diag(), Approx {1e-10}));
    REQUIRE_THAT(unfusedEllLs.rhs(), Equals(ellLs.rhs(), Approx {1e-10}));

    REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag(), Approx {1e-10}));
    REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs(), Approx {1e-10}));
    REQUIRE_THAT(
        csrLs.boundaryMatrix().values(), Equals(ellLs.boundaryMatrix().values(), Approx {1e-10})
    );
    REQUIRE_THAT(csrLs.boundaryRhs(), Equals(ellLs.boundaryRhs(), Approx {1e-10}));

    // Compare every logical (row,col) entry -- not the flat values() arrays, which have
    // different physical layouts (CSR compact vs ELL padded column-major).
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
}

// Corrected-scheme coverage for the fused ELL vertical slice above: linearUpwind (exercises
// addDivCorrectionToRhs's fused call site) on a genuinely non-orthogonal mesh (the sheared-cube
// technique from basicGeometryScheme.cpp / laplacianOperator.cpp, so
// computeLaplacianNonOrthCorrImpl also sees a nonzero correction, not just a dispatched no-op).
// Compares against the same mesh assembled with upwind/uncorrected schemes to prove the corrections
// actually changed the assembled rhs, then checks CSR and ELL agree on the corrected result.
TEST_CASE("Expression assembles fused GaussGreenDivLaplacian into ELL, matches CSR, corrected")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx n = 4;
    auto mesh = create3DUniformMesh(exec, n, n, n);
    const scalar s = 0.5;
    {
        auto ccH = mesh.cellCenters().copyToHost();
        auto v = ccH.view();
        for (localIdx i = 0; i < ccH.size(); ++i)
        {
            const Vec3 p = v[i];
            v[i] = Vec3 {p[0], p[1] + s * p[0], p[2]};
        }
        mesh.cellCenters() = ccH.copyToExecutor(exec);
    }
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto fluxV = faceFlux.internalVector().view();
    auto gammaV = gamma.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            fluxV[facei] = (facei % 2 == 0 ? 1.0 : -1.0) * (1.0 + 0.1 * static_cast<scalar>(facei));
            gammaV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
        }
    );
    fill(faceFlux.boundaryData().value(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    auto buildFvSchemes = [](const TokenList& divTokens, const std::string& snGradScheme)
    {
        Dictionary divSchemes;
        divSchemes.insert("div(faceFlux,phi)", divTokens);
        Dictionary lapSchemes;
        lapSchemes.insert(
            "laplacian(gamma,phi)",
            TokenList({std::string("Gauss"), std::string("linear"), snGradScheme})
        );
        Dictionary fvSchemes;
        fvSchemes.insert("divSchemes", divSchemes);
        fvSchemes.insert("laplacianSchemes", lapSchemes);
        return fvSchemes;
    };

    auto exprBaseline = dsl::imp::div(faceFlux, phi) + dsl::imp::laplacian(gamma, phi);
    auto optBaseline = dsl::optimize(exprBaseline);
    optBaseline.read(
        buildFvSchemes(TokenList({std::string("Gauss"), std::string("upwind")}), "uncorrected")
    );
    auto baselineLs = optBaseline.assemble<scalar, CSRMatrix>(mesh, 0.0, 0.0);

    auto exprCorrected = dsl::imp::div(faceFlux, phi) + dsl::imp::laplacian(gamma, phi);
    auto optCorrected = dsl::optimize(exprCorrected);
    optCorrected.read(buildFvSchemes(
        TokenList({std::string("Gauss"), std::string("linearUpwind"), std::string("Gauss")}),
        "corrected"
    ));

    auto csrLs = optCorrected.assemble<scalar, CSRMatrix>(mesh, 0.0, 0.0);
    auto ellLs = optCorrected.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0);

    // The div + non-orthogonal corrections are explicit rhs contributions only; on this sheared
    // mesh with a linearUpwind div scheme, they must actually change the rhs relative to the
    // upwind/uncorrected baseline, proving the generalized correction paths were exercised.
    auto baseRhsHost = baselineLs.rhs().copyToHost();
    auto corrRhsHost = csrLs.rhs().copyToHost();
    auto baseRhsV = baseRhsHost.view();
    auto corrRhsV = corrRhsHost.view();
    scalar maxDiff = 0.0;
    for (localIdx i = 0; i < corrRhsV.size(); ++i)
    {
        const scalar diff = mag(corrRhsV[i] - baseRhsV[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    REQUIRE(maxDiff > 1e-6);

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
}

} // namespace NeoN
