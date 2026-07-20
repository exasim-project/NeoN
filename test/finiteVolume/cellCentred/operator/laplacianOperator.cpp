// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;

namespace NeoN
{

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("laplacianOperator fixedValue", "[template]", scalar, Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 2.0);
    fill(gamma.boundaryData().value(), 2.0);

    auto [boundaryType, firstValue, lastValue] = GENERATE(
        std::tuple<std::string, scalar, scalar> {"fixedValue", 0.5, 10.5},
        std::tuple<std::string, scalar, scalar> {"fixedGradient", -10.0, 10}
    );

    SECTION(boundaryType)
    {
        std::vector<fvcc::VolumeBoundary<TestType>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<TestType>(
            mesh,
            Dictionary(
                {{"type", std::string(boundaryType)}, {boundaryType, firstValue * one<TestType>()}}
            ),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<TestType>(
            mesh,
            Dictionary(
                {{"type", std::string(boundaryType)}, {boundaryType, lastValue * one<TestType>()}}
            ),
            1
        ));

        auto phi = fvcc::VolumeField<TestType>(exec, "phi", mesh, bcs);
        parallelFor(
            phi.internalVector(),
            NEON_LAMBDA(const localIdx i) { return scalar(i + 1) * one<TestType>(); }
        );
        phi.correctBoundaryConditions();

        Input input =
            TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});

        SECTION("Construct from Token" + execName)
        {
            fvcc::LaplacianOperator<TestType> lapOp(
                dsl::Operator::Type::Implicit, gamma, phi, input
            );
        }

        SECTION("explicit laplacian operator for constant field on " + execName)
        {
            dsl::SpatialOperator lapOp = dsl::exp::laplacian(gamma, phi);
            lapOp.read(input);
            Vector<TestType> source(exec, nCells, zero<TestType>());
            lapOp.explicitOperation(source);
            auto sourceHost = source.copyToHost();
            auto sourceV = sourceHost.view();
            for (NeoN::localIdx i = 0; i < nCells; i++)
            {
                // the laplacian of a linear function is 0
                REQUIRE(mag(sourceV[i]) == Catch::Approx(0.0).margin(1e-8));
            }
        }

        auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);

        SECTION("implicit laplacian operator of constant field on " + execName)
        {
            dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
            lapOp.read(input);
            // currently only defined for scalar types
            if constexpr (std::is_same_v<TestType, scalar>)
            {
                lapOp.implicitOperation(ls);
                auto res = Vector<scalar>(phi.internalVector());
                fill(res, 1.0);

                computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

                auto resHost = res.copyToHost();
                auto resV = resHost.view();
                for (localIdx celli = 0; celli < resV.size(); celli++)
                {
                    // the laplacian of a linear function is 0
                    REQUIRE(resV[celli] == Catch::Approx(0.0).margin(1e-8));
                }
            }
        }

        SECTION("implicit laplacian operator scale" + execName)
        {
            if constexpr (std::is_same_v<TestType, scalar>)
            {
                ls.reset();
                dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
                lapOp.read(input);
                lapOp = dsl::Coeff(-0.5) * lapOp;

                lapOp.implicitOperation(ls);

                auto res = Vector<scalar>(phi.internalVector());
                computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

                auto resHost = res.copyToHost();
                auto resV = resHost.view();
                for (localIdx celli = 0; celli < resV.size(); celli++)
                {
                    // the laplacian of a linear function is 0
                    REQUIRE(resV[celli] == Catch::Approx(0.0).margin(1e-8));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("laplacianOperator boundary contributions are accumulated", "[template]", scalar)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    std::vector<fvcc::VolumeBoundary<TestType>> bcs;
    bcs.push_back(fvcc::VolumeBoundary<TestType>(
        mesh,
        Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", 0.5 * one<TestType>()}}),
        0
    ));
    bcs.push_back(fvcc::VolumeBoundary<TestType>(
        mesh,
        Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", 10.5 * one<TestType>()}}),
        1
    ));

    auto phi = fvcc::VolumeField<TestType>(exec, "phi", mesh, bcs);
    parallelFor(
        phi.internalVector(),
        NEON_LAMBDA(const localIdx i) { return scalar(i + 1) * one<TestType>(); }
    );
    phi.correctBoundaryConditions();

    Input input =
        TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});

    if constexpr (std::is_same_v<TestType, scalar>)
    {
        SECTION("bValues and bRhs accumulate on repeated calls on " + execName)
        {
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);
            dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
            lapOp.read(input);

            lapOp.implicitOperation(ls);
            auto bValuesFirst = ls.boundaryMatrix().values().copyToHost();
            auto bRhsFirst = ls.boundaryRhs().copyToHost();

            // second call without reset: bValues and bRhs must accumulate, not overwrite
            lapOp.implicitOperation(ls);
            auto bValuesSecond = ls.boundaryMatrix().values().copyToHost();
            auto bRhsSecond = ls.boundaryRhs().copyToHost();

            auto bValuesFirstV = bValuesFirst.view();
            auto bValuesSecondV = bValuesSecond.view();
            auto bRhsFirstV = bRhsFirst.view();
            auto bRhsSecondV = bRhsSecond.view();

            for (localIdx i = 0; i < bValuesFirstV.size(); i++)
            {
                // fixedValue BC: boundary contribution to diagonal is positive (atomic_sub
                // counterpart stored in bValues with matching sign)
                REQUIRE(bValuesFirstV[i] > 0);
                REQUIRE(bValuesSecondV[i] == Catch::Approx(2.0 * bValuesFirstV[i]).margin(1e-8));
            }
            for (localIdx i = 0; i < bRhsFirstV.size(); i++)
            {
                REQUIRE(bRhsFirstV[i] > 0);
                REQUIRE(bRhsSecondV[i] == Catch::Approx(2.0 * bRhsFirstV[i]).margin(1e-8));
            }
        }

        SECTION("removeBoundaryContributions restores interior-only diagonal on " + execName)
        {
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);
            dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
            lapOp.read(input);
            lapOp.implicitOperation(ls);

            auto lsNoBnd = la::removeBoundaryContributions(ls);
            auto diagHost = lsNoBnd.matrix().diag().copyToHost();
            auto diagV = diagHost.view();

            // interior cells have 2 interior face neighbors; boundary cells have 1.
            // after removing boundary contributions, each cell's diagonal reflects only
            // interior face connections — boundary cells should have exactly half the
            // magnitude of interior cells on a uniform 1D mesh.
            for (localIdx i = 1; i < nCells - 1; i++)
            {
                REQUIRE(diagV[i] == Catch::Approx(diagV[1]).margin(1e-8));
            }
            REQUIRE(diagV[0] == Catch::Approx(0.5 * diagV[1]).margin(1e-8));
            REQUIRE(diagV[nCells - 1] == Catch::Approx(0.5 * diagV[1]).margin(1e-8));
        }
    }
}

TEMPLATE_TEST_CASE(
    "Face based and cellbased iteration give same results", "[template]", NeoN::scalar
)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Input input =
        TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});

    auto lsFaceBased = la::createEmptyLinearSystem<TestType>(mesh);

    auto cellIterator = std::make_shared<la::CellBasedIterator>();
    auto lsCellBased = la::createEmptyLinearSystem<TestType>(mesh, cellIterator);

    dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
    lapOp.read(input);
    lapOp.implicitOperation(lsFaceBased);
    lapOp.implicitOperation(lsCellBased);

    REQUIRE_THAT(
        lsFaceBased.matrix().values(), Equals(lsCellBased.matrix().values(), Approx {1e-12})
    );
}

// Implicit slip/symmetry: the scalar-matrix Vec3 Laplacian assembly must populate the linear
// system's per-component diagonal correction (diagCmpt) with γ|S|·coeff·Δ·|n_c|, accumulated per
// cell, rather than touching the shared scalar diagonal. This is the assembly half of the implicit
// transform-BC path (the solver applies it column-by-column).
TEST_CASE("laplacianOperator implicit transform diagCmpt")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("implicit slip diagonal correction on " + execName)
    {
        auto mesh = createSingleCellMesh(exec);

        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
        fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
        const scalar gammaValue = 2.0;
        fill(gamma.internalVector(), gammaValue);
        fill(gamma.boundaryData().value(), gammaValue);

        std::vector<fvcc::VolumeBoundary<Vec3>> bcs;
        for (NeoN::localIdx patchID = 0; patchID < mesh.nBoundaries(); ++patchID)
        {
            bcs.push_back(fvcc::VolumeBoundary<Vec3>(
                mesh, Dictionary({{"type", std::string("slip")}, {"implicit", true}}), patchID
            ));
        }
        auto phi = VolumeField<Vec3>(exec, "phi", mesh, bcs);
        fill(phi.internalVector(), Vec3(1.0, -1.0, 0.5));
        phi.correctBoundaryConditions();

        const scalar coeff = 1.0;
        // GaussGreenLaplacian's constructor consumes interpolation/gradient tokens directly (the
        // DSL strips the leading "Gauss" operator name first), so pass only those two here.
        Input input = TokenList({std::string("linear"), std::string("uncorrected")});
        auto ls = NeoN::la::createEmptyLinearSystem<scalar, Vec3>(mesh);
        fvcc::GaussGreenLaplacian<Vec3, scalar> lapOp(exec, mesh, input);
        lapOp.laplacian(ls, gamma, phi, dsl::Coeff(coeff));

        // the implicit transform path must have lazily allocated diagCmpt, one Vec3 per cell
        REQUIRE(ls.diagCmpt());
        REQUIRE(ls.diagCmpt()->size() == mesh.nCells());

        auto [diagCmptH, nH, magSfH, deltaH, ownerH] = copyToHosts(
            *ls.diagCmpt(),
            mesh.boundaryMesh().faceUnitNormals(),
            mesh.boundaryMesh().faceAreas(),
            mesh.boundaryMesh().deltaCoeffs(),
            mesh.boundaryMesh().faceOwners()
        );

        // expected: per cell, sum over its boundary faces of γ|S|·coeff·Δ·|n_c|
        std::vector<Vec3> expected(static_cast<size_t>(mesh.nCells()), Vec3(0.0, 0.0, 0.0));
        for (NeoN::localIdx f = 0; f < mesh.nBoundaryFaces(); ++f)
        {
            const auto own = static_cast<size_t>(ownerH.view()[f]);
            const auto n = nH.view()[f];
            const scalar w = gammaValue * magSfH.view()[f] * coeff * deltaH.view()[f];
            expected[own][0] += w * std::abs(n[0]);
            expected[own][1] += w * std::abs(n[1]);
            expected[own][2] += w * std::abs(n[2]);
        }

        auto diagV = diagCmptH.view();
        for (NeoN::localIdx c = 0; c < mesh.nCells(); ++c)
        {
            for (auto d = 0u; d < 3; ++d)
            {
                REQUIRE(diagV[c][d] == Catch::Approx(expected[static_cast<size_t>(c)][d]));
            }
        }
        // no face on this mesh has a z-normal, so the z-component must be exactly zero
        REQUIRE(diagV[0][2] == Catch::Approx(0.0));
    }
}

// computeLaplacianIntImpl is the only piece of Laplacian assembly touching upperIdx()/lowerIdx()
// as well as diagIdx() -- called here directly on CSR and ELL systems, bypassing the still-CSR-
// only virtual GaussGreenLaplacian::laplacian(), to prove the assembly kernel itself is
// format-generic for a full (not diagonal-only) finite-volume stencil.
TEMPLATE_TEST_CASE(
    "computeLaplacianIntImpl matches for CSR and ELL", "[template]", NeoN::scalar, NeoN::Vec3
)
{
    using CSRMatrix = NeoN::la::CSRMatrix<TestType, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<TestType, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // 4x4 mesh: corner cells have 2 internal-face neighbours, edge cells 3, interior cells 4 --
    // gives ELL three distinct row widths (real padding, multiple diagonal slot positions),
    // unlike the uniform 1D mesh used elsewhere in this branch.
    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 2.0);
    fill(gamma.boundaryData().value(), 2.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    // FaceNormalGradientFactory only expects its own scheme token -- the 3-token
    // {"Gauss","linear","uncorrected"} form elsewhere in this file is consumed one token at a
    // time as it passes through GaussGreenLaplacian's own factory dispatch and
    // SurfaceInterpolation before reaching here; constructing FaceNormalGradient directly needs
    // just the remaining "uncorrected" token.
    Input faceNormalGradientInput = TokenList({std::string("uncorrected")});
    fvcc::FaceNormalGradient<TestType> faceNormalGradient(exec, mesh, faceNormalGradientInput);

    SECTION("logical entries, diag and padding match " + execName)
    {
        auto csrLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, CSRMatrix>(mesh);
        auto ellLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, ELLMatrix>(mesh);

        fvcc::computeLaplacianIntImpl(csrLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);
        fvcc::computeLaplacianIntImpl(ellLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);

        REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag(), Approx {1e-10}));

        // Compare every logical (row,col) entry -- not the flat values() arrays, which have
        // different physical layouts (CSR compact vs ELL padded column-major).
        auto csrLsHost = csrLs.copyToExecutor(SerialExecutor());
        auto ellLsHost = ellLs.copyToExecutor(SerialExecutor());
        auto csrSparsityView = csrLsHost.matrix().sparsity()->view();
        auto csrMatView = csrLsHost.matrix().view();
        auto ellMatView = ellLsHost.matrix().view();

        std::vector<TestType> csrEntries;
        std::vector<TestType> ellEntries;
        for (localIdx row = 0; row < nCells; ++row)
        {
            for (localIdx col = 0; col < nCells; ++col)
            {
                if (csrSparsityView.findEntry(row, col)
                    != decltype(csrSparsityView)::invalidIndex())
                {
                    csrEntries.push_back(csrMatView.entry(row, col));
                    ellEntries.push_back(ellMatView.entry(row, col));
                }
            }
        }
        REQUIRE(csrEntries.size() == ellEntries.size());
        REQUIRE_THAT(
            Vector<TestType>(SerialExecutor(), ellEntries), Equals(csrEntries, Approx {1e-10})
        );

        // Every ELL slot whose column index is the padding sentinel must stay untouched.
        auto colIdxHostV = ellLsHost.matrix().sparsity()->colIdxs().view();
        auto ellValuesHostV = ellLsHost.matrix().values().view();
        for (localIdx i = 0; i < colIdxHostV.size(); ++i)
        {
            if (colIdxHostV[i] == decltype(ellLsHost.matrix().sparsity()->view())::invalidIndex())
            {
                REQUIRE(ellValuesHostV[i] == zero<TestType>());
            }
        }
    }
}

// Segregated vector-solve form (scalar matrix, Vec3 rhs), matching the ELL instantiation added
// alongside GaussGreenLaplacian<Vec3, scalar>'s CSR support. computeLaplacianIntImpl only ever
// reads phi.mesh()/phi.exec() -- never phi's values -- so this performs the identical scalar
// arithmetic as the scalar/scalar case above; the point of this test is proving the
// {Vec3, scalar, ELLMatrix} instantiation actually links and runs, not a new numerical path.
TEST_CASE("computeLaplacianIntImpl matches for CSR and ELL, segregated vector-solve form")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 2.0);
    fill(gamma.boundaryData().value(), 2.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Input faceNormalGradientInput = TokenList({std::string("uncorrected")});
    fvcc::FaceNormalGradient<Vec3> faceNormalGradient(exec, mesh, faceNormalGradientInput);

    SECTION("diag matches " + execName)
    {
        auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, CSRMatrix>(mesh);
        auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, ELLMatrix>(mesh);

        fvcc::computeLaplacianIntImpl(csrLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);
        fvcc::computeLaplacianIntImpl(ellLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);

        REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag(), Approx {1e-10}));
    }
}

} // namespace NeoN
