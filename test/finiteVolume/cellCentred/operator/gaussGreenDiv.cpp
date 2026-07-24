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

TEMPLATE_TEST_CASE("DivOperator", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    // compute corresponding uniform faceFlux
    // TODO this should be handled outside of the unit test
    fvcc::SurfaceField<scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    fill(faceFlux.boundaryData().value(), 1.0);
    // left boundary face (patch 0) has opposite orientation to the flow
    auto bFaceFlux = faceFlux.boundaryData().value().view();
    parallelFor(
        exec, {0, 1}, NEON_LAMBDA(const localIdx i) { bFaceFlux[i] = -1.0; }
    );

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), one<TestType>());
    fill(phi.boundaryData().value(), one<TestType>());
    phi.correctBoundaryConditions();

    auto result = Vector<TestType>(exec, phi.size());
    fill(result, zero<TestType>());

    SECTION("Construct from Token" + execName)
    {
        Input input = TokenList({std::string("Gauss"), std::string("linear")});
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        Input input = Dictionary(
            {{std::string("DivOperator"), std::string("Gauss")},
             {std::string("surfaceInterpolation"), std::string("linear")}}
        );
        auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
        op.explicitOperation(result);

        // divergence of a uniform field should be zero
        auto outHost = result.copyToHost();
        auto outHostView = outHost.view();
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i] == zero<TestType>());
        }
    }

    SECTION("Implicit operation " + execName)
    {
        if constexpr (std::is_same_v<TestType, scalar>)
        {
            Input input = Dictionary(
                {{std::string("DivOperator"), std::string("Gauss")},
                 {std::string("surfaceInterpolation"), std::string("linear")}}
            );
            auto op = fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, input);
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);
            op.implicitOperation(ls);

            // the divergence of a uniform field under a conservative flux is zero,
            // so A*phi - b must vanish for every cell
            auto res = Vector<scalar>(exec, mesh.nCells(), 0.0);
            computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

            auto resExp = std::vector<NeoN::scalar>(res.size(), 0);
            REQUIRE_THAT(res, Equals(resExp, Approx {1e-12}));
        }
    }
}

// computeDivIntImp is the only piece of div assembly touching upperIdx()/lowerIdx() as well as
// diagIdx() -- called here directly on CSR and ELL systems, bypassing the still-CSR-only virtual
// GaussGreenDiv::div(), to prove the assembly kernel itself is format-generic, mirroring
// computeLaplacianIntImpl's equivalent test in laplacianOperator.cpp.
TEMPLATE_TEST_CASE(
    "computeDivIntImp matches for CSR and ELL", "[template]", NeoN::scalar, NeoN::Vec3
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
    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    fvcc::SurfaceField<scalar> weights(exec, "weights", mesh, surfaceBCs);

    // Face-dependent (not uniform) flux/weights, and a cell-dependent coefficient below --
    // uniform values could mask a swapped upperIdx()/lowerIdx() or misindexed face/cell, since
    // every face or cell would then contribute an identical value regardless of which one it
    // actually landed on.
    const auto nInternalFaces = mesh.nInternalFaces();
    auto fluxV = faceFlux.internalVector().view();
    auto weightsV = weights.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            fluxV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
            weightsV[facei] = 0.2 + 0.01 * static_cast<scalar>(facei % 30);
        }
    );
    fill(faceFlux.boundaryData().value(), 1.0);
    fill(weights.boundaryData().value(), 0.5);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Vector<scalar> coeffVec(exec, nCells, 0.0);
    auto coeffVecV = coeffVec.view();
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            coeffVecV[celli] = 1.0 + 0.05 * static_cast<scalar>(celli);
        }
    );
    dsl::Coeff coeff(coeffVec);

    SECTION("logical entries match " + execName)
    {
        auto csrLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, CSRMatrix>(mesh);
        auto ellLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, ELLMatrix>(mesh);

        fvcc::computeDivIntImp(csrLs, faceFlux, phi, weights, coeff);
        fvcc::computeDivIntImp(ellLs, faceFlux, phi, weights, coeff);

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
// alongside GaussGreenDiv<Vec3, scalar>'s CSR support. Compares every logical entry, not just
// the diagonal -- div is primarily a face-coupled operator, so the upper/lower entries matter
// here just as much as in the main comparison test above.
TEST_CASE("computeDivIntImp matches for CSR and ELL, segregated vector-solve form")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    fvcc::SurfaceField<scalar> weights(exec, "weights", mesh, surfaceBCs);

    const auto nInternalFaces = mesh.nInternalFaces();
    auto fluxV = faceFlux.internalVector().view();
    auto weightsV = weights.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            fluxV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
            weightsV[facei] = 0.2 + 0.01 * static_cast<scalar>(facei % 30);
        }
    );
    fill(faceFlux.boundaryData().value(), 1.0);
    fill(weights.boundaryData().value(), 0.5);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Vector<scalar> coeffVec(exec, nCells, 0.0);
    auto coeffVecV = coeffVec.view();
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            coeffVecV[celli] = 1.0 + 0.05 * static_cast<scalar>(celli);
        }
    );
    dsl::Coeff coeff(coeffVec);

    SECTION("logical entries match " + execName)
    {
        auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, CSRMatrix>(mesh);
        auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, ELLMatrix>(mesh);

        fvcc::computeDivIntImp(csrLs, faceFlux, phi, weights, coeff);
        fvcc::computeDivIntImp(ellLs, faceFlux, phi, weights, coeff);

        REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag(), Approx {1e-10}));

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
            Vector<scalar>(SerialExecutor(), ellEntries), Equals(csrEntries, Approx {1e-10})
        );
    }
}

TEST_CASE("DivOperator implicit boundary contributions are accumulated")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);

    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    fill(faceFlux.boundaryData().value(), 1.0);

    Input input = TokenList({std::string("Gauss"), std::string("linear")});

    SECTION("bRhs accumulates for fixedValue BC on " + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", NeoN::scalar(1.0)}}),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary({{"type", std::string("fixedValue")}, {"fixedValue", NeoN::scalar(2.0)}}),
            1
        ));
        auto phi = fvcc::VolumeField<NeoN::scalar>(exec, "phi", mesh, bcs);
        fill(phi.internalVector(), NeoN::scalar(1.0));
        phi.correctBoundaryConditions();

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        dsl::SpatialOperator divOp = dsl::imp::div(faceFlux, phi);
        divOp.read(input);

        divOp.implicitOperation(ls);
        auto bRhsFirst = ls.boundaryRhs().copyToHost();

        // second call without reset: bRhs must accumulate, not overwrite
        divOp.implicitOperation(ls);
        auto bRhsSecond = ls.boundaryRhs().copyToHost();

        auto bRhsFirstV = bRhsFirst.view();
        auto bRhsSecondV = bRhsSecond.view();
        for (localIdx i = 0; i < bRhsFirstV.size(); i++)
        {
            REQUIRE(NeoN::mag(bRhsFirstV[i]) > 0);
            REQUIRE(bRhsSecondV[i] == Catch::Approx(2.0 * bRhsFirstV[i]).margin(1e-8));
        }
    }

    SECTION("bValues has correct sign and accumulates for fixedGradient BC on " + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary(
                {{"type", std::string("fixedGradient")}, {"fixedGradient", NeoN::scalar(1.0)}}
            ),
            0
        ));
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(
            mesh,
            Dictionary(
                {{"type", std::string("fixedGradient")}, {"fixedGradient", NeoN::scalar(1.0)}}
            ),
            1
        ));
        auto phi = fvcc::VolumeField<NeoN::scalar>(exec, "phi", mesh, bcs);
        fill(phi.internalVector(), NeoN::scalar(1.0));
        phi.correctBoundaryConditions();

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        dsl::SpatialOperator divOp = dsl::imp::div(faceFlux, phi);
        divOp.read(input);

        divOp.implicitOperation(ls);
        auto bValuesFirst = ls.boundaryMatrix().values().copyToHost();
        auto bRhsFirst = ls.boundaryRhs().copyToHost();

        // second call without reset: contributions must accumulate, not overwrite
        divOp.implicitOperation(ls);
        auto bValuesSecond = ls.boundaryMatrix().values().copyToHost();
        auto bRhsSecond = ls.boundaryRhs().copyToHost();

        auto bValuesFirstV = bValuesFirst.view();
        auto bValuesSecondV = bValuesSecond.view();
        auto bRhsFirstV = bRhsFirst.view();
        auto bRhsSecondV = bRhsSecond.view();
        for (localIdx i = 0; i < bValuesFirstV.size(); i++)
        {
            // bValues stores the inverse of the diagonal contribution (bValues -= valueMat)
            // so bValues must be negative when the diagonal contribution is positive
            REQUIRE(bValuesFirstV[i] < 0);
            REQUIRE(bValuesSecondV[i] == Catch::Approx(2.0 * bValuesFirstV[i]).margin(1e-8));
        }
        for (localIdx i = 0; i < bRhsFirstV.size(); i++)
        {
            REQUIRE(bRhsSecondV[i] == Catch::Approx(2.0 * bRhsFirstV[i]).margin(1e-8));
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
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);

    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    fill(faceFlux.boundaryData().value(), 1.0);

    Input input = TokenList({std::string("Gauss"), std::string("linear")});

    auto lsFaceBased = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);

    auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
    auto lsCellBased = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh, cellIterator);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    dsl::SpatialOperator divOp = dsl::imp::div(faceFlux, phi);
    divOp.read(input);
    divOp.implicitOperation(lsFaceBased);
    divOp.implicitOperation(lsCellBased);

    REQUIRE_THAT(
        lsFaceBased.matrix().values(), Equals(lsCellBased.matrix().values(), Approx {1e-12})
    );
}

// Full vertical slice: dsl::imp::div (the production entry point) assembles into an ELL system
// via Expression::assemble<AssemblyType, SystemMatrixType>(), through DivOperator ->
// DivOperatorFactory -> GaussGreenDiv::div(ELL...) -- not by calling computeDivIntImp directly,
// unlike the TEMPLATE_TEST_CASEs above. Real boundary faces are in play here (computeDivBoundImpl
// is templated on SystemMatrixType too), so boundaryMatrix/boundaryRhs are compared as well.
TEST_CASE("Expression assembles div into ELL via DivOperator, matches CSR")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto fluxV = faceFlux.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) { fluxV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei); }
    );
    fill(faceFlux.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Dictionary divSchemes;
    divSchemes.insert(
        "div(faceFlux,phi)", TokenList({std::string("Gauss"), std::string("linear")})
    );
    Dictionary fvSchemes;
    fvSchemes.insert("divSchemes", divSchemes);

    dsl::Expression<scalar> expr(dsl::imp::div(faceFlux, phi));
    expr.read(fvSchemes);

    auto csrLs = expr.assemble<scalar, CSRMatrix>(mesh, 0.0, 0.0);
    auto ellLs = expr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0);

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

// Segregated vector-solve form (scalar matrix, Vec3 rhs) through the full DSL -- unlike
// "computeDivIntImp matches for CSR and ELL, segregated vector-solve form" above (which calls the
// kernel directly), this goes through dsl::imp::div -> Expression<Vec3> -> SpatialOperator's new
// segregated-ELL dispatch (HasImplicitOperatorScalarMtxELL/implicitOperationScalarMtxELL) ->
// DivOperator's new segregated implicitOperation<SystemMatrixType> entry point. The underlying
// kernel and GaussGreenDiv<Vec3, scalar>'s ELL override already existed (proven by the
// kernel-level test above); this proves the DSL can actually reach them.
TEST_CASE("Expression assembles div into ELL via DivOperator, matches CSR, segregated")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto fluxV = faceFlux.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) { fluxV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei); }
    );
    fill(faceFlux.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Dictionary divSchemes;
    divSchemes.insert(
        "div(faceFlux,phi)", TokenList({std::string("Gauss"), std::string("linear")})
    );
    Dictionary fvSchemes;
    fvSchemes.insert("divSchemes", divSchemes);

    dsl::Expression<Vec3> expr(dsl::imp::div(faceFlux, phi));
    expr.read(fvSchemes);

    auto csrLs = expr.assemble<scalar, CSRMatrix>(mesh, 0.0, 0.0);
    auto ellLs = expr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0);

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

// Corrected-scheme coverage for the ELL vertical slice above -- "Gauss linear" (used above) is
// uncorrected, so it never exercises addDivCorrectionToRhs's SystemMatrixType-generic path;
// linearUpwind's deferred gradient correction does.
TEST_CASE("Expression assembles div into ELL via DivOperator, matches CSR, linearUpwind")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    fvcc::SurfaceField<scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto fluxV = faceFlux.internalVector().view();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) { fluxV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei); }
    );
    fill(faceFlux.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    Dictionary divSchemes;
    divSchemes.insert(
        "div(faceFlux,phi)",
        TokenList({std::string("Gauss"), std::string("linearUpwind"), std::string("Gauss")})
    );
    Dictionary fvSchemes;
    fvSchemes.insert("divSchemes", divSchemes);

    dsl::Expression<scalar> expr(dsl::imp::div(faceFlux, phi));
    expr.read(fvSchemes);

    auto csrLs = expr.assemble<scalar, CSRMatrix>(mesh, 0.0, 0.0);
    auto ellLs = expr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0);

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
