// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEMPLATE_TEST_CASE("Matrix", "[template]", NeoN::scalar)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // sparse matrix
    // [ 1 .  . ]
    // [ . 5  6 ]
    // [ . 8 .  ]
    NeoN::Vector<TestType> valuesSparse(exec, {1.0, 5.0, 6.0, 8.0});
    NeoN::Vector<NeoN::localIdx> colIdxSparse(exec, {0, 1, 2, 1});
    NeoN::Vector<NeoN::localIdx> rowOffsSparse(exec, {0, 1, 3, 4});
    NeoN::la::CSRMatrix<TestType, NeoN::localIdx> sparseMatrix(
        valuesSparse, colIdxSparse, rowOffsSparse, {3, 3}
    );
    const NeoN::la::CSRMatrix<TestType, NeoN::localIdx> sparseMatrixConst(
        valuesSparse, colIdxSparse, rowOffsSparse, {3, 3}
    );

    // dense matrix
    NeoN::Vector<TestType> valuesDense(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    NeoN::Vector<NeoN::localIdx> colIdxDense(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    NeoN::Vector<NeoN::localIdx> rowOffsDense(exec, {0, 3, 6, 9});
    NeoN::la::CSRMatrix<TestType, NeoN::localIdx> denseMatrix(
        valuesDense, colIdxDense, rowOffsDense, {3, 3}
    );
    const NeoN::la::CSRMatrix<TestType, NeoN::localIdx> denseMatrixConst(
        valuesDense, colIdxDense, rowOffsDense, {3, 3}
    );

    // NOTE: The purpose of this test is to detect changes in the order
    // of the structured bindings
    SECTION("View Order " + execName)
    {
        auto denseMatrixHost = denseMatrix.copyToHost();
        auto [values, sparsity] = denseMatrixHost.view();
        auto [colIdxs, rowOffs] = sparsity;
        auto valuesDenseHost = valuesDense.copyToHost();
        auto valuesDenseHostView = valuesDenseHost.view();
        auto colIdxDenseHost = colIdxDense.copyToHost();
        auto colIdxDenseHostView = colIdxDenseHost.view();
        auto rowOffsDenseHost = rowOffsDense.copyToHost();
        auto rowOffsDenseHostView = rowOffsDenseHost.view();

        for (int i = 0; i < valuesDenseHostView.size(); ++i)
        {
            REQUIRE(valuesDenseHostView[i] == values[i]);
            REQUIRE(colIdxDenseHostView[i] == colIdxs[i]);
        }
        for (int i = 0; i < rowOffsDenseHostView.size(); ++i)
        {
            REQUIRE(rowOffsDenseHostView[i] == rowOffs[i]);
        }
    }

    SECTION("Read entry on " + execName)
    {
        // Sparse
        NeoN::Vector<NeoN::scalar> checkSparse(exec, 4);
        auto checkSparseView = checkSparse.view();
        auto csrView = sparseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                checkSparseView[0] = csrView.entry(0, 0);
                checkSparseView[1] = csrView.entry(1, 1);
                checkSparseView[2] = csrView.entry(1, 2);
                checkSparseView[3] = csrView.entry(2, 1);
            }
        );
        REQUIRE_THAT(checkSparse, Equals(I({1.0, 5.0, 6.0, 8.0})));

        // Dense
        NeoN::Vector<NeoN::scalar> checkDense(exec, 9);
        auto checkDenseView = checkDense.view();
        auto denseView = denseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                checkDenseView[0] = denseView.entry(0, 0);
                checkDenseView[1] = denseView.entry(0, 1);
                checkDenseView[2] = denseView.entry(0, 2);
                checkDenseView[3] = denseView.entry(1, 0);
                checkDenseView[4] = denseView.entry(1, 1);
                checkDenseView[5] = denseView.entry(1, 2);
                checkDenseView[6] = denseView.entry(2, 0);
                checkDenseView[7] = denseView.entry(2, 1);
                checkDenseView[8] = denseView.entry(2, 2);
            }
        );

        auto denseExp = std::vector<NeoN::scalar> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        REQUIRE_THAT(checkDense, Equals(denseExp));
    }

    SECTION("Can extract diagonal " + execName)
    {
        REQUIRE_THAT(sparseMatrix.diag(), Equals(I({1.0, 5.0, 0.0})));
    }

    SECTION("Can extract diagonal " + execName)
    {
        REQUIRE_THAT(denseMatrix.diag(), Equals(I({1.0, 5.0, 9.0})));
    }

    SECTION("Can extract upper " + execName)
    {
        auto upper = NeoN::la::upper(denseMatrix);
        REQUIRE_THAT(upper, Equals(I({2.0, 3.0, 6.0})));
    }

    SECTION("Update existing entry on " + execName)
    {
        // Sparse
        auto csrView = sparseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                csrView.entry(0, 0) = -1.0;
                csrView.entry(1, 1) = -5.0;
                csrView.entry(1, 2) = -6.0;
                csrView.entry(2, 1) = -8.0;
            }
        );
        REQUIRE_THAT(sparseMatrix.values(), Equals(I({-1.0, -5.0, -6.0, -8.0})));

        // Dense
        auto denseView = denseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                denseView.entry(0, 0) = -1.0;
                denseView.entry(0, 1) = -2.0;
                denseView.entry(0, 2) = -3.0;
                denseView.entry(1, 0) = -4.0;
                denseView.entry(1, 1) = -5.0;
                denseView.entry(1, 2) = -6.0;
                denseView.entry(2, 0) = -7.0;
                denseView.entry(2, 1) = -8.0;
                denseView.entry(2, 2) = -9.0;
            }
        );
        REQUIRE_THAT(
            denseMatrix.values(), Equals(I({-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0}))
        );
    }

    SECTION("Read directValue on " + execName)
    {
        // Sparse
        NeoN::Vector<NeoN::scalar> checkSparse(exec, 4);
        auto csrView = sparseMatrixConst.view();
        checkSparse.apply(NEON_LAMBDA(const NeoN::localIdx i) { return csrView.entry(i); });
        REQUIRE_THAT(checkSparse, Equals(I({1.0, 5.0, 6.0, 8.0})));

        // Dense
        NeoN::Vector<NeoN::scalar> checkDense(exec, 9);
        auto denseView = denseMatrixConst.view();
        checkDense.apply(NEON_LAMBDA(const NeoN::localIdx i) { return denseView.entry(i); });
        REQUIRE_THAT(checkDense, Equals(I({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0})));
    }

    SECTION("Update existing directValue on " + execName)
    {
        // Sparse
        auto csrView = sparseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                csrView.entry(0) = -1.0;
                csrView.entry(1) = -5.0;
                csrView.entry(2) = -6.0;
                csrView.entry(3) = -8.0;
            }
        );
        REQUIRE_THAT(sparseMatrix.values(), Equals(I({-1.0, -5.0, -6.0, -8.0})));

        // Dense
        auto denseView = denseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                denseView.entry(0) = -1.0;
                denseView.entry(1) = -2.0;
                denseView.entry(2) = -3.0;
                denseView.entry(3) = -4.0;
                denseView.entry(4) = -5.0;
                denseView.entry(5) = -6.0;
                denseView.entry(6) = -7.0;
                denseView.entry(7) = -8.0;
                denseView.entry(8) = -9.0;
            }
        );
        REQUIRE_THAT(
            denseMatrix.values(), Equals(I({-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0}))
        );
    }

    SECTION("View " + execName)
    {
        auto hostMatrix = sparseMatrix.copyToHost();
        auto [value, sparsity] = hostMatrix.view();
        auto [column, row] = sparsity;
        auto hostvaluesSparse = valuesSparse.copyToHost();
        auto hostcolIdxSparse = colIdxSparse.copyToHost();
        auto hostrowOffsSparse = rowOffsSparse.copyToHost();

        REQUIRE(hostvaluesSparse.size() == value.size());
        REQUIRE(hostcolIdxSparse.size() == column.size());
        REQUIRE(hostrowOffsSparse.size() == row.size());

        for (NeoN::localIdx i = 0; i < value.size(); ++i)
        {
            REQUIRE(hostvaluesSparse.view()[i] == value[i]);
            REQUIRE(hostcolIdxSparse.view()[i] == column[i]);
        }
        for (NeoN::localIdx i = 0; i < row.size(); ++i)
        {
            REQUIRE(hostrowOffsSparse.view()[i] == row[i]);
        }
    }
}

TEMPLATE_TEST_CASE("Matrix", "[template]", NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // sparse matrix
    std::vector<NeoN::Vec3> valuesSparseV {
        {1.0, 1.0, 1.0}, {5.0, 5.0, 5.0}, {6.0, 6.0, 6.0}, {8.0, 8.0, 8.0}
    };
    NeoN::Vector<TestType> valuesSparse(exec, valuesSparseV);
    NeoN::Vector<NeoN::localIdx> colIdxSparse(exec, {0, 1, 2, 1});
    NeoN::Vector<NeoN::localIdx> rowOffsSparse(exec, {0, 1, 3, 4});

    NeoN::la::CSRMatrix<TestType, NeoN::localIdx> sparseMatrix(
        valuesSparse, colIdxSparse, rowOffsSparse, {3, 3}
    );
    const NeoN::la::CSRMatrix<TestType, NeoN::localIdx> sparseMatrixConst(
        valuesSparse, colIdxSparse, rowOffsSparse, {3, 3}
    );

    SECTION("Read entry on " + execName)
    {
        // Sparse
        NeoN::Vector<NeoN::Vec3> checkSparse(exec, 4);
        auto checkSparseView = checkSparse.view();
        auto csrView = sparseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                checkSparseView[0] = csrView.entry(0, 0);
                checkSparseView[1] = csrView.entry(1, 1);
                checkSparseView[2] = csrView.entry(1, 2);
                checkSparseView[3] = csrView.entry(2, 1);
            }
        );

        auto checkHost = checkSparse.copyToHost();
        REQUIRE(checkHost.view()[0] == NeoN::Vec3 {1.0, 1.0, 1.0});
        REQUIRE(checkHost.view()[1] == NeoN::Vec3 {5.0, 5.0, 5.0});
        REQUIRE(checkHost.view()[2] == NeoN::Vec3 {6.0, 6.0, 6.0});
        REQUIRE(checkHost.view()[3] == NeoN::Vec3 {8.0, 8.0, 8.0});
    }
}

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;
using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vec3;
using NeoN::Vector;

// CSR-vs-ELL parity for the momentum-predictor utilities (scaledInverseDiag / rAU,
// scaledInvDiagNegLUx / HbyA), on a real assembled Laplacian -- same rationale as
// "applyMatrixRelaxation matches for CSR and ELL" in test/dsl/relaxation.cpp: proves the
// format-generic rewrite (SparsityView/EllSparsityView rowSize()/linearIndex()/invalidIndex(),
// plus FaceToMatrixAddress::view() for the O(1)-diagonal overloads) produces identical results
// on both formats, on a mesh (create2DUniformMesh) whose nonuniform row lengths genuinely
// exercise ELL padding.
TEST_CASE("scaledInverseDiag and scaledInvDiagNegLUx match for CSR and ELL")
{
    using CSRMatrix = NeoN::la::CSRMatrix<Vec3, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<Vec3, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto gammaV = gamma.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            gammaV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
        }
    );
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    NeoN::Input faceNormalGradientInput = NeoN::TokenList({std::string("uncorrected")});
    fvcc::FaceNormalGradient<Vec3> faceNormalGradient(exec, mesh, faceNormalGradientInput);

    auto csrLs = NeoN::la::createEmptyLinearSystem<Vec3, Vec3, CSRMatrix>(mesh);
    auto ellLs = NeoN::la::createEmptyLinearSystem<Vec3, Vec3, ELLMatrix>(mesh);
    fvcc::computeLaplacianIntImpl(csrLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);
    fvcc::computeLaplacianIntImpl(ellLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);

    // Confirm this mesh's nonuniform row lengths (corner/edge/interior cells have different
    // stencil sizes) actually pad the ELL storage -- the CSR/ELL agreement checks below are
    // only meaningful proof of correct padding-handling if padding is genuinely present.
    REQUIRE(ellLs.matrix().sparsity()->storageSize() > ellLs.matrix().sparsity()->nnz());

    // scaledInvDiagNegLUx computes rAU = vol/diag internally (not a caller-supplied scale), so
    // the standalone scaledInverseDiag calls below use vol as their own scale factor too -- that
    // makes the two functions' rAU outputs the same quantity, so the cross-check at the end
    // (fused vs standalone) is actually comparing like with like, on top of the CSR-vs-ELL checks.
    const auto vol = mesh.cellVolumes();

    // scaledInverseDiag, no FaceToMatrixAddress (linear-scan overload).
    auto rAUCsr = NeoN::la::scaledInverseDiag(csrLs.matrix(), vol);
    auto rAUEll = NeoN::la::scaledInverseDiag(ellLs.matrix(), vol);
    REQUIRE_THAT(rAUCsr, Equals(rAUEll, Approx {1e-10}));

    // scaledInverseDiag, with FaceToMatrixAddress (O(1)-diagonal overload).
    auto rAUCsrMi = NeoN::la::scaledInverseDiag(csrLs.matrix(), *csrLs.faceToMatrixAddress(), vol);
    auto rAUEllMi = NeoN::la::scaledInverseDiag(ellLs.matrix(), *ellLs.faceToMatrixAddress(), vol);
    REQUIRE_THAT(rAUCsrMi, Equals(rAUEllMi, Approx {1e-10}));
    REQUIRE_THAT(rAUCsrMi, Equals(rAUCsr, Approx {1e-10})); // both overloads agree, CSR side

    // scaledInvDiagNegLUx (rAU + HbyA together).
    Vector<Vec3> aVec(exec, nCells);
    Vector<Vec3> b(exec, nCells);
    Catch::randomizeVector(aVec);
    Catch::randomizeVector(b);

    Vector<scalar> rAUCsrFused(exec, nCells, NeoN::zero<scalar>());
    Vector<scalar> rAUEllFused(exec, nCells, NeoN::zero<scalar>());
    Vector<Vec3> hByACsr(exec, nCells, NeoN::zero<Vec3>());
    Vector<Vec3> hByAEll(exec, nCells, NeoN::zero<Vec3>());
    NeoN::la::scaledInvDiagNegLUx(csrLs.matrix(), aVec, b, vol, rAUCsrFused, hByACsr);
    NeoN::la::scaledInvDiagNegLUx(ellLs.matrix(), aVec, b, vol, rAUEllFused, hByAEll);

    REQUIRE_THAT(rAUCsrFused, Equals(rAUEllFused, Approx {1e-10}));
    REQUIRE_THAT(hByACsr, Equals(hByAEll, Approx {1e-10}));
    REQUIRE_THAT(
        rAUCsrFused, Equals(rAUCsr, Approx {1e-10})
    ); // fused rAU matches the standalone one
}

// Segregated vector-solve form (scalar matrix, Vec3 rhs) of the test above -- the momentum-
// equation shape (a Vec3 field assembled into a scalar coefficient matrix). Only the
// with-FaceToMatrixAddress scaledInverseDiag overload exists for this form.
TEST_CASE("scaledInverseDiag and scaledInvDiagNegLUx match for CSR and ELL, segregated")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto gammaV = gamma.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            gammaV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
        }
    );
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    NeoN::Input faceNormalGradientInput = NeoN::TokenList({std::string("uncorrected")});
    fvcc::FaceNormalGradient<Vec3> faceNormalGradient(exec, mesh, faceNormalGradientInput);

    auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, CSRMatrix>(mesh);
    auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, ELLMatrix>(mesh);
    fvcc::computeLaplacianIntImpl(csrLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);
    fvcc::computeLaplacianIntImpl(ellLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);

    // Confirm this mesh's nonuniform row lengths genuinely pad the ELL storage -- same rationale
    // as the same-type test above.
    REQUIRE(ellLs.matrix().sparsity()->storageSize() > ellLs.matrix().sparsity()->nnz());

    // Same rationale as the same-type test above: vol as the scale factor makes the standalone
    // scaledInverseDiag and fused scaledInvDiagNegLUx rAU outputs comparable below.
    const auto vol = mesh.cellVolumes();

    auto rAUCsr = NeoN::la::scaledInverseDiag(csrLs.matrix(), *csrLs.faceToMatrixAddress(), vol);
    auto rAUEll = NeoN::la::scaledInverseDiag(ellLs.matrix(), *ellLs.faceToMatrixAddress(), vol);
    REQUIRE_THAT(rAUCsr, Equals(rAUEll, Approx {1e-10}));

    Vector<Vec3> aVec(exec, nCells);
    Vector<Vec3> b(exec, nCells);
    Catch::randomizeVector(aVec);
    Catch::randomizeVector(b);

    Vector<scalar> rAUCsrFused(exec, nCells, NeoN::zero<scalar>());
    Vector<scalar> rAUEllFused(exec, nCells, NeoN::zero<scalar>());
    Vector<Vec3> hByACsr(exec, nCells, NeoN::zero<Vec3>());
    Vector<Vec3> hByAEll(exec, nCells, NeoN::zero<Vec3>());
    NeoN::la::scaledInvDiagNegLUx(csrLs.matrix(), aVec, b, vol, rAUCsrFused, hByACsr);
    NeoN::la::scaledInvDiagNegLUx(ellLs.matrix(), aVec, b, vol, rAUEllFused, hByAEll);

    REQUIRE_THAT(rAUCsrFused, Equals(rAUEllFused, Approx {1e-10}));
    REQUIRE_THAT(hByACsr, Equals(hByAEll, Approx {1e-10}));
    REQUIRE_THAT(rAUCsrFused, Equals(rAUCsr, Approx {1e-10}));
}

// Hand-computed oracle for scaledInverseDiag/scaledInvDiagNegLUx, independent of the CSR-vs-ELL
// parity checks above: those only prove CSR and ELL agree with each other, which a bug shared by
// both (they now walk rows through the same generic rowSize()/linearIndex() logic) would still
// pass. This builds both formats directly (not through DSL assembly) from known values and checks
// against by-hand-derived expected rAU/HbyA. Vec3-isotropic values only -- the segregated (scalar
// matrix) overload differs from this one solely in the (trivially inspectable) presence/absence
// of a [0] component index, so it isn't separately hand-verified here.
//
// Matrix (3 cells):
//   row 0: diag only,        value 2   -> [2 . .]
//   row 1: diag 4, off 1     (col 2)   -> [. 4 1]
//   row 2: off 2 (col 1),    diag 3    -> [. 2 3]
// nnz = 5 (1 + 2 + 2). ELL: numStoredElementsPerRow = 2, stride = 3 -> storageSize = 6, so row 0
// has one padded slot -- genuine ELL padding is exercised, not just a same-length-rows case.
TEST_CASE("scaledInverseDiag and scaledInvDiagNegLUx match a hand-computed oracle, CSR and ELL")
{
    using EllSparsityType = NeoN::la::EllSparsityPattern<localIdx>;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nRows = 3;
    const auto v3 = [](scalar s) { return Vec3(s, s, s); };

    Vector<Vec3> valuesCsr(exec, std::vector<Vec3> {v3(2.0), v3(4.0), v3(1.0), v3(2.0), v3(3.0)});
    Vector<localIdx> colIdxCsr(exec, std::vector<localIdx> {0, 1, 2, 1, 2});
    Vector<localIdx> rowOffsCsr(exec, std::vector<localIdx> {0, 1, 3, 5});
    NeoN::la::CSRMatrix<Vec3, localIdx> csrMtx(valuesCsr, colIdxCsr, rowOffsCsr, {nRows, nRows});

    const auto inv = la::EllSparsityView<localIdx>::invalidIndex();
    const localIdx numStoredElementsPerRow = 2;
    const localIdx stride = nRows;
    // column-major, padded: slot 0 = [row0=col0, row1=col1, row2=col1],
    //                        slot 1 = [row0=pad,  row1=col2, row2=col2]
    Vector<localIdx> colIdxEll(exec, std::vector<localIdx> {0, 1, 1, inv, 2, 2});
    auto sp = std::make_shared<const EllSparsityType>(
        std::move(colIdxEll), la::Dimensions {nRows, nRows}, numStoredElementsPerRow, stride, 5
    );
    Vector<Vec3> valuesEll(
        exec, std::vector<Vec3> {v3(2.0), v3(4.0), v3(2.0), v3(0.0), v3(1.0), v3(3.0)}
    );
    NeoN::la::ELLMatrix<Vec3, localIdx> ellMtx(valuesEll, sp);

    REQUIRE(sp->storageSize() > sp->nnz()); // confirms row 0's padded slot is genuine

    SECTION("scaledInverseDiag on " + execName)
    {
        Vector<scalar> a(exec, std::vector<scalar> {10.0, 20.0, 30.0});
        auto expected = std::vector<scalar> {5.0, 5.0, 10.0}; // a[i] / diag[i]

        auto rAUCsr = NeoN::la::scaledInverseDiag(csrMtx, a);
        auto rAUEll = NeoN::la::scaledInverseDiag(ellMtx, a);
        REQUIRE_THAT(rAUCsr, Equals(expected, Approx {1e-12}));
        REQUIRE_THAT(rAUEll, Equals(expected, Approx {1e-12}));
    }

    SECTION("scaledInvDiagNegLUx on " + execName)
    {
        Vector<Vec3> aVec(exec, nRows, v3(1.0));
        Vector<Vec3> b(exec, nRows, NeoN::zero<Vec3>());
        Vector<scalar> vol(exec, nRows, 1.0);

        // rAU = vol/diag = [0.5, 0.25, 1/3].
        // row 0: no off-diag       -> HbyA = rAU*(b - 0)               = 0
        // row 1: off-diag 1 * a[2] -> HbyA = 0.25*(0 - 1*1)            = -0.25
        // row 2: off-diag 2 * a[1] -> HbyA = (1/3)*(0 - 2*1)           = -2/3
        auto expectedRAU = std::vector<scalar> {0.5, 0.25, 1.0 / 3.0};
        auto expectedHbyA = std::vector<Vec3> {v3(0.0), v3(-0.25), v3(-2.0 / 3.0)};

        Vector<scalar> rAUCsr(exec, nRows, NeoN::zero<scalar>());
        Vector<Vec3> hByACsr(exec, nRows, NeoN::zero<Vec3>());
        NeoN::la::scaledInvDiagNegLUx(csrMtx, aVec, b, vol, rAUCsr, hByACsr);
        REQUIRE_THAT(rAUCsr, Equals(expectedRAU, Approx {1e-12}));
        REQUIRE_THAT(hByACsr, Equals(expectedHbyA, Approx {1e-12}));

        Vector<scalar> rAUEll(exec, nRows, NeoN::zero<scalar>());
        Vector<Vec3> hByAEll(exec, nRows, NeoN::zero<Vec3>());
        NeoN::la::scaledInvDiagNegLUx(ellMtx, aVec, b, vol, rAUEll, hByAEll);
        REQUIRE_THAT(rAUEll, Equals(expectedRAU, Approx {1e-12}));
        REQUIRE_THAT(hByAEll, Equals(expectedHbyA, Approx {1e-12}));
    }
}
