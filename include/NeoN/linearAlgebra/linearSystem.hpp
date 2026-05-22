// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/copyTo.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/distributed/communicationPattern.hpp"
#endif

#include <string>
#include <algorithm>
#include <numeric>
#include <vector>

namespace NeoN::la
{

/**
 * @struct LinearSystemView
 * @brief A view linear into a linear system's data.
 *
 * @tparam RHSValueType The value type of the rhs/solution vectors.
 * @tparam MatrixViewType The type representing the matrix view
 */
template<typename RHSValueType, typename MatrixViewType>
struct LinearSystemView
{
    LinearSystemView() = default;
    ~LinearSystemView() = default;

    LinearSystemView(
        MatrixViewType matrixView,
        View<RHSValueType> rhsView,
        MatrixViewType boundaryMatrixView,
        View<RHSValueType> boundaryRhsView
    )
        : matrix(matrixView), rhs(rhsView), boundaryMatrix(boundaryMatrixView),
          boundaryRhs(boundaryRhsView) {};

    MatrixViewType matrix;
    View<RHSValueType> rhs;

    MatrixViewType boundaryMatrix;
    View<RHSValueType> boundaryRhs;
};

/**
 * @class LinearSystem
 * @brief A class representing a linear system of equations.
 *
 * The LinearSystem class provides functionality to store and manipulate a linear system of
 * equations. It supports the storage of the coefficient matrix and the right-hand side vector, as
 * well as the solution vector.
 *
 * @tparam MatrixValueType The value type of the system and boundary matrix coefficients.
 * @tparam SystemMatrixType The sparse matrix type used for the system matrix (default:
 * CSRMatrix<MatrixValueType, localIdx>).
 * @tparam BoundaryMatrixType The sparse matrix type used for boundary and off-diagonal matrices
 * (default: COOMatrix<MatrixValueType, localIdx>).
 * @tparam RHSValueType The value type of the right-hand side and boundary rhs vectors. Defaults to
 * MatrixValueType, but may differ (e.g. scalar matrix with Vec3 rhs for segregated vector solves).
 */
template<
    typename MatrixValueType,
    typename SystemMatrixType = CSRMatrix<MatrixValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<MatrixValueType, localIdx>,
    typename RHSValueType = MatrixValueType>
class LinearSystem :
    public NeoN::SupportsCopyTo<
        LinearSystem<MatrixValueType, SystemMatrixType, BoundaryMatrixType, RHSValueType>>
{

    void validate()
    {
        NF_ASSERT(matrix_.exec() == rhs_.exec(), "Executors are not the same");
        NF_ASSERT(matrix_.nRows() == rhs_.size(), "Matrix and RHS size mismatch");
        NF_ASSERT(
            meshIteratorContext_ != nullptr,
            "Mesh iterator context must be set before validating the linear system"
        );
        NF_ASSERT(
            meshIteratorContext_->get() != nullptr,
            "Mesh iterator strategy must be set before validating the linear system"
        );
        // NF_ASSERT(
        //     boundaryMatrix_.nRows() == boundaryRhs_.size(), "BMatrix.nRows() !=
        //     boundaryRHS.size()"
        // );
    }


public:

    using LinearSystemIndexType = typename SystemMatrixType::MatrixSparsityType::SparsityIndexType;

    LinearSystem(
        const SystemMatrixType& matrix,
        const Vector<RHSValueType>& rhs,
        const BoundaryMatrixType& offDiagonalMatrix,
        const BoundaryMatrixType& boundaryMatrix,
        const Vector<RHSValueType>& boundaryRhs,
        std::shared_ptr<MeshIterationStrategy> strategy = std::make_shared<FaceBasedIterator>()
    )
        : matrix_(matrix), rhs_(rhs), boundaryMatrix_(boundaryMatrix),
          offDiagonalMatrix_(offDiagonalMatrix), boundaryRhs_(boundaryRhs),
          meshIteratorContext_(std::make_shared<MeshIteratorContext>())
    {
        meshIteratorContext_->setStrategy(strategy);
        validate();
    }

    LinearSystem(
        const SystemMatrixType& matrix,
        const Vector<RHSValueType>& rhs,
        const BoundaryMatrixType& boundaryMatrix,
        const Vector<RHSValueType>& boundaryRhs,
        std::shared_ptr<MeshIterationStrategy> strategy = std::make_shared<FaceBasedIterator>()
    )
        : LinearSystem(
            matrix, rhs, emptyMatrix(matrix.exec()), boundaryMatrix, boundaryRhs, strategy
        )
    {}

    LinearSystem(const LinearSystem& ls)
        : matrix_(ls.matrix_), rhs_(ls.rhs_), boundaryMatrix_(ls.boundaryMatrix_),
          offDiagonalMatrix_(ls.offDiagonalMatrix_), boundaryRhs_(ls.boundaryRhs_),
          meshIteratorContext_(ls.meshIteratorContext_)
#ifdef NF_WITH_MPI_SUPPORT
          ,
          commPattern_(ls.commPattern_)
#endif
    {
        validate();
    }

    ~LinearSystem() = default;

    [[nodiscard]] SystemMatrixType& matrix() { return matrix_; }

    [[nodiscard]] const SystemMatrixType& matrix() const { return matrix_; }

    [[nodiscard]] BoundaryMatrixType& offDiagonalMatrix() { return offDiagonalMatrix_; }

    [[nodiscard]] const BoundaryMatrixType& offDiagonalMatrix() const { return offDiagonalMatrix_; }

    [[nodiscard]] BoundaryMatrixType& boundaryMatrix() { return boundaryMatrix_; }

    [[nodiscard]] const BoundaryMatrixType& boundaryMatrix() const { return boundaryMatrix_; }

    [[nodiscard]] Vector<RHSValueType>& rhs() { return rhs_; }

    [[nodiscard]] const Vector<RHSValueType>& rhs() const { return rhs_; }

    [[nodiscard]] Vector<RHSValueType>& boundaryRhs() { return boundaryRhs_; }

    [[nodiscard]] const Vector<RHSValueType>& boundaryRhs() const { return boundaryRhs_; }

    [[nodiscard]] LinearSystem<MatrixValueType, SystemMatrixType, BoundaryMatrixType, RHSValueType>
    copyToExecutor(Executor exec) const override
    {
        LinearSystem<MatrixValueType, SystemMatrixType, BoundaryMatrixType, RHSValueType> ls {
            matrix_.copyToExecutor(exec),
            rhs_.copyToExecutor(exec),
            offDiagonalMatrix_.copyToExecutor(exec),
            boundaryMatrix_.copyToExecutor(exec),
            boundaryRhs_.copyToExecutor(exec),
            offDiagonalMatrix_.copyToExecutor(exec)
        };
#ifdef NF_WITH_MPI_SUPPORT
        ls.commPattern_ = commPattern_;
#endif
        return ls;
    }

    void reset()
    {
        fill(matrix_.values(), zero<MatrixValueType>());
        fill(rhs_, zero<RHSValueType>());
        fill(boundaryMatrix_.values(), zero<MatrixValueType>());
        fill(boundaryRhs_, zero<RHSValueType>());
        fill(offDiagonalMatrix_.values(), zero<MatrixValueType>());
    }

    [[nodiscard]] LinearSystemView<
        RHSValueType,
        MatrixView<
            MatrixValueType,
            SparsityView<typename SystemMatrixType::MatrixSparsityType::SparsityIndexType>>>
    view() && = delete;

    [[nodiscard]] LinearSystemView<
        RHSValueType,
        MatrixView<
            MatrixValueType,
            SparsityView<typename SystemMatrixType::MatrixSparsityType::SparsityIndexType>>>
    view() const&& = delete;

    [[nodiscard]] LinearSystemView<
        RHSValueType,
        MatrixView<MatrixValueType, SparsityView<LinearSystemIndexType>>>
    view() &
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    std::shared_ptr<const FaceToMatrixAddress> faceToMatrixAddress() const
    {
        return matrix_.faceToMatrixAddress();
    }

#ifdef NF_WITH_MPI_SUPPORT
    [[nodiscard]] const CommunicationPattern& commPattern() const { return commPattern_; }
    [[nodiscard]] CommunicationPattern& commPattern() { return commPattern_; }
#endif

    [[nodiscard]] LinearSystemView<
        const RHSValueType,
        const MatrixView<MatrixValueType, SparsityView<const LinearSystemIndexType>>>
    view() const&
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    std::shared_ptr<MeshIteratorContext> getMeshIterator() { return meshIteratorContext_; }

    const Executor& exec() const { return matrix_.exec(); }

private:

    static BoundaryMatrixType emptyMatrix(const Executor& exec)
    {
        using IndexType = typename BoundaryMatrixType::MatrixSparsityType::SparsityIndexType;
        auto sp = std::make_shared<const typename BoundaryMatrixType::MatrixSparsityType>(
            Vector<IndexType>(exec, 0), Vector<IndexType>(exec, 0), Dimensions {0, 0}
        );
        return BoundaryMatrixType(Vector<MatrixValueType>(exec, 0, zero<MatrixValueType>()), sp);
    }

    // internal values
    SystemMatrixType matrix_;

    Vector<RHSValueType> rhs_;

    // boundary values
    BoundaryMatrixType boundaryMatrix_;

    // store values on boundaries that are non local
    // eg on processor boundaries
    BoundaryMatrixType offDiagonalMatrix_;

    Vector<RHSValueType> boundaryRhs_;

    Dictionary auxiliaryCoefficients_;

    std::shared_ptr<MeshIteratorContext> meshIteratorContext_ = nullptr;

#ifdef NF_WITH_MPI_SUPPORT
    CommunicationPattern commPattern_;
#endif
};

/*@brief helper function that creates a zero initialised linear system based on a given mesh
 */
template<
    typename ValueType,
    typename SystemMatrixType = CSRMatrix<ValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<ValueType, localIdx>>
LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType> createEmptyLinearSystem(
    const UnstructuredMesh& mesh,
    std::shared_ptr<MeshIterationStrategy> strategy = std::make_shared<FaceBasedIterator>()
)
{
    auto [sp, mi] =
        createSparsityPatternFaceToMatrixAddress<typename SystemMatrixType::MatrixSparsityType>(mesh
        );
    auto bSp =
        createBoundarySparsityPattern<typename BoundaryMatrixType::MatrixSparsityType>(mesh, *mi);
    const auto exec = sp->exec();
    const auto nCells = static_cast<localIdx>(mesh.nCells());
    const auto nProcFaces = static_cast<localIdx>(mesh.nProcBoundaryFaces());
    using IndexType = typename BoundaryMatrixType::MatrixSparsityType::SparsityIndexType;

    Vector<IndexType> offDiagColIdxs(exec, nProcFaces, 0);
    Vector<IndexType> offDiagRowIdxs(exec, nProcFaces, 0);

#ifdef NF_WITH_MPI_SUPPORT
    auto commPattern = computeCommunicationPattern(mesh);
    if (nProcFaces > 0)
    {
        const localIdx nBoundaryFaces = static_cast<localIdx>(mesh.nBoundaryFaces());
        const auto faceOwnersH = mesh.boundaryMesh().faceOwners().copyToHost();
        const auto faceOwnersV = faceOwnersH.view();
        Vector<IndexType> rowH(SerialExecutor {}, nProcFaces, 0);
        Vector<IndexType> colH(SerialExecutor {}, nProcFaces, 0);
        auto rowHV = rowH.view();
        auto colHV = colH.view();
        const auto globalOffset = static_cast<IndexType>(mesh.globalOffset());
        for (localIdx i = 0; i < nProcFaces; ++i)
        {
            rowHV[i] = faceOwnersV[nBoundaryFaces + i] + globalOffset;
            colHV[i] = static_cast<IndexType>(commPattern.recvIdx[static_cast<std::size_t>(i)]);
        }
        offDiagRowIdxs = rowH.copyToExecutor(exec);
        offDiagColIdxs = colH.copyToExecutor(exec);

        // TODO: find a better home for this computation (together with
        // CommunicationPattern::offDiagRowSortPerm, possibly the boundary mesh) — it is
        // sparsity/topology setup, not linear-system assembly.
        // Precompute the row-sort permutation of the off-diagonal (processor-face) entries.
        // Ginkgo's CUDA Coo::apply2 requires the non-local COO sorted by global row; computing
        // the permutation here (once, at off-diagonal creation) lets createGkoMtxDist apply it
        // instead of re-sorting on every matrix build. Stable to keep a deterministic order
        // within a row. offDiagRowSortPerm[i] = proc-face index whose value belongs at sorted i.
        std::vector<localIdx> offDiagRowSortPerm(static_cast<std::size_t>(nProcFaces));
        std::iota(offDiagRowSortPerm.begin(), offDiagRowSortPerm.end(), localIdx {0});
        std::stable_sort(
            offDiagRowSortPerm.begin(),
            offDiagRowSortPerm.end(),
            [&](localIdx a, localIdx b) { return rowHV[a] < rowHV[b]; }
        );
        commPattern.offDiagRowSortPerm = std::move(offDiagRowSortPerm);
    }
#endif

    auto offDiagSp = std::make_shared<const typename BoundaryMatrixType::MatrixSparsityType>(
        std::move(offDiagColIdxs), std::move(offDiagRowIdxs), Dimensions {nCells, nCells}
    );

    LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType> ls {
        SystemMatrixType(Vector<ValueType>(sp->exec(), sp->nnz(), zero<ValueType>()), sp, mi),
        Vector<ValueType>(sp->exec(), sp->rows(), zero<ValueType>()),
        BoundaryMatrixType(Vector<ValueType>(exec, nProcFaces, zero<ValueType>()), offDiagSp),
        BoundaryMatrixType(Vector<ValueType>(bSp->exec(), bSp->nnz(), zero<ValueType>()), bSp),
        Vector<ValueType>(bSp->exec(), bSp->nnz(), zero<ValueType>()),
        BoundaryMatrixType(Vector<ValueType>(exec, nProcFaces, zero<ValueType>()), offDiagSp),
        strategy
    };

#ifdef NF_WITH_MPI_SUPPORT
    ls.commPattern() = std::move(commPattern);
#endif

    return ls;
}

/** @brief for testing purposes, this function reverses boundary contributions previously applied to
 * the matrix diagonal and RHS for some operators (e.g., div). **/
template<typename ValueType>
inline la::LinearSystem<ValueType>
removeBoundaryContributions(const la::LinearSystem<ValueType>& lsIn)
{
    auto ls = la::LinearSystem<ValueType>(lsIn);
    auto lsView = ls.view();
    auto& matrix = lsView.matrix;
    auto& rhs = lsView.rhs;
    auto& bMatrix = lsView.boundaryMatrix;
    auto& bRhs = lsView.boundaryRhs;

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    parallelFor(
        ls.exec(),
        {0, bMatrix.values.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto celli = bMatrix.sparsity.rowOffs[facei]; // cell index stored in rowOffs
            Kokkos::atomic_add(&matrix.values[ma.diagIdx(celli)], bMatrix.values[facei]);
            Kokkos::atomic_add(&rhs[celli], bRhs[facei]);
        },
        "removeBoundaryContributions"
    );

    return ls;
}

} // namespace NeoN::la
