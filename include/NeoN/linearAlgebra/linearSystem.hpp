// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"

#include <memory>
#include <optional>
#include <string>

namespace NeoN::la
{

/**
 * @struct LinearSystemView
 * @brief A view linear into a linear system's data.
 *
 * @tparam ValueType The value type of the linear system.
 * @tparam MatrixViewType The type representing the matrix view
 */
template<typename ValueType, typename MatrixViewType>
struct LinearSystemView
{
    LinearSystemView() = default;
    ~LinearSystemView() = default;

    LinearSystemView(
        MatrixViewType matrixView,
        View<ValueType> rhsView,
        MatrixViewType boundaryMatrixView,
        View<ValueType> boundaryRhsView
    )
        : matrix(matrixView), rhs(rhsView), boundaryMatrix(boundaryMatrixView),
          boundaryRhs(boundaryRhsView) {};

    MatrixViewType matrix;
    View<ValueType> rhs;

    MatrixViewType boundaryMatrix;
    View<ValueType> boundaryRhs;
};

/**
 * @class LinearSystem
 * @brief A class representing a linear system of equations.
 *
 * The LinearSystem class provides functionality to store and manipulate a linear system of
 * equations. It supports the storage of the coefficient matrix and the right-hand side vector, as
 * well as the solution vector.
 */
template<
    typename ValueType,
    typename SystemMatrixType = CSRMatrix<ValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<ValueType, localIdx>>
class LinearSystem
{

    void validate()
    {
        NF_ASSERT(matrix_.exec() == rhs_.exec(), "Executors are not the same");
        NF_ASSERT(matrix_.nRows() == rhs_.size(), "Matrix and RHS size mismatch");
        // NF_ASSERT(
        //     boundaryMatrix_.nRows() == boundaryRhs_.size(), "BMatrix.nRows() !=
        //     boundaryRHS.size()"
        // );
    }

public:

    using LinearSystemIndexType = typename SystemMatrixType::MatrixSparsityType::SparsityIndexType;

    LinearSystem(
        const SystemMatrixType& matrix,
        const Vector<ValueType>& rhs,
        const BoundaryMatrixType& boundaryMatrix,
        const Vector<ValueType>& boundaryRhs
    )
        : matrix_(matrix), rhs_(rhs), boundaryMatrix_(boundaryMatrix), boundaryRhs_(boundaryRhs)
    {
        validate();
    }

    /* @brief Distributed-aware ctor that also takes the non-local matrix
     * (proc-boundary off-diagonals) and the corresponding CommunicationPattern.
     *
     * For non-distributed runs the non-distributed 4-arg ctor above keeps
     * working; this overload adds the additional distributed coupling. */
    LinearSystem(
        const SystemMatrixType& matrix,
        const Vector<ValueType>& rhs,
        const BoundaryMatrixType& boundaryMatrix,
        const Vector<ValueType>& boundaryRhs,
        std::shared_ptr<BoundaryMatrixType> nonLocalMatrix,
        CommunicationPattern commPattern
    )
        : matrix_(matrix), rhs_(rhs), boundaryMatrix_(boundaryMatrix), boundaryRhs_(boundaryRhs),
          nonLocalMatrix_(std::move(nonLocalMatrix)),
          commPattern_(std::optional<CommunicationPattern> {std::move(commPattern)})
    {
        validate();
    }

    LinearSystem(const LinearSystem& ls)
        : matrix_(ls.matrix_), rhs_(ls.rhs_), boundaryMatrix_(ls.boundaryMatrix_),
          boundaryRhs_(ls.boundaryRhs_), nonLocalMatrix_(ls.nonLocalMatrix_),
          commPattern_(ls.commPattern_)
    {}

    ~LinearSystem() = default;

    [[nodiscard]] SystemMatrixType& matrix() { return matrix_; }

    [[nodiscard]] const SystemMatrixType& matrix() const { return matrix_; }

    [[nodiscard]] BoundaryMatrixType& boundaryMatrix() { return boundaryMatrix_; }

    [[nodiscard]] const BoundaryMatrixType& boundaryMatrix() const { return boundaryMatrix_; }

    [[nodiscard]] Vector<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const Vector<ValueType>& rhs() const { return rhs_; }

    [[nodiscard]] Vector<ValueType>& boundaryRhs() { return boundaryRhs_; }

    [[nodiscard]] const Vector<ValueType>& boundaryRhs() const { return boundaryRhs_; }

    /* @brief return the non-local (proc-boundary) coupling matrix.
     *
     * In serial / non-distributed runs nonLocalMatrix_ is a nullptr-equivalent
     * (default-constructed shared_ptr). Callers that may run in non-distributed
     * mode should gate on hasNonLocalMatrix() before dereferencing. */
    [[nodiscard]] BoundaryMatrixType& nonLocalMatrix() { return *nonLocalMatrix_; }

    [[nodiscard]] const BoundaryMatrixType& nonLocalMatrix() const { return *nonLocalMatrix_; }

    [[nodiscard]] bool hasNonLocalMatrix() const { return static_cast<bool>(nonLocalMatrix_); }

    /* @brief return the CommunicationPattern associated with this system.
     *
     * Returns a default-constructed (empty) CommunicationPattern when the
     * system is non-distributed. Solver dispatch (la::Solver) inspects
     * commPattern().sendCounts.size() to choose between solve / solveDist. */
    [[nodiscard]] CommunicationPattern commPattern() const
    {
        if (commPattern_)
        {
            return *commPattern_;
        }
        return CommunicationPattern {};
    }

    [[nodiscard]] LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType> copyToHost() const
    {
        if (nonLocalMatrix_)
        {
            return LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType>(
                matrix_.copyToHost(),
                rhs_.copyToHost(),
                boundaryMatrix_.copyToHost(),
                boundaryRhs_.copyToHost(),
                std::make_shared<BoundaryMatrixType>(nonLocalMatrix_->copyToHost()),
                commPattern_ ? *commPattern_ : CommunicationPattern {}
            );
        }
        return {
            matrix_.copyToHost(),
            rhs_.copyToHost(),
            boundaryMatrix_.copyToHost(),
            boundaryRhs_.copyToHost()
        };
    }

    void reset()
    {
        fill(matrix_.values(), zero<ValueType>());
        fill(rhs_, zero<ValueType>());
        fill(boundaryMatrix_.values(), zero<ValueType>());
        fill(boundaryRhs_, zero<ValueType>());
        if (nonLocalMatrix_)
        {
            fill(nonLocalMatrix_->values(), zero<ValueType>());
        }
    }

    /* @brief Communicate the non-local matrix values across procs and fold the
     * received off-diagonal coefficients into the local matrix diagonal.
     *
     * Folds processor-patch coupling into a fixed-value Dirichlet against the
     * exchanged ghost cell value of the unknown. Mirrors the pre-merge
     * LinearSystem::communicate() implementation. */
    void communicate(const CommunicationPattern& commPattern)
    {
        if (!nonLocalMatrix_)
        {
            return;
        }
        auto mpiEnv = commPattern.env;
        int commRanks = static_cast<int>(mpiEnv.sizeRank());

        // Build a per-nnz map from non-local row index to the local-CSR diagonal
        // value-array index. Uses the FaceToMatrixAddress borrowed by the local
        // matrix.
        const auto& rowIdxs = nonLocalMatrix_->sparsity()->rowIdxs();
        auto ftma = matrix_.faceToMatrixAddress();
        NF_ASSERT(ftma, "communicate(): LinearSystem.matrix has no FaceToMatrixAddress");
        const auto rowOffsLocalView = matrix_.sparsity()->rowOffs().view();

        Vector<localIdx> rowToDiagonalMap(exec(), rowIdxs.size());
        {
            auto retV = rowToDiagonalMap.view();
            const auto rowsV = rowIdxs.view();
            const auto diagOffsV = ftma->diagOffset().view();
            parallelFor(
                exec(),
                {0, rowToDiagonalMap.size()},
                KOKKOS_LAMBDA(const localIdx i) {
                    localIdx cell = rowsV[i];
                    retV[i] = rowOffsLocalView[cell] + diagOffsV[cell];
                },
                "LinearSystem::communicate::rowToDiagonalMap"
            );
        }

        auto commSize =
            static_cast<localIdx>(commPattern.sendCounts[static_cast<std::size_t>(commRanks)]);
        auto recvBuffer = Vector<ValueType>(exec(), commSize);

        auto sdispls = std::vector<int>(static_cast<std::size_t>(commRanks), 0);
        for (int i = 1; i < commRanks; i++)
        {
            sdispls[static_cast<std::size_t>(i)] =
                commPattern.sendCounts[static_cast<std::size_t>(i - 1)]
                + sdispls[static_cast<std::size_t>(i - 1)];
        }

        MPI_Alltoallv(
            nonLocalMatrix_->values().data(),
            commPattern.sendCounts.data(),
            sdispls.data(),
            mpi::getType<ValueType>(),
            recvBuffer.data(),
            commPattern.sendCounts.data(),
            sdispls.data(),
            mpi::getType<ValueType>(),
            mpiEnv.comm()
        );

        // Add received non-local coefficients into the local matrix diagonal.
        add(recvBuffer, rowToDiagonalMap, matrix_.values());
    }

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<
            ValueType,
            SparsityView<typename SystemMatrixType::MatrixSparsityType::SparsityIndexType>>>
    view() && = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<
            ValueType,
            SparsityView<typename SystemMatrixType::MatrixSparsityType::SparsityIndexType>>>
    view() const&& = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<LinearSystemIndexType>>>
    view() &
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    std::shared_ptr<const FaceToMatrixAddress> faceToMatrixAddress() const
    {
        return matrix_.faceToMatrixAddress();
    }

    [[nodiscard]] LinearSystemView<
        const ValueType,
        const MatrixView<ValueType, SparsityView<const LinearSystemIndexType>>>
    view() const&
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    const Executor& exec() const { return matrix_.exec(); }

private:

    // internal values
    SystemMatrixType matrix_;

    Vector<ValueType> rhs_;

    // boundary values
    BoundaryMatrixType boundaryMatrix_;

    Vector<ValueType> boundaryRhs_;

    // proc-boundary coupling (off-diagonals towards ghost cells on other ranks).
    // Optional via shared_ptr — empty for non-distributed runs.
    std::shared_ptr<BoundaryMatrixType> nonLocalMatrix_;

    // communication pattern matching nonLocalMatrix_. Empty optional in serial.
    std::optional<CommunicationPattern> commPattern_;

    Dictionary auxiliaryCoefficients_;
};

/*@brief helper function that creates a zero initialised linear system based on a given mesh
 *
 * In serial mode this constructs a non-distributed LinearSystem (4-arg ctor).
 * For distributed meshes (boundaryMesh().isDistributed() == true) this also
 * builds the proc-boundary (non-local) matrix and CommunicationPattern via
 * createDistributedSparsityPattern, and uses the distributed LinearSystem
 * ctor.
 */
template<
    typename ValueType,
    typename SystemMatrixType = CSRMatrix<ValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<ValueType, localIdx>>
LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType>
createEmptyLinearSystem(const UnstructuredMesh& mesh)
{
    if (mesh.boundaryMesh().isDistributed())
    {
        auto [systemSp, ftma, nonLocalSp, bSp, commPattern] = createDistributedSparsityPattern<
            typename SystemMatrixType::MatrixSparsityType,
            typename BoundaryMatrixType::MatrixSparsityType>(mesh);

        SystemMatrixType matrix(
            Vector<ValueType>(systemSp->exec(), systemSp->nnz(), zero<ValueType>()), systemSp, ftma
        );
        Vector<ValueType> rhs(systemSp->exec(), systemSp->rows(), zero<ValueType>());
        BoundaryMatrixType boundaryMatrix(
            Vector<ValueType>(bSp->exec(), bSp->nnz(), zero<ValueType>()), bSp
        );
        Vector<ValueType> boundaryRhs(bSp->exec(), bSp->nnz(), zero<ValueType>());

        auto nonLocalMatrix = std::make_shared<BoundaryMatrixType>(
            Vector<ValueType>(nonLocalSp->exec(), nonLocalSp->nnz(), zero<ValueType>()), nonLocalSp
        );

        return LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType>(
            matrix, rhs, boundaryMatrix, boundaryRhs, nonLocalMatrix, commPattern
        );
    }

    auto [systemSp, ftma] =
        createSparsityPatternFaceToMatrixAddress<typename SystemMatrixType::MatrixSparsityType>(mesh
        );
    auto bSp =
        createBoundarySparsityPattern<typename BoundaryMatrixType::MatrixSparsityType>(mesh, *ftma);
    return {
        SystemMatrixType(
            Vector<ValueType>(systemSp->exec(), systemSp->nnz(), zero<ValueType>()), systemSp, ftma
        ),
        Vector<ValueType>(systemSp->exec(), systemSp->rows(), zero<ValueType>()),
        BoundaryMatrixType(Vector<ValueType>(bSp->exec(), bSp->nnz(), zero<ValueType>()), bSp),
        Vector<ValueType>(bSp->exec(), bSp->nnz(), zero<ValueType>())
    };
}

/** @brief for testing purposes, this function reverses boundary contributions previously applied to
 * the matrix diagonal and RHS for some operators (e.g., div).
 *
 * Distributed-aware: when the input LinearSystem carries a non-local matrix
 * (proc-boundary coupling) the raw non-local off-diagonal values are also
 * folded back into the local diagonal. The non-local RHS contribution
 * (−D_f * x_G) requires the ghost values; see the 2-arg overload below.
 **/
template<typename ValueType>
inline la::LinearSystem<ValueType>
removeBoundaryContributions(const la::LinearSystem<ValueType>& lsIn)
{
    auto ls = la::LinearSystem<ValueType>(lsIn);
    auto lsView = ls.view();
    auto& matrix = lsView.matrix;
    auto& rhs = lsView.rhs;
    auto& bRhs = lsView.boundaryRhs;

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    // boundaryMatrix uses COO sparsity: per-nnz row indices live in rowIdxs().
    const auto bRowIdxs = ls.boundaryMatrix().sparsity()->rowIdxs().view();
    const auto bValuesV = ls.boundaryMatrix().values().view();

    parallelFor(
        ls.exec(),
        {0, bValuesV.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto celli = bRowIdxs[facei];
            Kokkos::atomic_add(&matrix.values[ma.diagIdx(celli)], bValuesV[facei]);
            Kokkos::atomic_add(&rhs[celli], bRhs[facei]);
        },
        "removeBoundaryContributions"
    );

    if (ls.hasNonLocalMatrix())
    {
        const auto nlRowIdxs = ls.nonLocalMatrix().sparsity()->rowIdxs().view();
        const auto nlValuesV = ls.nonLocalMatrix().values().view();
        parallelFor(
            ls.exec(),
            {0, nlValuesV.size()},
            NEON_LAMBDA(const localIdx facei) {
                const auto celli = nlRowIdxs[facei];
                Kokkos::atomic_add(&matrix.values[ma.diagIdx(celli)], nlValuesV[facei]);
                // Non-local RHS subtraction (-D_f * x_G) lives in the 2-arg
                // overload below — ghost values x_G must be threaded in by the
                // caller.
            },
            "removeBoundaryContributions::nonLocal"
        );
    }

    return ls;
}

/** @brief Two-arg overload of removeBoundaryContributions for distributed
 * (proc-boundary) systems.
 *
 * In addition to the single-arg behaviour (adds raw D_f back to local
 * diagonal), also subtracts the FVM non-local RHS term -D_f * x_G for each
 * proc face f with ghost cell G.
 *
 * @param lsIn                Input linear system (unchanged)
 * @param procFaceGhostValues Ghost cell values; size must equal
 *                            nonLocalMatrix.values().size(). In serial
 *                            (size == 0) the parallelFor is a no-op.
 */
template<typename ValueType>
inline la::LinearSystem<ValueType> removeBoundaryContributions(
    const la::LinearSystem<ValueType>& lsIn, const Vector<ValueType>& procFaceGhostValues
)
{
    auto ls = removeBoundaryContributions(lsIn);
    if (!ls.hasNonLocalMatrix())
    {
        return ls;
    }
    NF_ASSERT(
        ls.nonLocalMatrix().values().size() == procFaceGhostValues.size(),
        "removeBoundaryContributions two-arg: ghost values size must match nonLocalMatrix.values "
        "size"
    );

    auto lsView = ls.view();
    auto& rhs = lsView.rhs;
    const auto nlRowIdxs = ls.nonLocalMatrix().sparsity()->rowIdxs().view();
    const auto nlValuesV = ls.nonLocalMatrix().values().view();
    auto ghostV = procFaceGhostValues.view();

    parallelFor(
        ls.exec(),
        {0, nlValuesV.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto celli = nlRowIdxs[facei];
            Kokkos::atomic_sub(&rhs[celli], nlValuesV[facei] * ghostV[facei]);
        },
        "removeBoundaryContributions::nonLocalRhs"
    );
    return ls;
}

/** @brief computes out = rAU * ( -(L+U) x + b ) for distributed FV momentum
 *  step, threading the proc-boundary contribution through the non-local
 *  matrix and the exchanged ghost values aBound.
 *
 *  Re-attached from feat/gpu-distributed pre-CooSparsity (commit 1059b7d81c).
 */
inline void scaledInvDiagNegLUx(
    const la::LinearSystem<Vec3>& ls,
    const Vector<Vec3>& a,
    const Vector<Vec3>& aBound,
    const UnstructuredMesh& mesh,
    Vector<scalar>& rAU,
    Vector<Vec3>& out
)
{
    auto& mtx = ls.matrix();
    auto& vol = mesh.cellVolumes();
    auto& bMesh = mesh.boundaryMesh();
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto [rowOffsV, colIdxV, matrixV, rAUV, volV, aV, aBoundV, bV] = views(
        mtx.sparsity()->rowOffs(),
        ls.matrix().sparsity()->colIdxs(),
        mtx.values(),
        rAU,
        vol,
        a,
        aBound,
        ls.rhs()
    );

    auto outV = out.view();
    auto procFacesStart = bMesh.nBoundaryFaces();

    const bool hasNonLocal = ls.hasNonLocalMatrix();
    // We deliberately read these regardless of hasNonLocal (default-constructed
    // views are zero-size and the kernel guards on nonLocalRows.size()).
    View<const Vec3> nonLocalMtx {};
    View<const localIdx> nonLocalRows {};
    if (hasNonLocal)
    {
        nonLocalMtx = ls.nonLocalMatrix().values().view();
        nonLocalRows = ls.nonLocalMatrix().sparsity()->rowIdxs().view();
    }

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            outV[rowi] = zero<Vec3>();
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                auto colI = colIdxV[i];
                if (rowi == colI)
                {
                    rAUV[rowi] = volV[rowi] / matrixV[i][0];
                }
                else
                {
                    outV[rowi] -= matrixV[i] * aV[colI];
                }
            }

            // FIXME this scans every non-local entry per row. With small
            // proc-boundary widths (typical for partition cuts) this is
            // tolerable; revisit if profiling shows it.
            localIdx curNonLocalRow = 0;
            auto nonLocalVal = zero<Vec3>();
            for (auto i = 0; i < nonLocalRows.size(); i++)
            {
                if (nonLocalRows[i] == rowi)
                {
                    nonLocalVal = nonLocalMtx[i];
                    curNonLocalRow = i;
                    outV[rowi] -= nonLocalVal * aBoundV[procFacesStart + curNonLocalRow];
                }
            }

            outV[rowi] += bV[rowi];
            outV[rowi] *= rAUV[rowi] / volV[rowi];
        }
    );
}

} // namespace NeoN::la
