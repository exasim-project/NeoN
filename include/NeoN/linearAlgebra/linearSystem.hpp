// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"

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
    typename MatrixType = CSRMatrix<ValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<ValueType, localIdx>>
class LinearSystem
{

    void validate()
    {
        NF_ASSERT(matrix_.exec() == rhs_.exec(), "Executors are not the same");
        NF_ASSERT(matrix_.nRows() == rhs_.size(), "Matrix and RHS size mismatch");
        NF_ASSERT(
            boundaryMatrix_.nRows() == boundaryRhs_.size(), "BMatrix.nRows() != boundaryRHS.size()"
        );
    }

public:

    using LinearSystemIndexType = typename MatrixType::MatrixSparsityType::SparsityIndexType;

    LinearSystem(
        std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> faceToMatrixAddress,
        CommunicationPattern commPattern
    )
        : matrix_(
            Vector<ValueType>(
                faceToMatrixAddress->exec(), faceToMatrixAddress->localNonZeros(), zero<ValueType>()
            ),
            faceToMatrixAddress->sparsityPattern()
        ),
          nonLocalMatrix_(
              Vector<ValueType>(
                  faceToMatrixAddress->exec(),
                  faceToMatrixAddress->nonLocalNonZeros(),
                  zero<ValueType>()
              ),
              faceToMatrixAddress->nonLocalSparsityPattern()
          ),
          commPattern_(commPattern),
          rhs_(faceToMatrixAddress->exec(), faceToMatrixAddress->localRows(), zero<ValueType>()),
          boundaryMatrix_(
              Vector<ValueType>(
                  faceToMatrixAddress->exec(),
                  faceToMatrixAddress->boundaryNonZeros(),
                  zero<ValueType>()
              ),
              faceToMatrixAddress->boundarySparsityPattern()
          ),
          boundaryRhs_(
              faceToMatrixAddress->exec(),
              faceToMatrixAddress->boundaryNonZeros(),
              zero<ValueType>()
          ),
          faceToMatrixAddress_(faceToMatrixAddress)
    {
        validate();
    }

    LinearSystem(
        const MatrixType& matrix,
        const BoundaryMatrixType& nonLocalMatrix,
        const CommunicationPattern commPattern,
        const Vector<ValueType>& rhs,
        const BoundaryMatrixType& boundaryMatrix,
        const Vector<ValueType>& boundaryRhs,
        std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> mi
    )
        : matrix_(matrix), nonLocalMatrix_(nonLocalMatrix), commPattern_(commPattern), rhs_(rhs),
          boundaryMatrix_(boundaryMatrix), boundaryRhs_(boundaryRhs), faceToMatrixAddress_(mi)
    {
        validate();
    }

    LinearSystem(const LinearSystem& ls)
        : matrix_(ls.matrix_), nonLocalMatrix_(ls.nonLocalMatrix_), commPattern_(ls.commPattern_),
          rhs_(ls.rhs_), boundaryMatrix_(ls.boundaryMatrix_), boundaryRhs_(ls.boundaryRhs_),
          faceToMatrixAddress_(ls.faceToMatrixAddress_)
    {}

    ~LinearSystem() = default;

    [[nodiscard]] MatrixType& matrix() { return matrix_; }

    [[nodiscard]] const MatrixType& matrix() const { return matrix_; }

    [[nodiscard]] BoundaryMatrixType& boundaryMatrix() { return boundaryMatrix_; }

    [[nodiscard]] const BoundaryMatrixType& nonLocalMatrix() const { return nonLocalMatrix_; }

    [[nodiscard]] BoundaryMatrixType& nonLocalMatrix() { return nonLocalMatrix_; }

    [[nodiscard]] const BoundaryMatrixType& boundaryMatrix() const { return boundaryMatrix_; }

    [[nodiscard]] Vector<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const Vector<ValueType>& rhs() const { return rhs_; }

    [[nodiscard]] Vector<ValueType>& boundaryRhs() { return boundaryRhs_; }

    [[nodiscard]] const Vector<ValueType>& boundaryRhs() const { return boundaryRhs_; }

    [[nodiscard]] LinearSystem<ValueType, MatrixType> copyToHost() const
    {
        if (faceToMatrixAddress_ == nullptr)
        {
            return {
                matrix_.copyToHost(),
                nonLocalMatrix_.copyToHost(),
                commPattern_,
                rhs_.copyToHost(),
                boundaryMatrix_.copyToHost(),
                boundaryRhs_.copyToHost(),
                {}
            };
        }
        auto mi = std::make_shared<FaceToMatrixAddress<LinearSystemIndexType>>(
            faceToMatrixAddress_->ownerOffset().copyToHost(),
            faceToMatrixAddress_->neighbourOffset().copyToHost(),
            faceToMatrixAddress_->diagOffset().copyToHost(),
            std::make_shared<SparsityPattern<LinearSystemIndexType>>(
                faceToMatrixAddress_->sparsityPattern()->copyToHost()
            ),
            std::make_shared<CooSparsityPattern<LinearSystemIndexType>>(
                faceToMatrixAddress_->nonLocalSparsityPattern()->copyToHost()
            ),
            std::make_shared<CooSparsityPattern<LinearSystemIndexType>>(
                faceToMatrixAddress_->boundarySparsityPattern()->copyToHost()
            )
        );
        return {
            matrix_.copyToHost(),
            nonLocalMatrix_.copyToHost(),
            commPattern_,
            rhs_.copyToHost(),
            boundaryMatrix_.copyToHost(),
            boundaryRhs_.copyToHost(),
            mi
        };
    }

    /** @brief boundaryMatrixMap - bfaceIdx -> matrixAddr */
    void communicate(CommunicationPattern& commPattern)
    {
        auto mpiEnv = commPattern.env;
        int commRanks = mpiEnv.sizeRank();

        // auto boundaryMatrixMap = Vector<localIdx>(exec(), commPattern.boundaryMapVector);
        auto nsp = faceToMatrixAddress_->nonLocalSparsityPattern();
        auto rowToDiagonalMap = la::computeRowToDiagonalMap(nsp->rowOffs(), faceToMatrixAddress_);

        // 1. copy bValues which need to be communicated into sendBuffer
        auto commSize = commPattern.sendCounts[mpiEnv.sizeRank()];
        auto recvBuffer = Vector<ValueType>(exec(), commSize);

        // TODO compute using scan
        auto sdispls = std::vector<int>(commRanks, 0);
        for (int i = 1; i < sdispls.size(); i++)
        {
            auto prev = sdispls[i - 1];
            sdispls[i] = commPattern.sendCounts[i - 1] + prev;
        }

        MPI_Alltoallv(
            nonLocalMatrix_.values().data(),
            commPattern.sendCounts.data(),
            sdispls.data(),
            mpi::getType<ValueType>(),
            recvBuffer.data(),
            commPattern.sendCounts.data(),
            sdispls.data(),
            mpi::getType<ValueType>(),
            mpiEnv.comm()
        );

        std::cout << __FILE__ << ":" << __LINE__ << " rank " << mpiEnv.rank() << " recvBuffer "
                  << recvBuffer.view()[0] << " matrixValue " << matrix_.values().view()[9] << "\n";

        // 3. apply received values to corresponding matrix
        // add diagonal contributions
        add(recvBuffer, rowToDiagonalMap, matrix_.values());
        std::cout << __FILE__ << ":" << __LINE__ << " rank " << mpiEnv.rank() << " recvBuffer "
                  << recvBuffer.view()[0] << " matrixValue " << matrix_.values().view()[9] << "\n";
    }

    // FIXME needed?
    void reset()
    {
        matrix_.reset();
        boundaryMatrix_.reset();
        nonLocalMatrix_.reset();
        fill(rhs_, zero<ValueType>());
        fill(boundaryRhs_, zero<ValueType>());
    }

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<typename MatrixType::MatrixSparsityType>>>
    view() && = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<typename MatrixType::MatrixSparsityType>>>
    view() const&& = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<LinearSystemIndexType>>>
    view() &
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> faceToMatrixAddress() const
    {
        return faceToMatrixAddress_;
    }

    [[nodiscard]] LinearSystemView<
        const ValueType,
        const MatrixView<ValueType, SparsityView<const LinearSystemIndexType>>>
    view() const&
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    const Executor& exec() const { return matrix_.exec(); }

    const CommunicationPattern& commPattern() const { return commPattern_; }

private:

    // internal values
    MatrixType matrix_;

    // store values on boundaries that are non local
    // eg on processor boundaries
    BoundaryMatrixType nonLocalMatrix_;

    CommunicationPattern commPattern_;

    Vector<ValueType> rhs_;

    // store values on boundaries that are non local
    BoundaryMatrixType boundaryMatrix_;

    Vector<ValueType> boundaryRhs_;

    Dictionary auxiliaryCoefficients_;

    std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> faceToMatrixAddress_;
};

/*@brief helper function that creates a zero initialised linear system based on a given mesh
 */
template<
    typename ValueType,
    typename InnerMatrixType = CSRMatrix<ValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<ValueType, localIdx>>
LinearSystem<ValueType, InnerMatrixType, BoundaryMatrixType>
createEmptyLinearSystem(const UnstructuredMesh& mesh)
{
    auto [faceToMatrixAddr, commPattern] =
        createSparsityPatternFaceToMatrixAddress<NeoN::localIdx>(mesh);
    return LinearSystem<ValueType, InnerMatrixType, BoundaryMatrixType>(
        faceToMatrixAddr, commPattern
    );
}

/** @brief for testing purposes, this function reverses boundary contributions previously applied to
 * the matrix diagonal and RHS for some operators (e.g., div). **/
template<typename ValueType>
inline la::LinearSystem<ValueType>
removeBoundaryContributions(const la::LinearSystem<ValueType>& lsIn)
{
    auto ls = la::LinearSystem<ValueType>(lsIn);
    const auto matIt = ls.faceToMatrixAddress();
    auto lsView = ls.view();
    auto& matrix = lsView.matrix;
    auto& rhs = lsView.rhs;
    auto& bMatrix = lsView.boundaryMatrix;
    auto& bRhs = lsView.boundaryRhs;
    auto& mtx = ls.matrix();
    const auto [diagOffs, rowOffs] = views(matIt->diagOffset(), mtx.rowOffs());

    // Boundary matrix
    parallelFor(
        ls.exec(),
        {0, bMatrix.values.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto celli = bMatrix.sparsity.rowOffs[facei]; // cell index stored in rowOffs
            Kokkos::atomic_add(
                &matrix.values[rowOffs[celli] + diagOffs[celli]], bMatrix.values[facei]
            );
            Kokkos::atomic_add(&rhs[celli], bRhs[facei]);
        },
        "removeBoundaryContributions"
    );

    return ls;
}

inline void scaledInvDiagNegLUx(
    const la::LinearSystem<Vec3>& ls, // In,
    const Vector<Vec3>& a,
    const Vector<Vec3>& aBound,
    const UnstructuredMesh& mesh,
    Vector<scalar>& rAU,
    Vector<Vec3>& out
)
{
    // auto ls = removeBoundaryContributions(lsIn);
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
    const auto [nonLocalMtx, nonLocalRows] =
        views(ls.nonLocalMatrix().values(), ls.nonLocalMatrix().rowOffs());
    auto outV = out.view();

    auto procFacesStart = bMesh.nBoundaryFaces();

    // localIdx curNonLocalRow = 0;
    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            // local mtx first
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

            // check
            // FIXME this scans everytime all boundary values
            localIdx curNonLocalRow = 0;
            auto nonLocalVal = zero<Vec3>();
            for (auto i = 0; i < nonLocalRows.size(); i++)
            {
                if (nonLocalRows[i] == rowi)
                {
                    nonLocalVal = nonLocalMtx[i];
                    curNonLocalRow = i;
                    // FIXME is the sign correct?
                    outV[rowi] -= nonLocalVal * aBoundV[procFacesStart + curNonLocalRow];
                    // curNonLocalRow = 0;
                    // break;
                }
            }

            outV[rowi] += bV[rowi];
            outV[rowi] *= rAUV[rowi] / volV[rowi];
        }
    );
}

} // namespace NeoN::la
