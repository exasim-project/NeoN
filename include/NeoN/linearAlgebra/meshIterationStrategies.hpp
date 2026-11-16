// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "NeoN/core/segmentedVector.hpp"

#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoN::la
{

/** @brief Abstract base class defining the interface for mesh iteration strategies.
 *
 * Concrete strategies (e.g. FaceBasedIterator, CellBasedIterator) differ in how they
 * traverse the mesh when assembling linear-system contributions.
 */
class MeshIterationStrategy
{

protected:
public:

    /** @brief Returns the strategy name. */
    virtual std::string name() const = 0;

    /** @brief Returns the number of iteration elements (cells or faces). */
    virtual sizeType size() const = 0;

    virtual ~MeshIterationStrategy() {}
};

/** @brief Holds and exposes the active MeshIterationStrategy.
 *
 * Acts as a strategy context: callers set one concrete strategy and then query
 * it uniformly through @ref get and @ref name without knowing the concrete type.
 */
class MeshIteratorContext
{

    std::shared_ptr<MeshIterationStrategy> strategy {};

public:

    /** @brief Replaces the active strategy. */
    void setStrategy(std::shared_ptr<MeshIterationStrategy> strategy);

    /** @brief Returns the active strategy, or nullptr if none has been set. */
    std::shared_ptr<MeshIterationStrategy> get() const;

    /** @brief Returns the name of the active strategy, or an empty string if none is set. */
    std::string name() const;
};


/** @brief Iteration strategy that traverses internal faces.
 *
 * Used when assembling operator contributions face-by-face, visiting each
 * internal face once and scattering to the owner and neighbour cells.
 */
class FaceBasedIterator : public MeshIterationStrategy
{

public:

    std::string name() const override { return "FaceBased"; }

    sizeType size() const override { return 0; }

    ~FaceBasedIterator() override {}
};


/** @brief Iteration strategy that traverses all faces, internal and boundary, in one sweep.
 *
 * Like @ref FaceBasedIterator, but the operators assemble the boundary faces in the same kernel
 * as the internal faces instead of a separate boundary kernel.
 */
class GlobalFaceBasedIterator : public MeshIterationStrategy
{

public:

    std::string name() const override { return "GlobalFaceBased"; }

    sizeType size() const override { return 0; }

    ~GlobalFaceBasedIterator() override {}
};


/** @brief Iteration strategy that traverses cells and their face stencils.
 *
 * Pre-computes a cell-centric connectivity table (@ref CellBasedData) so that
 * each cell can independently gather all face contributions during assembly,
 * which is better suited for GPU-parallel execution than the face-based approach.
 */
class CellBasedIterator : public MeshIterationStrategy
{


public:

    CellBasedIterator();

    /** @brief Pre-computed connectivity required for cell-based assembly.
     *
     * All arrays are indexed by the flat position within @ref cellFaces, i.e.
     * for cell @c c the relevant slice runs from @c cellFaces.offset(c) to
     * @c cellFaces.offset(c+1).
     */
    struct CellBasedData
    {
        CellBasedData() = delete;

        /** @brief Construct from pre-computed connectivity arrays.
         *
         * A user-declared constructor is required (rather than relying on
         * aggregate initialisation) because @c std::make_shared builds the
         * object through @c std::construct_at, which performs parenthesised
         * placement-new: @c new(p)\ CellBasedData(args...). Parenthesised
         * initialisation of aggregates (C++20 P0960) is not implemented by the
         * CUDA 12.4 nvcc/EDG front-end, so @c make_shared<CellBasedData>(...)
         * fails there with "no instance of constructor ... matches".
         *
         * The container parameters are templated so the constructor adapts to
         * however the connectivity arrays are spelled at the call site
         * (@c Vector<localIdx> or @c Vector<label>), avoiding a brittle exact
         * type match across NeoN revisions.
         */
        template<
            typename CellFaces,
            typename FaceNeighbour,
            typename FaceSign,
            typename MatrixColumnIdx>
        CellBasedData(
            localIdx sizeIn,
            CellFaces&& cellFacesIn,
            FaceNeighbour&& faceNeighbourIn,
            FaceSign&& faceSignIn,
            MatrixColumnIdx&& matrixColumnIdxIn
        )
            : size(sizeIn), cellFaces(std::forward<CellFaces>(cellFacesIn)),
              faceNeighbour(std::forward<FaceNeighbour>(faceNeighbourIn)),
              faceSign(std::forward<FaceSign>(faceSignIn)),
              matrixColumnIdx(std::forward<MatrixColumnIdx>(matrixColumnIdxIn))
        {}

        localIdx size; ///< Number of cells.

        /** @brief Cell-to-face stencil; segments correspond to cells, values are face indices. */
        SegmentedVector<localIdx, localIdx> cellFaces;

        /** @brief For each stencil entry: the cell on the other side of the face. */
        Vector<localIdx> faceNeighbour;

        // TODO scalar is unreasonable large here
        /** @brief Sign of the face contribution: +1 if this cell owns the face, -1 otherwise. */
        Vector<scalar> faceSign;

        /** @brief Flat index into the CSR matrix values array for each stencil entry. */
        Vector<localIdx> matrixColumnIdx;
    };

    std::string name() const override;

    sizeType size() const override;

    ~CellBasedIterator() override {}

    /** @brief Returns the pre-computed connectivity data, or nullptr if not yet initialised. */
    std::shared_ptr<CellBasedData> getCellBasedData();

    /** @brief Computes and stores the cell-based connectivity data.
     *
     * @tparam SparsityType      Sparsity pattern type (e.g. CsrSparsityPattern,
     * CooSparsityPattern).
     * @param mesh               The mesh providing owner/neighbour topology.
     * @param sparsityPattern    Sparsity pattern of the linear system.
     * @param faceToMatrixAddress Mapping from faces to matrix row offsets.
     */
    template<typename SparsityType>
    void setComputeCellBasedData(
        const UnstructuredMesh& mesh,
        std::shared_ptr<const SparsityType> sparsityPattern,
        std::shared_ptr<const la::FaceToMatrixAddress> faceToMatrixAddress
    );

    /** @brief Builds and returns cell-based connectivity data without storing it.
     *
     * Launches a parallel kernel that, for every cell, iterates over its face
     * stencil and records the neighbour cell, the face sign, and the matrix
     * column index for each face.
     *
     * @tparam SparsityType      Sparsity pattern type (e.g. CsrSparsityPattern,
     * CooSparsityPattern).
     * @param mesh               The mesh providing owner/neighbour topology.
     * @param sparsityPattern    Sparsity pattern of the linear system.
     * @param faceToMatrixAddress Mapping from faces to matrix row offsets.
     * @return Shared pointer to the newly allocated CellBasedData.
     */
    template<typename SparsityType>
    std::shared_ptr<CellBasedData> computeCellBasedData(
        const UnstructuredMesh& mesh,
        std::shared_ptr<const SparsityType> sparsityPattern,
        const std::shared_ptr<const la::FaceToMatrixAddress> faceToMatrixAddress
    );

    std::shared_ptr<CellBasedData> cellBasedIteratorData_;
};

/** @brief Iteration strategy that traverses the internal faces batch by batch.
 *
 * Pre-computes the batches from an edge colouring of the cell graph (@ref BatchedFaceData): no
 * two faces of one cell share a batch, so the faces of one batch can be assembled in parallel
 * without atomic updates of the diagonal. Boundary faces are not batched: they see a single cell
 * and are assembled like in the face-based strategy. The iteration elements are the batches.
 */
class BatchedFaceIterator : public MeshIterationStrategy
{
public:

    BatchedFaceIterator();

    /** @brief Pre-computed face batches required for the batched face-based assembly.
     *
     * The segments of a SegmentedVector are the batches, its values the face indices of that
     * batch in ascending face order.
     */
    struct BatchedFaceData
    {
        BatchedFaceData() = delete;

        /** @brief Construct from pre-computed batch segments (user-declared for the same nvcc
         *  reason as CellBasedData). */
        template<typename Faces>
        BatchedFaceData(localIdx nBatchesIn, Faces&& batchFacesIn)
            : nBatches(nBatchesIn), batchFaces(std::forward<Faces>(batchFacesIn))
        {}

        localIdx nBatches;                              ///< Number of internal-face batches.
        SegmentedVector<localIdx, localIdx> batchFaces; ///< Internal faces by batch.
    };

    std::string name() const override;

    /** @brief Returns the number of internal-face batches. */
    sizeType size() const override;

    ~BatchedFaceIterator() override {}

    /** @brief Returns the pre-computed batches, or nullptr if not yet initialised. */
    std::shared_ptr<BatchedFaceData> getBatchedFaceData();

    /** @brief Computes and stores the face batches of the given mesh. */
    void setComputeBatchedFaceData(const UnstructuredMesh& mesh);

    /** @brief Builds and returns the face batches without storing them.
     *
     * A sequential greedy edge colouring on the host, O(nFaces), run once per mesh: every face
     * takes the lowest colour that neither of its cells has used yet, which gives at most
     * 2 * maxCellDegree - 1 batches.
     */
    std::shared_ptr<BatchedFaceData> computeBatchedFaceData(const UnstructuredMesh& mesh);

private:

    std::shared_ptr<BatchedFaceData> batchedFaceData_;
};

/** @brief The batches of a BatchedFaceIterator, computed on first use; nullptr for any other
 * strategy. */
std::shared_ptr<const BatchedFaceIterator::BatchedFaceData> batchedFaceData(
    const std::shared_ptr<MeshIterationStrategy>& strategy, const UnstructuredMesh& mesh
);

}
