// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/segmentedVector.hpp"

#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoN::la
{


class MeshIterationStrategy
{

protected:
public:

    virtual std::string name() const = 0;

    virtual size_t size() const = 0;

    virtual ~MeshIterationStrategy() {}
};

class MeshIteratorContext
{

    std::shared_ptr<MeshIterationStrategy> strategy {};

public:

    void setStrategy(std::shared_ptr<MeshIterationStrategy> strategy_)
    {
        strategy = std::move(strategy_);
    }

    /**@brief get underlying strategy*/
    std::shared_ptr<MeshIterationStrategy> get() const
    {
        if (strategy)
        {
            return strategy;
        }
    }

    std::string name() const
    {
        if (strategy)
        {
            return strategy->name();
        }
    }
};


// template<typename IndexType = localIdx, typename MeshType = UnstructuredMesh>
class FaceBasedIterator : public MeshIterationStrategy
{

public:

    std::string name() const override { return "FaceBased"; }

    // FIXME
    size_t size() const override { return 0; };

    ~FaceBasedIterator() override {}
};

// FIXME add template again
// template<typename IndexType = localIdx, typename MeshType = UnstructuredMesh>
class CellBasedIterator : public MeshIterationStrategy
{


public:

    CellBasedIterator() : cellBasedIteratorData_(nullptr) {}

    struct CellBasedData
    {
        localIdx size;

        // provide mapping between face and cell idx
        // faces per cell as offsets
        SegmentedVector<localIdx, localIdx> cellFaces;

        // mapping between face id and neighbour cell id
        Vector<localIdx> faceNeighbour;

        // TODO scalar is unreasonable large here
        // sign for face contribution (+1 if owner, -1 if neighbour)
        Vector<scalar> faceSign;

        // maps "row start + cell local face idx" to matrix address
        Vector<localIdx> matrixColumnIdx;
    };

    std::string name() const override { return "CellBased"; }

    size_t size() const override { return cellBasedIteratorData_->size; };

    ~CellBasedIterator() override {}

    std::shared_ptr<CellBasedData> getCellBasedData() { return cellBasedIteratorData_; }

    void setComputeCellBasedData(
        const UnstructuredMesh& mesh,
        std::shared_ptr<const la::FaceToMatrixAddress<localIdx>> faceToMatrixAddress
    )
    {
        cellBasedIteratorData_ = computeCellBasedData(mesh, faceToMatrixAddress);
    }


    // // Function to pre-compute cell-based connectivity
    std::shared_ptr<CellBasedData> computeCellBasedData(
        const UnstructuredMesh& mesh,
        const std::shared_ptr<const la::FaceToMatrixAddress<localIdx>> faceToMatrixAddress
    )
    {
        // Count faces per cell (owner and neighbour contributions)
        auto exec = mesh.exec();
        auto cellToFaceStencil = finiteVolume::cellCentred::CellToFaceStencil(mesh);
        auto cellFaces = cellToFaceStencil.computeStencil();
        auto owner = mesh.faceOwner();
        auto neighbour = mesh.faceNeighbour();

        // Pre-compute face neighbour, sign, and matrix column indices
        const auto totalFaceConnections = cellFaces.size();
        Vector<localIdx> faceNeighbour(exec, totalFaceConnections);
        Vector<scalar> faceSign(exec, totalFaceConnections);
        Vector<localIdx> matrixColumnIdx(exec, totalFaceConnections);

        const auto [diagOffs, ownOffs, neiOffs] = views(
            faceToMatrixAddress->diagOffset(),
            faceToMatrixAddress->ownerOffset(),
            faceToMatrixAddress->neighbourOffset()
        );

        const auto [ownerV, neighbourV] = views(owner, neighbour);
        const auto [matrixRowOffs] = views(faceToMatrixAddress->sparsityPattern()->rowOffs());

        auto [faceNeighbourV, faceSignV, matrixColumnIdxV] =
            views(faceNeighbour, faceSign, matrixColumnIdx);

        auto [cellFacesValues, cellFacesSegments] = cellFaces.views();

        parallelFor(
            exec,
            {0, mesh.nCells()},
            NEON_LAMBDA(const localIdx celli) {
                const auto faces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
                const auto startIdx = cellFacesSegments[celli];

                for (localIdx i = 0; i < faces; ++i)
                {
                    const auto faceIdx = cellFacesValues[startIdx + i];
                    const auto own = ownerV[faceIdx];
                    const auto nei = neighbourV[faceIdx];

                    const auto isOwner = (own == celli);
                    const auto neiCell = isOwner ? nei : own;

                    faceNeighbourV[startIdx + i] = neiCell;
                    faceSignV[startIdx + i] = isOwner ? 1.0 : -1.0;

                    // Compute matrix column index
                    const auto rowStart = matrixRowOffs[celli];
                    if (isOwner)
                    {
                        matrixColumnIdxV[startIdx + i] = rowStart + ownOffs[faceIdx];
                    }
                    else
                    {
                        matrixColumnIdxV[startIdx + i] = rowStart + neiOffs[faceIdx];
                    }
                }
            },
            "computeFaceData"
        );

        return std::make_shared<CellBasedData>(
            mesh.nCells(), cellFaces, faceNeighbour, faceSign, matrixColumnIdx
        );
    }

    std::shared_ptr<CellBasedData> cellBasedIteratorData_;
};

}
